#include <cuda_runtime.h>

extern "C" __global__ void reduce_weighted_landmarks(
    const float* keys,
    const float* values,
    const float* weights,
    const int* span_offsets,
    float* out_keys,
    float* out_values,
    int dim
) {
    int landmark_id = blockIdx.x;
    int lane = threadIdx.x;
    int start = span_offsets[landmark_id];
    int stop = span_offsets[landmark_id + 1];

    for (int d = lane; d < dim; d += blockDim.x) {
        float key_acc = 0.0f;
        float value_acc = 0.0f;
        float weight_sum = 0.0f;
        for (int token = start; token < stop; ++token) {
            float w = weights[token];
            key_acc += w * keys[token * dim + d];
            value_acc += w * values[token * dim + d];
            weight_sum += w;
        }
        if (weight_sum > 0.0f) {
            out_keys[landmark_id * dim + d] = key_acc / weight_sum;
            out_values[landmark_id * dim + d] = value_acc / weight_sum;
        } else {
            out_keys[landmark_id * dim + d] = 0.0f;
            out_values[landmark_id * dim + d] = 0.0f;
        }
    }
}

extern "C" __global__ void fused_landmark_attention(
    const float* queries,
    const float* keys,
    const float* values,
    float* output,
    int num_queries,
    int num_tokens,
    int dim,
    float scale
) {
    int qid = blockIdx.x;
    int lane = threadIdx.x;

    if (qid >= num_queries) {
        return;
    }

    extern __shared__ float scratch[];
    float* scores = scratch;

    for (int token = lane; token < num_tokens; token += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < dim; ++d) {
            dot += queries[qid * dim + d] * keys[token * dim + d];
        }
        scores[token] = dot * scale;
    }
    __syncthreads();

    if (lane == 0) {
        float max_score = scores[0];
        for (int token = 1; token < num_tokens; ++token) {
            if (scores[token] > max_score) {
                max_score = scores[token];
            }
        }

        float denom = 0.0f;
        for (int token = 0; token < num_tokens; ++token) {
            scores[token] = expf(scores[token] - max_score);
            denom += scores[token];
        }

        for (int d = 0; d < dim; ++d) {
            float acc = 0.0f;
            for (int token = 0; token < num_tokens; ++token) {
                float weight = scores[token] / denom;
                acc += weight * values[token * dim + d];
            }
            output[qid * dim + d] = acc;
        }
    }
}
