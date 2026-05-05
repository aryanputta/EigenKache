#include <cmath>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

extern "C" __global__ void reduce_weighted_landmarks(
    const float* keys,
    const float* values,
    const float* weights,
    const int* span_offsets,
    float* out_keys,
    float* out_values,
    int dim
);

namespace {

bool check_cuda(cudaError_t status, const char* label) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA error at " << label << ": " << cudaGetErrorString(status) << "\n";
        return false;
    }
    return true;
}

}  // namespace

int main() {
    constexpr int tokens = 4;
    constexpr int dim = 2;
    constexpr int landmarks = 2;

    const std::vector<float> h_keys = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        10.0f, 20.0f,
        30.0f, 40.0f,
    };
    const std::vector<float> h_values = h_keys;
    const std::vector<float> h_weights = {1.0f, 1.0f, 1.0f, 3.0f};
    const std::vector<int> h_offsets = {0, 2, 4};

    float *d_keys = nullptr, *d_values = nullptr, *d_weights = nullptr, *d_out_keys = nullptr, *d_out_values = nullptr;
    int* d_offsets = nullptr;
    if (!check_cuda(cudaMalloc(&d_keys, h_keys.size() * sizeof(float)), "cudaMalloc(d_keys)") ||
        !check_cuda(cudaMalloc(&d_values, h_values.size() * sizeof(float)), "cudaMalloc(d_values)") ||
        !check_cuda(cudaMalloc(&d_weights, h_weights.size() * sizeof(float)), "cudaMalloc(d_weights)") ||
        !check_cuda(cudaMalloc(&d_offsets, h_offsets.size() * sizeof(int)), "cudaMalloc(d_offsets)") ||
        !check_cuda(cudaMalloc(&d_out_keys, landmarks * dim * sizeof(float)), "cudaMalloc(d_out_keys)") ||
        !check_cuda(cudaMalloc(&d_out_values, landmarks * dim * sizeof(float)), "cudaMalloc(d_out_values)")) {
        return 1;
    }

    if (!check_cuda(cudaMemcpy(d_keys, h_keys.data(), h_keys.size() * sizeof(float), cudaMemcpyHostToDevice), "copy keys") ||
        !check_cuda(cudaMemcpy(d_values, h_values.data(), h_values.size() * sizeof(float), cudaMemcpyHostToDevice), "copy values") ||
        !check_cuda(cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(float), cudaMemcpyHostToDevice), "copy weights") ||
        !check_cuda(cudaMemcpy(d_offsets, h_offsets.data(), h_offsets.size() * sizeof(int), cudaMemcpyHostToDevice), "copy offsets")) {
        return 1;
    }

    reduce_weighted_landmarks<<<landmarks, 32>>>(d_keys, d_values, d_weights, d_offsets, d_out_keys, d_out_values, dim);
    if (!check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
        return 1;
    }

    std::vector<float> h_out(landmarks * dim, 0.0f);
    if (!check_cuda(cudaMemcpy(h_out.data(), d_out_keys, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy out")) {
        return 1;
    }

    const float expected[] = {2.0f, 3.0f, 25.0f, 35.0f};
    for (int i = 0; i < landmarks * dim; ++i) {
        if (std::fabs(h_out[i] - expected[i]) > 1e-4f) {
            std::cerr << "Mismatch at " << i << ": got " << h_out[i] << " expected " << expected[i] << "\n";
            return 1;
        }
    }

    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_weights);
    cudaFree(d_offsets);
    cudaFree(d_out_keys);
    cudaFree(d_out_values);
    std::cout << "OK: cuda_kernel_smoke\n";
    return 0;
}
