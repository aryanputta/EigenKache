# Data

This folder is for real trace artifacts only.

Expected files:

- `*.npz` trace files with `keys`, `values`, and `queries`

Recommended source process:

1. Run a real decoder model on `LongBench` or `PG19`
2. Capture one head or one merged trace slice at a chosen layer
3. Save:
   - `keys`: `(tokens, dim)`
   - `values`: `(tokens, dim)`
   - `queries`: `(held_out_queries, dim)`
4. Store metadata with task name, model name, layer, head, and prompt id

Do not benchmark on synthetic prompts when claiming project results.
