#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
OUTDIR="$ROOT/build"

mkdir -p "$OUTDIR"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc is not available on this machine"
  exit 0
fi

cmake -S "$ROOT" -B "$OUTDIR" >/dev/null
cmake --build "$OUTDIR" --target cuda_kernel_smoke >/dev/null
"$OUTDIR/cuda_kernel_smoke"
