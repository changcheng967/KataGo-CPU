# Hygon DCU support (`USE_BACKEND=ROCM` against DTK) — WORKING

## Status: official backend runs on DCU with ZERO code changes

KataGo's upstream CUDA/ROCm shared backend (`cudaandrocmbackend.inc` via
`rocmbackend.cpp`) configures, builds, and runs against Hygon DTK 26.04 on the
Z200SM_80 (gfx906) with no patches:

```
cmake -S cpp -B build-rocm -DUSE_BACKEND=ROCM \
  -DCMAKE_HIP_COMPILER=/opt/dtk/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DCMAKE_PREFIX_PATH=/opt/dtk -DCMAKE_BUILD_TYPE=Release
make -C build-rocm -j6 katago
```

Build notes: source must carry its `.git` dir (git-info step), and git needs
`safe.directory` after kubectl-cp ownership changes. `katago version` reports
"Using ROCm backend, HIP 6.3". fp16 is auto-selected (`useFP16 = true`).

## Device qualification (measured, cpp/scripts/gpubench*.cpp)

| metric | value |
|---|---|
| Device | Z200SM_80, 64 CU, gfx906 (GCN5 vector, no MFMA/BF16), 17.2 GB |
| fp32 GEMM | 10.4 TFLOP/s (96% of the 10.8 ceiling) |
| fp16 GEMM (hipblasHgemm) | 17.4 TFLOP/s (80% of 21.6) |
| HBM | 683 GB/s |

## Engine results (official auto-tuner, 7-core quota + 1 DCU)

Tuner verdict: **numSearchThreads=32 + numNNServerThreadsPerModel=2**
(it found the 2-server-thread +17.3% itself). avgBatch ~10-15.

| model | v/s | vs 8-core Zen4 CPU | per-visit Elo anchor |
|---|---|---|---|
| tf2-b10c384 | **556** | 10.3x (54) | 13,712 |
| tf3-b11c768 | 162 | ~8x | ~14,700 |
| zhizi-b40c768 | **117** | (unusable on small CPU boxes) | ~14,800 |

**Headline: the DCU inverts the CPU model ranking.** On CPU, zhizi-b40 was the
OOM-killer and tf2 was the only sane choice; on the DCU the strongest official
net (zhizi) runs at 117 v/s — 4.75x fewer visits than tf2 gets cannot buy back
~1,100 Elo of per-visit strength, so **strongest-per-wall-clock on DCU is
zhizi-b40c768**, with tf3-b11c768 as the balanced point.

Container logistics that mattered: no internet in the pod, pod→login blocked
(kubectl cp is the only supply path); media.katagotraining.org is directly
fetchable on the login node.
