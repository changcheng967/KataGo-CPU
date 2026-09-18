# Hygon DCU backend (`onnxProvider=dcu`, planned)

Native KataGo backend for Hygon DCU (DTK), following the flat backend-file
structure of `eigenbackend.cpp` / `onnxbackend.cpp`. Target device, measured
on the actual card (Z200SM_80, gfx906, 64 CU, DTK 26.04):

| metric | measured |
|---|---|
| fp32 GEMM (rocblas) | 10.4 TFLOP/s = 96% of device ceiling |
| fp16 GEMM (hipblasHgemm; GemmEx does not exist in DTK) | 17.4 TFLOP/s = 80% |
| HBM bandwidth | 683 GB/s |
| formats | no BF16, no MFMA (GCN5 vector datapath) |

fp16 is the deployment precision (10 mantissa bits > the bf16 CPU path's 7,
which was validated end-to-end at ~99% best-move agreement).

## Plan A (official path, first to try): build upstream cudaandrocm backend

Upstream ships `cudaandrocmbackend.inc` — the CUDA backend compiles against
ROCm/HIP. DTK is a HIP fork with miopen + rocblas present in the container;
a DTK build of the official backend may work with modest patches.

## Plan B (extraction path): custom executor, fixed op routing

Mirrors the OV-native provider pattern (ONNX stays the model contract):
hipblasHgemm for FC GEMMs (~85% FLOPs), hipblasHgemmStridedBatched for the
[N,H,361,361] attention GEMMs, fused custom HIP kernels for the
RMSNorm+SiLU+residual+mask chains, RoPE and 361-softmax (math already proven
in cpp/scripts/katflash/), im2col+GEMM for the two 3x3 convs. Projected
200-350 pos/s for tf2-b10c384 vs 46 evals/s on the 8-core CPU box.

## Dev loop (no GPU, no network needed)

- `cpp/scripts/gen_dev_graph.py` — small KataGo-shaped ONNX (same op mix)
- `cpp/scripts/ref_eval.py` — fp32 reference vectors via onnx.reference
- `cpp/scripts/gpubench.cpp`, `gpubench16.cpp` — device qualification
