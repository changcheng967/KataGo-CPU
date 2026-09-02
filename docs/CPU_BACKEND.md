# Fast CPU Inference: Native OpenVINO Provider ("ov")

This fork adds a **native OpenVINO execution path** to KataGo's ONNX backend, giving
CPU-only machines the same bf16 oneDNN kernels that dominate every other CPU option —
measured at up to **3.7x faster play** than the stock `cpu` (MLAS) provider and stronger
than the Eigen backend, while matching the Eigen fp32 reference to **0.15% of
KataGo's own error tolerance** (`testgpuerror -reference-file`).

It works with every model architecture — the classic convnets (b18/b28/b40) and the
new transformer nets (tf2/tf3), loaded from the usual `.bin.gz` files or `.onnx` files
(`katago dumponnx`).

## What was changed

| File | Change |
|---|---|
| `cpp/neuralnet/onnxbackend.cpp` | New `onnxProvider = ov`: builds the same ONNX graph bytes, then executes them on the OpenVINO runtime instead of ORT. Zero-copy tensor plumbing (the OV tensors wrap the same host buffers the ORT path views as `Ort::Value`). New config key `onnxOVThreads`. Also: an env-gated **NN input dumper** (`KATAGO_DUMP_INPUTS=<file>`) that writes real encoded positions — used for quantization calibration and accuracy testing. |
| `cpp/CMakeLists.txt` | New options `USE_OV_NATIVE` and `OV_ROOT` to compile and link the provider. |

No other files are modified; every existing backend and workflow is untouched.

## Building

Requirements are the same as the stock ONNX backend (ONNX Runtime, protobuf), plus an
OpenVINO installation with C++ headers — the **pip wheel works** (`pip install openvino`;
it ships `include/` and `libs/`):

```bash
OV_ROOT=$(python3 -c "import openvino, os; print(os.path.dirname(openvino.__file__))")
# the wheel ships a versioned soname only; give the linker an unversioned symlink:
ln -sf $OV_ROOT/libs/libopenvino.so.*[0-9] $OV_ROOT/libs/libopenvino.so

cmake -S cpp -B build -DUSE_BACKEND=ONNX \
  -DONNXRUNTIME_ROOT=<onnxruntime install> \
  -DUSE_OV_NATIVE=1 -DOV_ROOT=$OV_ROOT \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

At runtime, `LD_LIBRARY_PATH` (or PATH on Windows) must contain both the ONNX Runtime
`lib/` and `$OV_ROOT/libs/`.

## Running

In your config (or via `-override-config`):

```ini
onnxProvider = ov          # native OpenVINO execution
onnxOVThreads = 8          # pins OpenVINO's internal thread pool; set to your core count
```

Keep `numNNServerThreadsPerModel = 1` (the default) — one OpenVINO session with all
cores measured **4x faster** than splitting cores across several sessions.

```bash
./katago gtp -model kata1-tf3-b10c512-s3203M-d5937M.bin.gz \
  -config configs/gtp_example.cfg -override-config "onnxProvider=ov,onnxOVThreads=8"
```

On AVX512_BF16 CPUs (AMD Zen 4/Zen 5, Intel Sapphire Rapids+), OpenVINO automatically
runs the trunk in bf16 — validated on real positions at 98.5–98.8% best-move agreement
with fp32 and winrate error ≤0.3% mean / ≤1.9% max. `onnxOpenVINOPrecision` overrides
this if set (`FP32` to force exact precision).

## Measured performance (8-core Zen 4 @ ~1.5GHz, 8 search threads)

Engine visits/s, `katago benchmark -t 8`, all four columns measured on the same
host under the same load (2026-08-31, all backends built from this repo's tree;
Eigen from `-DUSE_BACKEND=EIGEN -DUSE_AVX2=1`, MLAS = the stock `onnxProvider=cpu`,
ov = `onnxProvider=ov`):

| Model | Elo (katagotraining.org) | v/s (ov) | v/s (Eigen) | v/s (cpu/MLAS) | Fixed-time ΔElo vs b18+Eigen* |
|---|---|---|---|---|---|
| tf2-b10c384 (10.5M params) | 13712 | **48.4** | 25.6 | 6.0 | **+178** |
| b18c384nbt (26.3M) | 13608 | 30.1 | 28.9 | 2.8 | +6 |
| tf3-b10c512 (28.5M) | 14170 | **20.4** | 11.0 | 2.6 | **+512** |
| b28c512nbt (72.8M) | 14105 | 11.7 | 10.9 | 1.1 | +367 |
| tf3-b11c768 (70.4M) | 14542 | 9.2 | 5.1 | 1.2 | **+769** |
| zhizi-b40c768nbt (232M) | 14549 | 3.8 | 4.1 | 0.4 | +659 |

\* net Elo plus a visits bonus of ~100 Elo per 2x visits (log2 of the best-of-three
backend speed ratio vs b18+Eigen). The provider ranking per model: ov dominates every
net except zhizi-b40c768nbt, where Eigen's conv-dense shape narrowly wins (4.1 vs
3.8); MLAS is 5-10x behind everywhere and should not be used for CPU play.

Recommendations: **tf3-b10c512** for live play, **tf3-b11c768** for maximum-strength
analysis, **tf2-b10c384** for fast analysis. The transformer nets deliver far more Elo
per GFLOP than convnets on CPU (tf2 is stronger than b18 at half the compute).

## OpenVINO IR models (.xml) and weight compression

`onnxProvider = ov` also loads OpenVINO IR model files directly (`-model
something.xml` with the weights in the sibling `something.bin`). Because an IR
carries no embedded KataGo metadata, a sidecar `something_meta.txt` with the
`katago.*` key=value block is required next to the `.xml` (multi-line values use
`|` in place of newlines). This makes any OpenVINO-processed model deployable in
the engine — quantized graphs, NNCF-compressed weights, runtime-transformed IRs —
without re-exporting to ONNX.

Measured on b18c384nbt: an int8 weight-compressed IR (NNCF `compress_weights`,
int8_asym per-channel, ~26 MB weights) runs at the **same engine speed** as the
plain bf16 model (30.9 vs 31.8 v/s; the earlier +10% from weight compression was
a batch-1-only microbenchmark effect that vanishes at the engine's real batch
sizes, where compute — not weight streaming — dominates) while degrading accuracy
to w8a's known level (96.9% best-move, 0.25% mean winrate error; fails KataGo's
strict testgpuerror gate at 11x limit). Conclusion: **do not use weight-only
compression for the transformer or conv nets on CPU**; plain bf16 through the ov
provider is the accuracy-preserving choice. The IR loader itself is neutral
infrastructure for future model formats.

## Inference-side exhaustion log (post-v1 checks)

Every remaining inference knob was measured at engine level after the main work:

- **`onnxOVThreads` sweep (4/5/6/7/8, search threads 8)**: monotone to 8 — 16.2 /
  20.0 / 23.0 / 26.2 / 31.4 v/s (b18); tf3-b10c512 same shape (10.8 / 15.2 / 19.8).
  Leaving cores for the search threads always loses; OV should get them all.
- **`nnMaxBatchSize` sweep (4/8/16/32)**: 30.9 / 30.9 / 29.7 / 30.2 — no effect at
  these visit counts (the search never fills batches anyway, avgBatch ~3.7).
- **Static-shape specialization** (fixed-batch model copies): 40.0 vs 43.4 v/s
  (b1) and 53.6 vs 56.4 (b8) — *slower*; OV's dynamic-batch handling is already
  optimal. Rejected.
- **Compile-time `PERFORMANCE_HINT` / `cache_dir`**: this OpenVINO (2026.3)
  rejects both property names at Core level (crash, then removed the knobs);
  hints were also nil-to-negative in standalone measurements. Not supported.

With these, the inference-side surface is exhausted: kernels at physics
(conv 98%), engine at its MCTS-sharing optimum, threading/batching/layout/
precision/weight-compression all measured to flat or negative. The remaining
levers are architectural (see the repo's history / research notes) and
training-side.

## Performance tuning (swept and settled)

Measured on 8 cores across search threads 4/6/8/12/16/24, `numNNServerThreadsPerModel`
1/2/3, and `onnxOVThreads` 1/2/4/8 (with shared and per-handle sessions):

- **Optimal is simply matching core counts**: `numSearchThreads ≈ cores`,
  `onnxOVThreads ≈ cores`, `numNNServerThreadsPerModel = 1` (the defaults).
- More search threads than cores fill bigger batches (avgBatch 3.8 → 10 at t=24) but
  do **not** raise evals/s — OpenVINO's kernels are already batch-efficient, and the
  search threads' tree work shares the same cores.
- Multiple server threads (each with its own InferRequest on one shared
  Core/CompiledModel) shrink batches without adding throughput on a saturated CPU.

## How close to the hardware limit? (honest accounting)

With bf16 (the accuracy-validated precision), measured on 8 Zen 4 cores @ ~1.5 GHz:

| Layer | Convnets | Transformers |
|---|---|---|
| Kernel throughput vs practical bf16 peak (best GEMM, 1107 GFLOP/s) | **98%** — physics | 65–83% (attention ops) |
| Kernel throughput vs absolute instruction ceiling (1536 GFLOP/s) | 71% | 47–60% |
| Engine vs standalone-kernel throughput | ~75% | ~60–75% |

The engine gap is **not recoverable overhead**: on CPU-only machines the MCTS tree
work and the NN kernels share the same cores (~20–25% of CPU is productive search
work that a GPU machine gets "for free" on separate silicon). Pipelining experiments
(shared sessions, parallel server threads, batch oversubscription) all measured flat.
The one genuine remaining lever is custom fused-attention kernels for the transformer
nets (potential +15–30%, bounded by the instruction ceiling) — a from-scratch kernel
project.

## Transformer kernel profiling (why there is little left to fuse)

Per-op profiling (`PERF_COUNT`, OpenVINO) of tf3-b11c768, batch 1 / batch 8:

| op class | b1 share | b8 share | state |
|---|---|---|---|
| FullyConnected (QKV/FFN GEMMs) | 74.3% | 68.6% | dense GEMM at practical peak — physics |
| attention `sv` Subgraph | 12.8% | 16.7% | **already fused by OpenVINO** (scores+softmax+values); remaining cost is structural to the 361×361×heads shapes |
| elementwise glue (SiLU muls, norms, gathers, reorders) | ~10% | ~13% | partially fused by OV snippets; the only addressable slice |

The graph layout was also A/B-tested: `onnxTransformerNHWC = false` (NCHW) is
**10–19% slower** on every transformer model — the NHWC default is optimal.

Conclusion: a hand-written fused-attention kernel has little to recover (the runtime
already emits fused attention subgraphs); the realistic custom-kernel target is the
~10% elementwise glue via an OpenVINO extension with real AVX-512 microkernels —
estimated +5–7% end-to-end for a multi-week kernel project.


All handles share one `ov::Core` and one `CompiledModel` per model (single OpenVINO
thread pool, single graph compilation per process); each NN-server thread takes its
own `InferRequest` from it.

## Accuracy tooling

- `katago testgpuerror -model X -reference-file ref.bin` — provider `ov` vs the Eigen
  fp32 reference: **0.15% of the tolerance limit** (b18c384nbt, real positions).
- `KATAGO_DUMP_INPUTS=<file> katago testgpuerror -model X ...` — dumps real encoded
  input rows (mask + spatial + global, float32) for calibration/metrics tooling.
- bf16 vs fp32 on real positions (400+ held-out): 98.5% best-move agreement
  (tf3-b11c768), 98.8% (tf3-b10c512 and b18).

## Quantization notes (experimental)

Post-training int8 (NNCF, trunk-only, real-position calibration) reaches 1.25–1.5x
bf16 speed on the b18 convnet but plateaus at ~90% best-move agreement vs fp32
(sensitivity-guided exclusion of blocks 11/14 + MSE-optimal activation clipping);
bf16 keeps 98.8%. Weight-only int8 (`compress_weights`) preserves accuracy but gives
no speedup on this runtime. Clean int8 would require QAT. The transformer nets are
the better CPU investment: they beat every convnet precision trick at stock bf16.

### QAT/distillation investigation and OpenVINO int8 fusion bug (2026-08-31)

A full QAT pipeline was built: the b18 graph converted to PyTorch (validated to
≤9e-6 vs OpenVINO fp32), per-tensor-u8/per-channel-s8 fake-quant wrappers with
straight-through estimation, calibrated on 512 real kata1 training positions
(katagoarchive.org daily shards), distilled from the fp32 teacher.

- The calibrated (untrained) fake-quant net measures **94.9% best-move agreement /
  0.33% mean winrate error in PyTorch** — real headroom above the ~90% PTQ plateau.
- Gradient QAT as configured (Adam, batch 16) degraded the model and was stopped.
- **Export to OpenVINO is blocked by an int8 fusion bug**: with quantization on a
  nested-bottleneck block's 1×1 boundary convs (`normactconvp` + `normactconvq`),
  PyTorch computes 96.9% best-move agreement while the numerically-identical OV
  FQ graph computes 53.1%. Probes confirm OV applies the FakeQuantize operations
  bit-exactly at the intended edges, yet the compiled result diverges — the
  low-precision fusion itself miscompiles this pattern (OV 2026.3.1 CPU plugin).
  Minimal repro: `scripts/torch_pq_probe.py` (PyTorch) vs `scripts/bisect_export.py
  "model.blocks.0.normactconvq.conv,model.blocks.0.normactconvp.conv"` (OV).
- Deployable status quo stands: NNCF's own placement (which avoids the broken
  pattern) reaches ~90% at int8 speed; the 94.9% recipe becomes deployable when
  the runtime compiles QDQ/FQ exactly (fixed OpenVINO, or a runtime with exact
  QDQ semantics).


## Benchmarking methodology

Speed: `katago benchmark -t 8 -v <visits> -n <positions>`, medians over multiple runs.
FLOP counts: exact per-layer MAC counts from the dumped ONNX graph. Accuracy:
held-out real positions harvested via the input dumper; metrics are true best-move
agreement over all 362 moves, top-5 retention, policy KL, winrate/scoreMean error.
