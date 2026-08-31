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

Engine visits/s, `katago benchmark`, provider `ov`:

| Model | Elo (katagotraining.org) | v/s (ov) | v/s (Eigen) | v/s (cpu/MLAS) | Fixed-time ΔElo vs b18+Eigen |
|---|---|---|---|---|---|
| tf2-b10c384 (10.5M params) | 13712 | **48.9** | n/a | ~13 | **+134** |
| b18c384nbt (26.3M) | 13608 | 31.9 | 30.7 | 18.9 | +7 |
| tf3-b10c512 (28.5M) | 14170 | **21.8** | n/a | ~9 | **+512** |
| b28c512nbt (72.8M) | 14105 | 11.7 | ~6 (est) | ~5 | +362 |
| tf3-b11c768 (70.4M) | 14542 | 9.1 | n/a | ~4 | **+774** |
| zhizi-b40c768nbt (232M) | 14549 | 3.8 | <2 (est) | ~1.5 | +650 |

Recommendations: **tf3-b10c512** for live play, **tf3-b11c768** for maximum-strength
analysis, **tf2-b10c384** for fast analysis. The transformer nets deliver far more Elo
per GFLOP than convnets on CPU (tf2 is stronger than b18 at half the compute).

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

## Benchmarking methodology

Speed: `katago benchmark -t 8 -v <visits> -n <positions>`, medians over multiple runs.
FLOP counts: exact per-layer MAC counts from the dumped ONNX graph. Accuracy:
held-out real positions harvested via the input dumper; metrics are true best-move
agreement over all 362 moves, top-5 retention, policy KL, winrate/scoreMean error.
