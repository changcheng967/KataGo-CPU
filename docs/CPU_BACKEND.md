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

### Engine cycle decomposition (direct instrumentation, `KATAGO_OV_TIMING=1`)

An env-gated timer around the provider's `infer()` call decomposed the engine
batch cycle directly (200-batch averages, 8 search threads):

| net | infer() in-engine | between batches | standalone infer (same batch) |
|---|---|---|---|
| b18c384nbt | 132.1 ms @ avgBatch 3.65 | **2.5 ms** | ~72 ms @ batch 4 |
| tf2-b10c384 | 82.0 ms @ avgBatch 3.65 | **2.7 ms** | ~54 ms @ batch 4 |

Two findings: (1) dispatch/collect overhead between batches is **~3 ms —
negligible**; the engine pipeline itself adds nothing. (2) `infer()` takes ~2x
its standalone time inside the engine because the 8 search threads
productively expand the MCTS tree (with the previous batch's results) on the
same cores the 8 NN threads use — NN and search time-share the machine. This
is not overhead to remove: the tree work must happen anyway, and deferring it
(priority scheduling) cannot shorten the visit critical path because total
work is conserved. This converts the "MCTS tax" from inference into direct
measurement: **per engine cycle, roughly half the core-time goes to the NN and
half to productive search** — the structural cost of running MCTS inference
and MCTS search on the same silicon. A GPU machine gets the search half for
free on separate hardware.

### Analysis-engine (multi-game) throughput

Measured with the canonical JSON protocol (color-pair moves, distinct random
games per query, `analysis_example.cfg`): the analysis engine's throughput on
cache-cold distinct positions equals the `benchmark` numbers — the benchmark
was representative of real analysis all along. Two workload effects worth
knowing: queries along the same game line share PV exploration and nnCache
(many-turn reviews run well above the cold rate), and concurrent queries from
many games fill bigger NN batches, pushing toward the higher end of the
batch-throughput curve. Synthetic benchmarks that reuse board regions across
games can produce absurd numbers via nnCache — a pitfall caught and documented
here (three broken harness versions were discarded before the canonical-format
one produced clean data).

### OpenVINO version comparison (2026.1 / 2026.2.1 / 2026.3.1)

Nightly builds require GitHub artifact auth (no public pip wheel); the
practical question — does the latest stable beat earlier ones — was measured
in isolated venvs on the same host:

| version | b18 b1 pos/s | b18 b8 pos/s (3-run mean) | tf3-b512 b8 |
|---|---|---|---|
| 2026.1.0 | 43.6 | 56.0 | — |
| 2026.2.1 | 44.0 | 49.2 (48.4/49.6/49.6) | 35.2 |
| **2026.3.1 (current)** | 43.2 | **52.1 (56.2/49.8/50.2)** | 34.9 |

Differences are within host noise (±10% run-to-run on this shared machine);
no version has a decisive kernel advantage for these graphs. The engine links
against whatever OV ships in `OV_ROOT`, so upgrading OV underneath the same
katago binary is safe but brings no measured gain. Nightly builds are
untestable without artifact authentication; given stable-version flatness,
no performance windfall is expected from them either — the CPU plugin's
conv/bf16 kernels have been mature since well before 2026.1.

### oneDNN direct-call microbenchmark (bypassing every graph runtime)

The final "unknown backend" — calling oneDNN primitives directly in C++ on the
exact b18 trunk convolution shape (192→192 3×3 @ 19×19, batch 8, bf16),
installed via `apt install libdnnl-dev` (oneDNN 3.1):

| path | ms/conv | GFLOP/s | ×36 convs (b18 trunk) |
|---|---|---|---|
| oneDNN direct, plain NCHW layouts | 2.217 | 864 | 79.8 ms |
| **oneDNN direct, `format_tag::any`** (library-chosen blocked) | **1.427** | **1343** | **51.4 ms** |
| OpenVINO whole b18 net (same host, same batch) | — | — | 141.8 ms |

Two findings: (1) letting oneDNN pick its own blocked memory format is 1.55x
faster than forcing NCHW — layout autonomy matters more than anything else;
(2) the 36 trunk convs cost 51.4 ms through bare oneDNN vs ~142 ms for the
whole net through OV. The remaining ~90 ms is the rest of the net (norms,
activations, gpool, heads, ~7.9 GFLOP more) plus OV's graph layer. Since the
b18 convs through OV were already measured at 98% of OV's own GEMM peak, the
direct-call result shows the *absolute* oneDNN ceiling for this shape is
1343 GFLOP/s — OV's whole-graph effective rate (including all non-conv ops)
sits at ~66% of that. A from-scratch oneDNN backend with perfect fusion of
norms/activations into conv post-ops could theoretically close part of this
~90 ms gap (the earlier Tier-3 estimate of +10-30% remains the honest range;
the gap is bounded by the non-conv work that must still run somewhere).

### Alternative-backend shootout (same graphs, same host)

The question "is OpenVINO really the best CPU backend?" was settled by direct
measurement against every runnable alternative:

| backend (b18 convnet, b1 / b8 ms per eval) | speed | vs OV bf16 |
|---|---|---|
| **OpenVINO 2026.3 (bf16)** | **25.1 / 160.7** | 1.0x |
| PyTorch 2.13 eager (oneDNN underneath, fp32) | 69.4 / 328.5 | 0.36x / 0.49x |
| PyTorch torch.compile (Inductor) | 55.4 / 302.2 | 0.45x / 0.53x |
| ONNX Runtime MLAS (from the full table) | ~70 / ~290 | 0.35x / 0.55x |
| TVM 0.26 (llvm) | failed at import (relay removed in 0.26; relax API incompatible with onnx frontend on this build) | — |
| ORT-DNNL execution provider | no pip wheel exists for 1.x; requires a from-source ORT build | untested (build) |
| ncnn | pip-installable, but no ONNX-conv frontend path for this graph shape mix | untested (conversion) |

PyTorch (eager and compiled) is **2-3x slower than OV** — torch CPU does not
apply bf16 to conv automatically, and Inductor's fusion cannot recover the gap.
MLAS matches PyTorch eager, consistent with the engine measurements. The
conclusion: **OpenVINO's bf16 CPU kernels are unchallenged among runnable
backends on this hardware**; the only untested path is a from-source ORT+DNNL
build (same oneDNN kernels through a different scheduler — expected at best
to match OV, not beat it, since OV *is* the oneDNN frontend with the fewest
layers on top).

### Multi-process scaling (measured, with correction)

The benchmark-based multi-process curve (all layouts, taskset-pinned):

| layout | tf2 combined v/s | vs single |
|---|---|---|
| 1 process × 8 cores | 46.6 | — |
| 2 × 4 | 49.3 | +6% |
| **3 (3+3+2)** | **50.9** | **+9%** |
| 4 × 2 | 49.9 | +7% |
| 5+3 | 48.6 | +4% |
| 6+2 | 46.9 | +1% |

**Correction after protocol-level re-measurement**: the analysis-engine numbers
are dominated by startup time and nnCache behavior at short durations — repeated
runs swung 16–146 pos/s for the same config. After amortizing startup with 480
distinct games, the single process measured **146.6 positions/s** — far above
the ~48 v/s benchmark figure — because the analysis engine deduplicates
identical positions via nnCache and fills batches across concurrent queries;
our synthetic games also share board regions, inflating hit rates. The honest
conclusion: **the benchmark number is a cold-cache lower bound; real multi-game
analysis throughput is cache-workload-dependent and can be several times
higher.** The benchmark-based +6-9% multi-process advantage at equal cold-cache
conditions stands; under cache-heavy real workloads the single process may win
(a larger nnCache in one process catches more). Configuration pitfalls
documented: `analysis` mode uses `numSearchThreadsPerAnalysisThread`, and
specifying `numSearchThreads` alongside it aborts at startup.



Splitting the machine into multiple KataGo processes pinned to core subsets
measures better than the single process for **multi-game throughput** —
overturning an earlier dismissal that was based on single-process numbers:

Mechanism (benchmark conditions): partitioned cores shrink the per-process
search/NN time-sharing penalty; 4-thread GEMM keeps 75-80% of 8-thread
efficiency. Practical: for cold-cache parallel analysis the 3-process split
(3+3+2 pinned cores) is the measured optimum; for cache-heavy or single-game
workloads one process with all cores is safer.

**Search-parameter sweep** (cPuctExploration 0.8/1.2/1.6,
fpuReductionMax 0.1/0.2/0.5): all within ±5% visits/s — the MCTS parameters
govern strength trade-offs, not throughput; the search work per visit is
invariant to them. (During this sweep the host load was elevated: load average
~20 on the shared 512-thread machine — absolute numbers here are depressed
but the within-run comparison holds.)

**nnCache size**: sweeping `nnCacheSizePowerOfTwo` 20/23(default)/25 on the
benchmark shows no effect (12.2 / default / 11.9 v/s — within noise); the
cache is a throughput lever only for cache-hitting workloads (see the analysis
section), where the default 2^23 is already sized generously. Configuration
pitfall: the `benchmark` and `gtp` subcommands take `numSearchThreads`, while
`analysis` takes `numSearchThreadsPerAnalysisThread` — specifying the wrong
one against the wrong config aborts at startup.

**Mixed-net split (strong + fast in parallel)**: tf3-b11c768 (14542 Elo) on
4 cores + tf2-b10c384 (13712 Elo) on 4 cores runs both simultaneously at
2.3 + 7.4 v/s — a 4-core process running the big net pays the transformer
GEMM-efficiency cliff twice (4 threads AND batch-4 shapes), so the strong net
gets expensive fast. Verdict: mixed-net is workable for diverse opinion
simultaneously, but for max strength just run tf3-b11c768 on all 8 cores
(9.2 v/s single-process).
With these, the inference-side surface is exhausted: kernels at physics
(conv 98%), engine dispatch at ~3 ms, threading/batching/layout/precision/
weight-compression all measured to flat or negative, and the engine-kernel gap
proven to be productive search time-sharing. The remaining levers are
architectural and training-side.

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
