"""Precompute tf3-b11c768 teacher outputs on training positions (OV fp32)."""
import sys
import time

import numpy as np
from openvino import Core

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
TEACHER = "/hyperai/home/katago-bench/onnx/kata1-tf3-b11c768-s11001M-d5973M.onnx"
CACHE = "/hyperai/home/katago-bench/traindata/inputs_u8.npz"
OUT = "/hyperai/home/katago-bench/traindata/teacher_out.npz"

cache = np.load(CACHE)
sp = cache["spatial"]
gl = cache["glob"]
n = min(N, len(sp))
print(f"teacher on {n} positions")

core = Core()
c = Core(); c.set_property("CPU", {"INFERENCE_PRECISION_HINT": "FP32"})
m = c.compile_model(core.read_model(TEACHER), "CPU")
names = [o.any_name for o in m.outputs]
outs = {k: [] for k in names}
t0 = time.perf_counter()
for s in range(0, n, 32):
    b = min(32, n - s)
    r = m({"InputSpatial": sp[s:s+b].astype(np.float32),
           "InputGlobal": gl[s:s+b].reshape(b, 19, 1, 1).astype(np.float32),
           "InputMask": sp[s:s+b, 0:1].astype(np.float32)})
    for k in names:
        outs[k].append(np.array(r[k], dtype=np.float16))  # fp16 storage, plenty for distill
print(f"teacher pass: {time.perf_counter()-t0:.0f}s")
np.savez(OUT, **{k: np.concatenate(v, 0) for k, v in outs.items()})
print(f"saved {OUT}")
print("TEACHER_PRECOMPUTE_DONE")
