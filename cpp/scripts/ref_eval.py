"""Reference evaluator + test-vector dumper for the DCU executor.

Runs the dev (or real) KataGo-shaped ONNX in fp32 via onnx.reference (pure
Python) and saves inputs + all reference outputs to npz. The HIP executor
diffs against these vectors kernel by kernel. Zero dependencies beyond
onnx+numpy; runs on any laptop.
"""
import sys

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

model_path = sys.argv[1] if len(sys.argv) > 1 else "dev_katago.onnx"
out_path = sys.argv[2] if len(sys.argv) > 2 else "ref_vectors.npz"

m = onnx.load(model_path)
ref = ReferenceEvaluator(m)

rng = np.random.default_rng(42)
feeds = {}
for i in m.graph.input:
    name, shape = i.name, [d.dim_value if d.dim_value else 1 for d in i.type.tensor_type.shape.dim]
    # batch 2 for all dynamic dims
    shape = [2 if d.dim_param else d.dim_value for d in i.type.tensor_type.shape.dim]
    feeds[name] = rng.normal(0, 0.3, size=shape).astype(np.float32)
# mask should be zeros (legal) to exercise softmax plainly
for k in feeds:
    if "Mask" in k:
        feeds[k][:] = 0.0

outs = ref.run(None, feeds)
out_names = [o.name for o in m.graph.output]
np.savez(out_path,
         **{f"in_{k}": v for k, v in feeds.items()},
         **{f"out_{k}": v for k, v in zip(out_names, outs)})
print(f"saved {out_path}: inputs {list(feeds)} outputs {out_names}")
for k, v in zip(out_names, outs):
    print(f"  out_{k}: shape {v.shape} mean {v.mean():+.4f} std {v.std():.4f}")
