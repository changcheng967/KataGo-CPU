"""Emit a small KataGo-shaped ONNX for kernel development (no GPU needed).

Mirrors the exact op mix of the tf2 transformer blocks at tiny width so every
custom HIP kernel can be diffed against onnx.reference on a laptop:
  stem 3x3 conv -> [normact chain -> q/k/v matmuls -> RoPE -> strided-batched
  scores -> scale/mask -> softmax -> sv -> merge -> out proj -> normact] x B
  -> policy head 3x3 -> value head pool+FC
Inputs/outputs use the KataGo contract names and the v3 layout (22+19 channels,
362-move policy with pass at index 0 of the flattened move logits).
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

C = 64        # trunk width (tf2 has 2 stacks x 192; tiny here)
H = 2         # heads
DH = 32       # head dim (same as real nets)
B = 2         # blocks
L = 19 * 19

def t(name, arr):
    if arr.dtype == np.int64:
        return helper.make_tensor(name, TensorProto.INT64, arr.shape, arr.tobytes(), raw=True)
    return helper.make_tensor(name, TensorProto.FLOAT, arr.shape, arr.astype(np.float32).tobytes(), raw=True)

inits = []
nodes = []

w = {}
rng = np.random.default_rng(7)
def param(name, shape, scale=0.05):
    arr = rng.normal(0, scale, size=shape).astype(np.float32)
    inits.append(t(name, arr))
    return name

# stem: 3x3 conv 22 -> C
param("stem_w", (C, 22, 3, 3), 0.08)
nodes.append(helper.make_node("Conv", ["InputSpatial", "stem_w"], ["stem_c"], pads=[1,1,1,1]))
nodes.append(helper.make_node("Transpose", ["stem_c"], ["stem_out"], perm=[0,2,3,1]))

# global: broadcast-add of Linear(19 -> C) per position
param("glob_w", (19, C), 0.1)
nodes.append(helper.make_node("MatMul", ["InputGlobalFlat", "glob_w"], ["glob_mm"]))
nodes.append(helper.make_node("Reshape", ["glob_mm", "shape_n11c"], ["glob_proj"]))
inits.append(t("shape_n11c", np.array([-1, 1, 1, C], np.int64)))
nodes.append(helper.make_node("Add", ["stem_out", "glob_proj"], ["stem2_4d"]))
nodes.append(helper.make_node("Reshape", ["stem2_4d", "shape3"], ["stem2"]))

inits.append(t("shape3", np.array([-1, L, C], np.int64)))
inits.append(t("shape4", np.array([-1, L, H, DH], np.int64)))

def normact(inp, out, tag):
    # RMSNorm (Pow/ReduceMean/Sqrt/Div) + SiLU (Sigmoid*Mul) as KataGo graphs do
    nodes.append(helper.make_node("Pow", [inp, "c2"], [tag+"_sq"]))
    inits.append(t("c2", np.float32(2.0)))
    nodes.append(helper.make_node("ReduceMean", [tag+"_sq"], [tag+"_ms"], axes=[2], keepdims=1))
    nodes.append(helper.make_node("Add", [tag+"_ms", "eps"], [tag+"_den"]))
    inits.append(t("eps", np.float32(1e-5)))
    nodes.append(helper.make_node("Sqrt", [tag+"_den"], [tag+"_std"]))
    nodes.append(helper.make_node("Div", [inp, tag+"_std"], [tag+"_norm"]))
    nodes.append(helper.make_node("Sigmoid", [tag+"_norm"], [tag+"_sig"]))
    nodes.append(helper.make_node("Mul", [tag+"_norm", tag+"_sig"], [out]))

for b in range(B):
    p = f"b{b}"
    normact("stem2" if b == 0 else f"{p-1}_out" if b > 0 else "stem2", f"{p}_na1", f"{p}_na1") if False else None
    prev = "stem2" if b == 0 else f"b{b-1}_out"
    normact(prev, f"{p}_na", f"{p}_na")
    # q/k/v projections
    for nm in "qkv":
        param(f"{p}_{nm}_w", (C, C), 0.08)
        nodes.append(helper.make_node("MatMul", [f"{p}_na", f"{p}_{nm}_w"], [f"{p}_{nm}"]))
    # RoPE: swap-gather + cos/sin tables (per-head, full-dim, as real nets)
    swap = np.arange(DH) ^ 1
    inits.append(t(f"{p}_swap", swap.astype(np.int64)))
    theta = np.arange(DH // 2) * (2.0 / DH)
    pos = np.arange(L)
    ang = pos[:, None] * theta[None, :]          # [L, DH/2]
    cos_half = np.cos(ang); sin_half = np.sin(ang)
    cos = np.broadcast_to(np.concatenate([cos_half, cos_half], 1)[None, None], (1, H, L, DH)).astype(np.float32)
    sin = np.broadcast_to(np.concatenate([sin_half, sin_half], 1)[None, None], (1, H, L, DH)).astype(np.float32)
    inits.append(t(f"{p}_cos", cos))
    inits.append(t(f"{p}_sin", sin))
    for nm in "qkv":
        nodes.append(helper.make_node("Reshape", [f"{p}_{nm}", f"shape4"], [f"{p}_{nm}r"]))
        nodes.append(helper.make_node("Transpose", [f"{p}_{nm}r"], [f"{p}_{nm}h"], perm=[0, 2, 1, 3]))
    for nm in "qk":
        nodes.append(helper.make_node("Gather", [f"{p}_{nm}h", f"{p}_swap"], [f"{p}_{nm}s"], axis=3))
        nodes.append(helper.make_node("Mul", [f"{p}_{nm}h", f"{p}_cos"], [f"{p}_{nm}t1"]))
        nodes.append(helper.make_node("Mul", [f"{p}_{nm}s", f"{p}_sin"], [f"{p}_{nm}t2"]))
        nodes.append(helper.make_node("Add", [f"{p}_{nm}t1", f"{p}_{nm}t2"], [f"{p}_{nm}_rope"]))
    nodes.append(helper.make_node("Transpose", [f"{p}_k_rope"], [f"{p}_kT"], perm=[0, 1, 3, 2]))
    nodes.append(helper.make_node("MatMul", [f"{p}_q_rope", f"{p}_kT"], [f"{p}_scores"]))
    inits.append(t(f"{p}_scale", np.float32(1.0 / np.sqrt(DH))))
    nodes.append(helper.make_node("Mul", [f"{p}_scores", f"{p}_scale"], [f"{p}_sc"]))
    nodes.append(helper.make_node("Add", [f"{p}_sc", "InputMaskFlat361"], [f"{p}_scm"]))
    nodes.append(helper.make_node("Softmax", [f"{p}_scm"], [f"{p}_probs"], axis=3))
    nodes.append(helper.make_node("MatMul", [f"{p}_probs", f"{p}_vh"], [f"{p}_sv"]))
    nodes.append(helper.make_node("Transpose", [f"{p}_sv"], [f"{p}_svT"], perm=[0, 2, 1, 3]))
    nodes.append(helper.make_node("Reshape", [f"{p}_svT", "shape3"], [f"{p}_attn"]))
    param(f"{p}_out_w", (C, C), 0.08)
    nodes.append(helper.make_node("MatMul", [f"{p}_attn", f"{p}_out_w"], [f"{p}_proj"]))
    nodes.append(helper.make_node("Add", [prev, f"{p}_proj"], [f"{p}_out"]))

# heads
prev = f"b{B-1}_out"
normact(prev, "head_na", "head_na")
nodes.append(helper.make_node("Reshape", ["head_na", "shape_nchw"], ["head_4d"]))
nodes.append(helper.make_node("Transpose", ["head_4d"], ["head_nchw"], perm=[0,3,1,2]))
inits.append(t("shape_nchw", np.array([-1, 19, 19, C], np.int64)))
param("pol_w", (2, C, 3, 3), 0.1)
nodes.append(helper.make_node("Conv", ["head_nchw", "pol_w"], ["OutputPolicy"], pads=[1,1,1,1]))
nodes.append(helper.make_node("ReduceMean", ["head_na"], ["pool"], axes=[1], keepdims=0))
param("val_w", (C, 3), 0.1)
nodes.append(helper.make_node("MatMul", ["pool", "val_w"], ["val_p"]))


graph = helper.make_graph(
    nodes, "dev_katago",
    [helper.make_tensor_value_info("InputSpatial", TensorProto.FLOAT, ["N", 22, 19, 19]),
     helper.make_tensor_value_info("InputGlobalFlat", TensorProto.FLOAT, ["N", 1, 19]),
     helper.make_tensor_value_info("InputMaskFlat361", TensorProto.FLOAT, ["N", 1, 1, L])],
    [helper.make_tensor_value_info("OutputPolicy", TensorProto.FLOAT, ["N", 2, 19, 19]),
     helper.make_tensor_value_info("val_p", TensorProto.FLOAT, ["N", 3])],
    inits)
m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 8
onnx.save(m, "dev_katago.onnx")
ops = {}
for n in nodes:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
print(f"saved dev_katago.onnx: {len(nodes)} nodes, ops={ops}")
