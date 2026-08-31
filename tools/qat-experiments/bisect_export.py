"""Bisect the OV-FQ export: quantize only the first k trunk convs (graph order) and
measure agreement vs fp32 for growing k to localize the divergence."""
import sys

import numpy as np
from openvino import Core, Model, save_model
from openvino import opset8 as ops

sys.path.insert(0, "/hyperai/home/katago-bench/scripts")
import torch
import torch.nn as nn
from katago_torch import load_torch_model

ONNX_PATH = "/hyperai/home/katago-bench/onnx/b18.onnx"
CKPT = "/hyperai/home/katago-bench/traindata/qat_student.pt"

torch.set_num_threads(4)
base, _ = load_torch_model(ONNX_PATH)
import copy

class QuantConv2d(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv
        self.register_buffer("act_scale", torch.tensor(1.0))
        self.register_buffer("act_zp", torch.tensor(0.0))
        self.register_buffer("w_scale", (conv.weight.data.abs().amax(dim=(1, 2, 3), keepdim=True) / 127.0).clamp_min(1e-12))
        self.learn_act_scale = nn.Parameter(torch.tensor(1.0))
        self.learn_act_zp = nn.Parameter(torch.tensor(0.0))

student = copy.deepcopy(base)
for name, mod in list(student.named_modules()):
    if isinstance(mod, nn.Conv2d) and "blocks" in name:
        parent, _, leaf = name.rpartition(".")
        setattr(student.get_submodule(parent) if parent else student, leaf, QuantConv2d(mod))
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
student.load_state_dict(ckpt["student"])
student.eval()
qmap = {}
with torch.no_grad():
    for name, mod in student.named_modules():
        if isinstance(mod, QuantConv2d):
            s = float(mod.learn_act_scale.abs().clamp_min(1e-12))
            zp = float(torch.round(mod.learn_act_zp))
            w = mod.conv.weight.data
            ws = (w.abs().amax(dim=(1, 2, 3)) / 127.0).clamp_min(1e-12)
            wq = torch.clamp(torch.round(w / ws.view(-1, 1, 1, 1)), -127, 127)
            w_deq = (wq * ws.view(-1, 1, 1, 1)).numpy()
            qmap[name.replace("/", ".")] = (s, zp, w_deq, ws.numpy())

raw = np.fromfile("/hyperai/home/katago-bench/calib_real.f32", dtype=np.float32)
ROW = 361 + 22 * 361 + 19
rows = raw[: (len(raw) // ROW) * ROW].reshape(-1, ROW)
N = 128
masks = rows[:N, :361].reshape(-1, 1, 19, 19).astype(np.float32)
sp = rows[:N, 361:361 + 7942].reshape(-1, 22, 19, 19).astype(np.float32)
gl = rows[:N, 361 + 7942:].reshape(-1, 19, 1, 1).astype(np.float32)

core = Core()
c32 = Core(); c32.set_property("CPU", {"INFERENCE_PRECISION_HINT": "FP32"})
m32 = c32.compile_model(core.read_model(ONNX_PATH), "CPU")

def outs(mm):
    names = [o.any_name for o in mm.outputs]
    r = {}
    for s in range(0, N, 32):
        o = mm({"InputMask": masks[s:s+32], "InputSpatial": sp[s:s+32], "InputGlobal": gl[s:s+32]})
        for k in names:
            r.setdefault(k, []).append(np.array(o[k]))
    return {k: np.concatenate(v, 0) for k, v in r.items()}

ref = outs(m32)
la = np.concatenate([ref["OutputPolicyPass"][:, 0].reshape(N, -1), ref["OutputPolicy"][:, 0].reshape(N, -1)], 1)

def export_subset(k):
    m = core.read_model(ONNX_PATH)
    convs = [op for op in m.get_ops() if op.get_type_name() == "Convolution" and op.get_friendly_name() in qmap]
    convs.sort(key=lambda op: op.get_friendly_name())
    n_q = 0
    for op in convs[:k]:
        name = op.get_friendly_name()
        s, zp, w_deq, ws = qmap[name]
        vmin = -zp * s
        vmax = vmin + 255.0 * s
        src = op.input(0).get_source_output()
        fq = ops.fake_quantize(src, ops.constant(np.float32(vmin)), ops.constant(np.float32(vmax)),
                               ops.constant(np.float32(vmin)), ops.constant(np.float32(vmax)), 256)
        op.input(0).replace_source_output(fq.output(0))
        shape = w_deq.shape
        lo = (-127.0 * ws.reshape(-1, 1, 1, 1)).astype(np.float32)
        hi = (127.0 * ws.reshape(-1, 1, 1, 1)).astype(np.float32)
        fq_w = ops.fake_quantize(ops.constant(w_deq.astype(np.float32)), ops.constant(lo), ops.constant(hi),
                                 ops.constant(lo), ops.constant(hi), 255)
        op.input(1).replace_source_output(fq_w.output(0))
        n_q += 1
    c = Core(); c.set_property("CPU", {"INFERENCE_PRECISION_HINT": "FP32"})
    cm = c.compile_model(m, "CPU")
    return outs(cm), n_q

ONLY = sys.argv[1] if len(sys.argv) > 1 else None
if ONLY:
    keep = dict(qmap)
    qmap.clear()
    for nm in ONLY.split(","):
        qmap[nm] = keep[nm]
    ks = [len(qmap)]
else:
    ks = [1, 2, 4, 8, 16, 32, 64, 118]
for k in ks:
    o, n_q = export_subset(k)
    lb = np.concatenate([o["OutputPolicyPass"][:, 0].reshape(N, -1), o["OutputPolicy"][:, 0].reshape(N, -1)], 1)
    best = (la.argmax(1) == lb.argmax(1)).mean() * 100
    polmax = np.abs(la - lb).max()
    print(f"{n_q:3d} convs quantized: bestMove {best:5.1f}%  policyMaxAbs {polmax:.3f}")
print("BISECT_DONE")
