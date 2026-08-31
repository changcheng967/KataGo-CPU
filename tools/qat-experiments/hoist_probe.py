"""Probe: does OV hoist a consumer-edge FQ onto the shared producer tensor?"""
import copy
import sys

sys.path.insert(0, "/hyperai/home/katago-bench/scripts")
import numpy as np
import torch
import torch.nn as nn
from openvino import Core, Model
from openvino import opset8 as ops
from katago_torch import load_torch_model

torch.set_num_threads(4)
base, _ = load_torch_model("/hyperai/home/katago-bench/onnx/b18.onnx")

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
ckpt = torch.load("/hyperai/home/katago-bench/traindata/qat_student.pt", map_location="cpu", weights_only=False)
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
            qmap[name.replace("/", ".")] = (s, zp, (wq * ws.view(-1, 1, 1, 1)).numpy(), ws.numpy())

core = Core()
raw = np.fromfile("/hyperai/home/katago-bench/calib_real.f32", dtype=np.float32)
ROW = 361 + 22 * 361 + 19
rows = raw[: (len(raw) // ROW) * ROW].reshape(-1, ROW)
masks = rows[:8, :361].reshape(-1, 1, 19, 19).astype(np.float32)
sp = rows[:8, 361:361 + 7942].reshape(-1, 22, 19, 19).astype(np.float32)
gl = rows[:8, 361 + 7942:].reshape(-1, 19, 1, 1).astype(np.float32)

QNAME = "model.blocks.0.normactconvq.conv"
PNAME = "model.blocks.0.normactconvp.conv"

def build(quant_names):
    m = core.read_model("/hyperai/home/katago-bench/onnx/b18.onnx")
    op_by = {op.get_friendly_name(): op for op in m.get_ops() if op.get_type_name() == "Convolution"}
    for nm in quant_names:
        op = op_by[nm]
        s, zp, w_deq, ws = qmap[nm]
        vmin = -zp * s
        vmax = vmin + 255.0 * s
        fq = ops.fake_quantize(op.input(0).get_source_output(), ops.constant(np.float32(vmin)), ops.constant(np.float32(vmax)),
                               ops.constant(np.float32(vmin)), ops.constant(np.float32(vmax)), 256)
        op.input(0).replace_source_output(fq.output(0))
        shape = w_deq.shape
        lo = (-127.0 * ws.reshape(-1, 1, 1, 1)).astype(np.float32)
        hi = (127.0 * ws.reshape(-1, 1, 1, 1)).astype(np.float32)
        fq_w = ops.fake_quantize(ops.constant(w_deq.astype(np.float32)), ops.constant(lo), ops.constant(hi),
                                 ops.constant(lo), ops.constant(hi), 255)
        op.input(1).replace_source_output(fq_w.output(0))
    return m

ref = None
for tag, qn in [("fp32", []), ("p-only", [PNAME]), ("p+q", [PNAME, QNAME])]:
    m = build(qn)
    qop = next(op for op in m.get_ops() if op.get_type_name() == "Convolution" and op.get_friendly_name() == QNAME)
    src = qop.input(0).get_source_output()
    cap = Model([src], list(m.parameters), "cap")
    cm = core.compile_model(cap, "CPU")
    r = cm({"InputMask": masks, "InputSpatial": sp, "InputGlobal": gl})
    t = np.array(r[cm.outputs[0]])
    if ref is None:
        ref = t.copy()
    print(f"{tag:8s} q-input[0,0,0,:4]: {np.round(t[0,0,0,:4], 5)}")

s, zp, _, _ = qmap[QNAME]
vmin = -zp * s
q_of_ref = np.round((ref - vmin) / s).clip(0, 255) * s + vmin
print("fq(fp32):", np.round(q_of_ref[0, 0, 0, :4], 5), " <- p+q equaling this means FQ hoisted to the shared tensor")
