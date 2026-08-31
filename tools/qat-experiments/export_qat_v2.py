"""Export v2: build the int8 model via OV-native FakeQuantize insertion.
For each trunk conv: FQ(activations, u8 256 levels asymmetric) + FQ(weights, s8 255
levels per-channel symmetric) -> CPU plugin LPT fuses to int8 convolutions."""
import sys

import numpy as np
import torch
import torch.nn as nn
from openvino import Core, save_model
from openvino import opset8 as ops

sys.path.insert(0, "/hyperai/home/katago-bench/scripts")
from katago_torch import load_torch_model

ONNX_PATH = "/hyperai/home/katago-bench/onnx/b18.onnx"
CKPT = "/hyperai/home/katago-bench/traindata/qat_student.pt"
OUT = "/hyperai/home/katago-bench/onnx/b18-fq-unfused.xml"

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
            w_deq = (wq * ws.view(-1, 1, 1, 1))  # pre-quantized weight VALUES
            qmap[name.replace("/", ".")] = (s, zp, w_deq.numpy(), ws.numpy())
print(f"params for {len(qmap)} convs")

core = Core()
ovm = core.read_model(ONNX_PATH)

def const(arr):
    return ops.constant(arr.astype(np.float32))

n_act, n_w = 0, 0
for op in list(ovm.get_ops()):
    if op.get_type_name() != "Convolution":
        continue
    name = op.get_friendly_name()
    if name not in qmap:
        continue
    s, zp, w_deq, ws = qmap[name]
    # activation FQ: u8-style, 256 levels, asymmetric [vmin, vmin+255*s]
    vmin = -zp * s
    vmax = vmin + 255.0 * s
    src = op.input(0).get_source_output()
    fq = ops.fake_quantize(src, const(np.array(vmin)), const(np.array(vmax)),
                           const(np.array(vmin)), const(np.array(vmax)), 256)
    op.input(0).replace_source_output(fq.output(0))
    n_act += 1
    # weight: replace the constant with pre-quantized values + per-channel symmetric FQ (255 levels)
    w_src = op.input(1).get_source_output()
    w_node = w_src.get_node()
    if w_node.get_type_name() == "Convert":
        w_node = w_node.input(0).get_source_output().get_node()
    if w_node.get_type_name() == "Constant":
        shape = w_deq.shape  # (C,kH,kW,C_in)? OV conv weight layout (C_out, C_in, kH, kW)
        wsc = np.ones(shape, dtype=np.float32)
        # OV Convolution weight layout: [out, in, h, w] -> broadcast per out-channel
        wsc = ws.reshape(-1, *([1] * (len(shape) - 1))).astype(np.float32)
        new_w = ops.constant(w_deq.astype(np.float32))
        lo = np.broadcast_to(-127.0 * ws.reshape(-1, 1, 1, 1), shape).astype(np.float32).copy()
        hi = np.broadcast_to(127.0 * ws.reshape(-1, 1, 1, 1), shape).astype(np.float32).copy()
        fq_w = ops.fake_quantize(new_w, ops.constant(lo), ops.constant(hi),
                                 ops.constant(lo), ops.constant(hi), 255)
        op.input(1).replace_source_output(fq_w.output(0))
        n_w += 1
print(f"inserted: {n_act} activation FQs, {n_w} weight FQs")
save_model(ovm, OUT)
print(f"saved {OUT}")
print("EXPORT_V2_DONE")
