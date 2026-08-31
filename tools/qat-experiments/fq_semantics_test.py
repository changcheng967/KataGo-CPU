"""Isolate the FQ semantics: single conv, torch QuantConv2d vs OV FQ-inserted graph."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from openvino import Core
from openvino import opset8 as ops

rng = np.random.default_rng(5)
C_in, C_out, K = 8, 16, 3
x = rng.normal(0, 1, (1, C_in, 19, 19)).astype(np.float32) * 0.5
w = rng.normal(0, 0.1, (C_out, C_in, K, K)).astype(np.float32)
b = rng.normal(0, 0.01, C_out).astype(np.float32)

# torch side
conv = nn.Conv2d(C_in, C_out, K, padding=1, bias=True)
with torch.no_grad():
    conv.weight.copy_(torch.tensor(w)); conv.bias.copy_(torch.tensor(b))
ws = np.abs(w).max(axis=(1, 2, 3)) / 127.0
amin, amax = float(x.min()), float(x.max())
s = (amax - amin) / 255.0
zp = round(-amin / s)
def fake_q(x, scale, zp, qmin, qmax):
    q = torch.clamp(torch.round(x / scale + zp), qmin, qmax)
    dq = (q - zp) * scale
    return x + (dq - x).detach()
with torch.no_grad():
    xq = fake_q(torch.tensor(x), torch.tensor(s), torch.tensor(float(zp)), 0, 255)
    wq = fake_q(conv.weight, torch.tensor(ws.reshape(-1, 1, 1, 1)), 0.0, -127, 127)
    y_t = F.conv2d(xq, wq, conv.bias, 1, 1).numpy()

# OV side
core = Core()
x_p = ops.parameter([1, C_in, 19, 19], np.float32)
w_c = ops.constant(w)
b_c = ops.constant(b)
y = ops.convolution(x_p, w_c, [1, 1], [1, 1], [1, 1], [1, 1])
model = None
from openvino import Model
m = Model([y], [x_p])
conv_op = None
for op in m.get_ops():
    if op.get_type_name() == "Convolution":
        conv_op = op
        break
vmin = -zp * s
vmax = vmin + 255.0 * s
src = conv_op.input(0).get_source_output()
fq = ops.fake_quantize(src, ops.constant(np.float32(vmin)), ops.constant(np.float32(vmax)),
                       ops.constant(np.float32(vmin)), ops.constant(np.float32(vmax)), 256)
conv_op.input(0).replace_source_output(fq.output(0))
wq_np = np.clip(np.round(w / ws.reshape(-1, 1, 1, 1)), -127, 127) * ws.reshape(-1, 1, 1, 1)
shape = w.shape
lo = np.broadcast_to(-127.0 * ws.reshape(-1, 1, 1, 1), shape).astype(np.float32).copy()
hi = np.broadcast_to(127.0 * ws.reshape(-1, 1, 1, 1), shape).astype(np.float32).copy()
fq_w = ops.fake_quantize(ops.constant(wq_np.astype(np.float32)), ops.constant(lo), ops.constant(hi),
                         ops.constant(lo), ops.constant(hi), 255)
conv_op.input(1).replace_source_output(fq_w.output(0))
def run(m):
    compiled = core.compile_model(m, "CPU")
    return np.array(compiled({0: x})[compiled.outputs[0]])

y_o = run(m)
print(f"both-FQ   : max diff {np.abs(y_t - y_o).max():.6f}")

# act-only variant
m2 = Model([y], [x_p])
c2 = next(op for op in m2.get_ops() if op.get_type_name() == "Convolution")
fq2 = ops.fake_quantize(c2.input(0).get_source_output(),
                        ops.constant(np.float32(vmin)), ops.constant(np.float32(vmax)),
                        ops.constant(np.float32(vmin)), ops.constant(np.float32(vmax)), 256)
c2.input(0).replace_source_output(fq2.output(0))
with torch.no_grad():
    y_t_act = F.conv2d(xq, conv.weight, conv.bias, 1, 1).numpy()
print(f"act-only  : max diff {np.abs(y_t_act - run(m2)).max():.6f}")

# weight-only variant
m3 = Model([y], [x_p])
c3 = next(op for op in m3.get_ops() if op.get_type_name() == "Convolution")
lo3 = np.broadcast_to(-127.0 * ws.reshape(-1, 1, 1, 1), shape).astype(np.float32).copy()
hi3 = np.broadcast_to(127.0 * ws.reshape(-1, 1, 1, 1), shape).astype(np.float32).copy()
fq3 = ops.fake_quantize(ops.constant(wq_np.astype(np.float32)), ops.constant(lo3), ops.constant(hi3),
                        ops.constant(lo3), ops.constant(hi3), 255)
c3.input(1).replace_source_output(fq3.output(0))
with torch.no_grad():
    y_t_w = F.conv2d(torch.tensor(x), wq, conv.bias, 1, 1).numpy()
print(f"weight-only: max diff {np.abs(y_t_w - run(m3)).max():.6f}")
