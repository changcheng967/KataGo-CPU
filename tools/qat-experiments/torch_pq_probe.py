"""Torch p+q-only quantization eval, using the trainer's exact build code."""
import copy
import sys

sys.path.insert(0, "/hyperai/home/katago-bench/scripts")
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from katago_torch import load_torch_model

torch.set_num_threads(8)
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

    def fake_q(self, x, scale, zp, qmin, qmax):
        q = torch.clamp(torch.round(x / scale + zp), qmin, qmax)
        dq = (q - zp) * scale
        return x + (dq - x).detach()

    def forward(self, x):
        scale = self.learn_act_scale.abs().clamp_min(1e-12)
        zp = torch.round(self.learn_act_zp)
        xq = self.fake_q(x, scale, zp, 0, 255)
        wq = self.fake_q(self.conv.weight, self.w_scale, 0.0, -127, 127)
        return F.conv2d(xq, wq, self.conv.bias, self.conv.stride,
                        self.conv.padding, self.conv.dilation, self.conv.groups)

ckpt = torch.load("/hyperai/home/katago-bench/traindata/qat_student.pt", map_location="cpu", weights_only=False)
sd = ckpt["student"]

student = copy.deepcopy(base)
targets = [n for n, mod in student.named_modules() if isinstance(mod, nn.Conv2d) and "blocks" in n]
keep = {"model/blocks/0/normactconvp/conv", "model/blocks/0/normactconvq/conv"}
n_q = 0
for name in targets:
    if name not in keep:
        continue
    wrapper = QuantConv2d(student.get_submodule(name))
    with torch.no_grad():
        wrapper.learn_act_scale.copy_(sd[name + ".learn_act_scale"])
        wrapper.learn_act_zp.copy_(sd[name + ".learn_act_zp"])
        wrapper.w_scale.copy_(sd[name + ".w_scale"])
    student.__setattr__(name, wrapper)   # GraphModule children are slash-named top-level
    n_q += 1
print(f"wrapped: {n_q}")

teacher = copy.deepcopy(base)
student.eval(); teacher.eval()

raw = np.fromfile("/hyperai/home/katago-bench/calib_real.f32", dtype=np.float32)
ROW = 361 + 22 * 361 + 19
rows = raw[: (len(raw) // ROW) * ROW].reshape(-1, ROW)
N = 128
masks = rows[:N, :361].reshape(-1, 1, 19, 19).astype(np.float32)
sp = rows[:N, 361:361 + 7942].reshape(-1, 22, 19, 19).astype(np.float32)
gl = rows[:N, 361 + 7942:].reshape(-1, 19, 1, 1).astype(np.float32)

best = 0
with torch.no_grad():
    for s in range(0, N, 32):
        so = student(torch.tensor(sp[s:s+32]), torch.tensor(gl[s:s+32]), torch.tensor(masks[s:s+32]))
        to = teacher(torch.tensor(sp[s:s+32]), torch.tensor(gl[s:s+32]), torch.tensor(masks[s:s+32]))
        lb = torch.cat([so[0][:, 0].flatten(1), so[1][:, 0].flatten(1)], 1)
        lt = torch.cat([to[0][:, 0].flatten(1), to[1][:, 0].flatten(1)], 1)
        best += (lb.argmax(1) == lt.argmax(1)).sum().item()
print(f"TORCH p+q only: bestMove {100 * best / N:.1f}%   (OV p+q measured 53.1%)")
print("TORCH_PQ_DONE")
