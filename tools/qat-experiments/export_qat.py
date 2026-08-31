"""Export the QAT-trained student to QDQ-ONNX (QuantizeLinear/DequantizeLinear), loadable
by OpenVINO/oneDNN as real int8 kernels."""
import sys

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper

sys.path.insert(0, "/hyperai/home/katago-bench/scripts")
from katago_torch import load_torch_model

ONNX_PATH = "/hyperai/home/katago-bench/onnx/b18.onnx"
CKPT = "/hyperai/home/katago-bench/traindata/qat_student.pt"
OUT = "/hyperai/home/katago-bench/onnx/b18-int8-qat.onnx"

torch.set_num_threads(4)
base, _ = load_torch_model(ONNX_PATH)

# rebuild student architecture (same as trainer)
import copy
import torch.nn as nn
import torch.nn.functional as F

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
print(f"loaded checkpoint at step {ckpt['step']}")

# map: onnx conv node name -> (act scale, zp, int8 weights, w scale vec)
qmap = {}
with torch.no_grad():
    for name, mod in student.named_modules():
        if isinstance(mod, QuantConv2d):
            scale = float(mod.learn_act_scale.abs().clamp_min(1e-12))
            zp = float(torch.round(mod.learn_act_zp))
            w = mod.conv.weight.data
            ws = (w.abs().amax(dim=(1, 2, 3)) / 127.0).clamp_min(1e-12)  # (out,)
            wq = torch.clamp(torch.round(w / ws.view(-1, 1, 1, 1)), -127, 127).to(torch.int8)
            qmap[name.replace("/", ".")] = (scale, zp, wq.numpy(), ws.numpy())
print(f"export params for {len(qmap)} convs")

model = onnx.load(ONNX_PATH)
graph = model.graph
inits = {i.name: i for i in graph.initializer}

new_nodes = []
new_inits = []
node_by_output = {}
for n in graph.node:
    for o in n.output:
        node_by_output[o] = n

for node in graph.node:
    if node.op_type == "Conv" and node.name in qmap and node.input[1] in inits:
        scale, zp, wq, ws = qmap[node.name]
        base_w = numpy_helper.to_array(inits[node.input[1]])
        if tuple(wq.shape) != tuple(base_w.shape):
            print(f"  SKIP (shape mismatch) {node.name}")
            new_nodes.append(node)
            continue
        # --- activation QDQ on the data input edge
        x = node.input[0]
        xq, xdq = node.name + "_xq", node.name + "_xdq"
        xs_name, xz_name = node.name + "_xscale", node.name + "_xzp"
        new_inits.append(numpy_helper.from_array(np.array(scale, dtype=np.float32), name=xs_name))
        new_inits.append(numpy_helper.from_array(np.array(int(round(zp)), dtype=np.uint8), name=xz_name))
        new_nodes.append(helper.make_node("QuantizeLinear", [x, xs_name, xz_name], [xq], name=node.name + "_qx"))
        new_nodes.append(helper.make_node("DequantizeLinear", [xq, xs_name, xz_name], [xdq], name=node.name + "_dqx"))
        # --- weight as int8 initializer + per-channel QDQ (axis=0)
        wq_name = node.name + "_wq"
        ws_name = node.name + "_wscale"
        wz_name = node.name + "_wzp"
        wdq = node.name + "_wdq"
        new_inits.append(numpy_helper.from_array(wq, name=wq_name))
        new_inits.append(numpy_helper.from_array(ws.astype(np.float32), name=ws_name))
        new_inits.append(numpy_helper.from_array(np.zeros(len(ws), dtype=np.int8), name=wz_name))
        new_nodes.append(helper.make_node("DequantizeLinear", [wq_name, ws_name, wz_name], [wdq], name=node.name + "_dqw", axis=0))
        new_nodes.append(helper.make_node("Conv", [xdq, wdq] + list(node.input[2:]), list(node.output), name=node.name))
    else:
        new_nodes.append(node)

del graph.node[:]
graph.node.extend(new_nodes)
graph.initializer.extend(new_inits)
onnx.save(model, OUT)
print(f"saved {OUT}")
print("EXPORT_DONE")
