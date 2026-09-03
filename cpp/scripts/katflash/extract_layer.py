"""Extract one attention window (nodes 33-54 of block 0 stack 0) as a standalone
ONNX graph: inputs q_proj/k_proj/v_proj outputs + InputMask bias; output =
attnnhwc reshape output. Also emit a variant with the fused op for A/B."""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

m = onnx.load(sys.argv[1])
g = m.graph
byname = {n.name: n for n in g.node}
prefix = 'model.blocks.0.blockstack.0'

window = [
    f'{prefix}/q/reshape', f'{prefix}/qh/transpose',
    f'{prefix}/k/reshape', f'{prefix}/kh/transpose',
    f'{prefix}/v/reshape', f'{prefix}/vh/transpose',
    f'{prefix}/qrope/rope_swap', f'{prefix}/qrope/rope_t1',
    f'{prefix}/qrope/rope_t2', f'{prefix}/qrope/rope_out',
    f'{prefix}/krope/rope_swap', f'{prefix}/krope/rope_t1',
    f'{prefix}/krope/rope_t2', f'{prefix}/krope/rope_out',
    f'{prefix}/khT/transpose',
    f'{prefix}/scores', f'{prefix}/scoresscaled', f'{prefix}/scoresmasked',
    f'{prefix}/softmax', f'{prefix}/sv',
    f'{prefix}/svT/transpose', f'{prefix}/attnnhwc/reshape',
]
nodes = [byname[w] for w in window]

# graph inputs: the exact external tensor names the window consumes
def make_in(name, dims):
    return helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, dims)
qname = byname[f'{prefix}/q/reshape'].input[0]
kname = byname[f'{prefix}/k/reshape'].input[0]
vname = byname[f'{prefix}/v/reshape'].input[0]
mname = byname[f'{prefix}/scoresmasked'].input[1]
ins = [
    make_in(qname, ['batch', 19, 19, 192]),
    make_in(kname, ['batch', 19, 19, 192]),
    make_in(vname, ['batch', 19, 19, 192]),
    make_in(mname, ['batch', 1, 1, 361]),
]
out = helper.make_tensor_value_info(f'{prefix}/attnnhwc/reshape/70', onnx.TensorProto.FLOAT, ['batch', 19, 19, 192])

# collect initializers referenced by the window (rope tables, shapes, scale)
init_names = set()
for n in nodes:
    for i in n.input:
        if i not in byname and not any(i in g.output for g in [g]) :
            init_names.add(i)
inits = [i for i in g.initializer if i.name in init_names]

ng = helper.make_graph(nodes, 'attn0', ins, [out], inits)
nm = helper.make_model(ng, opset_imports=list(m.opset_import))
nm.ir_version = m.ir_version
onnx.save(nm, sys.argv[2])
print('wrote', sys.argv[2], 'nodes:', len(nodes), 'inits:', len(inits))
