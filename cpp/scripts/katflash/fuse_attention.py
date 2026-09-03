"""Fuse KataGo transformer attention into one custom ONNX op (KatFlashAttention).

Replaces, per attention layer, the 22-node window
  3x(Reshape,Transpose) + RoPE(q) + RoPE(k) + khT + scores + scale + mask
  + Softmax + sv + svT + mergeReshape
with a single node:

  KatFlashAttention(q_proj_out[N,361,192], k_proj_out, v_proj_out,
                    mask_bias[N,1,1,361],
                    ropecos[1,H,361,dh], ropesin[1,H,361,dh], swapidx[dh],
                    scale[1])
      -> [N,19,19,192]   (feeds out_proj unchanged)

Everything the kernel needs is derived from input shapes; the node has no
attributes. Metadata properties are preserved so KataGo can still load the file.

Usage: python3 fuse_attention.py <in.onnx> <out.onnx>
"""
import sys

import numpy as np
import onnx
from onnx import helper


def fuse(in_path, out_path):
    m = onnx.load(in_path)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    byname = {n.name: n for n in g.node}
    byout = {}
    for n in g.node:
        for o in n.output:
            byout[o] = n

    # attention anchors: the '/sv' matmul of every layer
    sv_nodes = [n for n in g.node if n.name.endswith('/sv')]
    assert len(sv_nodes) == 20, f"expected 20 attention layers, got {len(sv_nodes)}"

    new_nodes = []
    first_node_to_fused = {}
    removed = set()

    for sv in sv_nodes:
        prefix = sv.name[: -len('/sv')]
        # walk up from sv to collect the whole window
        names = [
            '/q/reshape', '/q/reshape',  # placeholder; collected below by name
        ]
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
        for w in window:
            assert w in byname, f'missing node {w}'
            removed.add(w)

        # proj nodes use dots before the leaf, window nodes use slashes
        stack_prefix = prefix  # model.blocks.B.blockstack.S
        q_proj = byname[f'{stack_prefix}.q_proj/nhwc'].output[0]
        k_proj = byname[f'{stack_prefix}.k_proj/nhwc'].output[0]
        v_proj = byname[f'{stack_prefix}.v_proj/nhwc'].output[0]
        mask = byname[f'{prefix}/scoresmasked'].input[1]  # InputMask/bias/reshape/...
        attn_out = byname[f'{prefix}/attnnhwc/reshape'].output[0]

        # constants: prefer q-side rope tables (verified equal to k-side)
        cos_name = byname[f'{prefix}/qrope/rope_t1'].input[1]
        sin_name = byname[f'{prefix}/qrope/rope_t2'].input[1]
        swap_name = byname[f'{prefix}/qrope/rope_swap'].input[1]
        scale_name = byname[f'{prefix}/scoresscaled'].input[1]

        fused = helper.make_node(
            'KatFlashAttention',
            [q_proj, k_proj, v_proj, mask, cos_name, sin_name, swap_name, scale_name],
            [attn_out],
            name=f'{prefix}/flashattn',
        )
        new_nodes.append(fused)
        first_node_to_fused[f'{prefix}/q/reshape'] = fused
        print(f'fused {prefix}: out={attn_out}')

    kept_nodes = []
    for n in g.node:
        if n.name in removed:
            # emit the fused op in place of the window's first node
            fused = first_node_to_fused.get(n.name)
            if fused is not None:
                kept_nodes.append(fused)
            continue
        kept_nodes.append(n)

    new_g = helper.make_graph(
        kept_nodes, g.name, g.input, g.output, list(g.initializer),
        doc_string=g.doc_string, value_info=g.value_info,
    )
    new_m = helper.make_model(
        new_g, opset_imports=list(m.opset_import), producer_name=m.producer_name,
    )
    new_m.ir_version = m.ir_version
    # preserve KataGo metadata contract
    for prop in m.metadata_props:
        new_m.metadata_props.append(prop)

    # checker cannot validate custom ops; ONNX does not require checking to save
    onnx.save(new_m, out_path)
    print(f'wrote {out_path}: {len(removed)} nodes removed, {len(new_nodes)} fused ops added')


if __name__ == '__main__':
    fuse(sys.argv[1], sys.argv[2])
