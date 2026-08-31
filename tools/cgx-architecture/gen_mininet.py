"""Generate full KataGo-shaped transformer mini-nets as ONNX for CPU design sweeps.
Faithful cost structure: nbt-wrapped pre-norm attention+SwiGLU blocks with honest glue
(RMSNorm chains, attention transposes, board masking). Validation gate: the tf3-b10c512
configuration must reproduce the real net's measured latency."""
import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def make_net(C, inner, F, n_blocks, n_sub, heads, attention, out_path, seed=0, attn_every=1, fused_qkv=False, use_mask=True, attn_width=None, head_dim=32, linear_attn=False, no_nbt=False, stem=3):
    rng = np.random.default_rng(seed)
    inits, nodes = [], []

    def w(name, shape, scale=0.05):
        arr = (rng.standard_normal(shape) * scale).astype(np.float32)
        inits.append(numpy_helper.from_array(arr, name=name))
        return name

    def k(name, arr):
        inits.append(numpy_helper.from_array(arr, name=name))
        return name

    def n(op, inputs, outputs, **kw):
        nodes.append(helper.make_node(op, inputs, outputs, **kw))

    def rmsnorm(x, width, name):
        n("Mul", [x, x], [name + "sq"])
        n("ReduceMean", [name + "sq"], [name + "m"], axes=[1], keepdims=1)
        n("Add", [name + "m", k(name + "e", np.array(1e-5, dtype=np.float32))], [name + "me"])
        n("Sqrt", [name + "me"], [name + "s"])
        n("Div", [x, name + "s"], [name + "d"])
        n("Mul", [name + "d", w(name + "w", (1, width, 1, 1))], [name + "o"])
        return name + "o"

    def silu(x, name):
        n("Sigmoid", [x], [name + "sg"])
        n("Mul", [x, name + "sg"], [name + "o"])
        return name + "o"

    def conv1x1(x, cin, cout, name):
        n("Conv", [x, w(name + "w", (cout, cin, 1, 1))], [name + "o"])
        return name + "o"

    def attention_sub(x, inner_, name):
        aw = attn_width if attn_width else inner_
        hd = head_dim
        nh = aw // hd
        h = rmsnorm(x, inner_, name + "n")
        if fused_qkv:
            qkv = conv1x1(h, inner_, 3 * aw, name + "qkv")
            parts = []
            for i, t in enumerate("qkv"):
                n("Slice", [qkv, k(name + t + "st", np.array([i * aw], dtype=np.int64)),
                            k(name + t + "en", np.array([(i + 1) * aw], dtype=np.int64)),
                            k(name + t + "ax", np.array([1], dtype=np.int64))], [name + t + "sl"])
                parts.append(name + t + "sl")
        else:
            parts = [conv1x1(h, inner_, aw, name + t) for t in "qkv"]
        qk = []
        for t, v in zip("qkv", parts):
            n("Reshape", [v, k(name + t + "r", np.array([0, nh, hd, 361], dtype=np.int64))], [name + t + "rs"])
            if t == "k":
                qk.append(name + t + "rs")
            else:
                n("Transpose", [name + t + "rs"], [name + t + "th"], perm=[0, 1, 3, 2])
                qk.append(name + t + "th")
        if linear_attn:
            # phi(x) = elu(x)+1 on Q and K; y = phiQ @ (phiK @ V), heads dim hd
            pq, pk = qk[0], qk[1]  # Q: [N,H,361,hd], K: [N,H,hd,361]
            n("Elu", [pq], [name + "qe"]); n("Add", [name + "qe", k(name + "q1", np.array(1.0, dtype=np.float32))], [name + "pq"])
            n("Elu", [pk], [name + "ke"]); n("Add", [name + "ke", k(name + "k1", np.array(1.0, dtype=np.float32))], [name + "pk2"])
            v = qk[2]  # V: [N,H,361,hd]
            if use_mask:
                # zero off-board K columns and V rows so they contribute nothing
                n("Reshape", [mask_in, k(name + "mr", np.array([0, 1, 361], dtype=np.int64))], [name + "mk"])
                n("Unsqueeze", [name + "mk", k(name + "mu", np.array([2], dtype=np.int64))], [name + "mkq"])  # [N,1,1,361]
                n("Mul", [name + "pk2", name + "mkq"], [name + "pkm"])
                n("Unsqueeze", [name + "mk", k(name + "mu2", np.array([3], dtype=np.int64))], [name + "mkv"])  # [N,1,361,1]
                n("Mul", [v, name + "mkv"], [name + "vm"])
            else:
                name = name  # keep names
                n("Identity", [name + "pk2"], [name + "pkm"]); n("Identity", [v], [name + "vm"])
            n("MatMul", [name + "pkm", name + "vm"], [name + "kv"])      # [N,H,hd,hd]
            n("MatMul", [name + "pq", name + "kv"], [name + "av"])       # [N,H,361,hd]
            n("Transpose", [name + "av"], [name + "avt2"], perm=[0, 1, 3, 2])
            n("Reshape", [name + "avt2", k(name + "avr", np.array([0, aw, 19, 19], dtype=np.int64))], [name + "av2"])
            o = conv1x1(name + "av2", aw, inner_, name + "o")
            n("Add", [x, o], [name + "res"])
            return name + "res"
        n("MatMul", [qk[0], qk[1]], [name + "sc"])
        if use_mask:
            n("Mul", [name + "sc", k(name + "scs", np.array(1.0 / np.sqrt(hd), dtype=np.float32))], [name + "sc2"])
            n("Reshape", [mask_in, k(name + "mr", np.array([0, 1, 361], dtype=np.int64))], [name + "mk"])
            n("Sub", [name + "mk", k(name + "m1", np.array(1.0, dtype=np.float32))], [name + "mk0"])
            n("Mul", [name + "mk0", k(name + "mb", np.array(1e4, dtype=np.float32))], [name + "mneg"])
            n("Unsqueeze", [name + "mneg", k(name + "mu", np.array([2], dtype=np.int64))], [name + "muo"])
            n("Add", [name + "sc2", name + "muo"], [name + "scm"])
        else:
            n("Mul", [name + "sc", k(name + "scs", np.array(1.0 / np.sqrt(hd), dtype=np.float32))], [name + "scm"])
        n("Softmax", [name + "scm"], [name + "sm"], axis=-1)
        n("MatMul", [name + "sm", qk[2]], [name + "av"])
        n("Transpose", [name + "av"], [name + "avt2"], perm=[0, 1, 3, 2])
        n("Reshape", [name + "avt2", k(name + "avr", np.array([0, aw, 19, 19], dtype=np.int64))], [name + "av2"])
        o = conv1x1(name + "av2", aw, inner_, name + "o")
        n("Add", [x, o], [name + "res"])
        return name + "res"

    def ff_sub(x, inner_, F_, name):
        h = rmsnorm(x, inner_, name + "n")
        g = conv1x1(h, inner_, F_, name + "g")
        u = conv1x1(h, inner_, F_, name + "u")
        gs = silu(g, name + "gs")
        n("Mul", [gs, u], [name + "m"])
        d = conv1x1(name + "m", F_, inner_, name + "d")
        n("Add", [x, d], [name + "res"])
        return name + "res"

    mask_in = "mask"
    # ---- stem
    if stem == 3:
        n("Conv", ["spatial", w("stemw", (C, 22, 3, 3))], ["s1"], pads=[1, 1, 1, 1])
    else:
        n("Conv", ["spatial", w("stemw", (C, 22, 5, 5))], ["s1"], pads=[2, 2, 2, 2])
    n("Reshape", ["global", k("gr", np.array([0, 19], dtype=np.int64))], ["g2"])
    n("MatMul", ["g2", w("gw", (19, C))], ["g3"])
    n("Reshape", ["g3", k("gr2", np.array([-1, C, 1, 1], dtype=np.int64))], ["g4"])
    n("Add", ["s1", "g4"], ["x0"])

    x = "x0"
    for b in range(n_blocks):
        res = x
        if no_nbt:
            wdt, ff_in = C, F
            y = x
            for s in range(n_sub):
                if attention and (b * n_sub + s) % attn_every == 0:
                    y = attention_sub(y, C, f"b{b}s{s}a")
                y = ff_sub(y, C, F, f"b{b}s{s}f")
            n("Add", [res, y], [f"b{b}o"])
        else:
            xb = rmsnorm(x, C, f"b{b}pn")
            xb = silu(xb, f"b{b}pa")
            xb = conv1x1(xb, C, inner, f"b{b}p")
            y = xb
            for s in range(n_sub):
                if attention and (b * n_sub + s) % attn_every == 0:
                    y = attention_sub(y, inner, f"b{b}s{s}a")
                y = ff_sub(y, inner, F, f"b{b}s{s}f")
            y = rmsnorm(y, inner, f"b{b}qn")
            y = silu(y, f"b{b}qa")
            y = conv1x1(y, inner, C, f"b{b}q")
            n("Add", [res, y], [f"b{b}o"])
        x = f"b{b}o"

    x = rmsnorm(x, C, "tipn")
    x = silu(x, "tipa")

    # ---- heads
    n("Conv", [x, w("polw", (2, C, 3, 3))], ["OutputPolicy"], pads=[1, 1, 1, 1])
    n("Conv", [x, w("ownw", (1, C, 1, 1))], ["OutputOwnership"])
    n("Mul", [x, mask_in], ["xm"])
    n("ReduceMean", ["xm"], ["pmean"], axes=[2, 3], keepdims=0)
    n("ReduceMax", ["xm"], ["vmaxx"], axes=[2, 3], keepdims=0)
    n("Concat", ["pmean", "vmaxx"], ["vcat"], axis=1)
    n("MatMul", ["pmean", w("ppw", (C, 2))], ["ppl"])
    n("Reshape", ["ppl", k("pplr", np.array([-1, 2, 1, 1], dtype=np.int64))], ["OutputPolicyPass"])
    n("MatMul", ["vcat", w("vw", (2 * C, 3))], ["vl"])
    n("Reshape", ["vl", k("vlr", np.array([-1, 3, 1, 1], dtype=np.int64))], ["OutputValue"])
    n("MatMul", ["vcat", w("sw", (2 * C, 6))], ["sl"])
    n("Reshape", ["sl", k("slr", np.array([-1, 6, 1, 1], dtype=np.int64))], ["OutputScoreValue"])

    graph = helper.make_graph(
        nodes, "mininet",
        [helper.make_tensor_value_info(t, TensorProto.FLOAT, s) for t, s in
         [("spatial", [None, 22, 19, 19]), ("global", [None, 19, 1, 1]), ("mask", [None, 1, 19, 19])]],
        [helper.make_tensor_value_info(t, TensorProto.FLOAT, s) for t, s in
         [("OutputPolicyPass", [None, 2, 1, 1]), ("OutputPolicy", [None, 2, 19, 19]),
          ("OutputValue", [None, 3, 1, 1]), ("OutputScoreValue", [None, 6, 1, 1]),
          ("OutputOwnership", [None, 1, 19, 19])]],
        initializer=inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, out_path)
    npar = sum(int(np.prod(t.dims)) for t in inits)
    return npar


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--C", type=int, default=512)
    p.add_argument("--inner", type=int, default=256)
    p.add_argument("--F", type=int, default=768)
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--sub", type=int, default=3)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--no-attention", action="store_true")
    p.add_argument("--attn-every", type=int, default=1)
    p.add_argument("--fused-qkv", action="store_true")
    p.add_argument("--no-mask", action="store_true")
    p.add_argument("--attn-width", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=32)
    p.add_argument("--linear-attn", action="store_true")
    p.add_argument("--no-nbt", action="store_true")
    p.add_argument("--stem", type=int, default=3)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    npars = make_net(a.C, a.inner, a.F, a.blocks, a.sub, a.heads, not a.no_attention, a.out, attn_every=a.attn_every, fused_qkv=a.fused_qkv, use_mask=not a.no_mask, attn_width=a.attn_width, head_dim=a.head_dim, linear_attn=a.linear_attn, no_nbt=a.no_nbt, stem=a.stem)
    print(f"saved {a.out}: {npars/1e6:.2f}M params (C={a.C} inner={a.inner} F={a.F} "
          f"blocks={a.blocks}x{a.sub} heads={a.heads} attn={not a.no_attention})")
