"""Generate full KataGo-shaped transformer mini-nets as ONNX for CPU design sweeps.
Faithful cost structure: nbt-wrapped pre-norm attention+SwiGLU blocks with honest glue
(RMSNorm chains, attention transposes, board masking). Validation gate: the tf3-b10c512
configuration must reproduce the real net's measured latency."""
import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def make_net(C, inner, F, n_blocks, n_sub, heads, attention, out_path, seed=0, attn_every=1):
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

    def attention_sub(x, inner_, heads_, name):
        h = rmsnorm(x, inner_, name + "n")
        qk = []
        for t in "qkv":
            v = conv1x1(h, inner_, inner_, name + t)
            n("Reshape", [v, k(name + t + "r", np.array([0, heads_, 32, 361], dtype=np.int64))], [name + t + "rs"])
            if t == "k":
                qk.append(name + t + "rs")
            else:
                n("Transpose", [name + t + "rs"], [name + t + "th"], perm=[0, 1, 3, 2])
                qk.append(name + t + "th")
        n("MatMul", [qk[0], qk[1]], [name + "sc"])
        n("Mul", [name + "sc", k(name + "scs", np.array(1.0 / 32.0, dtype=np.float32))], [name + "sc2"])
        n("Reshape", [mask_in, k(name + "mr", np.array([0, 1, 361], dtype=np.int64))], [name + "mk"])
        n("Sub", [name + "mk", k(name + "m1", np.array(1.0, dtype=np.float32))], [name + "mk0"])
        n("Mul", [name + "mk0", k(name + "mb", np.array(1e4, dtype=np.float32))], [name + "mneg"])
        n("Unsqueeze", [name + "mneg", k(name + "mu", np.array([2], dtype=np.int64))], [name + "muo"])
        n("Add", [name + "sc2", name + "muo"], [name + "scm"])
        n("Softmax", [name + "scm"], [name + "sm"], axis=-1)
        n("MatMul", [name + "sm", qk[2]], [name + "av"])
        n("Transpose", [name + "av"], [name + "avt2"], perm=[0, 1, 3, 2])
        n("Reshape", [name + "avt2", k(name + "avr", np.array([0, inner_, 19, 19], dtype=np.int64))], [name + "av2"])
        o = conv1x1(name + "av2", inner_, inner_, name + "o")
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
    n("Conv", ["spatial", w("stemw", (C, 22, 3, 3))], ["s1"], pads=[1, 1, 1, 1])
    n("Reshape", ["global", k("gr", np.array([0, 19], dtype=np.int64))], ["g2"])
    n("MatMul", ["g2", w("gw", (19, C))], ["g3"])
    n("Reshape", ["g3", k("gr2", np.array([-1, C, 1, 1], dtype=np.int64))], ["g4"])
    n("Add", ["s1", "g4"], ["x0"])

    x = "x0"
    for b in range(n_blocks):
        res = x
        xb = rmsnorm(x, C, f"b{b}pn")
        xb = silu(xb, f"b{b}pa")
        xb = conv1x1(xb, C, inner, f"b{b}p")
        y = xb
        for s in range(n_sub):
            if attention and (b * n_sub + s) % attn_every == 0:
                y = attention_sub(y, inner, heads, f"b{b}s{s}a")
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
    p.add_argument("--out", required=True)
    a = p.parse_args()
    npars = make_net(a.C, a.inner, a.F, a.blocks, a.sub, a.heads, not a.no_attention, a.out, attn_every=a.attn_every)
    print(f"saved {a.out}: {npars/1e6:.2f}M params (C={a.C} inner={a.inner} F={a.F} "
          f"blocks={a.blocks}x{a.sub} heads={a.heads} attn={not a.no_attention})")
