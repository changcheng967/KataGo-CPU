"""QAT by distillation: fake-quantize the trunk convs of the b18 torch model and
fine-tune it to match its own fp32 teacher on real training positions."""
import copy
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, "/hyperai/home/katago-bench/scripts")
from katago_torch import load_torch_model

ONNX_PATH = "/hyperai/home/katago-bench/onnx/b18.onnx"
CACHE = "/hyperai/home/katago-bench/traindata/inputs_u8.npz"
CALIB = "/hyperai/home/katago-bench/calib_real.f32"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
BATCH = 16
LR_W, LR_S = 3e-5, 5e-4

torch.set_num_threads(8)

# ---------------- data ----------------
cache = np.load(CACHE)
SP = cache["spatial"]
GL = cache["glob"]
N_TRAIN = len(SP) - 2048
print(f"train positions: {N_TRAIN}, held-out: 2048")

raw = np.fromfile(CALIB, dtype=np.float32)
ROW = 361 + 22 * 361 + 19
rows = raw[: (len(raw) // ROW) * ROW].reshape(-1, ROW)
eval_mask = rows[-256:, :361].reshape(-1, 1, 19, 19)
eval_sp = rows[-256:, 361:361 + 7942].reshape(-1, 22, 19, 19)
eval_gl = rows[-256:, 361 + 7942:].reshape(-1, 19, 1, 1)

def to_t(x):
    return torch.from_numpy(np.ascontiguousarray(x)).float()

# ---------------- models ----------------
base_model, input_names = load_torch_model(ONNX_PATH)
teacher = copy.deepcopy(base_model)
for p in teacher.parameters():
    p.requires_grad_(False)

# ---------------- fake quant ----------------
class QuantConv2d(nn.Module):
    """Conv2d with per-tensor u8 activation fake-quant and per-out-channel s8 weight
    fake-quant (oneDNN semantics), straight-through estimator."""

    def __init__(self, conv):
        super().__init__()
        self.conv = conv
        self.register_buffer("act_scale", torch.tensor(1.0))
        self.register_buffer("act_zp", torch.tensor(0.0))
        w = conv.weight.data
        self.register_buffer("w_scale", (w.abs().amax(dim=(1, 2, 3), keepdim=True) / 127.0).clamp_min(1e-12))
        self.learn_act_scale = nn.Parameter(torch.tensor(1.0))
        self.learn_act_zp = nn.Parameter(torch.tensor(0.0))

    def init_act(self, amin, amax):
        with torch.no_grad():
            s = torch.tensor((amax - amin) / 255.0).clamp_min(1e-12)
            self.act_scale.copy_(s)
            self.act_zp.copy_(torch.round(-amin / s))
            self.learn_act_scale.copy_(s)
            self.learn_act_zp.copy_(torch.round(-amin / s))

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

def build_student(model):
    student = copy.deepcopy(model)
    targets = []
    for name, mod in list(student.named_modules()):
        if isinstance(mod, nn.Conv2d) and "blocks" in name:
            targets.append(name)
    for name in targets:
        parent, _, leaf = name.rpartition(".")
        setattr(student.get_submodule(parent) if parent else student, leaf,
                QuantConv2d(student.get_submodule(name)))
    print(f"quantized conv modules: {len(targets)}")
    return student, targets

student, target_names = build_student(base_model)

# calibrate activation ranges with hooks through the teacher
print("calibrating activation ranges...")
mins, maxs = {}, {}
hooks = []
def mk_hook(name):
    def hook(mod, inp, out):
        x = inp[0]
        amin = float(x.min()); amax = float(x.max())
        if name not in mins:
            mins[name] = [amin, amax]
        else:
            mins[name][0] = min(mins[name][0], amin)
            mins[name][1] = max(mins[name][1], amax)
    return hook
handle_map = {}
for name, mod in teacher.named_modules():
    if isinstance(mod, nn.Conv2d) and "blocks" in name:
        handle_map[name] = mod.register_forward_hook(mk_hook(name))
with torch.no_grad():
    for i in range(0, 512, 32):
        teacher(to_t(SP[i:i+32]), to_t(GL[i:i+32].reshape(-1, 19, 1, 1)), to_t(SP[i:i+32, 0:1]))
for h in handle_map.values():
    h.remove()
n_inited = 0
for name, mod in student.named_modules():
    if isinstance(mod, QuantConv2d):
        tname = name
        if tname in mins:
            mod.init_act(mins[tname][0], mins[tname][1])
            n_inited += 1
print(f"calibrated {n_inited}/{len(mins)} quantizers")

# ---------------- loss ----------------
def forward_all(m, sp, gl, mk):
    outs = m(sp, gl, mk)
    if isinstance(outs, torch.Tensor):
        outs = [outs]
    return outs  # order: PolicyPass, Policy, Value, ScoreValue, Ownership

def distill_loss(s_outs, t_outs):
    # policy: KL over 362 base-channel moves (pass + board)
    def move_logits(o):
        return torch.cat([o[0][:, 0].flatten(1), o[1][:, 0].flatten(1)], dim=1)
    ls, lt = move_logits(s_outs), move_logits(t_outs)
    pt = F.softmax(lt, dim=1)
    logps = F.log_softmax(ls, dim=1)
    pol = -(pt * logps).sum(1).mean()
    # value: softmax-prob MSE
    vs = F.softmax(s_outs[2].flatten(1), dim=1)
    vt = F.softmax(t_outs[2].flatten(1), dim=1)
    val = F.mse_loss(vs, vt)
    # scoreValue: std-normalized MSE per channel
    st_ = t_outs[3].flatten(2)[..., 0]  # (N,6)
    ss = s_outs[3].flatten(2)[..., 0]
    std = st_.std(0, keepdim=True).clamp_min(1e-3)
    score = (((ss - st_) / std) ** 2).mean()
    # ownership
    os_ = s_outs[4]; ot = t_outs[4]
    ostd = ot.std().clamp_min(1e-3)
    own = (((os_ - ot) / ostd) ** 2).mean()
    return pol + 1.0 * val + 0.5 * score + 0.25 * own, pol.item(), val.item(), score.item(), own.item()

# ---------------- eval ----------------
def evaluate():
    student.eval()
    stats = {"best": 0, "n": 0, "wr": [], "kl": []}
    with torch.no_grad():
        for i in range(0, 256, 64):
            sp = to_t(eval_sp[i:i+64]); gl = to_t(eval_gl[i:i+64]); mk = to_t(eval_mask[i:i+64])
            so = forward_all(student, sp, gl, mk)
            to = forward_all(teacher, sp, gl, mk)
            ls = torch.cat([so[0][:, 0].flatten(1), so[1][:, 0].flatten(1)], 1)
            lt = torch.cat([to[0][:, 0].flatten(1), to[1][:, 0].flatten(1)], 1)
            stats["best"] += (ls.argmax(1) == lt.argmax(1)).sum().item()
            stats["n"] += len(ls)
            ps, pt = F.softmax(ls, 1), F.softmax(lt, 1)
            stats["kl"] += (pt * (torch.log(pt + 1e-12) - torch.log(ps + 1e-12))).sum(1).tolist()
            vs = F.softmax(so[2].flatten(1), 1)[:, 0]
            vt = F.softmax(to[2].flatten(1), 1)[:, 0]
            stats["wr"] += (vs - vt).abs().mul(100).tolist()
    student.train()
    kl = float(np.mean(stats["kl"])); wr = float(np.mean(stats["wr"])); wrmax = float(np.max(stats["wr"]))
    print(f"  EVAL bestMove {100*stats['best']/stats['n']:.1f}%  KL {kl:.6f}  wrMean {wr:.3f}% wrMax {wrmax:.2f}%")
    return 100 * stats["best"] / stats["n"]

# ---------------- train ----------------
print("initial eval (fake-quant, untrained):")
evaluate()

params_w, params_s = [], []
for name, p in student.named_parameters():
    if "learn_act" in name:
        params_s.append(p)
    else:
        params_w.append(p)
opt = torch.optim.AdamW([
    {"params": params_w, "lr": LR_W},
    {"params": params_s, "lr": LR_S},
], weight_decay=0.0)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

rng = np.random.default_rng(0)
student.train()
t0 = time.perf_counter()
best_agree = 0.0
for step in range(STEPS):
    idx = rng.integers(0, N_TRAIN, BATCH)
    sp = to_t(SP[idx]); gl = to_t(GL[idx].reshape(-1, 19, 1, 1)); mk = to_t(SP[idx, 0:1])
    with torch.no_grad():
        t_outs = forward_all(teacher, sp, gl, mk)
    s_outs = forward_all(student, sp, gl, mk)
    loss, pl, vl, sc, ow = distill_loss(s_outs, t_outs)
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 50 == 0:
        el = time.perf_counter() - t0
        print(f"step {step:5d}  loss {loss.item():.4f} (pol {pl:.4f} val {vl:.5f} sc {sc:.4f} ow {ow:.4f})  {el:.0f}s", flush=True)
    if step > 0 and step % 400 == 0:
        agree = evaluate()
        best_agree = max(best_agree, agree)
        torch.save({"student": student.state_dict(), "target_names": target_names,
                    "step": step}, "/hyperai/home/katago-bench/traindata/qat_student.pt")

print("final eval:")
agree = evaluate()
torch.save({"student": student.state_dict(), "target_names": target_names, "step": STEPS},
           "/hyperai/home/katago-bench/traindata/qat_student.pt")
print(f"QAT_DONE bestAgree={best_agree:.1f} finalAgree={agree:.1f}")
