"""Round K: train CGX variants by distillation from tf3 teacher outputs, then
measure teacher-agreement (policy KL, best-move, value) on held-out positions."""
import copy
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import onnx
from onnx2torch import convert

torch.set_num_threads(8)

ONNX_IN = sys.argv[1]
TAG = sys.argv[2]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 300
BATCH = 32

CACHE = "/hyperai/home/katago-bench/traindata/inputs_u8.npz"
TEACH = "/hyperai/home/katago-bench/traindata/teacher_out.npz"

print(f"[{TAG}] loading {ONNX_IN}")
model = convert(onnx.load(ONNX_IN))
for p in model.parameters():
    p.requires_grad_(True)

cache = np.load(CACHE)
sp_all, gl_all = cache["spatial"], cache["glob"]
t = np.load(TEACH)
tp_pol = t["OutputPolicy"].astype(np.float32)   # [N,2,19,19]
tp_pass = t["OutputPolicyPass"].astype(np.float32)  # [N,2,1,1]
tval = t["OutputValue"].astype(np.float32)      # [N,3,1,1]

N = min(len(sp_all), len(tp_pol))
N_TRAIN = N - 1024
N_TEST = 512
print(f"[{TAG}] positions: {N}, train {N_TRAIN}, test {N_TEST}")

def batch(idx):
    sp = torch.from_numpy(sp_all[idx].astype(np.float32))
    gl = torch.from_numpy(gl_all[idx].reshape(len(idx), 19, 1, 1).astype(np.float32))
    mk = sp[:, 0:1]
    return sp, gl, mk

def teacher_targets(idx):
    # base policy over 362 moves
    pol = tp_pol[idx][:, 0].reshape(len(idx), -1)
    pas = tp_pass[idx][:, 0].reshape(len(idx), -1)
    logits = np.concatenate([pas, pol], 1)
    return torch.from_numpy(logits), torch.from_numpy(tval[idx][:, :, 0, 0])

LR = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3
WARMUP = 100
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=WARMUP/STEPS)
rng = np.random.default_rng(0)
model.train()
t0 = time.perf_counter()
for step in range(STEPS):
    idx = rng.integers(0, N_TRAIN, BATCH)
    sp, gl, mk = batch(idx)
    out = model(sp, gl, mk)
    out = out if isinstance(out, list) else [out]
    # find policy outputs by shape: (B,2,1,1)=pass, (B,2,19,19)=policy, (B,3,1,1)=value
    pas = pol = val = None
    for o in out:
        if o.dim() == 4 and o.shape[1] == 2 and o.shape[2] == 1:
            pas = o
        elif o.dim() == 4 and o.shape[1] == 2 and o.shape[2] == 19:
            pol = o
        elif o.dim() == 4 and o.shape[1] == 3:
            val = o
    s_logits = torch.cat([pas[:, 0].flatten(1), pol[:, 0].flatten(1)], 1)
    tl, tv = teacher_targets(idx)
    loss_pol = F.cross_entropy(s_logits, tl.softmax(1))
    loss_val = F.mse_loss(val.flatten(1).softmax(1), tv.softmax(1))
    loss = loss_pol + 0.5 * loss_val
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 25 == 0:
        print(f"[{TAG}] step {step:4d} loss {loss.item():.4f} (pol {loss_pol.item():.4f} val {loss_val.item():.5f}) {time.perf_counter()-t0:.0f}s", flush=True)

# ---- evaluation on held-out
model.eval()
test_idx = np.arange(N_TRAIN, N_TRAIN + N_TEST)
agree = 0
kls = []
with torch.no_grad():
    for s in range(0, N_TEST, 64):
        idx = test_idx[s:s+64]
        sp, gl, mk = batch(idx)
        out = model(sp, gl, mk)
        out = out if isinstance(out, list) else [out]
        pas = pol = None
        for o in out:
            if o.dim() == 4 and o.shape[1] == 2 and o.shape[2] == 1:
                pas = o
            elif o.dim() == 4 and o.shape[1] == 2 and o.shape[2] == 19:
                pol = o
        sl = torch.cat([pas[:, 0].flatten(1), pol[:, 0].flatten(1)], 1).numpy()
        tl, _ = teacher_targets(idx)
        tlp = tl.numpy()
        agree += (sl.argmax(1) == tlp.argmax(1)).sum()
        ps = np.exp(sl - sl.max(1, keepdims=True)); ps /= ps.sum(1, keepdims=True)
        pt = np.exp(tlp - tlp.max(1, keepdims=True)); pt /= pt.sum(1, keepdims=True)
        kls.append((pt * (np.log(pt + 1e-12) - np.log(ps + 1e-12))).sum(1))
kl = np.concatenate(kls)
print(f"[{TAG}] RESULT bestMoveAgree {100*agree/N_TEST:.1f}%  policyKL {kl.mean():.4f}  median {np.median(kl):.4f}")
print(f"[{TAG}] TRAIN_DONE")
