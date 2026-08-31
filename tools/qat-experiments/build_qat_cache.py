"""Extract N training chunks from the daily tgz and cache unpacked inputs for QAT."""
import subprocess
import sys

import numpy as np

N_CHUNKS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
TGZ = "/hyperai/home/katago-bench/downloads/traindata.tgz"
OUT = "/hyperai/home/katago-bench/traindata/inputs_cache.npz"

listing = subprocess.run(["tar", "tzf", TGZ], capture_output=True, text=True).stdout.splitlines()
npzs = [l for l in listing if l.endswith(".npz")][:N_CHUNKS]
print(f"extracting {len(npzs)} chunks")
with open("/tmp/npzlist.txt", "w") as f:
    f.write("\n".join(npzs))
subprocess.run(["tar", "xzf", TGZ, "-C", "/hyperai/home/katago-bench/traindata", "-T", "/tmp/npzlist.txt"], check=True)

import glob

files = sorted(glob.glob("/hyperai/home/katago-bench/traindata/2026-08-25npzs/*/*.npz"))
print(f"extracted {len(files)} npz files")
spatials, globals_ = [], []
for fn in files:
    d = np.load(fn)
    spatials.append(d["binaryInputNCHWPacked"])
    globals_.append(d["globalInputNC"])
packed = np.concatenate(spatials, 0)
g = np.concatenate(globals_, 0)
bits = np.unpackbits(packed, axis=2)          # (N,22,368) msb-first
sp = bits[:, :, :361].reshape(-1, 22, 19, 19).astype(np.float32)
print(f"positions: {len(sp)}")
# sanity: channel 0 is the on-board mask -> every row exactly 361-ish ones
print(f"mask sum stats: mean {sp[:,0].sum((1,2)).mean():.1f} min {sp[:,0].sum((1,2)).min():.0f} max {sp[:,0].sum((1,2)).max():.0f}")
np.savez(OUT, spatial=sp, glob=g)
print(f"saved {OUT}")
# cleanup the tiny npz files to save quota
subprocess.run(["rm", "-rf", "/hyperai/home/katago-bench/traindata/2026-08-25npzs"])
print("CACHE_DONE")
