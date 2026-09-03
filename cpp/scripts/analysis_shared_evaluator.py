"""Shared-evaluator multi-game analysis: ONE process, numAnalysisThreads=N,
each with its own search threads, all feeding the SAME NNEvaluator.
Compare against analysis_mp.py's multi-process split at the same total threads."""
import json
import random
import subprocess
import sys
import threading
import time

N_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 96
MODEL = sys.argv[2] if len(sys.argv) > 2 else "kata1-tf2-b10c384-s2941M-d5872M"
VISITS = int(sys.argv[3]) if len(sys.argv) > 3 else 160
NAT = int(sys.argv[4]) if len(sys.argv) > 4 else 3       # numAnalysisThreads
NST = int(sys.argv[5]) if len(sys.argv) > 5 else 3       # search threads per analysis thread
OVT = int(sys.argv[6]) if len(sys.argv) > 6 else 8       # onnxOVThreads

proc = subprocess.Popen(
    ["./build-ov/katago", "analysis", "-model", f"models/{MODEL}.bin.gz",
     "-config", "src/KataGo/cpp/configs/analysis_example.cfg",
     "-override-config",
     (f"onnxProvider=ov,onnxOVThreads={OVT},"
      f"numAnalysisThreads={NAT},numSearchThreadsPerAnalysisThread={NST}")],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    text=True, bufsize=1)

ok = []
def reader():
    for line in proc.stdout:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" not in r and "warning" not in r and "moveInfos" in r:
            ok.append(r)
t = threading.Thread(target=reader, daemon=True)
t.start()

COLS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
t0 = time.perf_counter()
for g in range(N_GAMES):
    rg = random.Random(3000 + g)
    moves = []
    occupied = set()
    while len(moves) < 30:
        mv = COLS[rg.randrange(8)] + str(2 + rg.randrange(8))
        if mv not in occupied:
            occupied.add(mv)
            moves.append(["b" if len(moves) % 2 == 0 else "w", mv])
    q = {"id": f"g{g}", "moves": moves, "initialStone": [],
         "rules": "chinese", "komi": 7.5, "boardXSize": 19, "boardYSize": 19,
         "maxVisits": VISITS, "analyzeTurns": [30]}
    proc.stdin.write(json.dumps(q) + "\n")
proc.stdin.flush()

while len(ok) < N_GAMES and time.perf_counter() - t0 < 900:
    time.sleep(1.0)
dt = time.perf_counter() - t0
try:
    proc.stdin.write(json.dumps({"id": "quit", "quit": True}) + "\n")
    proc.stdin.flush()
except Exception:
    pass
proc.kill()

print(f"shared nat={NAT}x{NST}st ovT={OVT} games={N_GAMES} ok={len(ok)} wall={dt:.1f}s")
if len(ok):
    print(f"ANALYSIS THROUGHPUT: {VISITS*len(ok)/dt:.1f} visits/s, {len(ok)/dt:.2f} positions/s")
