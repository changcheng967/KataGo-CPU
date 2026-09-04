"""Tree-reuse test v2: sequential turns of ONE game in a single query (real
analysis workload shape) vs the same turns as shuffled independent queries.
Games are short and sparse (60 moves on a 12x15 region) to avoid illegal states;
each run first probes legality with a maxVisits=2 query and aborts if rejected."""
import json
import random
import subprocess
import sys
import threading
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else "sequential"
NAT = int(sys.argv[2]) if len(sys.argv) > 2 else 1
NST = int(sys.argv[3]) if len(sys.argv) > 3 else 8
EXTRA = sys.argv[4] if len(sys.argv) > 4 else ""
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 777

proc = subprocess.Popen(
    ["./build-ov/katago", "analysis", "-model", "models/kata1-tf2-b10c384-s2941M-d5872M.bin.gz",
     "-config", "src/KataGo/cpp/configs/analysis_example.cfg",
     "-override-config",
     (f"onnxProvider=ov,onnxOVThreads=8,numAnalysisThreads={NAT},"
      f"numSearchThreadsPerAnalysisThread={NST}" + ("," + EXTRA if EXTRA else ""))],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    text=True, bufsize=1)

events = []
def reader():
    for line in proc.stdout:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        events.append(r)
threading.Thread(target=reader, daemon=True).start()

rg = random.Random(SEED)
COLS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
moves = []
occ = set()
while len(moves) < 60:
    mv = COLS[rg.randrange(12)] + str(2 + rg.randrange(15))
    if mv not in occ:
        occ.add(mv)
        moves.append(["b" if len(moves) % 2 == 0 else "w", mv])

# legality probe
proc.stdin.write(json.dumps({"id": "probe", "moves": moves, "rules": "chinese", "komi": 7.5,
                             "boardXSize": 19, "boardYSize": 19, "maxVisits": 2,
                             "analyzeTurns": [59]}) + "\n")
proc.stdin.flush()
t0 = time.perf_counter()
while time.perf_counter() - t0 < 120:
    if any(r.get("id") == "probe" for r in events):
        break
    time.sleep(0.3)
pr = [r for r in events if r.get("id") == "probe"]
if not pr or "error" in pr[0] or "moveInfos" not in pr[0]:
    print(f"ABORT seed={SEED}: probe rejected: {json.dumps(pr[0])[:200] if pr else 'no response'}")
    proc.kill()
    sys.exit(1)
events.clear()

N_TURNS = 40
t0 = time.perf_counter()
if MODE == "sequential":
    q = {"id": "seq", "moves": moves, "rules": "chinese", "komi": 7.5,
         "boardXSize": 19, "boardYSize": 19, "maxVisits": 160,
         "analyzeTurns": list(range(20, 60))}
    proc.stdin.write(json.dumps(q) + "\n")
else:
    turns = list(range(20, 60))
    random.Random(5).shuffle(turns)  # random order breaks reuse chains
    for i, tn in enumerate(turns):
        proc.stdin.write(json.dumps({"id": f"t{i}", "moves": moves[:tn], "rules": "chinese",
                                     "komi": 7.5, "boardXSize": 19, "boardYSize": 19,
                                     "maxVisits": 160, "analyzeTurns": [tn]}) + "\n")
proc.stdin.flush()

# count individual turn results: sequential emits one response per turn
while time.perf_counter() - t0 < 900:
    good = [r for r in events if "moveInfos" in r and "warning" not in r and "error" not in r]
    if len(good) >= N_TURNS:
        break
    time.sleep(0.5)
dt = time.perf_counter() - t0
good = [r for r in events if "moveInfos" in r and "warning" not in r and "error" not in r]
proc.kill()
print(f"{MODE} nat={NAT}x{NST}: {len(good)} turns in {dt:.1f}s = {len(good)/dt:.2f} pos/s"
      + (f" [{EXTRA}]" if EXTRA else ""))
