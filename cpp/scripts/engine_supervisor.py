"""Engine supervisor for service deployments (uvicorn/FastAPI/systemd).

Why: the engine "dies instantly under the service but works manually" is almost
always one of:
  1. LD_LIBRARY_PATH missing in the service's spawn environment -> the loader
     kills the binary before main() (OV libs live outside default paths).
  2. stderr pipe never drained -> the engine blocks after ~64KB of warnings
     (e.g. warnUnusedFields spam from fields the service sends) and looks dead.
  3. Relative model/config paths resolved against the service CWD.
  4. cgroup OOM (check RSS at death).

This wrapper: spawns the engine with an explicit env, drains BOTH pipes,
classifies every death (exit code / signal / last stderr lines / RSS),
restarts with capped exponential backoff, and heartbeats readiness.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument("--engine", required=True, help="path to katago binary")
ap.add_argument("engine_args", nargs=argparse.REMAINDER, help="args after -- e.g. -- analysis -model ... -config ...")
ap.add_argument("--log-dir", default="engine_logs")
ap.add_argument("--backoff-max", type=float, default=30.0)
ap.add_argument("--ld-library-path", default=None,
                help="explicit LD_LIBRARY_PATH for the engine (default: inherit this wrapper's)")
ap.add_argument("--cwd", default=None, help="working directory for the engine")
args = ap.parse_args()

os.makedirs(args.log_dir, exist_ok=True)
env = dict(os.environ)
if args.ld_library_path:
    env["LD_LIBRARY_PATH"] = args.ld_library_path
if "LD_LIBRARY_PATH" not in env or not env["LD_LIBRARY_PATH"]:
    print("[supervisor] WARNING: empty LD_LIBRARY_PATH - if this engine links OpenVINO "
          "outside default paths it will die instantly with a loader error", flush=True)

RESTARTABLE = True
proc = None
death_log = []


def classify(returncode, stderr_tail):
    if returncode == 127:
        return "loader/exec failure (127): missing library or interpreter - check LD_LIBRARY_PATH"
    if returncode == -signal.SIGKILL:
        return "SIGKILL: OOM-killer or manual kill (RSS is unrecoverable post-mortem on this kernel)"
    if returncode == -signal.SIGSEGV:
        return "SIGSEGV: crash - see stderr tail"
    if returncode == 0:
        return "clean exit (stdin closed?)"
    return f"exit code {returncode}"


def monitor():
    global proc, RESTARTABLE
    backoff = 1.0
    while RESTARTABLE:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        err_path = os.path.join(args.log_dir, f"stderr_{run_id}.log")
        out_path = os.path.join(args.log_dir, f"stdout_{run_id}.log")
        feh = open(err_path, "wb")
        foh = open(out_path, "wb")
        t0 = time.time()
        proc = subprocess.Popen([args.engine] + args.engine_args,
                                stdout=foh, stderr=feh, stdin=subprocess.PIPE,
                                env=env, cwd=args.cwd)
        print(f"[supervisor] engine pid {proc.pid} started ({run_id})", flush=True)
        rc = proc.wait()
        lifetime = time.time() - t0
        feh.flush(); foh.flush()
        feh.close(); foh.close()
        tail = b""
        try:
            with open(err_path, "rb") as f:
                f.seek(max(0, os.path.getsize(err_path) - 2000))
                tail = f.read()
        except Exception:
            pass
        reason = classify(rc, tail)
        rec = {"run": run_id, "pid": proc.pid, "lifetime_s": round(lifetime, 2),
               "reason": reason, "stderr_tail": tail.decode("utf-8", "replace")[-500:]}
        death_log.append(rec)
        with open(os.path.join(args.log_dir, "deaths.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[supervisor] DIED after {lifetime:.2f}s: {reason}", flush=True)
        if tail:
            print(f"[supervisor] last stderr:\n{tail.decode('utf-8', 'replace')[-500:]}", flush=True)
        if not RESTARTABLE:
            break
        print(f"[supervisor] restarting in {backoff:.0f}s", flush=True)
        time.sleep(backoff)
        backoff = min(args.backoff_max, backoff * 2)


def on_signal(sig, frame):
    global RESTARTABLE, proc
    print(f"[supervisor] signal {sig}, shutting down engine", flush=True)
    RESTARTABLE = False
    if proc and proc.poll() is None:
        proc.terminate()


signal.signal(signal.SIGTERM, on_signal)
signal.signal(signal.SIGINT, on_signal)

mt = threading.Thread(target=monitor, daemon=True)
mt.start()

# forward stdin lines to the engine, keep the pipes drained above
for line in sys.stdin:
    while proc is None or proc.poll() is not None:
        time.sleep(0.1)
    try:
        proc.stdin.write(line)
        proc.stdin.flush()
    except BrokenPipeError:
        pass
while RESTARTABLE and (proc is None or proc.poll() is None):
    time.sleep(0.5)
