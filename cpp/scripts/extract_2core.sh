#!/bin/bash
# Final extraction on the 2-core quota box: pinning vs drift, thread variants,
# model choice at (2,2). Interleaved order, RSS tracked.
H=/hyperai/home/katago-bench
cd $H
export LD_LIBRARY_PATH=$H/tools/onnxruntime-linux-x64-1.29.0/lib:$H/venvs/ov/lib/python3.12/site-packages/openvino/libs:$LD_LIBRARY_PATH
CFG=$H/src/KataGo/cpp/configs/gtp_example.cfg
M=kata1-tf2-b10c384-s2941M-d5872M

run() { # label, taskset-prefix, threads, ovthreads, model
  local LABEL=$1; local PRE=$2; local T=$3; local OV=$4; local MO=$5
  local LOG=/tmp/ex_${LABEL}.log
  $PRE ./build-ov/katago benchmark -model models/$MO.bin.gz -config $CFG -t $T -v 100 -n 3 \
    -override-config "numSearchThreads=$T,onnxProvider=ov,onnxOVThreads=$OV" > $LOG 2>&1 &
  local BP=$!; local PEAK=0
  while kill -0 $BP 2>/dev/null; do
    R=$(ps -o rss= -p $BP 2>/dev/null | tr -d ' ')
    [ -n "$R" ] && [ "$R" -gt "$PEAK" ] && PEAK=$R
    sleep 1
  done
  local VS=$(tr '\r' '\n' < $LOG | grep -oE 'visits/s = [0-9.]+' | sed -n 2p | grep -oE '[0-9.]+')
  echo "$LABEL -> v/s=$VS peakRSS=$((PEAK/1024))MB"
}

for REP in 1 2; do
  run "rep$REP-drift-2x2-tf2"  ""              2 2 $M
  run "rep$REP-pin01-2x2-tf2"   "taskset -c 0,1" 2 2 $M
  run "rep$REP-pinSMT-2x2-tf2"  "taskset -c 0,128" 2 2 $M
done
run "pin01-3x2-tf2"  "taskset -c 0,1,2" 3 2 $M
run "pin01-4x2-tf2"  "taskset -c 0,1,2,3" 4 2 $M
run "pin01-2x2-b18"  "taskset -c 0,1" 2 2 kata1-b18c384nbt-latest
echo EXTRACT_DONE
