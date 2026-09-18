#!/bin/bash
# 2-core box config matrix: search threads x OV threads, plus RSS tracking
H=/hyperai/home/katago-bench
cd $H
export LD_LIBRARY_PATH=$H/tools/onnxruntime-linux-x64-1.29.0/lib:$H/venvs/ov/lib/python3.12/site-packages/openvino/libs:$LD_LIBRARY_PATH
CFG=$H/src/KataGo/cpp/configs/gtp_example.cfg
M=kata1-tf2-b10c384-s2941M-d5872M
for CFGX in "2 2" "2 1" "3 1" "4 1" "6 1"; do
  set -- $CFGX
  T=$1; OV=$2
  LOG=/tmp/matrix_${T}_${OV}.log
  ./build-ov/katago benchmark -model models/$M.bin.gz -config $CFG -t $T -v 100 -n 3 \
    -override-config "numSearchThreads=$T,onnxProvider=ov,onnxOVThreads=$OV" > $LOG 2>&1 &
  BPID=$!
  PEAK=0
  while kill -0 $BPID 2>/dev/null; do
    R=$(ps -o rss= -p $BPID 2>/dev/null | tr -d ' ')
    [ -n "$R" ] && [ "$R" -gt "$PEAK" ] && PEAK=$R
    sleep 1
  done
  VS=$(tr '\r' '\n' < $LOG | grep -oE 'visits/s = [0-9.]+' | sed -n 2p | grep -oE '[0-9.]+')
  BS=$(tr '\r' '\n' < $LOG | grep -oE 'avgBatchSize = [0-9.]+' | sed -n 2p | grep -oE '[0-9.]+')
  echo "threads=$T ovT=$OV -> v/s=$VS avgBatch=$BS peakRSS=$((PEAK/1024))MB"
done
echo MATRIX_DONE
