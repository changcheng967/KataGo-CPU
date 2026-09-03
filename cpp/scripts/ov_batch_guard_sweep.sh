#!/bin/bash
# Batch guard sweep: 0/250/500/1000/2000us, interleaved with rotating order, v=200 and 400.
# Reports visits/s AND avgBatchSize (does coalescing actually improve?)
H=/hyperai/home/katago-bench
cd $H
export LD_LIBRARY_PATH=$H/tools/onnxruntime-linux-x64-1.29.0/lib:$H/venvs/ov/lib/python3.12/site-packages/openvino/libs:$LD_LIBRARY_PATH
CFG=$H/src/KataGo/cpp/configs/gtp_example.cfg
M=kata1-tf2-b10c384-s2941M-d5872M
GUARDS=(0 250 500 1000 2000)
for V in 200 400; do
  for REP in 1 2 3; do
    # rotate guard order per rep to cancel order bias
    OFF=$(( (REP-1) % 5 ))
    for K in 0 1 2 3 4; do
      IDX=$(( (K + OFF) % 5 ))
      G=${GUARDS[$IDX]}
      LOG=logs/guard_${V}_g${G}_r${REP}.log
      KATAGO_BATCH_GUARD_US=$G ./build-ov/katago benchmark -model models/$M.bin.gz -config $CFG -t 8 -v $V -n 3 \
        -override-config "numSearchThreads=8,onnxProvider=ov" > $LOG 2>&1
      VS=$(tr '\r' '\n' < $LOG | grep -oE 'visits/s = [0-9.]+' | tail -1 | grep -oE '[0-9.]+')
      BS=$(tr '\r' '\n' < $LOG | grep -oE 'avgBatchSize = [0-9.]+' | tail -1 | grep -oE '[0-9.]+')
      echo "v=$V rep=$REP guard=${G}us visits/s=$VS avgBatch=$BS"
    done
  done
done
echo GUARD_SWEEP_DONE
