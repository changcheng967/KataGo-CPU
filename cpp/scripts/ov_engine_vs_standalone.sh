#!/bin/bash
# Alternated engine vs standalone OV throughput at matched batch size, same load period.
H=/hyperai/home/katago-bench
cd $H
export LD_LIBRARY_PATH=$H/tools/onnxruntime-linux-x64-1.29.0/lib:$H/venvs/ov/lib/python3.12/site-packages/openvino/libs:$LD_LIBRARY_PATH
CFG=$H/src/KataGo/cpp/configs/gtp_example.cfg
M=kata1-tf2-b10c384-s2941M-d5872M
for REP in 1 2 3; do
  LOG=logs/ratio_engine_$REP.log
  ./build-ov/katago benchmark -model models/$M.bin.gz -config $CFG -t 8 -v 200 -n 3 \
    -override-config "numSearchThreads=8,onnxProvider=ov" > $LOG 2>&1
  E=$(tr '\r' '\n' < $LOG | grep -oE 'nnEvals/s = [0-9.]+' | tail -1 | grep -oE '[0-9.]+')
  B=$(tr '\r' '\n' < $LOG | grep -oE 'avgBatchSize = [0-9.]+' | tail -1 | grep -oE '[0-9.]+')
  echo "rep=$REP ENGINE nnEvals/s=$E avgBatch=$B"
  S=$(taskset -c 0-7 $H/venvs/ov/bin/python3 $H/ov_sweep.py $H/onnx/$M.onnx 8 --batches 4 2>/dev/null | grep -E '^batch' | grep -oE '[0-9.]+ pos/s')
  echo "rep=$REP STANDALONE batch4 $S"
done
echo RATIO_DONE
