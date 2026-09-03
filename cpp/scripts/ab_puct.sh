#!/bin/bash
# A/B round 3: counter-balanced order (vec=1 FIRST, vec=0 second) to detect order bias
H=/hyperai/home/katago-bench
cd $H
export LD_LIBRARY_PATH=$H/tools/onnxruntime-linux-x64-1.29.0/lib:$H/venvs/ov/lib/python3.12/site-packages/openvino/libs:$LD_LIBRARY_PATH
CFG=$H/src/KataGo/cpp/configs/gtp_example.cfg
M=kata1-tf2-b10c384-s2941M-d5872M
for V in 200 400; do
  for REP in 1 2 3 4 5; do
    for MODE in 1 0; do
      LOG=logs/ab3_puct_${V}_${MODE}_${REP}.log
      KATAGO_PUCT_VEC=$MODE ./build-ov/katago benchmark -model models/$M.bin.gz -config $CFG -t 8 -v $V -n 3 \
        -override-config "numSearchThreads=8,onnxProvider=ov" > $LOG 2>&1
      R=$(tr '\r' '\n' < $LOG | grep -oE 'visits/s = [0-9.]+' | tail -1)
      echo "v=$V rep=$REP vec=$MODE $R"
    done
  done
done
echo AB3_DONE
