#!/bin/bash
# Interleaved A/B: 3-process split (deployment recipe) vs 1-process shared
# evaluator (numAnalysisThreads=3 x 3 search threads), same games, same visits.
H=/hyperai/home/katago-bench
cd $H
export LD_LIBRARY_PATH=$H/tools/onnxruntime-linux-x64-1.29.0/lib:$H/venvs/ov/lib/python3.12/site-packages/openvino/libs:$LD_LIBRARY_PATH
G=96
V=160
for REP in 1 2 3; do
  A=$(python3 analysis_mp.py $G kata1-tf2-b10c384-s2941M-d5872M $V 3 2>/dev/null | grep -oE '[0-9.]+ positions/s')
  B=$(python3 analysis_shared.py $G kata1-tf2-b10c384-s2941M-d5872M $V 3 3 8 2>/dev/null | grep -oE '[0-9.]+ positions/s')
  echo "rep=$REP  mp3proc=$A  shared1proc=$B"
done
echo AB_SHARED_DONE
