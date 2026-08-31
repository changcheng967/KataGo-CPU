# QAT / int8 accuracy research scripts

Research pipeline used to investigate int8 accuracy for the b18 convnet on CPU.
See docs/CPU_BACKEND.md (section "QAT/distillation investigation") for results,
including the minimal reproduction of an OpenVINO int8 fusion bug.

- katago_torch.py: convert the dumped ONNX graph to a validated PyTorch module
- build_qat_cache.py: unpack real training positions from katagoarchive.org shards
- qat_distill.py: fake-quant wrappers + calibration + distillation training
- export_qat.py / export_qat_v2.py: exports (QDQ-ONNX and OV FakeQuantize IR)
- bisect_export.py: quantize the first k convs / a named subset, measure accuracy
  (`--only name1,name2` for subsets)
- torch_pq_probe.py: PyTorch-side reference for the minimal OV-bug repro
- fq_semantics_test.py / hoist_probe.py: FakeQuantize semantics and placement probes
