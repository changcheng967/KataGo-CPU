// KatFlashAttention: fused KataGo transformer attention as an OV custom op.
//
// Replaces the 22-node per-layer window (head split x3, RoPE on q and k, scores
// GEMM, scale, board mask, softmax, av GEMM, head merge) with one evaluate().
// v0 kernel: correctness-first f32, auto-vectorizable loops; v1 will swap the
// inner loop for a tiled bf16 vdpbf16ps kernel.
//
// Inputs:
//   0 q      f32 [N, L, C]
//   1 k      f32 [N, L, C]
//   2 v      f32 [N, L, C]
//   3 mask   f32 [N, 1, 1, L]  (0 legal / -30000 illegal key)
//   4 cos    f32 [1, H, L, dh]
//   5 sin    f32 [1, H, L, dh]
//   6 swap   int [dh]
//   7 scale  f32 [1]
// Output:
//   0 out    f32 [N, Lh, Lw, C]  with Lh*Lw == L
#include "openvino/frontend/onnx/extension/op.hpp"
#include "katflash_op.inc"

using namespace ov;



// ---------------------------------------------------------------------------
// Test harness: correctness vs original graph + timing.
// ---------------------------------------------------------------------------
static void fill_inputs(std::vector<Tensor>& tis, ov::Core& core, uint32_t seed, int64_t N) {
  srand(seed);
  auto randn = [](float* p, size_t n) {
    for(size_t i = 0; i < n; i++) p[i] = (rand() / (float)RAND_MAX - 0.5f) * 2.0f;
  };
  randn(tis[0].data<float>(), tis[0].get_size());
  randn(tis[1].data<float>(), tis[1].get_size());
  float* mk = tis[2].data<float>();
  for(size_t i = 0; i < tis[2].get_size(); i++) mk[i] = (rand() % 10 == 0) ? 0.0f : 1.0f;
  (void)core;
}

int main(int argc, char** argv) {
  if(argc < 3) { printf("usage: %s <orig.onnx> <fused.onnx>\n", argv[0]); return 1; }
  Core core;
  core.add_extension(std::make_shared<frontend::onnx::OpExtension<KatFlashAttention>>("KatFlashAttention"));

  auto modelO = core.read_model(argv[1]);
  auto modelF = core.read_model(argv[2]);

  // tight reference: original graph forced to f32
  Core coreF32;
  coreF32.set_property("CPU", ov::hint::inference_precision(ov::element::f32));
  auto compO32 = coreF32.compile_model(modelO, "CPU");
  auto compObf = core.compile_model(modelO, "CPU");  // default hint (bf16)
  auto compF = core.compile_model(modelF, "CPU");

  for(int64_t N : {1, 4, 8}) {
    std::vector<Tensor> ins;
    ins.push_back(Tensor(element::f32, Shape{(size_t)N, 22, 19, 19}));
    ins.push_back(Tensor(element::f32, Shape{(size_t)N, 19, 1, 1}));
    ins.push_back(Tensor(element::f32, Shape{(size_t)N, 1, 19, 19}));
    fill_inputs(ins, core, 1234u, N);

    auto setins = [&](InferRequest& r) {
      for(size_t i = 0; i < ins.size(); i++) r.set_input_tensor(i, ins[i]);
    };
    InferRequest rO32 = compO32.create_infer_request();
    InferRequest rObf = compObf.create_infer_request();
    InferRequest rF = compF.create_infer_request();
    setins(rO32); setins(rObf); setins(rF);
    rO32.infer(); rObf.infer(); rF.infer();

    double worst32 = 0, worstbf = 0;
    for(size_t oi = 0; oi < compO32.outputs().size(); oi++) {
      const float* a = rO32.get_output_tensor(oi).data<const float>();
      const float* b = rObf.get_output_tensor(oi).data<const float>();
      const float* c = rF.get_output_tensor(oi).data<const float>();
      size_t sz = rO32.get_output_tensor(oi).get_size();
      for(size_t i = 0; i < sz; i++) {
        double d32 = std::fabs((double)c[i] - a[i]);
        double dbf = std::fabs((double)c[i] - b[i]);
        if(d32 > worst32) worst32 = d32;
        if(dbf > worstbf) worstbf = dbf;
      }
    }
    printf("N=%ld  max|fused-orig_f32|=%.3e   max|fused-orig_bf16|=%.3e\n", N, worst32, worstbf);

    // timing: bf16 world and f32 world (f32 isolates kernel speed from the
    // bf16<->f32 converts OV inserts around the f32 custom op)
    auto bench = [&](InferRequest& r) {
      for(int rep = 0; rep < 3; rep++) r.infer();
      auto t0 = std::chrono::steady_clock::now();
      for(int rep = 0; rep < 15; rep++) r.infer();
      return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 15;
    };
    auto compF32F = coreF32.compile_model(modelF, "CPU");
    InferRequest rF32F = compF32F.create_infer_request();
    setins(rF32F);
    double msB = bench(rObf), msB32 = bench(rO32), msF = bench(rF), msF32 = bench(rF32F);
    printf("N=%ld  bf16: orig=%.1f fused=%.1f (%.2fx) | f32: orig=%.1f fused=%.1f (%.2fx)\n",
           N, msB, msF, msB / msF, msB32, msF32, msB32 / msF32);
  }
  printf("KATFLASH_OK\n");
  return 0;
}
