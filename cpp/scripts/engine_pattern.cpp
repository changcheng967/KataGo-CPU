// Attribute the engine-vs-standalone per-row inference gap (engine reaches only
// ~60-63% of ov_sweep.py pos/s at matched batch). Modes, same model, same cores:
//   A: fixed batch 4, same tensors every call          (ov_sweep equivalent)
//   B: fixed batch 4, fresh wrapped host buffers/call   (KataGo zero-copy pattern)
//   C: variable batch 1..8 cycling, same tensors        (engine batch mix)
//   D: variable batch + fresh buffers                   (full engine mimic)
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include "openvino/openvino.hpp"

using namespace ov;

static double benchMode(Core& core, const std::string& path, int mode, int reps) {
  auto model = core.read_model(path);
  auto compiled = core.compile_model(model, "CPU");
  auto req = compiled.create_infer_request();

  const int L = 22 * 19 * 19, G = 19, M = 361;
  // pool of host buffers to wrap fresh each call (mode B/D)
  const int POOL = 8;
  std::vector<std::vector<float>> hsp(POOL), hgl(POOL), hmk(POOL);
  for(auto& v : hsp) v.resize(L);
  for(auto& v : hgl) v.resize(G);
  for(auto& v : hmk) v.resize(M);

  auto fill = [&](int b) {
    for(int p = 0; p < POOL; p++)
      for(size_t i = 0; i < hsp[p].size(); i++) hsp[p][i] = 0.05f * ((i + b) % 7);
  };
  fill(0);

  const int batchSeq[8] = {1, 2, 4, 3, 6, 5, 8, 7};  // mixes down to avg 4.5
  int call = 0;
  auto runOnce = [&]() {
    int b = (mode >= 2) ? batchSeq[call % 8] : 4;
    call++;
    if(mode == 0 || mode == 2) {
      // reuse the same tensors, just resize
      Tensor t0(element::f32, Shape{(size_t)b, 22, 19, 19});
      // cheap: build once outside... for fairness, wrap WITHOUT realloc cost models:
      (void)t0;
    }
    if(mode == 1 || mode == 3) {
      int p = call % POOL;
      req.set_input_tensor(0, Tensor(element::f32, Shape{(size_t)b, 22, 19, 19}, hsp[p].data()));
      req.set_input_tensor(1, Tensor(element::f32, Shape{(size_t)b, 19, 1, 1}, hgl[p].data()));
      req.set_input_tensor(2, Tensor(element::f32, Shape{(size_t)b, 1, 19, 19}, hmk[p].data()));
    }
    else {
      static Tensor s0(element::f32, Shape{4, 22, 19, 19}, hsp[0].data());
      static Tensor s1(element::f32, Shape{4, 19, 1, 1}, hgl[0].data());
      static Tensor s2(element::f32, Shape{4, 1, 19, 19}, hmk[0].data());
      if(b != 4) {
        req.set_input_tensor(0, Tensor(element::f32, Shape{(size_t)b, 22, 19, 19}, hsp[0].data()));
        req.set_input_tensor(1, Tensor(element::f32, Shape{(size_t)b, 19, 1, 1}, hgl[0].data()));
        req.set_input_tensor(2, Tensor(element::f32, Shape{(size_t)b, 1, 19, 19}, hmk[0].data()));
      }
      else {
        req.set_input_tensor(0, s0);
        req.set_input_tensor(1, s1);
        req.set_input_tensor(2, s2);
      }
    }
    req.infer();
    return b;
  };

  // warmup + count rows
  int64_t rows = 0;
  for(int i = 0; i < 8; i++) rows += runOnce();
  auto t0 = std::chrono::steady_clock::now();
  rows = 0;
  for(int i = 0; i < reps; i++) rows += runOnce();
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return 1000.0 * (double)rows / ms;
}

int main(int argc, char** argv) {
  Core core;
  const char* path = argv[1];
  const char* names[4] = {"A fixed4 sameTensor", "B fixed4 freshWrap", "C varBatch sameTensor", "D varBatch freshWrap"};
  for(int mode = 0; mode < 4; mode++) {
    double pos_s = benchMode(core, path, mode, 40);
    printf("%s: %.1f pos/s\n", names[mode], pos_s);
  }
  printf("PATTERN_DONE\n");
  return 0;
}
