// Direct A/B of one attention layer: original 22-node window vs KatFlashAttention.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/frontend/onnx/extension/op.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/op/op.hpp"

using namespace ov;

#include "katflash_op.inc"

int main(int argc, char** argv) {
  Core core;
  core.add_extension(std::make_shared<frontend::onnx::OpExtension<KatFlashAttention>>("KatFlashAttention"));
  auto model = core.read_model(argv[1]);
  auto compiled = core.compile_model(model, "CPU");
  auto req = compiled.create_infer_request();

  const int64_t N = 2;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  Tensor tq(element::f32, Shape{(size_t)N, 19, 19, 192});
  Tensor tk(element::f32, Shape{(size_t)N, 19, 19, 192});
  Tensor tv(element::f32, Shape{(size_t)N, 19, 19, 192});
  for(auto* t : {&tq, &tk, &tv})
    for(size_t i = 0; i < t->get_size(); i++) t->data<float>()[i] = nd(rng);
  Tensor tmask(element::f32, Shape{(size_t)N, 1, 1, 361});
  for(size_t i = 0; i < tmask.get_size(); i++) tmask.data<float>()[i] = (rng() % 10 == 0) ? -30000.f : 0.f;

  req.set_input_tensor(0, tq);
  req.set_input_tensor(1, tk);
  req.set_input_tensor(2, tv);
  req.set_input_tensor(3, tmask);
  req.infer();
  const float* ref = req.get_output_tensor().data<const float>();

  // lift the layer's constants out of the model
  std::shared_ptr<ov::op::v0::Constant> ccos, csin, cswap, cscale;
  for(auto& n : model->get_ops()) {
    auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(n);
    if(!c) continue;
    std::string f = c->get_friendly_name();
    if(f.find("ropecos") != std::string::npos && !ccos) ccos = c;
    if(f.find("ropesinsigned") != std::string::npos && !csin) csin = c;
    if(f.find("ropeswapidx") != std::string::npos && !cswap) cswap = c;
    if(f.find("/scale/") != std::string::npos && !cscale) cscale = c;
  }

  auto opq = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape(tq.get_shape()));
  auto opk = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape(tk.get_shape()));
  auto opv = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape(tv.get_shape()));
  auto opm = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape(tmask.get_shape()));
  Tensor tcos(element::f32, ccos->get_output_shape(0), (float*)ccos->get_data_ptr());
  Tensor tsin(element::f32, csin->get_output_shape(0), (float*)csin->get_data_ptr());
  Tensor tscale(element::f32, cscale->get_output_shape(0), (float*)cscale->get_data_ptr());
  auto ccosN = std::make_shared<ov::op::v0::Constant>(tcos);
  auto csinN = std::make_shared<ov::op::v0::Constant>(tsin);
  auto cscaleN = std::make_shared<ov::op::v0::Constant>(tscale);
  std::shared_ptr<ov::op::v0::Constant> cswapN;
  if(cswap->get_output_element_type(0) == element::i64)
    cswapN = std::make_shared<ov::op::v0::Constant>(element::i64, cswap->get_output_shape(0), (int64_t*)cswap->get_data_ptr());
  else
    cswapN = std::make_shared<ov::op::v0::Constant>(element::i32, cswap->get_output_shape(0), (int32_t*)cswap->get_data_ptr());

  ov::OutputVector opInputs{opq, opk, opv, opm, ccosN, csinN, cswapN, cscaleN};
  auto node = std::make_shared<KatFlashAttention>(opInputs);
  auto f = std::make_shared<ov::Model>(ov::OutputVector{node}, ov::ParameterVector{opq, opk, opv, opm});

  auto compF = core.compile_model(f, "CPU");
  auto reqF = compF.create_infer_request();
  reqF.set_input_tensor(0, tq); reqF.set_input_tensor(1, tk);
  reqF.set_input_tensor(2, tv); reqF.set_input_tensor(3, tmask);
  reqF.infer();
  const float* got = reqF.get_output_tensor().data<const float>();

  // timing: original window vs fused op, same inputs, 30 reps each
  auto bench = [&](InferRequest& r) {
    for(int i = 0; i < 5; i++) r.infer();
    auto t0 = std::chrono::steady_clock::now();
    for(int i = 0; i < 30; i++) r.infer();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 30;
  };
  double msW = bench(req), msO = bench(reqF);
  printf("WINDOW(orig)=%.3f ms  OP(fused)=%.3f ms  (%.2fx)  [N=%ld]\n", msW, msO, msW / msO, N);

  size_t sz = reqF.get_output_tensor().get_size();
  double maxd = 0; size_t argmax = 0; double sumref = 0;
  for(size_t i = 0; i < sz; i++) {
    double d = std::fabs((double)got[i] - ref[i]);
    sumref += ref[i];
    if(d > maxd) { maxd = d; argmax = i; }
  }
  printf("LAYER A/B: maxdiff=%.4e at %zu (ref=%.5f got=%.5f) over %zu elems, sumref=%.1f\n",
         maxd, argmax, ref[argmax], got[argmax], sz, sumref);
  printf("LAYER_TEST_DONE\n");
  return 0;
}
