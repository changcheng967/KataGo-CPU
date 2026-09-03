// Hello-world: prove OV custom-op injection works end to end.
// A custom ONNX node "KatFlashAttention" gets mapped to this op via
// frontend::onnx::OpExtension and executed by the CPU plugin through evaluate().
#include <chrono>
#include <cstdio>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/frontend/onnx/extension/op.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/op/op.hpp"

using namespace ov;

class KatFlashAttention : public op::Op {
public:
  OPENVINO_OP("KatFlashAttention", "org.katago");

  KatFlashAttention() = default;
  KatFlashAttention(const OutputVector& args) : op::Op(args) {
    constructor_validate_and_infer_types();
  }

  void validate_and_infer_types() override {
    // out shape/type = first input's
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
  }

  std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_inputs) const override {
    return std::make_shared<KatFlashAttention>(new_inputs);
  }

  bool has_evaluate() const override { return true; }

  bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override {
    const Tensor& a = inputs[0];
    const Tensor& b = inputs[1];
    if(outputs[0].get_element_type() != element::f32 || outputs[0].get_size() != a.get_size())
      outputs[0] = Tensor(element::f32, a.get_shape());
    const float* pa = a.data<const float>();
    const float* pb = b.data<const float>();
    float* po = outputs[0].data<float>();
    size_t n = a.get_size();
    // exercise ov::parallel_for to confirm in-evaluate threading
    ov::parallel_for(static_cast<int64_t>(n), [&](int64_t i) { po[i] = pa[i] + pb[i]; });
    return true;
  }
};

int main(int argc, char** argv) {
  Core core;
  core.add_extension(std::make_shared<frontend::onnx::OpExtension<KatFlashAttention>>("KatFlashAttention"));
  auto model = core.read_model(argv[1]);
  auto compiled = core.compile_model(model, "CPU");
  auto req = compiled.create_infer_request();

  const size_t N = 1 << 20; // 1M floats: big enough to see threading
  Tensor ta(element::f32, Shape{N}), tb(element::f32, Shape{N});
  float* pa = ta.data<float>(); float* pb = tb.data<float>();
  for(size_t i = 0; i < N; i++) { pa[i] = 1.0f; pb[i] = 2.0f; }
  req.set_input_tensor(0, ta);
  req.set_input_tensor(1, tb);
  req.infer();
  const float* po = req.get_output_tensor().data<const float>();
  double s = 0; for(size_t i = 0; i < N; i++) s += po[i];
  printf("result sum=%.1f (expect %.1f)\n", s, 3.0 * N);

  auto t0 = std::chrono::steady_clock::now();
  for(int r = 0; r < 20; r++) req.infer();
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 20.0;
  printf("per-infer %.3f ms for 2x4MB f32 add -> %.2f GB/s (RAM-bound implies threaded)\n",
         ms, (2.0 * 4.0 * 2) / (ms / 1000.0));
  printf("HELLO_EXT_OK\n");
  return 0;
}
