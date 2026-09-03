"""Build a minimal ONNX with one custom node KatFlashAttention(a, b) -> out."""
import numpy as np
import onnx
from onnx import helper, TensorProto

N = 1 << 20
a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [N])
b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [N])
o = helper.make_tensor_value_info("out", TensorProto.FLOAT, [N])
node = helper.make_node("KatFlashAttention", ["a", "b"], ["out"], name="hello0")
g = helper.make_graph([node], "hello", [a, b], [o])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 8
onnx.save(m, "hello_kat.onnx")
print("saved hello_kat.onnx")
