"""Shared: convert KataGo's dumped ONNX graph into a validated PyTorch module."""
import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference, version_converter
from onnx2torch import convert


def load_torch_model(onnx_path):
    model = onnx.load(onnx_path)
    input_names = [i.name for i in model.graph.input]

    # onnx2torch lacks ReduceMean: rewrite as ReduceSum + Div
    inferred = shape_inference.infer_shapes(model)
    vi = {}
    for v in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        vi[v.name] = [d.dim_value for d in v.type.tensor_type.shape.dim]
    shape_by_name = {i.name: list(i.dims) for i in model.graph.initializer}

    new_nodes, new_inits = [], []
    for node in model.graph.node:
        if node.op_type == "ReduceMean":
            axes = next((list(a.ints) for a in node.attribute if a.name == "axes"), None)
            keepdims = next((a.i for a in node.attribute if a.name == "keepdims"), 1)
            x = node.input[0]
            xshape = vi.get(x) or shape_by_name.get(x)
            if axes is None:
                axes = list(range(len(xshape)))
            n = 1
            for ax in axes:
                n *= xshape[ax]
            sum_name = node.output[0] + "_sumonly"
            axes_c = numpy_helper.from_array(np.array(axes, dtype=np.int64), name=node.name + "_axc")
            div_c = numpy_helper.from_array(np.array(n, dtype=np.float32), name=node.output[0] + "_divc")
            rs = helper.make_node("ReduceSum", [x, axes_c.name], [sum_name], keepdims=keepdims, name=node.name + "_rs")
            div = helper.make_node("Div", [sum_name, div_c.name], [node.output[0]], name=node.name + "_div")
            new_nodes += [rs, div]
            new_inits += [axes_c, div_c]
        else:
            new_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_inits)
    model = version_converter.convert_version(model, 17)
    torch_model = convert(model)
    torch_model.eval()
    return torch_model, input_names
