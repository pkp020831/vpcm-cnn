from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import onnx

from .models import ModelInfo


@dataclass(frozen=True)
class GraphReport:
    conv_layers: int
    linear_layers: int
    relax_functions: int

    @property
    def weight_layers(self) -> int:
        return self.conv_layers + self.linear_layers


def validate_with_tvm(model_path: Path) -> GraphReport:
    import tvm
    from tvm.relax.frontend.onnx import from_onnx

    info = ModelInfo()
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    operator_types = [node.op_type for node in model.graph.node]
    conv_layers = operator_types.count("Conv")
    linear_layers = sum(operator_types.count(name) for name in ("Gemm", "MatMul"))
    relax_module = from_onnx(
        model,
        shape_dict={info.input_name: list(info.input_shape)},
        dtype_dict="float32",
        keep_params_in_input=False,
    )
    relax_functions = len(relax_module.get_global_vars())
    if conv_layers != 8 or linear_layers != 1:
        raise ValueError(
            f"Unexpected VGG11 graph: {conv_layers} conv and {linear_layers} linear layers"
        )
    if relax_functions < 1 or not isinstance(relax_module, tvm.IRModule):
        raise ValueError("TVM did not produce a Relax IRModule")
    return GraphReport(conv_layers, linear_layers, relax_functions)
