# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""thnn_conv2d: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_thnn_conv2d.py:77 (test_thnn_conv2d)
Native expression: torch.ops.aten.thnn_conv2d
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import (
    BenchmarkCasePlan,
    generate_tensor_input,
    case_inputs,
    export_contract,
)


def _case_fn(shape, dtype):
    del dtype
    inp_shape, weight_shape, kernel_size, stride, padding = shape
    yield BenchmarkCasePlan(
        shape={"input": inp_shape, "weight": weight_shape},
        params={
            "kernel_size": kernel_size,
            "bias": True,
            "stride": stride,
            "padding": padding,
        },
        builder_args=(inp_shape, weight_shape, kernel_size, stride, padding),
    )


def _build_inputs_fn(plan, dtype, device):
    inp_shape, weight_shape, kernel_size, stride, padding = plan.builder_args
    inp = generate_tensor_input(inp_shape, dtype, device)
    weight = generate_tensor_input(weight_shape, dtype, device)
    bias = generate_tensor_input((weight_shape[0],), dtype, device)
    return (inp, weight, kernel_size, bias, stride, padding, {})


def _resolve_native():
    return torch.ops.aten.thnn_conv2d


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(export_contract("thnn_conv2d", DTYPES, _resolve_native, _inputs))
