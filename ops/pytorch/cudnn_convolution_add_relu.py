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


"""cudnn_convolution_add_relu: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_cudnn_convolution_add_relu.py:155 (test_cudnn_convolution_add_relu)
Native expression: torch.ops.aten.cudnn_convolution_add_relu
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

_BIAS_FORMS = ("none", "channel", "channel_11", "broadcast_1C11")

_BENCH_DTYPES = [
    dtype
    for dtype in [torch.float16, torch.float32, torch.bfloat16]
    if dtype is not torch.bfloat16 or True
]


def _output_shape(x_shape, w_shape, stride, padding, dilation):
    return (x_shape[0], w_shape[0]) + tuple(
        (
            (x_shape[i + 2] + 2 * padding[i] - dilation[i] * (w_shape[i + 2] - 1) - 1)
            // stride[i]
            + 1
            for i in range(len(stride))
        )
    )


def _case_fn(case, dtype):
    del dtype
    x_shape, w_shape, stride, padding, dilation, groups, bias_form, alpha = case
    if bias_form not in _BIAS_FORMS:
        raise ValueError("unknown bias form: " + repr(bias_form))
    out_shape = _output_shape(x_shape, w_shape, stride, padding, dilation)
    yield BenchmarkCasePlan(
        shape={
            "input": tuple(x_shape),
            "weight": tuple(w_shape),
            "residual": tuple(out_shape),
        },
        params={
            "stride": list(stride),
            "padding": list(padding),
            "dilation": list(dilation),
            "groups": groups,
            "bias": bias_form,
            "alpha": alpha,
        },
        builder_args=(tuple(x_shape), tuple(w_shape), tuple(out_shape)),
    )


def _build_inputs_fn(plan, dtype, device):
    x_shape, w_shape, out_shape = plan.builder_args
    inp = generate_tensor_input(x_shape, dtype, device)
    weight = generate_tensor_input(w_shape, dtype, device)
    z = generate_tensor_input(out_shape, dtype, device)
    channels = out_shape[1]
    bias_form = plan.params["bias"]
    if bias_form == "none":
        bias = None
    elif bias_form == "channel":
        bias = generate_tensor_input((channels,), dtype, device)
    elif bias_form == "channel_11":
        bias = generate_tensor_input((channels, 1, 1), dtype, device)
    elif bias_form == "broadcast_1C11":
        bias = generate_tensor_input((1, channels, 1, 1), dtype, device)
    else:
        raise ValueError("unknown bias form: " + repr(bias_form))
    return (
        inp,
        weight,
        z,
        plan.params["alpha"],
        bias,
        list(plan.params["stride"]),
        list(plan.params["padding"]),
        list(plan.params["dilation"]),
        plan.params["groups"],
    )


def _resolve_native():
    return torch.ops.aten.cudnn_convolution_add_relu


DTYPES = _BENCH_DTYPES

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(
    export_contract("cudnn_convolution_add_relu", DTYPES, _resolve_native, _inputs)
)
