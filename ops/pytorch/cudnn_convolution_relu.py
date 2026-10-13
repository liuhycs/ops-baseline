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


"""cudnn_convolution_relu: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_cudnn_convolution_relu.py:125 (test_cudnn_convolution_relu)
Native expression: torch.ops.aten.cudnn_convolution_relu
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

_BENCH_DTYPES = [torch.float32, torch.float16]


def _case_fn(case, dtype):
    del dtype
    shape, out_channels, kernel, stride, padding, dilation, groups, bias = case
    spatial = len(shape) - 2
    yield BenchmarkCasePlan(
        shape={
            "input": list(shape),
            "out_channels": out_channels,
            "kernel_size": kernel,
            "groups": groups,
        },
        params={
            "stride": [stride] * spatial,
            "padding": [padding] * spatial,
            "dilation": [dilation] * spatial,
            "groups": groups,
            "bias": bias,
        },
        builder_args=(
            shape,
            out_channels,
            kernel,
            stride,
            padding,
            dilation,
            groups,
            bias,
        ),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, out_channels, kernel, stride, padding, dilation, groups, bias = (
        plan.builder_args
    )
    spatial = len(shape) - 2
    inp = generate_tensor_input(shape, dtype, device)
    weight = generate_tensor_input(
        (out_channels, shape[1] // groups) + (kernel,) * spatial, dtype, device
    )
    bias_t = generate_tensor_input((out_channels,), dtype, device) if bias else None
    return (
        inp,
        weight,
        bias_t,
        [stride] * spatial,
        [padding] * spatial,
        [dilation] * spatial,
        groups,
        {},
    )


def _resolve_native():
    return torch.ops.aten.cudnn_convolution_relu


DTYPES = _BENCH_DTYPES

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(
    export_contract("cudnn_convolution_relu", DTYPES, _resolve_native, _inputs)
)
