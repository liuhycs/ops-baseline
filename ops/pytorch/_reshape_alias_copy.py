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


"""_reshape_alias_copy: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test__reshape_alias_copy.py:115 (test__reshape_alias_copy)
Native expression: torch.ops.aten._reshape_alias_copy
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

import math

_BENCH_DTYPES = [
    dtype
    for dtype in [torch.float16, torch.float32, torch.bfloat16]
    if dtype is not torch.bfloat16 or True
]


def _int_extent(value, what):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{what} must be a non-bool integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{what} must not be negative, got {value}")
    return value


def _decode(case):
    if not isinstance(case, (tuple, list)) or len(case) != 3:
        raise TypeError(f"case must be (input_shape, size, stride), got {case!r}")
    shape, size, stride = case
    parts = (("input_shape", shape), ("size", size), ("stride", stride))
    for name, sequence in parts:
        if not isinstance(sequence, (tuple, list)):
            raise TypeError(f"{name} must be a sequence of ints, got {sequence!r}")
    shape = tuple((_int_extent(dim, "input_shape dim") for dim in shape))
    size = tuple((_int_extent(dim, "size dim") for dim in size))
    stride = tuple((_int_extent(step, "stride dim") for step in stride))
    if len(size) != len(stride):
        raise ValueError(f"size rank {len(size)} != stride rank {len(stride)}")
    reach = 0
    for dim, step in zip(size, stride):
        if dim > 0:
            reach += (dim - 1) * step
    numel = math.prod(shape)
    if all((dim > 0 for dim in size)) and (numel == 0 or reach >= numel):
        raise ValueError(
            f"size={size} stride={stride} reaches element {reach}, outside an input storage of {numel} elements"
        )
    return (shape, size, stride)


def _case_fn(case, dtype):
    del dtype
    shape, size, stride = _decode(case)
    yield BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"size": list(size), "stride": list(stride)},
        builder_args=(shape, size, stride),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, size, stride = plan.builder_args
    inp = generate_tensor_input(shape, dtype, device)
    return (inp, size, stride)


def _resolve_native():
    return torch.ops.aten._reshape_alias_copy


DTYPES = _BENCH_DTYPES

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(
    export_contract("_reshape_alias_copy", DTYPES, _resolve_native, _inputs)
)
