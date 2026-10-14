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


"""_cast_Short: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test__cast_Short.py:94 (test__cast_Short)
Native expression: torch.ops.aten._cast_Short
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

NON_BLOCKING_VALUES = [None, False, True]

BENCH_DTYPES = (
    [torch.float16, torch.float32, torch.bfloat16]
    if True
    else [
        dtype
        for dtype in [torch.float16, torch.float32, torch.bfloat16]
        if dtype != torch.bfloat16
    ]
)


def _case_fn(shape, dtype):
    del dtype
    for non_blocking in NON_BLOCKING_VALUES:
        params = {} if non_blocking is None else {"non_blocking": non_blocking}
        yield BenchmarkCasePlan(
            shape={"input": shape}, params=params, builder_args=(shape, non_blocking)
        )


def _build_inputs_fn(plan, dtype, device):
    shape, non_blocking = plan.builder_args
    inp = generate_tensor_input(shape, dtype, device)
    if non_blocking is None:
        return (inp, {})
    return (inp, {"non_blocking": non_blocking})


def _resolve_native():
    return torch.ops.aten._cast_Short


DTYPES = BENCH_DTYPES

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(export_contract("_cast_Short", DTYPES, _resolve_native, _inputs))
