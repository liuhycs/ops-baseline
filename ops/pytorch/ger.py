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


"""ger: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_ger.py:71 (test_ger)
Native expression: torch.ops.aten.ger
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

BENCH_DTYPES = [torch.float16, torch.float32]


def _case_fn(shape, dtype):
    del dtype
    self_len, vec2_len = shape
    yield BenchmarkCasePlan(
        shape={"self": self_len, "vec2": vec2_len},
        params={},
        builder_args=(self_len, vec2_len),
    )


def _build_inputs_fn(plan, dtype, device):
    self_len, vec2_len = plan.builder_args
    self_vec = generate_tensor_input((self_len,), dtype, device)
    vec2 = generate_tensor_input((vec2_len,), dtype, device)
    return (self_vec, vec2)


def _resolve_native():
    return torch.ops.aten.ger


DTYPES = BENCH_DTYPES

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(export_contract("ger", DTYPES, _resolve_native, _inputs))
