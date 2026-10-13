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


"""special_scaled_modified_bessel_k0: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_special_scaled_modified_bessel_k0.py:86 (test_special_scaled_modified_bessel_k0)
Native expression: torch.ops.aten.special_scaled_modified_bessel_k0
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

_DTYPES = [torch.float32]


def _case_fn(shape, dtype):
    del dtype
    yield BenchmarkCasePlan(shape={"input": shape}, builder_args=(shape,))


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    if dtype == torch.float64:
        return (torch.randn(shape, dtype=dtype, device=device), {})
    return (generate_tensor_input(shape, dtype, device), {})


def _resolve_native():
    return torch.ops.aten.special_scaled_modified_bessel_k0


DTYPES = _DTYPES

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(
    export_contract(
        "special_scaled_modified_bessel_k0", DTYPES, _resolve_native, _inputs
    )
)
