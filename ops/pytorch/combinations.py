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


"""combinations: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_combinations.py:85 (test_combinations)
Native expression: torch.ops.aten.combinations
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
    n, r, with_replacement = shape
    yield BenchmarkCasePlan(
        shape={"input": (n,)},
        params={"r": r, "with_replacement": with_replacement},
        builder_args=(n, r, with_replacement),
    )


def _build_inputs_fn(plan, dtype, device):
    n, r, with_replacement = plan.builder_args
    inp = generate_tensor_input((n,), dtype, device)
    return (inp, {"r": r, "with_replacement": with_replacement})


def _resolve_native():
    return torch.ops.aten.combinations


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(export_contract("combinations", DTYPES, _resolve_native, _inputs))
