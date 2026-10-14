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


"""cartesian_prod: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_cartesian_prod.py:79 (test_cartesian_prod)
Native expression: torch.ops.aten.cartesian_prod
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


def _case_fn(sizes, dtype):
    del dtype
    yield BenchmarkCasePlan(
        shape={"inputs": list(sizes)},
        params={"n_inputs": len(sizes)},
        builder_args=(sizes,),
    )


def _build_inputs_fn(plan, dtype, device):
    sizes = plan.builder_args[0]
    tensors = [generate_tensor_input(size, dtype, device) for size in sizes]
    return (tensors, {})


def _resolve_native():
    return torch.ops.aten.cartesian_prod


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(export_contract("cartesian_prod", DTYPES, _resolve_native, _inputs))
