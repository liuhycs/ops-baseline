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


"""glu_backward_jvp: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_glu_backward_jvp.py:73 (test_glu_backward_jvp)
Native expression: torch.ops.aten.glu_backward_jvp
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

DTYPES = [
    dtype
    for dtype in [torch.float16, torch.float32, torch.bfloat16]
    if dtype is not torch.bfloat16 or True
]


def _case_fn(shape, dtype):
    del dtype
    dim = len(shape) - 1
    grad_glu_shape = list(shape)
    grad_glu_shape[dim] //= 2
    yield BenchmarkCasePlan(
        shape={"x": list(shape), "grad_glu": grad_glu_shape},
        params={"dim": dim},
        builder_args=(tuple(shape), dim, tuple(grad_glu_shape)),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, dim, grad_glu_shape = plan.builder_args
    grad_x = generate_tensor_input(shape, dtype, device)
    grad_glu = generate_tensor_input(grad_glu_shape, dtype, device)
    x = generate_tensor_input(shape, dtype, device)
    dgrad_glu = generate_tensor_input(grad_glu_shape, dtype, device)
    dx = generate_tensor_input(shape, dtype, device)
    return (grad_x, grad_glu, x, dgrad_glu, dx, {"dim": dim})


def _resolve_native():
    return torch.ops.aten.glu_backward_jvp


DTYPES = DTYPES

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(export_contract("glu_backward_jvp", DTYPES, _resolve_native, _inputs))
