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


"""glu_jvp: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_glu_jvp.py:82 (test_glu_jvp)
Native expression: torch.ops.aten.glu_jvp
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

_SUPPORT_BF16 = True

_BENCH_DTYPES = [
    dtype
    for dtype in [torch.float16, torch.float32, torch.bfloat16]
    if dtype != torch.bfloat16 or _SUPPORT_BF16
]


def _benchmark_dims(shape):
    """halving dims benchmarked for shape: the last dim, plus dim 0 if it exists.

    A primal extent of zero is a valid native workload (the output is empty),
    so no shape is filtered out here and every requested case is timed.
    """
    return [-1] + ([0] if len(shape) > 1 else [])


def _case_fn(shape, dtype):
    del dtype
    for dim in _benchmark_dims(shape):
        glu_shape = list(shape)
        glu_shape[dim] = shape[dim] // 2
        yield BenchmarkCasePlan(
            shape={"x": list(shape), "glu": glu_shape},
            params={"dim": dim},
            builder_args=(tuple(shape), dim),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, dim = plan.builder_args
    glu_shape = list(shape)
    glu_shape[dim] = shape[dim] // 2
    glu = generate_tensor_input(tuple(glu_shape), dtype, device)
    x = generate_tensor_input(shape, dtype, device)
    dx = generate_tensor_input(shape, dtype, device)
    return (glu, x, dx, dim)


def _resolve_native():
    return torch.ops.aten.glu_jvp


DTYPES = _BENCH_DTYPES

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(export_contract("glu_jvp", DTYPES, _resolve_native, _inputs))
