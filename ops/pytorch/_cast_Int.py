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


"""_cast_Int: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test__cast_Int.py:91 (test__cast_Int)
Native expression: torch.ops.aten._cast_Int
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

_DTYPE_GATES = {torch.bfloat16: True}

_BENCH_DTYPES = [
    dtype
    for dtype in [torch.float16, torch.float32, torch.bfloat16]
    + [torch.int16, torch.int32]
    if _DTYPE_GATES.get(dtype, True)
]

_CALL_FORMS = (None, False, True)


def _checked_shape(shape):
    """Reject unusable shape metadata while listing, without allocating."""
    dims = []
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise ValueError(f"shape dimensions must be ints, got {dim!r}")
        if dim < 0:
            raise ValueError(f"shape dimensions must be non-negative, got {dim!r}")
        dims.append(dim)
    return tuple(dims)


def _case_fn(shape, dtype):
    del dtype
    dims = _checked_shape(shape)
    for non_blocking in _CALL_FORMS:
        yield BenchmarkCasePlan(
            shape={"input": dims},
            params={} if non_blocking is None else {"non_blocking": non_blocking},
            builder_args=(dims, non_blocking),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, non_blocking = plan.builder_args
    inp = generate_tensor_input(shape, dtype, device)
    if non_blocking is None:
        return (inp,)
    return (inp, {"non_blocking": non_blocking})


def _resolve_native():
    return torch.ops.aten._cast_Int


DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
    torch.int16,
]  # CUDA-supported subset; see coverage.json

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(export_contract("_cast_Int", DTYPES, _resolve_native, _inputs))
