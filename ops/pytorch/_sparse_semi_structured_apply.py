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


"""_sparse_semi_structured_apply: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test__sparse_semi_structured_apply.py:114 (test__sparse_semi_structured_apply)
Native expression: torch.ops.aten._sparse_semi_structured_apply
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

APPLY_DTYPES = [torch.float16] + ([torch.bfloat16] if True else [])


def _tile_problem(shape):
    """Describe why ``shape`` is not a legal 2:4 activation tile, else ``None``.

    The native kernel takes rank-2 tiles with positive extents, ``M % 32 == 0``
    and ``K % 64 == 0``, and raises for any other shape. A caller-supplied shape
    file can also carry entries written for other operators, so an illegal tile is
    reported while the cases are listed rather than failing only inside the timed
    call. The descriptor is validated before any arithmetic, so a malformed entry
    (a float or bool extent, a string, or something that is not a sequence of two
    extents) yields this diagnostic instead of a TypeError from the modulo below.
    """
    if not isinstance(shape, (list, tuple)):
        return (
            f"the tile must be a list or tuple of extents, got {type(shape).__name__}"
        )
    if len(shape) != 2:
        return f"the native kernel takes a rank-2 (M, K) tile, got rank {len(shape)}"
    for extent in shape:
        if isinstance(extent, bool) or not isinstance(extent, int):
            return f"tile extents must be integers, got {shape!r}"
    rows, cols = shape
    if rows <= 0 or cols <= 0:
        return f"the native kernel rejects zero-extent tiles, got {shape!r}"
    if rows % 32:
        return f"M must be a positive multiple of 32, got {rows}"
    if cols % 64:
        return f"K must be a positive multiple of 64, got {cols}"
    return None


def _case_fn(shape, dtype):
    problem = _tile_problem(shape)
    if problem is not None:
        raise ValueError(
            f"illegal _sparse_semi_structured_apply workload {shape!r}: {problem}"
        )
    if dtype not in APPLY_DTYPES:
        raise ValueError(
            f"_sparse_semi_structured_apply takes {APPLY_DTYPES}, got {dtype}"
        )
    rows, cols = shape
    yield BenchmarkCasePlan(
        shape={"input": shape},
        params={"mask_rows": rows // 8, "mask_cols": cols // 8},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = generate_tensor_input(shape, dtype, device)
    masks = torch.randint(
        0, 256, (shape[0] // 8, shape[1] // 8, 8), dtype=torch.uint8, device=device
    )
    return (inp, masks)


def _resolve_native():
    return torch.ops.aten._sparse_semi_structured_apply


DTYPES = APPLY_DTYPES

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(
    export_contract("_sparse_semi_structured_apply", DTYPES, _resolve_native, _inputs)
)
