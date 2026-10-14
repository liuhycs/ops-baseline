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


"""_sparse_mm: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test__sparse_mm.py:130 (test__sparse_mm)
Native expression: torch.ops.aten._sparse_mm
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

BENCH_DTYPES = [torch.float32]
BENCH_DTYPES.extend([torch.float64])  # Pinned NVIDIA capabilities.


def _resolve_descriptor(rows, inner, cols, nnz_per_row):
    """Return the ``(stored_nnz, coalesced)`` pair the builder will create.

    Planning and construction share this helper, so the metadata reported by
    ``--list-cases`` cannot disagree with the tensors built for that case. Every
    field must be a plain non-negative integer: a fractional, boolean or string
    field is rejected while the case metadata is planned instead of being
    silently truncated when the case runs.
    """
    descriptor = (rows, inner, cols, nnz_per_row)
    for name, value in zip(("rows", "inner", "cols", "nnz_per_row"), descriptor):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be a plain integer: {descriptor}")
        if value < 0:
            raise ValueError(f"{name} must be non-negative: {descriptor}")
    if inner == 0 and nnz_per_row > 0:
        raise ValueError(
            f"inner==0 leaves no column to store into; use nnz_per_row=0: {descriptor}"
        )
    if rows == 0 or inner == 0 or nnz_per_row == 0:
        return (0, True)
    return (rows * nnz_per_row, nnz_per_row <= inner)


def _case_fn(shape, dtype):
    del dtype
    rows, inner, cols, nnz_per_row = shape
    stored_nnz, coalesced = _resolve_descriptor(rows, inner, cols, nnz_per_row)
    yield BenchmarkCasePlan(
        shape={"rows": rows, "inner": inner, "cols": cols, "nnz_per_row": nnz_per_row},
        params={"stored_nnz": stored_nnz, "coalesced": coalesced},
        builder_args=(rows, inner, cols, nnz_per_row),
    )


def _random_values(shape, dtype, device):
    if dtype in [torch.float16, torch.float32, torch.bfloat16]:
        return generate_tensor_input(shape, dtype, device)
    return torch.randn(shape, dtype=dtype, device=device)


def _build_inputs_fn(plan, dtype, device):
    rows, inner, cols, nnz_per_row = plan.builder_args
    stored_nnz, coalesced = _resolve_descriptor(rows, inner, cols, nnz_per_row)
    if stored_nnz == 0:
        indices = torch.empty((2, 0), dtype=torch.long, device=device)
        values = torch.empty((0,), dtype=dtype, device=device)
    else:
        row_idx = torch.arange(rows, device=device).repeat_interleave(nnz_per_row)
        col_idx = (torch.arange(nnz_per_row, device=device) % inner).repeat(rows)
        indices = torch.stack([row_idx, col_idx])
        values = _random_values((stored_nnz,), dtype, device)
    sparse = torch.sparse_coo_tensor(
        indices, values, (rows, inner), device=device, is_coalesced=coalesced
    )
    dense = _random_values((inner, cols), dtype, device)
    return (sparse, dense)


def _resolve_native():
    return torch.ops.aten._sparse_mm


DTYPES = BENCH_DTYPES

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(export_contract("_sparse_mm", DTYPES, _resolve_native, _inputs))
