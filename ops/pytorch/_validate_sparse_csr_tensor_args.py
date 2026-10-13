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


"""_validate_sparse_csr_tensor_args: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test__validate_sparse_csr_tensor_args.py:217 (test_validate_sparse_csr_tensor_args)
Native expression: torch.ops.aten._validate_sparse_csr_tensor_args
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import BenchmarkCasePlan, case_inputs, export_contract

BENCH_INDEX_DTYPES = [torch.int32]

BENCH_VALUES_DTYPES = [torch.float32, torch.float16]


def _require_dim(value):
    """An exact non-negative integer CSR dimension.

    A float, a bool or a negative value is rejected instead of being truncated,
    coerced or clamped, so a requested extent is never silently changed.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"CSR dimensions must be integers, got {value!r}")
    if value < 0:
        raise ValueError(f"CSR dimensions must be non-negative, got {value!r}")
    return value


def _normalize_size(shape):
    """Return the CSR size for a default or shape-file entry.

    A bare int or a rank-1 entry is read as the side of a square size, because a
    CSR description needs both a rows and a cols dimension; every rank >= 2
    entry is used exactly as requested. Dimensions are validated before any
    metadata or allocation, and no extent is ever shrunk.
    """
    if isinstance(shape, bool):
        raise ValueError(
            f"CSR size must be an int or a sequence of ints, got {shape!r}"
        )
    if isinstance(shape, int):
        dim = _require_dim(shape)
        return (dim, dim)
    if isinstance(shape, (list, tuple)):
        dims = [_require_dim(dim) for dim in shape]
        if not dims:
            return (0, 0)
        if len(dims) == 1:
            return (dims[0], dims[0])
        return tuple(dims)
    raise ValueError(f"CSR size must be an int or a sequence of ints, got {shape!r}")


def _row_counts(rows, cols, pattern):
    """Stored entries per row.

    `one_per_row` stores exactly one entry per row, so the per-batch NNZ is
    `rows` even when rows > cols: the same column may appear again in another
    row, which is a valid CSR description, so the requested NNZ is never
    reduced. `cols == 0` is the only case that must store nothing, because every
    column index would break the 0 <= col < cols bound.
    """
    if cols == 0 or pattern == "empty":
        return [0] * rows
    if pattern == "one_per_row":
        return [1] * rows
    if pattern == "sparse":
        return [1 if row % 2 == 0 else 0 for row in range(rows)]
    if pattern == "uneven":
        counts = [2, 0] + [1] * max(rows - 2, 0)
        return counts[:rows]
    raise ValueError(f"unknown CSR pattern: {pattern}")


def _patterns_for(size):
    rows, cols = (size[-2], size[-1])
    if cols == 0:
        return ["empty"]
    patterns = ["one_per_row"]
    if rows >= 2:
        patterns.append("sparse")
    if rows >= 2 and cols >= 2:
        patterns.append("uneven")
    return patterns


def _describe(size, pattern, index_dtype_name):
    """Metadata for one (size, pattern, index dtype) workload.

    Pure Python on purpose: --list-cases and the input builder both call it, so
    the listed metadata and the executed case can never disagree.
    """
    rows, cols = (size[-2], size[-1])
    batch = tuple(size[:-2])
    batches = 1
    for dim in batch:
        batches *= dim
    counts = _row_counts(rows, cols, pattern)
    nnz = sum(counts)
    return {
        "pattern": pattern,
        "index_dtype": index_dtype_name,
        "batch": list(batch),
        "rows": rows,
        "cols": cols,
        "batches": batches,
        "nnz": nnz,
        "crow_shape": list(batch) + [rows + 1],
        "col_shape": list(batch) + [nnz],
        "values_shape": list(batch) + [nnz],
    }


def _case_fn(shape, dtype):
    del dtype
    size = _normalize_size(shape)
    for pattern in _patterns_for(size):
        for index_dtype in BENCH_INDEX_DTYPES:
            yield BenchmarkCasePlan(
                shape={"size": list(size)},
                params=_describe(size, pattern, str(index_dtype)),
                builder_args=(size, pattern, index_dtype),
            )


def _make_values(shape, dtype, device):
    """`values` is never inspected, but an empty description still needs a
    correctly shaped tensor."""
    if 0 in shape:
        return torch.empty(shape, dtype=dtype, device=device)
    return torch.randn(shape, dtype=torch.float32, device=device).to(dtype)


def _build_inputs_fn(plan, dtype, device):
    size, pattern, index_dtype = plan.builder_args
    description = _describe(size, pattern, str(index_dtype))
    rows, cols = (description["rows"], description["cols"])
    nnz, batches = (description["nnz"], description["batches"])
    batch = tuple(description["batch"])
    counts = torch.tensor(
        _row_counts(rows, cols, pattern), dtype=index_dtype, device=device
    )
    crow = torch.zeros(rows + 1, dtype=index_dtype, device=device)
    crow[1:] = counts.cumsum(0, dtype=index_dtype)
    if pattern == "one_per_row":
        col = torch.arange(nnz, dtype=index_dtype, device=device).remainder(cols)
    else:
        col = torch.zeros(nnz, dtype=index_dtype, device=device)
        if pattern == "uneven" and nnz >= 2:
            col[1] = 1
    crow_indices = crow.repeat(batches).reshape(batch + (rows + 1,))
    col_indices = col.repeat(batches).reshape(batch + (nnz,))
    values = _make_values(tuple(description["values_shape"]), dtype, device)
    return (crow_indices, col_indices, values, {"size": list(size)})


def _resolve_native():
    return torch.ops.aten._validate_sparse_csr_tensor_args


DTYPES = BENCH_VALUES_DTYPES

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(
    export_contract(
        "_validate_sparse_csr_tensor_args", DTYPES, _resolve_native, _inputs
    )
)
