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


"""_validate_compressed_sparse_indices: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test__validate_compressed_sparse_indices.py:161 (test_validate_compressed_sparse_indices)
Native expression: torch.ops.aten._validate_compressed_sparse_indices
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import BenchmarkCasePlan, case_inputs, export_contract

import itertools

import numbers

_INDEX_DTYPES = [torch.int32]


def _row_lengths(cdim):
    """Compressed row lengths holding at most 8 plain entries per row.

    A row's plain indices are ``arange(count)``, so the longest row bounds the
    largest index; keeping it at 8 or below lets ``dim`` stay small for every
    default shape.  The lengths sum to ``cdim`` and the caller pads the list to
    exactly ``cdim`` rows, so the requested compressed dimension is never
    shrunk to the number of filled rows.
    """
    if cdim <= 0:
        return ()
    block = min(8, cdim)
    return (block,) * (cdim // block) + ((cdim % block,) if cdim % block else ())


def _validated_shape(shape):
    """Exact non-negative integer extents, without coercion or clamping.

    A float or bool extent, a non-sequence, or a negative extent is a malformed
    shape request and is rejected instead of being truncated into a different
    descriptor.  Zero extents stay valid.
    """
    if not isinstance(shape, (tuple, list)):
        raise TypeError(f"a benchmark shape must be a sequence, got {shape!r}")
    extents = []
    for entry in shape:
        if isinstance(entry, bool) or not isinstance(entry, numbers.Integral):
            raise TypeError(f"shape extents must be integers, got {entry!r}")
        if entry < 0:
            raise ValueError(f"shape extents must be non-negative, got {entry!r}")
        extents.append(int(entry))
    if not extents:
        raise ValueError("a compressed descriptor needs at least one dimension")
    return tuple(extents)


def _case_fn(shape, dtype):
    shape = _validated_shape(shape)
    batch, cdim = (shape[:-1], shape[-1])
    filled = _row_lengths(cdim)
    counts = filled + (0,) * (cdim - len(filled))
    nnz = sum(counts)
    dim = min(8, cdim) or 1
    for is_crow in (False, True):
        yield BenchmarkCasePlan(
            shape={
                "batch": list(batch),
                "cdim": cdim,
                "compressed": [*batch, cdim + 1],
                "plain": [*batch, nnz],
            },
            params={
                "is_crow": is_crow,
                "cdim": cdim,
                "nnz": nnz,
                "dim": dim,
                "index_dtype": str(dtype).removeprefix("torch."),
            },
            builder_args=(batch, cdim, counts, dim, is_crow),
        )


def _build_inputs_fn(plan, dtype, device):
    batch, cdim, counts, dim, is_crow = plan.builder_args
    nnz = sum(counts)
    compressed = torch.tensor(
        [0, *itertools.accumulate(counts)], dtype=dtype, device=device
    )
    plain = torch.tensor(
        [value for count in counts for value in range(count)],
        dtype=dtype,
        device=device,
    )
    if batch:
        compressed = compressed.expand(*batch, -1).contiguous()
        plain = plain.expand(*batch, -1).contiguous()
    return (is_crow, compressed, plain, cdim, dim, nnz)


def _resolve_native():
    return torch.ops.aten._validate_compressed_sparse_indices


DTYPES = _INDEX_DTYPES

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(
    export_contract(
        "_validate_compressed_sparse_indices", DTYPES, _resolve_native, _inputs
    )
)
