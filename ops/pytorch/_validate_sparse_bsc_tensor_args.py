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


"""_validate_sparse_bsc_tensor_args: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test__validate_sparse_bsc_tensor_args.py:209 (test__validate_sparse_bsc_tensor_args)
Native expression: torch.ops.aten._validate_sparse_bsc_tensor_args
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

_INDEX_DTYPE_KEYS = {"int32": torch.int32, "int64": torch.int64}

_INDEX_KEY_BY_DTYPE = {dtype: key for key, dtype in _INDEX_DTYPE_KEYS.items()}


def _resolve_index_dtype(value):
    """Normalize a descriptor index dtype to a torch dtype.

    Accepts the documented string keys and an already-resolved torch dtype, so a
    YAML/JSON descriptor and a Python descriptor describe the same workload.
    """
    if isinstance(value, str):
        if value not in _INDEX_DTYPE_KEYS:
            raise ValueError(
                f"unknown index dtype {value!r}; expected one of {sorted(_INDEX_DTYPE_KEYS)}"
            )
        return _INDEX_DTYPE_KEYS[value]
    if isinstance(value, torch.dtype):
        if value not in _INDEX_DTYPE_KEYS.values():
            raise ValueError(f"unsupported index dtype {value}")
        return value
    raise ValueError(
        f"index dtype must be a string key or a torch.dtype, got {type(value).__name__}"
    )


def _require_int(value, what, *, minimum):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{what} must be >= {minimum}, got {value}")
    return value


def _descriptor_parts(descriptor):
    """Validate one benchmark descriptor and return its normalized parts.

    A descriptor is (size, blocksize, fill, index_dtype) with an optional
    trailing dense extent list, e.g. ([256, 256, 4], [16, 16], 2, 'int64', [4]).
    Arity, non-bool integer extents, positive block extents, divisibility,
    `fill <= nrow_blocks` and a dense tail matching the logical size tail are
    enforced here, before any tensor is allocated: the native validator rejects
    the same descriptors, so listing them would advertise workloads that can
    never execute. Zero logical extents, `fill == 0` and a matching zero dense
    tail stay valid, because the native worker accepts them, and no metadata is
    silently converted, clamped or truncated.
    """
    if not isinstance(descriptor, (list, tuple)) or isinstance(descriptor, str):
        raise ValueError(f"descriptor must be a list or tuple, got {descriptor!r}")
    if len(descriptor) not in (4, 5):
        raise ValueError(
            f"descriptor must have 4 or 5 entries (size, blocksize, fill, index_dtype[, dense]), got {len(descriptor)}"
        )
    size, blocksize, fill, index_key = descriptor[:4]
    dense = descriptor[4] if len(descriptor) == 5 else []
    if not isinstance(size, (list, tuple)) or isinstance(size, str):
        raise ValueError(f"size must be a list of integers, got {size!r}")
    dims = [_require_int(dim, "logical extent", minimum=0) for dim in size]
    if not isinstance(blocksize, (list, tuple)) or isinstance(blocksize, str):
        raise ValueError(f"blocksize must be a 2-entry list, got {blocksize!r}")
    if len(blocksize) != 2:
        raise ValueError(f"blocksize must have 2 entries, got {len(blocksize)}")
    rb, cb = (
        _require_int(extent, "blocksize extent", minimum=1) for extent in blocksize
    )
    if not isinstance(dense, (list, tuple)) or isinstance(dense, str):
        raise ValueError(f"dense extents must be a list of integers, got {dense!r}")
    dense = [_require_int(extent, "dense extent", minimum=0) for extent in dense]
    fill = _require_int(fill, "fill", minimum=0)
    index_dtype = _resolve_index_dtype(index_key)
    if len(dims) < len(dense) + 2:
        raise ValueError(
            f"size {dims} cannot hold {len(dense)} dense extents plus 2 base extents"
        )
    if dense and list(dims[-len(dense) :]) != list(dense):
        raise ValueError(
            f"dense extents {dense} must equal the logical size tail {dims[-len(dense) :]}"
        )
    nrows = dims[len(dims) - 2 - len(dense)]
    ncols = dims[len(dims) - 1 - len(dense)]
    if nrows % rb or ncols % cb:
        raise ValueError(
            f"blocksize {rb}x{cb} does not tile logical shape {nrows}x{ncols}"
        )
    nrow_blocks = nrows // rb
    if fill > nrow_blocks:
        raise ValueError(
            f"fill {fill} exceeds the {nrow_blocks} block rows of {nrows} rows"
        )
    return (dims, (rb, cb), dense, fill, index_dtype)


_BENCH_DTYPES = [
    dtype
    for dtype in [torch.float16, torch.float32, torch.bfloat16]
    if dtype != torch.bfloat16 or True
]


def _case_fn(case, dtype):
    del dtype
    dims, blocksize, dense, fill, index_dtype = _descriptor_parts(case)
    yield BenchmarkCasePlan(
        shape={"size": list(dims), "blocksize": [blocksize[0], blocksize[1]]},
        params={
            "dense": list(dense),
            "fill": fill,
            "index_dtype": _INDEX_KEY_BY_DTYPE[index_dtype],
        },
        builder_args=(dims, blocksize, fill, index_dtype, dense),
    )


def _build_inputs_fn(plan, dtype, device):
    dims, blocksize, dense, fill, index_dtype = _descriptor_parts(plan.builder_args)
    rb, cb = blocksize
    batch = tuple(dims[: len(dims) - 2 - len(dense)])
    ncols = dims[len(dims) - 1 - len(dense)]
    counts = torch.full((ncols // cb,), fill, dtype=torch.int64)
    offsets = torch.cumsum(counts, 0)
    nnz = int(counts.sum())
    ccol = torch.cat((torch.zeros(1, dtype=torch.int64), offsets))
    row = torch.arange(nnz) - (offsets - counts).repeat_interleave(counts)
    if batch:
        ccol = ccol.expand(batch + (-1,)).contiguous()
        row = row.expand(batch + (-1,)).contiguous()
    values = generate_tensor_input(batch + (nnz, rb, cb) + tuple(dense), dtype, device)
    return (
        ccol.to(index_dtype).to(device),
        row.to(index_dtype).to(device),
        values,
        list(dims),
    )


def _resolve_native():
    return torch.ops.aten._validate_sparse_bsc_tensor_args


DTYPES = _BENCH_DTYPES

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(
    export_contract(
        "_validate_sparse_bsc_tensor_args", DTYPES, _resolve_native, _inputs
    )
)
