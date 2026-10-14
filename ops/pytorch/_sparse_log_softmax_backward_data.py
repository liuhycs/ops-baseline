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


"""_sparse_log_softmax_backward_data: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test__sparse_log_softmax_backward_data.py:174 (test__sparse_log_softmax_backward_data_benchmark)
Native expression: torch.ops.aten._sparse_log_softmax_backward_data
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import BenchmarkCasePlan, case_inputs, export_contract

import math

_DTYPES = [torch.float32]
_DTYPES.extend([torch.float64])  # Pinned NVIDIA capabilities.


def _plan_fields(entry):
    """Validate a (shape, nnz[, sparse_dim, dim]) descriptor without altering it."""
    entry = tuple(entry)
    if entry and isinstance(entry[0], (tuple, list)):
        shape = tuple(entry[0])
        nnz = entry[1] if len(entry) > 1 else math.prod(shape)
        sparse_dim = entry[2] if len(entry) > 2 else len(shape)
        dim = entry[3] if len(entry) > 3 else len(shape) - 1
    else:
        shape = tuple(entry)
        nnz, sparse_dim, dim = (math.prod(shape), len(shape), len(shape) - 1)
    nnz, sparse_dim, dim = (int(nnz), int(sparse_dim), int(dim))
    if not 0 <= sparse_dim <= len(shape):
        raise ValueError(
            f"invalid case {entry!r}: sparse_dim must be in [0, {len(shape)}]"
        )
    if nnz < 0:
        raise ValueError(f"invalid case {entry!r}: nnz must not be negative")
    if len(shape) == 0:
        if dim != -1:
            raise ValueError(
                f"invalid case {entry!r}: a scalar shape only accepts dim=-1"
            )
    elif not -len(shape) <= dim < len(shape):
        raise ValueError(
            f"invalid case {entry!r}: dim must be in [{-len(shape)}, {len(shape) - 1}]"
        )
    return (shape, nnz, sparse_dim, dim)


def _coords(shape, sparse_dim, dim, nnz, device):
    """Support coordinates built on `device` in O(nnz).

    A normalization group is identified by every sparse coordinate except the one along
    `dim`, so the group ids enumerate those axes and the `dim` coordinate is inserted
    back at its own position. Several entries are placed in each group so the reduction
    has a within-group sum; the three operands of a case share this support and differ
    only in their values, otherwise a group of a single random entry sums to zero about
    half the time and the workload degrades to the trivial result.
    """
    shape = tuple(shape)
    if sparse_dim == 0:
        return torch.empty((0, nnz), dtype=torch.long, device=device)
    if nnz == 0:
        return torch.empty((sparse_dim, 0), dtype=torch.long, device=device)
    if dim < 0:
        dim += len(shape)
    if dim < sparse_dim:
        dim_size = shape[dim]
        group_shape = tuple((shape[a] for a in range(sparse_dim) if a != dim))
        n_groups = math.prod(group_shape) if group_shape else 1
        cols = max(1, min(dim_size, nnz))
        index = torch.arange(nnz, device=device)
        used = max(1, -(-nnz // cols))
        step = max(1, n_groups // used)
        group_id = index // cols * step % n_groups
        dim_coord = index % cols
        parts = torch.unravel_index(group_id, group_shape) if group_shape else ()
        axes = iter(parts)
        flat = torch.stack(
            [dim_coord if a == dim else next(axes) for a in range(sparse_dim)]
        )
    else:
        spread = math.prod(shape[:sparse_dim])
        step = max(1, spread // nnz)
        flat = torch.stack(
            torch.unravel_index(
                torch.arange(nnz, device=device) * step % spread, shape[:sparse_dim]
            )
        )
    return flat.to(torch.long)


def _coo(shape, sparse_dim, dim, nnz, dtype, device):
    size = tuple(shape)
    values = torch.randn((nnz,) + size[sparse_dim:], dtype=dtype, device=device)
    indices = _coords(shape, sparse_dim, dim, nnz, device)
    return torch.sparse_coo_tensor(indices, values, torch.Size(size), device=device)


def _case_fn(shape, dtype):
    del dtype
    sparse_shape, nnz, sparse_dim, dim = _plan_fields(shape)
    yield BenchmarkCasePlan(
        shape={"sparse_shape": list(sparse_shape), "nnz": nnz},
        params={
            "sparse_dim": sparse_dim,
            "dense_dim": len(sparse_shape) - sparse_dim,
            "dim": dim,
        },
        builder_args=(sparse_shape, nnz, sparse_dim, dim),
    )


def _build_inputs_fn(plan, dtype, device):
    sparse_shape, nnz, sparse_dim, dim = plan.builder_args
    grad_output = _coo(sparse_shape, sparse_dim, dim, nnz, dtype, device)
    output = _coo(sparse_shape, sparse_dim, dim, nnz, dtype, device)
    self_ = _coo(sparse_shape, sparse_dim, dim, nnz, dtype, device)
    return (grad_output, output, dim, self_, {})


def _resolve_native():
    return torch.ops.aten._sparse_log_softmax_backward_data


DTYPES = _DTYPES

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(
    export_contract(
        "_sparse_log_softmax_backward_data", DTYPES, _resolve_native, _inputs
    )
)
