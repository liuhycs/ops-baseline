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


"""_sparse_softmax_backward_data: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test__sparse_softmax_backward_data.py:99 (test__sparse_softmax_backward_data)
Native expression: torch.ops.aten._sparse_softmax_backward_data
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import BenchmarkCasePlan, case_inputs, export_contract

_DTYPES = [torch.float32] + ([torch.float64] if True else [])


def _case_fn(shape, dtype):
    del dtype
    sparse_shape, nnz, sparse_dim, dim = shape
    yield BenchmarkCasePlan(
        shape={"input": sparse_shape},
        params={"nnz": nnz, "sparse_dim": sparse_dim, "dim": dim},
        builder_args=(sparse_shape, nnz, sparse_dim, dim),
    )


def _build_inputs_fn(plan, dtype, device):
    sparse_shape, nnz, sparse_dim, dim = plan.builder_args
    if sparse_dim == 0:
        indices = torch.empty(0, nnz, dtype=torch.long, device=device)
    else:
        indices = torch.stack(
            [
                torch.randint(0, extent, (nnz,), dtype=torch.long, device=device)
                for extent in sparse_shape[:sparse_dim]
            ]
        )
    value_shape = (nnz,) + tuple(sparse_shape[sparse_dim:])
    grad = torch.sparse_coo_tensor(
        indices,
        torch.randn(value_shape, dtype=dtype, device=device),
        sparse_shape,
        device=device,
    )
    output = torch.sparse_coo_tensor(
        indices.clone(),
        torch.randn(value_shape, dtype=dtype, device=device),
        sparse_shape,
        device=device,
    )
    self_tensor = torch.randn(1, dtype=dtype, device=device)
    return (grad, output, dim, self_tensor, {})


def _resolve_native():
    return torch.ops.aten._sparse_softmax_backward_data


DTYPES = _DTYPES

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(
    export_contract("_sparse_softmax_backward_data", DTYPES, _resolve_native, _inputs)
)
