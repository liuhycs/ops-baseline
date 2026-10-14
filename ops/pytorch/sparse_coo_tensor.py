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


"""sparse_coo_tensor: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_sparse_coo_tensor.py:89 (test_sparse_coo_tensor)
Native expression: torch.ops.aten.sparse_coo_tensor
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


def _make_coo_inputs(shape, nnz, dtype, device, seed=0):
    gen = torch.Generator("cpu").manual_seed(seed)
    sparse_shape = shape[:2]
    dense_shape = shape[2:]
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    ).to(device)
    values = generate_tensor_input((nnz,) + tuple(dense_shape), dtype, device)
    return (indices, values)


def _case_fn(shape, dtype):
    del dtype
    tensor_shape, nnz = shape
    yield BenchmarkCasePlan(
        shape={"input": tensor_shape},
        params={"nnz": nnz},
        builder_args=(tensor_shape, nnz),
    )


def _build_inputs_fn(plan, dtype, device):
    tensor_shape, nnz = plan.builder_args
    indices, values = _make_coo_inputs(tensor_shape, nnz, dtype, device)
    return (indices, values, list(tensor_shape), {"dtype": dtype, "device": device})


def _resolve_native():
    return torch.ops.aten.sparse_coo_tensor


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(export_contract("sparse_coo_tensor", DTYPES, _resolve_native, _inputs))
