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


"""gather_sparse_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_gather_sparse_backward.py:48 (test_gather_sparse_backward)
Native expression: torch.ops.aten._gather_sparse_backward.default
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _gather_sparse_backward_input_fn(shape, dtype, device):
    """Yield ``(self, dim, index, grad)`` tuples for the gather sparse backward.

    `self` keeps the full dim while `index`/`grad` sample half of it along the
    gather dimension (a common real-world gradient shape).
    """
    inp = torch.randn(shape, dtype=dtype, device=device)
    for dim in range(len(shape)):
        index_shape = list(shape)
        index_shape[dim] = max(1, shape[dim] // 2)
        index = torch.randint(
            0, shape[dim], tuple(index_shape), dtype=torch.long, device=device
        )
        grad = torch.randn(tuple(index_shape), dtype=dtype, device=device)
        yield (inp, dim, index, grad)


def _resolve_native():
    return torch.ops.aten._gather_sparse_backward.default


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _gather_sparse_backward_input_fn

globals().update(
    export_contract("gather_sparse_backward", DTYPES, _resolve_native, _inputs)
)
