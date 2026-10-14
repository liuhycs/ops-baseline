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


"""index_select_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_index_select_backward.py:54 (test_index_select_backward)
Native expression: torch.ops.aten.index_select_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml index_select_backward or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

import random


def _input_fn(shape, dtype, device):
    dim = random.randint(0, len(shape) - 1)
    index_len = shape[dim]
    dim_size_out = index_len + 8
    self_sizes = list(shape)
    self_sizes[dim] = dim_size_out
    self_sizes = tuple(self_sizes)
    grad = torch.randn(shape, dtype=dtype, device=device)
    index = torch.randint(0, dim_size_out, (index_len,), device=device)
    yield (grad, self_sizes, dim, index)


def _resolve_native():
    return torch.ops.aten.index_select_backward


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_fn

globals().update(
    export_contract("index_select_backward", DTYPES, _resolve_native, _inputs)
)
