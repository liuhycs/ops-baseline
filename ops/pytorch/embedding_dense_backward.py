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


"""embedding_dense_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_embedding_dense_backward.py:56 (test_embedding_dense_backward)
Native expression: torch.ops.aten.embedding_dense_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_fn(shape, dtype, device):
    B, M, D, num_weights = shape
    grad_output = torch.randn((B, M, D), device=device, dtype=dtype)
    indices = torch.randint(0, num_weights, (B, M), device=device, dtype=torch.long)

    def inject_padding_idx(cur_indices, padding_idx):
        if padding_idx < 0:
            return cur_indices
        mask = torch.rand((B, M), device=device) < 0.25
        return torch.where(mask, torch.full_like(cur_indices, padding_idx), cur_indices)

    test_cases = [(-1, False), (0, True), (5, False)]
    for padding_idx, scale_grad_by_freq in test_cases:
        cur_indices = inject_padding_idx(indices, padding_idx)
        yield (grad_output, cur_indices, num_weights, padding_idx, scale_grad_by_freq)


def _resolve_native():
    return torch.ops.aten.embedding_dense_backward


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_fn

globals().update(
    export_contract("embedding_dense_backward", DTYPES, _resolve_native, _inputs)
)
