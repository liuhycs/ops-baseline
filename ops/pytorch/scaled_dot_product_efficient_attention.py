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


"""scaled_dot_product_efficient_attention: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_scaled_dot_product_efficient_attention.py:37 (test_scaled_dot_product_efficient_attention)
Native expression: torch.ops.aten._scaled_dot_product_efficient_attention
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        batch, num_heads, seq_len, head_dim = shape
        query = torch.randn(
            batch, num_heads, seq_len, head_dim, dtype=dtype, device=device
        )
        key = torch.randn(
            batch, num_heads, seq_len, head_dim, dtype=dtype, device=device
        )
        value = torch.randn(
            batch, num_heads, seq_len, head_dim, dtype=dtype, device=device
        )
        yield (query, key, value, None, False)


def _resolve_native():
    return torch.ops.aten._scaled_dot_product_efficient_attention


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(
    export_contract(
        "scaled_dot_product_efficient_attention", DTYPES, _resolve_native, _inputs
    )
)
