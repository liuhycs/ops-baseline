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


"""scaled_dot_product_attention_math: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_scaled_dot_product_attention_math.py:39 (test_scaled_dot_product_attention_math)
Native expression: sdpa_math
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

sdpa_math = torch.ops.aten._scaled_dot_product_attention_math


def scaled_dot_product_attention_math_input_fn(shape, dtype, device):
    query = torch.randn(shape, dtype=dtype, device=device)
    key = torch.randn(shape, dtype=dtype, device=device)
    value = torch.randn(shape, dtype=dtype, device=device)
    yield (query, key, value)


def _resolve_native():
    return sdpa_math


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = scaled_dot_product_attention_math_input_fn

globals().update(
    export_contract(
        "scaled_dot_product_attention_math", DTYPES, _resolve_native, _inputs
    )
)
