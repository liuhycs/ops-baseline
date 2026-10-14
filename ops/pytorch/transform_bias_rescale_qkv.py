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


"""transform_bias_rescale_qkv: PyTorch baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_transform_bias_rescale_qkv.py:68 (test__transform_bias_rescale_qkv)
Native expression: torch_op
Constructor state was explicitly lowered; source custom_shapes passed to constructor.
Core source parameters and filters are preserved.
"""

import torch

from _kernelgen import export_contract


def _transform_bias_rescale_qkv_input_fn(shape, dtype, device):
    """Generate inputs for _transform_bias_rescale_qkv benchmark.

    shape is expected to be (batch, seq_len, num_heads, head_dim)
    """
    if len(shape) != 4:
        return
    batch, seq_len, num_heads, head_dim = shape
    hidden_dim = num_heads * head_dim
    total_dim = 3 * hidden_dim
    qkv = torch.randn(batch, seq_len, total_dim, dtype=dtype, device=device)
    bias = torch.randn(total_dim, dtype=dtype, device=device)
    yield (qkv, bias, num_heads)


def torch_op(qkv, bias, num_heads):
    return torch._transform_bias_rescale_qkv(qkv, bias, num_heads)


DTYPES = [torch.float16, torch.float32, torch.bfloat16]


def _resolve_native():
    return torch_op


_inputs = _transform_bias_rescale_qkv_input_fn

globals().update(
    export_contract("transform_bias_rescale_qkv", DTYPES, _resolve_native, _inputs)
)
