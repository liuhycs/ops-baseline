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


"""_transformer_encoder_layer_fwd: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test__transformer_encoder_layer_fwd.py:103 (test__transformer_encoder_layer_fwd)
Native expression: torch._transformer_encoder_layer_fwd
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml Benchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def input_fn(shape, dtype, device):
    """Generate inputs for transformer encoder layer benchmark.

    Args:
        shape: Tuple specifying tensor size (used as reference for dimensions)
        dtype: Data type for tensors
        device: Device to create tensors on
    """
    batch_size = 8
    seq_len = 128
    embed_dim = 512
    num_heads = 8
    src = torch.randn(batch_size, seq_len, embed_dim, dtype=dtype, device=device)
    qkv_weight = torch.randn(3 * embed_dim, embed_dim, dtype=dtype, device=device)
    qkv_bias = torch.randn(3 * embed_dim, dtype=dtype, device=device)
    proj_weight = torch.randn(embed_dim, embed_dim, dtype=dtype, device=device)
    proj_bias = torch.randn(embed_dim, dtype=dtype, device=device)
    norm_weight_1 = torch.randn(embed_dim, dtype=dtype, device=device)
    norm_bias_1 = torch.randn(embed_dim, dtype=dtype, device=device)
    norm_weight_2 = torch.randn(embed_dim, dtype=dtype, device=device)
    norm_bias_2 = torch.randn(embed_dim, dtype=dtype, device=device)
    ffn_weight_1 = torch.randn(4 * embed_dim, embed_dim, dtype=dtype, device=device)
    ffn_bias_1 = torch.randn(4 * embed_dim, dtype=dtype, device=device)
    ffn_weight_2 = torch.randn(embed_dim, 4 * embed_dim, dtype=dtype, device=device)
    ffn_bias_2 = torch.randn(embed_dim, dtype=dtype, device=device)
    use_gelu = True
    norm_first = False
    eps = 1e-05
    yield (
        src,
        embed_dim,
        num_heads,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        use_gelu,
        norm_first,
        eps,
        norm_weight_1,
        norm_bias_1,
        norm_weight_2,
        norm_bias_2,
        ffn_weight_1,
        ffn_bias_1,
        ffn_weight_2,
        ffn_bias_2,
    )


def _resolve_native():
    return torch._transformer_encoder_layer_fwd


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = input_fn

globals().update(
    export_contract("_transformer_encoder_layer_fwd", DTYPES, _resolve_native, _inputs)
)
