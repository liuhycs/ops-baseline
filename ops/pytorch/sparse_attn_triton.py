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


"""sparse_attn_triton: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_sparse_attention.py:89 (test_sparse_attn_triton)
Native expression: torch_sparse_attention
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

import math


def torch_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale):
    batch, seq_len, heads, dim = q.shape
    topk = topk_idxs.shape[-1]
    kv_expanded = kv[:, None, :, :].expand(batch, seq_len, -1, dim)
    idx_expanded = topk_idxs[:, :, :, None].expand(batch, seq_len, topk, dim).long()
    gathered_kv = torch.gather(kv_expanded, 2, idx_expanded)
    scores = (
        torch.einsum("bmhd,bmtd->bmht", q.float(), gathered_kv.float()) * softmax_scale
    )
    sink = attn_sink[None, None, :, None].expand(batch, seq_len, heads, 1)
    attn = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)
    out = torch.einsum("bmht,bmtd->bmhd", attn[:, :, :, :-1], gathered_kv.float())
    return out.to(q.dtype)


def _input_iter(shape, dtype, device):
    for seed, (batch, seq_len, kv_len, topk, heads, dim) in enumerate([shape]):
        torch.manual_seed(2026 + seed)
        q = torch.randn((batch, seq_len, heads, dim), dtype=dtype, device=device)
        kv = torch.randn((batch, kv_len, dim), dtype=dtype, device=device)
        attn_sink = torch.zeros((heads,), dtype=torch.float32, device=device)
        topk_idxs = torch.randint(
            0, kv_len, (batch, seq_len, topk), dtype=torch.int32, device=device
        )
        yield (q, kv, attn_sink, topk_idxs, 1.0 / math.sqrt(dim))


def _resolve_native():
    return torch_sparse_attention


DTYPES = [torch.bfloat16]

_inputs = _input_iter

globals().update(
    export_contract("sparse_attn_triton", DTYPES, _resolve_native, _inputs)
)
