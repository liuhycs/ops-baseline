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


"""mixed_dtypes_linear: PyTorch baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_mixed_dtypes_linear.py:128 (test_mixed_dtypes_linear)
Native expression: mixed_dtypes_linear_torch
Constructor state was explicitly lowered; source MIXED_DTYPES_LINEAR_SHAPES and FP16/BF16 dtypes.
Core source parameters and filters are preserved.
"""

import torch

from _kernelgen import export_contract

FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]


def _input_iter(shape, dtype, device):
    yield from mixed_dtypes_linear_input_fn(shape, dtype, device)


FP16_BF16_DTYPES = [d for d in FLOAT_DTYPES if d != torch.float32]


def mixed_dtypes_linear_input_fn(shape, dtype, device):
    """Yield tuples of (input, weight_uint8, scale, bias, activation)."""
    M, K, N, mode, has_bias, activation = shape
    inp = torch.randn((M, K), dtype=dtype, device=device)
    ncols = N if mode == "int8" else N // 2
    weight = torch.randint(0, 256, (K, ncols), dtype=torch.uint8, device=device)
    scale = torch.randn((N,), dtype=dtype, device=device)
    bias = torch.randn((N,), dtype=dtype, device=device) if has_bias else None
    yield (inp, weight, scale, bias, activation)


def mixed_dtypes_linear_torch(inp, weight, scale, bias, activation):
    """Torch baseline: dequantize uint8 weights and compute matmul.

    Matches kernel order of operations (dtype-split scale placement).
    """
    K, ncols = weight.shape
    N = scale.shape[0]
    if ncols == N:
        w = weight.to(torch.int32) - 128
    else:
        lo = (weight.to(torch.int32) & 15) - 8
        hi = (weight.to(torch.int32) >> 4 & 15) - 8
        w = torch.stack([lo, hi], dim=-1).reshape(K, N)
    if inp.dtype == torch.float16:
        wf = w.to(inp.dtype)
        acc = torch.matmul(inp.to(torch.float32), wf.to(torch.float32))
        acc = acc * scale.to(torch.float32)[None, :]
    else:
        wdq = w.to(torch.float32) * scale.to(torch.float32)[None, :]
        wdq = wdq.to(inp.dtype)
        acc = torch.matmul(inp.to(torch.float32), wdq.to(torch.float32))
    if bias is not None:
        acc = acc + bias.to(torch.float32)[None, :]
    if activation == "relu":
        acc = torch.relu(acc)
    elif activation == "silu":
        acc = acc * torch.sigmoid(acc)
    return acc.to(inp.dtype)


DTYPES = FP16_BF16_DTYPES


def _resolve_native():
    return mixed_dtypes_linear_torch


_inputs = _input_iter

globals().update(
    export_contract("mixed_dtypes_linear", DTYPES, _resolve_native, _inputs)
)
