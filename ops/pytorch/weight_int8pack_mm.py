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


"""weight_int8pack_mm: PyTorch baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_weight_int8pack_mm.py:89 (test_weight_int8pack_mm)
Native expression: weight_int8pack_mm_torch
Constructor state was explicitly lowered; explicit source set_shapes.
Core source parameters and filters are preserved.
"""

import torch

from _kernelgen import export_contract

FP16_BF16_DTYPES = [torch.float16, torch.bfloat16]


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        yield from weight_int8pack_mm_input_fn(shape, dtype, device)


def weight_int8pack_mm_input_fn(shape, dtype, device):
    """Yield tuples of (A, B_int8, scales) for the given shape and dtype."""
    M, N, K = shape
    A = torch.randn((M, K), dtype=dtype, device=device)
    B = torch.randint(-128, 127, (N, K), dtype=torch.int8, device=device)
    scales = torch.randn((N,), dtype=dtype, device=device)
    yield (A, B, scales)


def weight_int8pack_mm_torch(A, B, scales):
    """Torch baseline: dequantize int8 weights and compute matmul with scaling."""
    B_fp = B.to(A.dtype)
    result = torch.matmul(A, B_fp.T)
    result = result * scales.unsqueeze(0)
    return result


DTYPES = FP16_BF16_DTYPES


def _resolve_native():
    return weight_int8pack_mm_torch


_inputs = _input_iter

globals().update(
    export_contract("weight_int8pack_mm", DTYPES, _resolve_native, _inputs)
)
