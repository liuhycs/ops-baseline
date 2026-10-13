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


"""weight_int4pack_mm_with_scales_and_zeros: PyTorch baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_weight_int4pack_mm_with_scales_and_zeros.py:164 (test_weight_int4pack_mm_with_scales_and_zeros)
Native expression: _naive_dequant_mm
Constructor state was explicitly lowered; explicit source set_shapes.
Core source parameters and filters are preserved.
"""

import torch

from _kernelgen import export_contract

QGROUP_SIZES = (2, 4, 8, 16)


def _naive_dequant_mm(A, mat2, qGroupSize, qScale, qZeros):
    """Baseline: naive dequantize then matmul in PyTorch."""
    M, K = A.shape
    N = mat2.shape[0]
    W_f32 = torch.empty((K, N), dtype=torch.float32, device=A.device)
    for k in range(K):
        byte_idx = k // 2
        group = k // qGroupSize
        bytes_val = mat2[:, byte_idx].to(torch.int32)
        if k % 2 == 0:
            int4_vals = bytes_val & 15
        else:
            int4_vals = bytes_val >> 4 & 15
        scales = qScale[group, :]
        zeros = qZeros[group, :]
        w_k = (int4_vals.to(torch.float32) - zeros) * scales
        W_f32[k, :] = w_k
    return torch.mm(A.to(torch.float32), W_f32).to(A.dtype)


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        yield from _input_fn(shape, device, dtype)


def _input_fn(shape, device, dtype):
    """Yield inputs for the benchmark from a valid random test case."""
    M, K, N = shape
    K2 = K // 2
    for qGroupSize in QGROUP_SIZES:
        if K % qGroupSize != 0:
            continue
        A = torch.randn((M, K), dtype=dtype, device=device) * 0.1
        W_int32 = torch.randint(0, 16, (K, N), dtype=torch.int32, device=device)
        mat2 = torch.empty((N, K2), dtype=torch.uint8, device=device)
        for k in range(K):
            byte_idx = k // 2
            if k % 2 == 0:
                mat2[:, byte_idx] = (W_int32[k, :].to(torch.uint8) & 15).to(torch.uint8)
            else:
                mat2[:, byte_idx] = (
                    mat2[:, byte_idx].to(torch.int32)
                    | (W_int32[k, :].to(torch.uint8) & 15).to(torch.int32) << 4
                ).to(torch.uint8)
        G = K // qGroupSize
        qScale = torch.randn((G, N), dtype=dtype, device=device).abs() + 0.01
        qZeros = torch.randn((G, N), dtype=dtype, device=device) * 2.0
        yield (A, mat2, qGroupSize, qScale, qZeros)


DTYPES = [torch.float16, torch.float32, torch.bfloat16]


def _resolve_native():
    return _naive_dequant_mm


_inputs = _input_iter

globals().update(
    export_contract(
        "weight_int4pack_mm_with_scales_and_zeros", DTYPES, _resolve_native, _inputs
    )
)
