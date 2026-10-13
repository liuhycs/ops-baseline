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


"""linalg_matrix_exp: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_linalg_matrix_exp.py:82 (test_linalg_matrix_exp)
Native expression: _torch_matrix_exp
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

import math


def _small_ops_matrix_exp(A):
    n = A.shape[-1]
    if n == 0:
        return A.clone()
    batch_shape = A.shape[:-2]
    B = math.prod(batch_shape) if batch_shape else 1
    A_flat = A.reshape(B, n, n)
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    out = torch.empty_like(A_flat)
    for i in range(B):
        m = A_flat[i]
        norm = m.abs().sum(-2).max().item()
        s = math.ceil(math.log2(norm)) if norm > 1.0 else 0
        ms = m * 2.0 ** (-s)
        r = eye.clone()
        term = eye.clone()
        for k in range(1, 31):
            term = term @ ms / k
            r = r + term
        for _ in range(s):
            r = r @ r
        out[i] = r
    return out.reshape(batch_shape + (n, n) if batch_shape else (n, n))


def _torch_matrix_exp(A):
    if A.device.type == "npu":
        return _small_ops_matrix_exp(A)
    return torch.linalg.matrix_exp(A)


MATRIX_EXP_DTYPES = [torch.float32] + ([torch.float64] if True else [])


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        A = torch.randn(shape, dtype=dtype, device=device)
        yield (A,)


def _resolve_native():
    return _torch_matrix_exp


DTYPES = MATRIX_EXP_DTYPES

_inputs = _input_iter

globals().update(export_contract("linalg_matrix_exp", DTYPES, _resolve_native, _inputs))
