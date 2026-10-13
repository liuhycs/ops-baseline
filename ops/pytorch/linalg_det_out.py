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


"""linalg_det_out: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_linalg_det.py:110 (test_linalg_det_out)
Native expression: _torch_det_out
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

import math


def _small_ops_det(A):
    n = A.shape[-1]
    batch_shape = A.shape[:-2]
    B = math.prod(batch_shape) if batch_shape else 1
    LU = A.clone().reshape(B, n, n)
    sign = torch.ones(B, dtype=A.dtype, device=A.device)
    bidx = torch.arange(B, device=A.device)
    for k in range(n):
        p = LU[:, k:, k].abs().argmax(dim=-1) + k
        swap = p != k
        sign = torch.where(swap, -sign, sign)
        row_k = LU[:, k, :].clone()
        row_p = LU[bidx, p, :].clone()
        LU[:, k, :] = row_p
        LU[bidx[swap], p[swap], :] = row_k[swap]
        pivot = LU[:, k, k]
        safe_pivot = torch.where(pivot == 0, torch.ones_like(pivot), pivot)
        col = LU[:, k + 1 :, k]
        mult = torch.where(
            (pivot == 0).unsqueeze(-1),
            torch.zeros_like(col),
            col / safe_pivot.unsqueeze(-1),
        )
        LU[:, k + 1 :, k] = mult
        LU[:, k + 1 :, k + 1 :] -= mult.unsqueeze(-1) * LU[:, k : k + 1, k + 1 :]
    det = LU.diagonal(dim1=-2, dim2=-1).prod(dim=-1) * sign
    return det.reshape(batch_shape)


def _torch_det(A):
    if A.device.type == "npu":
        return _small_ops_det(A)
    return torch.linalg.det(A)


def _torch_det_out(A, *, out):
    if A.device.type == "npu":
        out.copy_(_torch_det(A))
        return out
    return torch.linalg.det(A, out=out)


DET_DTYPES = [torch.float32] + ([torch.float64] if True else [])


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        A = torch.randn(shape, dtype=dtype, device=device)
        out = torch.empty(shape[:-2], dtype=dtype, device=device)
        yield (A, {"out": out})


def _resolve_native():
    return _torch_det_out


DTYPES = DET_DTYPES

_inputs = _input_iter

globals().update(export_contract("linalg_det_out", DTYPES, _resolve_native, _inputs))
