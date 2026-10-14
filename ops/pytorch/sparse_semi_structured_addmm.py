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


"""sparse_semi_structured_addmm: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_sparse_semi_structured_addmm.py:88 (test_sparse_semi_structured_addmm)
Native expression: _sparse_semi_structured_addmm_ref
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _sparse_semi_structured_addmm_ref(
    input_tensor, mat1, mat1_meta, mat2, alpha=1.0, beta=1.0
):
    """Pure-PyTorch reference implementation.

    The native aten op uses a different input format and a different
    algorithm than the Gems implementation, so the native op cannot serve as
    an equivalent baseline. This reference shares both the input format and
    the algorithm with the Gems op, so it is used as the benchmark baseline.
    """
    M, K4 = mat1_meta.shape
    N = mat2.shape[1]
    mat1_reshaped = mat1.view(M, K4, 4)
    mat2_reshaped = mat2.view(K4, 4, N)
    meta_mask = torch.cat(
        [
            mat1_meta.unsqueeze(2),
            mat1_meta.unsqueeze(2),
            (~mat1_meta).unsqueeze(2),
            (~mat1_meta).unsqueeze(2),
        ],
        dim=2,
    )
    mat1_masked = torch.where(meta_mask, mat1_reshaped, torch.zeros_like(mat1_reshaped))
    result = torch.zeros((M, N), dtype=mat1.dtype, device=mat1.device)
    for pos in range(4):
        a = mat1_masked[:, :, pos]
        b = mat2_reshaped[:, pos, :]
        result += torch.mm(a, b)
    result = alpha * result + beta * input_tensor
    return result


def _input_iter(shape, dtype, device):
    K4 = 32
    for shape in [shape]:
        M, N = shape
        input_tensor = torch.randn(M, N, dtype=dtype, device=device)
        mat1 = torch.randn(M, 4 * K4, dtype=dtype, device=device)
        mat1_meta = torch.randint(0, 2, (M, K4), dtype=torch.bool, device=device)
        mat2 = torch.randn(4 * K4, N, dtype=dtype, device=device)
        yield (input_tensor, mat1, mat1_meta, mat2)


def _resolve_native():
    return _sparse_semi_structured_addmm_ref


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(
    export_contract("sparse_semi_structured_addmm", DTYPES, _resolve_native, _inputs)
)
