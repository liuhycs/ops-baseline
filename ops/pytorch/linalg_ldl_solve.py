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


"""linalg_ldl_solve: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_linalg_ldl_solve.py:48 (test_linalg_ldl_solve)
Native expression: torch.linalg.ldl_solve
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

LDL_SOLVE_DTYPES = [torch.float32, torch.float64, torch.complex64]
LDL_SOLVE_DTYPES.extend([torch.complex128])  # Pinned NVIDIA capabilities.


def _make_ldl_inputs(n, k, dtype, device):
    A = torch.randn(n, n, dtype=dtype, device=device)
    A = A @ A.mT + torch.eye(n, dtype=dtype, device=device) * n
    LD, pivots, info = torch.linalg.ldl_factor_ex(A)
    B = torch.randn(n, k, dtype=dtype, device=device)
    return (LD, pivots, B)


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        n, k = shape
        LD, pivots, B = _make_ldl_inputs(n, k, dtype, device)
        yield (LD, pivots, B)


def _resolve_native():
    return torch.linalg.ldl_solve


DTYPES = LDL_SOLVE_DTYPES

_inputs = _input_iter

globals().update(export_contract("linalg_ldl_solve", DTYPES, _resolve_native, _inputs))
