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


"""linalg_solve_ex: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_linalg_solve_ex.py:64 (test_linalg_solve_ex)
Native expression: torch.linalg.solve_ex
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

SOLVE_EX_DTYPES = [torch.float32]


def _solve_ex_input_fn(batch, n, k, dtype, device):
    """Create well-conditioned batched A and random B."""
    A = torch.randn(batch, n, n, dtype=dtype, device=device)
    eye = torch.eye(n, dtype=dtype, device=device).unsqueeze(0).expand(batch, n, n)
    A = A @ A.mT + eye * n
    B = torch.randn(batch, n, k, dtype=dtype, device=device)
    return (A, B)


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        batch, n, k = shape
        A, B = _solve_ex_input_fn(batch, n, k, dtype, device)
        yield (A, B)


def _resolve_native():
    return torch.linalg.solve_ex


DTYPES = SOLVE_EX_DTYPES

_inputs = _input_iter

globals().update(export_contract("linalg_solve_ex", DTYPES, _resolve_native, _inputs))
