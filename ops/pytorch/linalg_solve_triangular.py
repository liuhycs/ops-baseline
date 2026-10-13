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


"""linalg_solve_triangular: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_linalg_solve_triangular.py:65 (test_linalg_solve_triangular)
Native expression: torch.linalg.solve_triangular
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

SOLVE_TRI_DTYPES = [torch.float32]


def _make_triangular_input(n, k, dtype, device, upper):
    """Generate a well-conditioned triangular matrix: A = I + 0.1 * tri(randn)"""
    A = torch.randn(n, n, dtype=dtype, device=device)
    if upper:
        A = A.triu(diagonal=1)
    else:
        A = A.tril(diagonal=-1)
    A.mul_(0.1)
    A.add_(torch.eye(n, dtype=dtype, device=device))
    B = torch.randn(n, k, dtype=dtype, device=device)
    return (A, B)


def _input_iter(shape, dtype, device):
    for n, k in [shape]:
        for upper in (False, True):
            A, B = _make_triangular_input(n, k, dtype, device, upper)
            yield (A, B, {"upper": upper})


def _resolve_native():
    return torch.linalg.solve_triangular


DTYPES = SOLVE_TRI_DTYPES

_inputs = _input_iter

globals().update(
    export_contract("linalg_solve_triangular", DTYPES, _resolve_native, _inputs)
)
