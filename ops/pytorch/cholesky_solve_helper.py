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


"""cholesky_solve_helper: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_cholesky_solve_helper.py:42 (test_cholesky_solve_helper)
Native expression: torch.ops.aten._cholesky_solve_helper
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    for n, nrhs in [shape]:
        matrix = torch.randn((n, n), dtype=dtype, device=device)
        matrix = matrix @ matrix.mT + n * torch.eye(n, dtype=dtype, device=device)
        factor = torch.linalg.cholesky(matrix)
        rhs = torch.randn((n, nrhs), dtype=dtype, device=device)
        yield (rhs, factor, False)


def _resolve_native():
    return torch.ops.aten._cholesky_solve_helper


DTYPES = [torch.float32]

_inputs = _input_iter

globals().update(
    export_contract("cholesky_solve_helper", DTYPES, _resolve_native, _inputs)
)
