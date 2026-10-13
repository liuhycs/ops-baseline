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


"""linalg_cholesky: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_linalg_cholesky.py:56 (test_linalg_cholesky)
Native expression: torch.ops.aten.linalg_cholesky
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

fp64_is_supported = True


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        n = shape[-1]
        B = torch.randn(shape, dtype=dtype, device=device)
        A = B @ B.transpose(-2, -1) + torch.eye(n, dtype=dtype, device=device) * 0.1
        yield (A,)


def _resolve_native():
    return torch.ops.aten.linalg_cholesky


DTYPES = [torch.float32] + ([torch.float64] if fp64_is_supported else [])

_inputs = _input_iter

globals().update(export_contract("linalg_cholesky", DTYPES, _resolve_native, _inputs))
