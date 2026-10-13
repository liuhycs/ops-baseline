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


"""linalg_ldl_factor: PyTorch baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_linalg_ldl_factor.py:53 (test_linalg_ldl_factor)
Native expression: torch.linalg.ldl_factor
Constructor state was explicitly lowered; explicit source set_shapes.
Core source parameters and filters are preserved.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        n = shape[0]
        A = torch.randn(shape, dtype=dtype, device=device)
        A = A @ A.transpose(-2, -1) + torch.eye(n, dtype=dtype, device=device) * n
        yield (A,)


DTYPES = [torch.float32, torch.float64]


def _resolve_native():
    return torch.linalg.ldl_factor


_inputs = _input_iter

globals().update(export_contract("linalg_ldl_factor", DTYPES, _resolve_native, _inputs))
