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


"""linalg_vecdot: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_linalg_vecdot.py:50 (test_linalg_vecdot_benchmark)
Native expression: torch.linalg.vecdot
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

DTYPES = [torch.float32]
DTYPES.extend([torch.float64])  # Pinned NVIDIA capabilities.


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        x = torch.randn(shape, dtype=dtype, device=device)
        y = torch.randn(shape, dtype=dtype, device=device)
        yield (x, y)


def _resolve_native():
    return torch.linalg.vecdot


DTYPES = DTYPES

_inputs = _input_iter

globals().update(export_contract("linalg_vecdot", DTYPES, _resolve_native, _inputs))
