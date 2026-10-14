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


"""lu_unpack: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_lu_unpack.py:21 (test_lu_unpack)
Native expression: torch.ops.aten.lu_unpack
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml GenericBenchmark2DOnly or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def lu_unpack_input_fn(shape, dtype, device):
    if len(shape) != 2:
        return
    m, n = shape
    A = torch.randn(m, n, dtype=dtype, device=device)
    LU, pivots = torch.linalg.lu_factor(A)
    yield (LU, pivots)


def _resolve_native():
    return torch.ops.aten.lu_unpack


DTYPES = [torch.float32, torch.float64]

_inputs = lu_unpack_input_fn

globals().update(export_contract("lu_unpack", DTYPES, _resolve_native, _inputs))
