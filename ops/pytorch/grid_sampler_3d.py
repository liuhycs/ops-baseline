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


"""grid_sampler_3d: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_grid_sampler_3d.py:32 (test_grid_sampler_3d)
Native expression: torch.ops.aten.grid_sampler_3d
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        N, C, ID, IH, IW = shape
        OD, OH, OW = (ID // 2, IH // 2, IW // 2)
        inp = torch.randn(shape, dtype=dtype, device=device)
        grid = torch.randn(N, OD, OH, OW, 3, dtype=dtype, device=device)
        grid = grid * 1.5
        yield (inp, grid, 0, 0, False)


def _resolve_native():
    return torch.ops.aten.grid_sampler_3d


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(export_contract("grid_sampler_3d", DTYPES, _resolve_native, _inputs))
