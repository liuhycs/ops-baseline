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


"""affine_grid_generator: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_affine_grid_generator.py:47 (test_affine_grid_generator)
Native expression: torch.affine_grid_generator
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        N, H, W = shape
        theta = torch.randn((N, 2, 3), dtype=dtype, device=device)
        size = [N, 3, H, W]
        align_corners = False
        yield (theta, size, align_corners)


def _resolve_native():
    return torch.affine_grid_generator


DTYPES = [torch.float32]

_inputs = _input_iter

globals().update(
    export_contract("affine_grid_generator", DTYPES, _resolve_native, _inputs)
)
