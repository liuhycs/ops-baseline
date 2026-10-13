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


"""grid_sampler_3d_backward: PyTorch baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_grid_sampler_3d_backward.py:53 (test_grid_sampler_3d_backward)
Native expression: torch.ops.aten.grid_sampler_3d_backward.default
Constructor state was explicitly lowered; explicit source set_shapes.
Core source parameters and filters are preserved.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    for config in [shape]:
        N, C, iD, iH, iW, oD, oH, oW = config
        grad_output = torch.randn(N, C, oD, oH, oW, device=device, dtype=dtype)
        input_tensor = torch.randn(N, C, iD, iH, iW, device=device, dtype=dtype)
        grid = torch.rand(N, oD, oH, oW, 3, device=device, dtype=dtype) * 2 - 1
        yield (grad_output, input_tensor, grid, 0, 0, True, [True, True])


DTYPES = [torch.float32]


def _resolve_native():
    return torch.ops.aten.grid_sampler_3d_backward.default


_inputs = _input_iter

globals().update(
    export_contract("grid_sampler_3d_backward", DTYPES, _resolve_native, _inputs)
)
