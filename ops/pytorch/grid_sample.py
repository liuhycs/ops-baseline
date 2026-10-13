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


"""grid_sample: PyTorch baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_grid_sample.py:208 (test_grid_sample)
Native expression: torch.nn.functional.grid_sample
Constructor state was explicitly lowered; source constructor calls set_more_shapes.
Core source parameters and filters are preserved.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def _input_iter(shape, dtype, device):
    """Generate input tensors with various grid_sample parameters."""
    for shape in [shape]:
        inp = generate_tensor_input(shape, dtype, device)
        is_5d = len(shape) == 5
        if is_5d:
            N, C, D_in, H_in, W_in = shape
            D_out, H_out, W_out = (D_in, H_in, W_in)
            grid = torch.randn(N, D_out, H_out, W_out, 3, dtype=dtype, device=device)
            grid = torch.clamp(grid, -0.9, 0.9)
        else:
            N, C, H_in, W_in = shape
            H_out, W_out = (H_in, W_in)
            grid = torch.randn(N, H_out, W_out, 2, dtype=dtype, device=device)
            grid = torch.clamp(grid, -0.9, 0.9)
        yield (
            inp,
            grid,
            {"mode": "nearest", "padding_mode": "zeros", "align_corners": False},
        )
        yield (
            inp,
            grid,
            {"mode": "bilinear", "padding_mode": "zeros", "align_corners": False},
        )
        yield (
            inp,
            grid,
            {"mode": "bilinear", "padding_mode": "border", "align_corners": True},
        )
        if not is_5d:
            yield (
                inp,
                grid,
                {"mode": "bicubic", "padding_mode": "zeros", "align_corners": False},
            )


DTYPES = [torch.float16, torch.float32, torch.bfloat16]


def _resolve_native():
    return torch.nn.functional.grid_sample


_inputs = _input_iter

globals().update(export_contract("grid_sample", DTYPES, _resolve_native, _inputs))
