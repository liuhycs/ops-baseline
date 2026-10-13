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


"""im2col: PyTorch baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_im2col.py:48 (test_im2col)
Native expression: torch.ops.aten.im2col
Constructor state was explicitly lowered; explicit source set_shapes.
Core source parameters and filters are preserved.
"""

import torch

from _kernelgen import export_contract

IM2COL_CONFIGS = [
    ((3, 3), (1, 1), (1, 1), (1, 1)),
    ((3, 3), (1, 1), (0, 0), (2, 2)),
    ((5, 4), (2, 2), (2, 1), (1, 2)),
    ((1, 1), (1, 1), (0, 0), (1, 1)),
]


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        yield from _source_im2col_input_fn(shape, dtype, device)


def _source_im2col_input_fn(shape, dtype, device):
    for kernel_size, dilation, padding, stride in IM2COL_CONFIGS:
        x = torch.randn(shape, dtype=dtype, device=device)
        yield (x, kernel_size, dilation, padding, stride)


DTYPES = [torch.float16, torch.float32, torch.bfloat16]


def _resolve_native():
    return torch.ops.aten.im2col


_inputs = _input_iter

globals().update(export_contract("im2col", DTYPES, _resolve_native, _inputs))
