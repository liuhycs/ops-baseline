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


"""conv_transpose1d: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_conv_transpose1d.py:63 (test_conv_transpose1d)
Native expression: torch.nn.functional.conv_transpose1d
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: upstream fixed get_input_iter loop; first tensor shape identifies geometry.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def conv_transpose1d_input_fn(shape, dtype, device):
    batch, input_c, input_l, out_c, kernel, stride, padding, groups = shape
    input_shape = (batch, input_c, input_l)
    weight_shape = (input_c, out_c // groups, kernel)
    inp = generate_tensor_input(input_shape, dtype, device)
    weight = generate_tensor_input(weight_shape, dtype, device)
    yield (inp, weight, None, stride, padding, 0, groups)


def _input_iter(shape, dtype, device):
    shapes = [
        (32, 64, 128, 64, 3, 1, 0, 1),
        (64, 48, 256, 128, 5, 2, 2, 1),
        (16, 24, 512, 96, 7, 1, 3, 1),
        (8, 16, 1024, 32, 3, 2, 1, 2),
        (4, 8, 2048, 16, 5, 1, 2, 1),
    ]
    for shape in shapes:
        yield from conv_transpose1d_input_fn(shape, dtype, device)


def _resolve_native():
    return torch.nn.functional.conv_transpose1d


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(export_contract("conv_transpose1d", DTYPES, _resolve_native, _inputs))
