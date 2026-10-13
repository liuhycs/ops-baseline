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


"""cudnn_convolution: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_cudnn_convolution.py:70 (test_cudnn_convolution)
Native expression: torch.ops.aten.cudnn_convolution.default
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: upstream fixed get_input_iter loop; first tensor shape identifies geometry.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def cudnn_convolution_input_fn(shape, dtype, device):
    (
        batch,
        input_c,
        input_h,
        input_w,
        out_c,
        kernel_h,
        kernel_w,
        stride,
        padding,
        groups,
    ) = shape
    input_shape = (batch, input_c, input_h, input_w)
    weight_shape = (out_c, input_c // groups, kernel_h, kernel_w)
    inp = generate_tensor_input(input_shape, dtype, device)
    weight = generate_tensor_input(weight_shape, dtype, device)
    yield (
        inp,
        weight,
        [padding, padding],
        [stride, stride],
        [1, 1],
        groups,
        False,
        False,
        False,
    )


def _input_iter(shape, dtype, device):
    shapes = [
        (32, 64, 128, 128, 32, 3, 3, 1, 2, 1),
        (32, 64, 210, 210, 16, 5, 5, 2, 1, 1),
        (16, 32, 12, 12, 24, 3, 3, 2, 1, 1),
        (16, 32, 24, 24, 24, 3, 3, 2, 2, 2),
        (16, 32, 24, 24, 24, 3, 3, 1, 2, 2),
    ]
    for shape in shapes:
        yield from cudnn_convolution_input_fn(shape, dtype, device)


def _resolve_native():
    return torch.ops.aten.cudnn_convolution.default


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(export_contract("cudnn_convolution", DTYPES, _resolve_native, _inputs))
