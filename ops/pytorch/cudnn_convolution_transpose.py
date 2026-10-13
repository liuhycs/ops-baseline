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


"""cudnn_convolution_transpose: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_cudnn_convolution_transpose.py:84 (test_cudnn_convolution_transpose)
Native expression: torch.cudnn_convolution_transpose
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: upstream fixed get_input_iter loop; first tensor shape identifies geometry.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def cudnn_conv_transpose_input_fn(shape, dtype, device):
    N, C_in, H, W, C_out, kH, kW, stride, padding, groups = shape
    input_shape = (N, C_in, H, W)
    weight_shape = (C_in, C_out // groups, kH, kW)
    inp = generate_tensor_input(input_shape, dtype, device)
    weight = generate_tensor_input(weight_shape, dtype, device)
    output_padding_h = (stride - 1) * H - 2 * padding + (kW - 1)
    output_padding_w = (stride - 1) * W - 2 * padding + (kH - 1)
    output_padding = (output_padding_h % stride, output_padding_w % stride)
    yield (
        {
            "input": inp,
            "weight": weight,
            "stride": (stride, stride),
            "padding": (padding, padding),
            "output_padding": output_padding,
            "groups": groups,
            "dilation": (1, 1),
            "benchmark": False,
            "deterministic": False,
            "allow_tf32": False,
        },
    )


def _input_iter(shape, dtype, device):
    shapes = [
        (1, 8, 32, 32, 8, 3, 3, 1, 0, 1),
        (2, 16, 24, 24, 16, 3, 3, 1, 1, 1),
        (4, 32, 16, 16, 32, 3, 3, 2, 1, 1),
        (8, 64, 8, 8, 64, 3, 3, 1, 0, 1),
    ]
    for shape in shapes:
        yield from cudnn_conv_transpose_input_fn(shape, dtype, device)


def _resolve_native():
    return torch.cudnn_convolution_transpose


DTYPES = [torch.float16, torch.float32]

_inputs = _input_iter

globals().update(
    export_contract("cudnn_convolution_transpose", DTYPES, _resolve_native, _inputs)
)
