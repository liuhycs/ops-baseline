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


"""convolution_double_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_convolution_double_backward.py:169 (test_convolution_double_backward)
Native expression: torch.ops.aten._convolution_double_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        transposed = shape[-2]
        if len(shape) == 13:
            (
                n,
                cin,
                h,
                w,
                cout,
                kh,
                kw,
                stride,
                padding,
                dilation,
                groups,
                transposed,
                output_padding,
            ) = shape
            x = torch.randn((n, cin, h, w), dtype=dtype, device=device)
            if not transposed:
                weight_shape = (cout, cin // groups, kh, kw)
            else:
                weight_shape = (cin, cout // groups, kh, kw)
            weight = torch.randn(weight_shape, dtype=dtype, device=device)
            bias = torch.randn(cout, dtype=dtype, device=device)
            if not transposed:
                y = torch.nn.functional.conv2d(
                    x,
                    weight,
                    bias=bias,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
            else:
                y = torch.nn.functional.conv_transpose2d(
                    x,
                    weight,
                    bias=bias,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    output_padding=output_padding,
                    groups=groups,
                )
            spatial_ndim = 2
            opad = list(output_padding)
        else:
            (
                n,
                cin,
                l,
                cout,
                kl,
                stride,
                padding,
                dilation,
                groups,
                transposed,
                output_padding,
            ) = shape
            x = torch.randn((n, cin, l), dtype=dtype, device=device)
            if not transposed:
                weight_shape = (cout, cin // groups, kl)
            else:
                weight_shape = (cin, cout // groups, kl)
            weight = torch.randn(weight_shape, dtype=dtype, device=device)
            bias = torch.randn(cout, dtype=dtype, device=device)
            if not transposed:
                y = torch.nn.functional.conv1d(
                    x,
                    weight,
                    bias=bias,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
            else:
                y = torch.nn.functional.conv_transpose1d(
                    x,
                    weight,
                    bias=bias,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    output_padding=output_padding,
                    groups=groups,
                )
            spatial_ndim = 1
            opad = [output_padding] * spatial_ndim
        gO = torch.randn_like(y)
        ggI = torch.randn_like(x)
        ggW = torch.randn_like(weight)
        ggb = torch.randn_like(bias)
        output_mask = (True, True, True)
        yield (
            ggI,
            ggW,
            ggb,
            gO,
            weight,
            x,
            [stride] * spatial_ndim,
            [padding] * spatial_ndim,
            [dilation] * spatial_ndim,
            transposed,
            opad,
            groups,
            output_mask,
        )


def _resolve_native():
    return torch.ops.aten._convolution_double_backward


DTYPES = [torch.float32, torch.float16]

_inputs = _input_iter

globals().update(
    export_contract("convolution_double_backward", DTYPES, _resolve_native, _inputs)
)
