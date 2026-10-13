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


"""conv3d: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_conv3d.py:81 (test_conv3d)
Native expression: torch.nn.functional.conv3d
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml Conv3DBenchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def conv3d_input_fn(shape, dtype, device):
    (
        batch,
        input_c,
        input_d,
        input_h,
        input_w,
        out_c,
        kernel_d,
        kernel_h,
        kernel_w,
        stride,
        padding,
        groups,
    ) = shape
    input_shape = (batch, input_c, input_d, input_h, input_w)
    weight_shape = (out_c, input_c // groups, kernel_d, kernel_h, kernel_w)
    input = torch.randn(size=input_shape, device=device, dtype=dtype)
    weight = torch.randn(size=weight_shape, device=device, dtype=dtype)
    yield (
        {
            "input": input,
            "weight": weight,
            "bias": None,
            "groups": groups,
            "stride": stride,
            "padding": padding,
        },
    )


def _resolve_native():
    return torch.nn.functional.conv3d


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = conv3d_input_fn

globals().update(export_contract("conv3d", DTYPES, _resolve_native, _inputs))
