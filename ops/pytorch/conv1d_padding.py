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


"""conv1d_padding: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_conv1d.py:110 (test_conv1d_padding)
Native expression: torch.nn.functional.conv1d
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: upstream fixed get_input_iter loop; first tensor shape identifies geometry.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def conv1d_input_fn(shape, dtype, device):
    batch, input_c, input_l, out_c, kernel, stride, padding, groups = shape
    input_shape = (batch, input_c, input_l)
    weight_shape = (out_c, input_c // groups, kernel)
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


def _input_iter(shape, dtype, device):
    shapes = [
        (32, 64, 512, 64, 3, 1, 0, 1),
        (32, 64, 512, 64, 3, 1, 1, 1),
        (32, 64, 512, 64, 3, 1, 2, 1),
        (64, 48, 1024, 128, 5, 2, 0, 1),
        (64, 48, 1024, 128, 5, 2, 2, 1),
        (64, 48, 1024, 128, 5, 2, 4, 1),
        (16, 24, 2048, 96, 7, 1, 0, 2),
        (16, 24, 2048, 96, 7, 1, 3, 2),
        (16, 24, 2048, 96, 7, 1, 6, 2),
        (32, 64, 512, 64, 3, 1, "valid", 1),
        (64, 48, 1024, 128, 5, 2, "valid", 1),
        (16, 24, 2048, 96, 7, 1, "same", 2),
        (8, 8, 8192, 16, 11, 1, "same", 1),
    ]
    for shape in shapes:
        yield from conv1d_input_fn(shape, dtype, device)


def _resolve_native():
    return torch.nn.functional.conv1d


DTYPES = [torch.float16, torch.float32]

_inputs = _input_iter

globals().update(export_contract("conv1d_padding", DTYPES, _resolve_native, _inputs))
