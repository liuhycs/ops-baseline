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


"""conv_tbc_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_conv_tbc_backward.py:35 (test_conv_tbc_backward)
Native expression: torch.ops.aten.conv_tbc_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: upstream fixed get_input_iter loop; first tensor shape identifies geometry.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def conv_tbc_backward_input_fn(shape, dtype, device):
    ilen, batch, in_c, out_c, kw, pad = shape
    olen = ilen - kw + 1 + 2 * pad
    inp = torch.randn((ilen, batch, in_c), dtype=dtype, device=device)
    weight = torch.randn((kw, in_c, out_c), dtype=dtype, device=device)
    bias = torch.randn((out_c,), dtype=dtype, device=device)
    grad_output = torch.randn((olen, batch, out_c), dtype=dtype, device=device)
    yield (grad_output, inp, weight, bias, pad)


def _input_iter(shape, dtype, device):
    shapes = [
        (64, 16, 64, 64, 3, 1),
        (128, 32, 128, 128, 3, 1),
        (256, 16, 256, 256, 5, 2),
        (512, 8, 128, 128, 3, 1),
        (1024, 4, 64, 64, 7, 3),
    ]
    for shape in shapes:
        yield from conv_tbc_backward_input_fn(shape, dtype, device)


def _resolve_native():
    return torch.ops.aten.conv_tbc_backward


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(export_contract("conv_tbc_backward", DTYPES, _resolve_native, _inputs))
