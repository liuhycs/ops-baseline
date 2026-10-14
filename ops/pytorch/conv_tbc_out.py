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


"""conv_tbc_out: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_conv_tbc.py:63 (test_conv_tbc_out)
Native expression: torch.ops.aten.conv_tbc.out
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: upstream fixed get_input_iter loop; first tensor shape identifies geometry.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

CONV_TBC_BENCH_SHAPES = [
    (128, 32, 128, 128, 3, 1),
    (256, 16, 256, 256, 3, 1),
    (512, 8, 128, 256, 5, 2),
    (64, 64, 512, 512, 3, 1),
    (1024, 4, 64, 128, 7, 3),
]


def conv_tbc_out_input_fn(shape, dtype, device):
    T, B, Cin, Cout, kW, pad = shape
    input = torch.randn((T, B, Cin), device=device, dtype=dtype)
    weight = torch.randn((kW, Cin, Cout), device=device, dtype=dtype)
    bias = torch.randn((Cout,), device=device, dtype=dtype)
    Tout = T + 2 * pad - kW + 1
    out = torch.empty((Tout, B, Cout), device=device, dtype=dtype)
    yield (input, weight, bias, pad, {"out": out})


def _input_iter(shape, dtype, device):
    for shape in CONV_TBC_BENCH_SHAPES:
        yield from conv_tbc_out_input_fn(shape, dtype, device)


def _resolve_native():
    return torch.ops.aten.conv_tbc.out


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(export_contract("conv_tbc_out", DTYPES, _resolve_native, _inputs))
