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


"""avg_pool2d: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_avg_pool2d.py:90 (test_avg_pool2d)
Native expression: torch.ops.aten.avg_pool2d
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: upstream fixed get_input_iter loop; first tensor shape identifies geometry.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def avg_pool2d_input_fn(shape, dtype, device):
    inp = generate_tensor_input(shape, dtype, device)
    yield (
        inp,
        {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "ceil_mode": False,
            "count_include_pad": True,
            "divisor_override": None,
        },
    )


def _input_iter(shape, dtype, device):
    shapes_4d = [
        (4, 3, 224, 224),
        (16, 64, 56, 56),
        (32, 128, 28, 28),
        (64, 256, 14, 14),
        (128, 512, 7, 7),
    ]
    for shape in shapes_4d:
        yield from avg_pool2d_input_fn(shape, dtype, device)


def _resolve_native():
    return torch.ops.aten.avg_pool2d


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(export_contract("avg_pool2d", DTYPES, _resolve_native, _inputs))
