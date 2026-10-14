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


"""pad: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_pad.py:42 (test_pad)
Native expression: torch.nn.functional.pad
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml Benchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

import random


def _input_fn(shape, dtype, device):
    input = torch.randn(shape, device=device, dtype=dtype)
    rank = input.ndim
    pad_params = [random.randint(0, 10) for _ in range(rank * 2)]
    pad_value = float(torch.randint(0, 1024, [1]))
    yield (input, {"pad": pad_params, "mode": "constant", "value": pad_value})


def _resolve_native():
    return torch.nn.functional.pad


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_fn

globals().update(export_contract("pad", DTYPES, _resolve_native, _inputs))
