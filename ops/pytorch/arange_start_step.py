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


"""arange_start_step: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_arange.py:103 (test_arange_start_step)
Native expression: torch.arange
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml Benchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

import math


def _input_fn_start_step(shape, dtype, device):
    yield (
        {
            "start": 0,
            "end": math.prod(shape),
            "step": 1,
            "device": device,
            "dtype": dtype,
        },
    )


def _resolve_native():
    return torch.arange


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_fn_start_step

globals().update(export_contract("arange_start_step", DTYPES, _resolve_native, _inputs))
