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


"""kaiser_window_beta: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_kaiser_window.py:55 (test_kaiser_window_beta)
Native expression: torch.kaiser_window
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml Benchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

import math


def _input_fn(shape, dtype, device):
    yield (
        {
            "window_length": math.prod(shape),
            "periodic": True,
            "beta": 12.0,
            "dtype": dtype,
            "device": device,
        },
    )


def _resolve_native():
    return torch.kaiser_window


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_fn

globals().update(
    export_contract("kaiser_window_beta", DTYPES, _resolve_native, _inputs)
)
