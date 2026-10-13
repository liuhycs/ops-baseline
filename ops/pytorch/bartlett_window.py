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


"""bartlett_window: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_bartlett_window.py:39 (test_bartlett_window)
Native expression: torch.bartlett_window
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml Benchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

BARTLETT_DTYPES = [torch.float16, torch.float32, torch.bfloat16] + [torch.float64]


def _input_fn(shape, dtype, device):
    window_length = shape[0] if isinstance(shape, (list, tuple)) else shape
    yield (
        {
            "window_length": window_length,
            "periodic": True,
            "dtype": dtype,
            "device": device,
        },
    )


def _resolve_native():
    return torch.bartlett_window


DTYPES = BARTLETT_DTYPES

_inputs = _input_fn

globals().update(export_contract("bartlett_window", DTYPES, _resolve_native, _inputs))
