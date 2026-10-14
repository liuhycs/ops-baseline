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


"""logspace: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_logspace.py:45 (test_logspace)
Native expression: torch.logspace
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml Benchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

import math


def _input_fn(shape, dtype, device):
    base = 1.05
    limit = math.log2(torch.finfo(dtype).max - 1) / math.log2(base)
    end = int(limit)
    yield (
        {
            "start": 0,
            "end": end,
            "steps": math.prod(shape),
            "base": base,
            "dtype": dtype,
            "device": device,
        },
    )


def _resolve_native():
    return torch.logspace


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_fn

globals().update(export_contract("logspace", DTYPES, _resolve_native, _inputs))
