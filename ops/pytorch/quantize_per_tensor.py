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


"""quantize_per_tensor: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_quantize_per_tensor.py:42 (test_quantize_per_tensor)
Native expression: torch.quantize_per_tensor
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml GenericBenchmarkExcluse1D or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

SCALE = 0.1

ZERO_POINT = 10

Q_DTYPE = torch.quint8


def _input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=torch.float32, device=device)
    yield (inp, SCALE, ZERO_POINT, Q_DTYPE)


def _resolve_native():
    return torch.quantize_per_tensor


DTYPES = [torch.float32]

_inputs = _input_fn

globals().update(
    export_contract("quantize_per_tensor", DTYPES, _resolve_native, _inputs)
)
