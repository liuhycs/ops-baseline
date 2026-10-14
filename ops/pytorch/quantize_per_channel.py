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


"""quantize_per_channel: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_quantize_per_channel.py:59 (test_quantize_per_channel)
Native expression: torch.quantize_per_channel
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

QUANT_DTYPES = [torch.quint8, torch.qint8, torch.qint32]


def _quantize_input_fn(shape, dtype, device):
    """Yield (input, scales, zero_points, axis, dtype) for torch.quantize_per_channel."""
    axis = len(shape) - 1
    n_channels = shape[axis]
    inp = torch.randn(shape, dtype=torch.float32, device=device) * 10.0
    scales = torch.rand(n_channels, device=device) * 0.5 + 0.01
    zero_points = torch.randint(0, 50, (n_channels,), device=device, dtype=torch.int32)
    yield (inp, scales, zero_points, axis, dtype)


def _resolve_native():
    return torch.quantize_per_channel


DTYPES = QUANT_DTYPES

_inputs = _quantize_input_fn

globals().update(
    export_contract("quantize_per_channel", DTYPES, _resolve_native, _inputs)
)
