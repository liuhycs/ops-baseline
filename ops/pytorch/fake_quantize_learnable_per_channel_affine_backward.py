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


"""fake_quantize_learnable_per_channel_affine_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_fake_quantize_learnable_per_channel_affine_backward.py:69 (test_fake_quantize_learnable_per_channel_affine_backward)
Native expression: ATEN_OP
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

ATEN_OP = torch.ops.aten._fake_quantize_learnable_per_channel_affine_backward


def _input_iter(shape, dtype, device):
    for shape, axis, quant_min, quant_max, grad_factor in [shape]:
        C = shape[axis]
        scale = torch.rand(C, dtype=dtype, device=device) * 0.4 + 0.05
        zero_point = torch.rand(C, dtype=dtype, device=device) * (quant_max * 0.5)
        sat_factor = float(quant_max) * 0.05
        grad = torch.randn(shape, dtype=dtype, device=device)
        self_t = torch.randn(shape, dtype=dtype, device=device) * sat_factor
        yield (grad, self_t, scale, zero_point, axis, quant_min, quant_max, grad_factor)


def _resolve_native():
    return ATEN_OP


DTYPES = [torch.float32]

_inputs = _input_iter

globals().update(
    export_contract(
        "fake_quantize_learnable_per_channel_affine_backward",
        DTYPES,
        _resolve_native,
        _inputs,
    )
)
