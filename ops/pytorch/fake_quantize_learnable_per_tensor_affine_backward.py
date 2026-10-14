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


"""fake_quantize_learnable_per_tensor_affine_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_fake_quantize_learnable_per_tensor_affine_backward.py:64 (test_fake_quantize_learnable_per_tensor_affine_backward)
Native expression: torch.ops.aten._fake_quantize_learnable_per_tensor_affine_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml Benchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _fake_quantize_learnable_per_tensor_affine_backward_input_fn(shape, dtype, device):
    self_t = torch.randn(shape, dtype=torch.float32, device=device)
    grad_t = torch.randn(shape, dtype=torch.float32, device=device)
    scale_t = torch.tensor([0.1], dtype=torch.float32, device=device)
    zero_point_t = torch.tensor([128.0], dtype=torch.float32, device=device)
    yield (grad_t, self_t, scale_t, zero_point_t, 0, 255, 1.0)


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        yield from _fake_quantize_learnable_per_tensor_affine_backward_input_fn(
            shape, dtype, device
        )


def _resolve_native():
    return torch.ops.aten._fake_quantize_learnable_per_tensor_affine_backward


DTYPES = [torch.float32]

_inputs = _input_iter

globals().update(
    export_contract(
        "fake_quantize_learnable_per_tensor_affine_backward",
        DTYPES,
        _resolve_native,
        _inputs,
    )
)
