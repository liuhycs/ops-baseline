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


"""native_dropout_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_dropout.py:46 (test_native_dropout_backward)
Native expression: torch.ops.aten.native_dropout_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml Benchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def _dropout_backward_input_fn(shape, dtype, device):
    grad_output = generate_tensor_input(shape, dtype, device)
    mask = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    scale = 1.0 / 0.5
    yield (grad_output, mask, {"scale": scale})


def _resolve_native():
    return torch.ops.aten.native_dropout_backward


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _dropout_backward_input_fn

globals().update(
    export_contract("native_dropout_backward", DTYPES, _resolve_native, _inputs)
)
