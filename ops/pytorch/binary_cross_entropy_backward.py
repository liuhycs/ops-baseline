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


"""binary_cross_entropy_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_binary_cross_entropy_backward.py:35 (test_binary_cross_entropy_backward)
Native expression: torch.ops.aten.binary_cross_entropy_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        self_t = torch.rand(shape, dtype=dtype, device=device)
        self_t = torch.clamp(self_t, 0.0001, 1 - 0.0001)
        target = torch.randint(0, 2, shape, dtype=dtype, device=device).to(dtype=dtype)
        grad_output = torch.ones_like(self_t)
        weight = torch.rand(shape, dtype=dtype, device=device)
        yield (grad_output, self_t, target, weight, 0)


def _resolve_native():
    return torch.ops.aten.binary_cross_entropy_backward


DTYPES = [torch.float16, torch.float32]

_inputs = _input_iter

globals().update(
    export_contract("binary_cross_entropy_backward", DTYPES, _resolve_native, _inputs)
)
