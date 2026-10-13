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


"""adaptive_max_pool3d_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_adaptive_max_pool3d_backward.py:47 (test_adaptive_max_pool3d_backward)
Native expression: torch.ops.aten.adaptive_max_pool3d_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        x = torch.randn(shape, dtype=dtype, device=device)
        output_size = (shape[2] // 2, shape[3] // 2, shape[4] // 2)
        _, indices = torch.nn.functional.adaptive_max_pool3d(
            x, output_size=output_size, return_indices=True
        )
        grad_output = torch.ones_like(_)
        yield (grad_output, x, indices)


def _resolve_native():
    return torch.ops.aten.adaptive_max_pool3d_backward


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(
    export_contract("adaptive_max_pool3d_backward", DTYPES, _resolve_native, _inputs)
)
