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


"""upsample_nearest_exact3d_backward_grad_input: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_upsample_nearest_exact3d_backward.py:80 (test_upsample_nearest_exact3d_backward_grad_input)
Native expression: torch.ops.aten._upsample_nearest_exact3d_backward.grad_input
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    for input_size, output_size, scales in [shape]:
        grad_output = torch.randn(
            (*input_size[:2], *output_size), dtype=dtype, device=device
        )
        grad_input = torch.empty(input_size, dtype=dtype, device=device)
        yield (
            grad_output,
            output_size,
            input_size,
            *scales,
            {"grad_input": grad_input},
        )


def _resolve_native():
    return torch.ops.aten._upsample_nearest_exact3d_backward.grad_input


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(
    export_contract(
        "upsample_nearest_exact3d_backward_grad_input", DTYPES, _resolve_native, _inputs
    )
)
