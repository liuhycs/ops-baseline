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


"""softmax_backward_out: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_softmax.py:64 (test_softmax_backward_out)
Native expression: torch.ops.aten._softmax_backward_data.out
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml softmax_backward_out or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def softmax_backward_out_input_fn(shape, dtype, device):
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    output = torch.randn(shape, dtype=dtype, device=device)
    grad_input = torch.empty(shape, dtype=dtype, device=device)
    dim = 1 if len(shape) > 1 else 0
    yield (grad_output, output, dim, dtype, {"grad_input": grad_input})


def _resolve_native():
    return torch.ops.aten._softmax_backward_data.out


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = softmax_backward_out_input_fn

globals().update(
    export_contract("softmax_backward_out", DTYPES, _resolve_native, _inputs)
)
