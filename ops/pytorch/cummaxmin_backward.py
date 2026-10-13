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


"""cummaxmin_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_cummaxmin_backward.py:30 (test_cummaxmin_backward)
Native expression: torch.ops.aten.cummaxmin_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml GenericBenchmark2DOnly or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def input_fn(shape, dtype, device):
    inp = generate_tensor_input(shape, dtype, device)
    grad = generate_tensor_input(shape, dtype, device)
    _, indices = torch.cummax(inp, dim=1)
    yield (grad, inp, indices, 1)


def _resolve_native():
    return torch.ops.aten.cummaxmin_backward


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = input_fn

globals().update(
    export_contract("cummaxmin_backward", DTYPES, _resolve_native, _inputs)
)
