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


"""matmul_bias_activation: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_matmul_bias_activation.py:45 (test_matmul_bias_activation)
Native expression: mma_torch_op
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml BlasBenchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def matmul_bias_activation_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    input_tensor = torch.randn([m, k], dtype=cur_dtype, device=device)
    weight = torch.randn([k, n], dtype=cur_dtype, device=device)
    bias = torch.randn([n], dtype=cur_dtype, device=device)
    yield (input_tensor, weight, bias)


def mma_torch_op(input, weight, bias):
    return torch.relu(torch.mm(input, weight) + bias)


def _resolve_native():
    return mma_torch_op


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = lambda shape, dtype, device: matmul_bias_activation_input_fn(
    *shape, dtype, device, False
)

globals().update(
    export_contract("matmul_bias_activation", DTYPES, _resolve_native, _inputs)
)
