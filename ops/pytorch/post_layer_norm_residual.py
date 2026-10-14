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


"""post_layer_norm_residual: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_post_layer_norm_residual.py:53 (test_post_layer_norm_residual)
Native expression: torch_op
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def torch_op(input, residual, normalized_shape, weight, bias, eps):
    return torch.layer_norm(input, normalized_shape, weight, bias, eps) + residual


def _input_iter(shape, dtype, device):
    for shape, normalized_shape in [shape]:
        input = torch.randn(shape, dtype=dtype, device=device)
        residual = torch.randn_like(input)
        weight = torch.randn(normalized_shape, dtype=dtype, device=device)
        bias = torch.randn_like(weight)
        yield (input, residual, normalized_shape, weight, bias, 1e-05)


def _resolve_native():
    return torch_op


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(
    export_contract("post_layer_norm_residual", DTYPES, _resolve_native, _inputs)
)
