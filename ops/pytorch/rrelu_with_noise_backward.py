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


"""rrelu_with_noise_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_rrelu_with_noise_backward.py:43 (test_rrelu_with_noise_backward)
Native expression: torch.ops.aten.rrelu_with_noise_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml UnaryPointwiseBenchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        inp = generate_tensor_input(shape, dtype, device)
        grad_out = torch.randn_like(inp)
        noise = torch.rand_like(inp)
        lower = 0.125
        upper = 1.0 / 3.0
        training = True
        self_is_result = False
        yield (grad_out, inp, noise, lower, upper, training, self_is_result)


def _resolve_native():
    return torch.ops.aten.rrelu_with_noise_backward


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(
    export_contract("rrelu_with_noise_backward", DTYPES, _resolve_native, _inputs)
)
