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


"""binary_cross_entropy_out: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_binary_cross_entropy.py:70 (test_binary_cross_entropy_out)
Native expression: torch_binary_cross_entropy_out
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml GenericBenchmark2DOnly or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def binary_cross_entropy_out_input_fn(shape, dtype, device):
    inp = torch.sigmoid(generate_tensor_input(shape, dtype, device))
    target = torch.randint(0, 2, shape, device=device).to(dtype)
    out = torch.empty((), dtype=dtype, device=device)
    yield (inp, target, {"out": out})


def torch_binary_cross_entropy_out(inp, target, **kwargs):
    out = kwargs.pop("out")
    result = torch.nn.functional.binary_cross_entropy(inp, target, **kwargs)
    out.copy_(result)
    return out


def _resolve_native():
    return torch_binary_cross_entropy_out


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = binary_cross_entropy_out_input_fn

globals().update(
    export_contract("binary_cross_entropy_out", DTYPES, _resolve_native, _inputs)
)
