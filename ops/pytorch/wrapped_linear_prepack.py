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


"""wrapped_linear_prepack: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_wrapped_linear_prepack.py:38 (test_wrapped_linear_prepack_benchmark)
Native expression: torch.ops.aten._wrapped_linear_prepack
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    for N, K in [shape]:
        weight = torch.randn((N, K), dtype=dtype, device=device)
        scale = torch.tensor(0.03125, dtype=dtype, device=device)
        zero_point = torch.tensor(-3, dtype=torch.int64, device=device)
        bias = torch.randn((N,), dtype=dtype, device=device)
        yield (weight, scale, zero_point, bias)


def _resolve_native():
    return torch.ops.aten._wrapped_linear_prepack


DTYPES = [torch.float32]

_inputs = _input_iter

globals().update(
    export_contract("wrapped_linear_prepack", DTYPES, _resolve_native, _inputs)
)
