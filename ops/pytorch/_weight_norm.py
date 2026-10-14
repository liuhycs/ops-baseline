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


"""_weight_norm: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test__weight_norm.py:63 (test_underscore_weight_norm_dim0)
Native expression: torch.ops.aten._weight_norm.default
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml _weight_norm or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def weight_norm_input_fn(shape, dtype, device):
    dim = 0
    v = torch.randn(shape, dtype=dtype, device=device)
    g = torch.randn(
        [1 if i != dim else shape[i] for i in range(len(shape))],
        dtype=dtype,
        device=device,
    )
    yield (v, g, dim)


def _resolve_native():
    return torch.ops.aten._weight_norm.default


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = weight_norm_input_fn

globals().update(export_contract("_weight_norm", DTYPES, _resolve_native, _inputs))
