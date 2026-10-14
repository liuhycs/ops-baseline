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


"""unsafe_masked_index: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_unsafe_masked_index.py:51 (test_unsafe_masked_index)
Native expression: torch._unsafe_masked_index
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        n = shape[0]
        inp = torch.randn(shape, dtype=dtype, device=device)
        mask = torch.rand(shape, device=device) > 0.3
        indices = torch.randint(0, max(n, 1), shape, device=device)
        fill = 0.0
        yield (inp, mask, [indices], fill)


def _resolve_native():
    return torch._unsafe_masked_index


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(
    export_contract("unsafe_masked_index", DTYPES, _resolve_native, _inputs)
)
