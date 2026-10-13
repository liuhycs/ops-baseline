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


"""binary_cross_entropy_with_logits: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_binary_cross_entropy_with_logits.py:15 (test_binary_cross_entropy_with_logits)
Native expression: torch.ops.aten.binary_cross_entropy_with_logits
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml Benchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    target = torch.rand(shape, dtype=dtype, device=device)
    yield (inp, target)


def _resolve_native():
    return torch.ops.aten.binary_cross_entropy_with_logits


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_fn

globals().update(
    export_contract(
        "binary_cross_entropy_with_logits", DTYPES, _resolve_native, _inputs
    )
)
