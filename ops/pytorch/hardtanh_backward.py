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


"""hardtanh_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_hardtanh_backward.py:38 (test_hardtanh_backward)
Native expression: torch.ops.aten.hardtanh_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml Benchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def hardtanh_backward_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    min_val = -1.0
    max_val = 1.0
    yield (grad_output, inp, min_val, max_val)


def _resolve_native():
    return torch.ops.aten.hardtanh_backward


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = hardtanh_backward_input_fn

globals().update(export_contract("hardtanh_backward", DTYPES, _resolve_native, _inputs))
