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


"""nll_loss2d: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_nll_loss2d_combined.py:55 (test_nll_loss2d)
Native expression: torch.ops.aten.nll_loss2d
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def nll_loss2d_input_fn(shape, cur_dtype, device):
    N, C, H, W = shape
    inp = generate_tensor_input(shape, cur_dtype, device)
    target = torch.randint(0, C, (N, H, W), device=device)
    yield (inp, target, None, 1, -100)


def _resolve_native():
    return torch.ops.aten.nll_loss2d


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = nll_loss2d_input_fn

globals().update(export_contract("nll_loss2d", DTYPES, _resolve_native, _inputs))
