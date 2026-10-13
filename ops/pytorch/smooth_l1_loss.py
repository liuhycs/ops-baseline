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


"""smooth_l1_loss: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_smooth_l1_loss.py:45 (test_smooth_l1_loss)
Native expression: torch.ops.aten.smooth_l1_loss
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml smooth_l1_loss or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        inp = torch.randn(shape, device=device, dtype=dtype)
        target = torch.randn(shape, device=device, dtype=dtype)
        yield (inp, target, 1, 1.0)


def _resolve_native():
    return torch.ops.aten.smooth_l1_loss


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(export_contract("smooth_l1_loss", DTYPES, _resolve_native, _inputs))
