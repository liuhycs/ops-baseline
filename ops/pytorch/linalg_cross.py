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


"""linalg_cross: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_linalg_cross.py:58 (test_linalg_cross)
Native expression: torch.linalg.cross
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _randn(shape, dtype, device):
    if dtype.is_complex and "nvidia" == "ascend":
        return torch.randn(shape, dtype=dtype).to(device)
    return torch.randn(shape, dtype=dtype, device=device)


def _input_iter(shape, dtype, device):
    for input_shape, other_shape, dim in [shape]:
        input = _randn(input_shape, dtype, device)
        other = _randn(other_shape, dtype, device)
        yield (input, other, {"dim": dim})


def _resolve_native():
    return torch.linalg.cross


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(export_contract("linalg_cross", DTYPES, _resolve_native, _inputs))
