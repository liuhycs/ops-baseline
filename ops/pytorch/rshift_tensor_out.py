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


"""rshift_tensor_out: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_rshift.py:98 (test_rshift_tensor_out)
Native expression: torch.ops.aten.__rshift__.Tensor_out
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        value = torch.randint(0, 100, shape, dtype=dtype, device=device)
        shift = torch.randint(0, 8, shape, dtype=dtype, device=device)
        out = torch.empty_like(value)
        yield (value, shift, {"out": out})


def _resolve_native():
    return torch.ops.aten.__rshift__.Tensor_out


DTYPES = [torch.int16, torch.int32] + [torch.int8, torch.uint8, torch.int64]

_inputs = _input_iter

globals().update(export_contract("rshift_tensor_out", DTYPES, _resolve_native, _inputs))
