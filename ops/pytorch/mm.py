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


"""mm: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_mm.py:38 (test_mm)
Native expression: torch.Tensor.mm
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml BlasBenchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def mm_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=cur_dtype, device=device)
        yield (inp1, inp2.t())
    else:
        inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
        yield (inp1, inp2)


def _resolve_native():
    return torch.Tensor.mm


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = lambda shape, dtype, device: mm_input_fn(*shape, dtype, device, False)

globals().update(export_contract("mm", DTYPES, _resolve_native, _inputs))
