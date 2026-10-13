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


"""mm_self_transpose: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_mm.py:75 (test_mm_self_transpose_benchmark)
Native expression: torch_mm_self_transpose
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml mm_self_transpose or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_fn(shape, cur_dtype, device):
    m, k = shape
    inp = torch.randn([k, m], dtype=cur_dtype, device=device).t()
    yield (inp,)


def torch_mm_self_transpose(inp):
    return torch.mm(inp, inp.t())


def _resolve_native():
    return torch_mm_self_transpose


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_fn

globals().update(export_contract("mm_self_transpose", DTYPES, _resolve_native, _inputs))
