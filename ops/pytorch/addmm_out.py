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


"""addmm_out: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_addmm.py:176 (test_addmm_out)
Native expression: torch.addmm
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml BlasBenchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_fn_out(b, m, n, k, dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=dtype, device=device)
    bias = torch.randn([m, n], dtype=dtype, device=device)
    out = torch.empty([m, n], dtype=dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=dtype, device=device)
        yield (bias, inp1, inp2.t(), {"out": out})
    else:
        inp2 = torch.randn([k, n], dtype=dtype, device=device)
        yield (bias, inp1, inp2, {"out": out})


def _resolve_native():
    return torch.addmm


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = lambda shape, dtype, device: _input_fn_out(*shape, dtype, device, False)

globals().update(export_contract("addmm_out", DTYPES, _resolve_native, _inputs))
