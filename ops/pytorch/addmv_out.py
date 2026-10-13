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


"""addmv_out: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_addmv.py:62 (test_addmv_out)
Native expression: torch.addmv
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml GenericBenchmark2DOnly or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_fn_out(m, n, cur_dtype, device):
    mat = torch.randn([m, n], dtype=cur_dtype, device=device)
    vec = torch.randn([n], dtype=cur_dtype, device=device)
    bias = torch.randn([m], dtype=cur_dtype, device=device)
    out = torch.empty([m], dtype=cur_dtype, device=device)
    yield (bias, mat, vec, {"out": out})


def _input_iter(shape, dtype, device):
    for m, n in [shape]:
        yield from _input_fn_out(m, n, dtype, device)


def _resolve_native():
    return torch.addmv


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(export_contract("addmv_out", DTYPES, _resolve_native, _inputs))
