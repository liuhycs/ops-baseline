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


"""matrix_power: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_matrix_power.py:36 (test_matrix_power)
Native expression: torch.matrix_power
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml BlasBenchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

MATRIX_POWER_N = 5


def _input_fn(m, cur_dtype, device):
    inp = torch.randn([m, m], dtype=cur_dtype, device=device)
    yield (inp, MATRIX_POWER_N)


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        m = shape[-1]
        yield from _input_fn(m, dtype, device)


def _resolve_native():
    return torch.matrix_power


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(export_contract("matrix_power", DTYPES, _resolve_native, _inputs))
