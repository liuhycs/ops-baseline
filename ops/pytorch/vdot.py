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


"""vdot: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_vdot.py:40 (test_vdot)
Native expression: torch.Tensor.vdot
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml vdot or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def vdot_input_fn(m, cur_dtype, device):
    inp1 = torch.randn([m], dtype=cur_dtype, device=device)
    inp2 = torch.randn([m], dtype=cur_dtype, device=device)
    yield (inp1, inp2)


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        m = shape[0]
        yield from vdot_input_fn(m, dtype, device)


def _resolve_native():
    return torch.Tensor.vdot


DTYPES = [torch.complex64] + [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(export_contract("vdot", DTYPES, _resolve_native, _inputs))
