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


"""tensordot_out: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_tensordot.py:69 (test_tensordot_out)
Native expression: torch.tensordot
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_fn_out(m, n, k1, k2, dtype, device):
    a = torch.randn((m, k1, k2), dtype=dtype, device=device)
    b = torch.randn((k1, k2, n), dtype=dtype, device=device)
    out = torch.empty((m, n), dtype=dtype, device=device)
    yield (a, b, 2, {"out": out})


def _input_iter(shape, dtype, device):
    for m, n, k1, k2 in [shape]:
        yield from _input_fn_out(m, n, k1, k2, dtype, device)


def _resolve_native():
    return torch.tensordot


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(export_contract("tensordot_out", DTYPES, _resolve_native, _inputs))
