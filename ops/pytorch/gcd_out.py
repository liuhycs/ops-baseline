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


"""gcd_out: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_gcd.py:71 (test_gcd_out)
Native expression: torch.gcd
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml Benchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def gcd_out_input_fn(shape, dtype, device):
    inp1 = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape,
        dtype=dtype,
        device=device,
    )
    inp2 = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape,
        dtype=dtype,
        device=device,
    )
    out = torch.empty(shape, dtype=dtype, device=device)
    yield (inp1, inp2, {"out": out})


def _resolve_native():
    return torch.gcd


DTYPES = [torch.int16, torch.int32]

_inputs = gcd_out_input_fn

globals().update(export_contract("gcd_out", DTYPES, _resolve_native, _inputs))
