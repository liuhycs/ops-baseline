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


"""nanmedian_out: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_nanmedian.py:97 (test_nanmedian_out)
Native expression: torch.ops.aten.nanmedian.out
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml nanmedian_out or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _filter_reference_supported(dtypes):
    return dtypes


NANMEDIAN_DTYPES = _filter_reference_supported(
    [torch.float16, torch.float32, torch.bfloat16]
    + [torch.int8, torch.int16, torch.int32, torch.uint8]
)


def _make_input(shape, dtype, device):
    if dtype is torch.uint8:
        return torch.randint(0, 101, shape, dtype=dtype, device="cpu").to(device)
    if dtype in (torch.int8, torch.int16, torch.int32):
        return torch.randint(-100, 101, shape, dtype=dtype, device="cpu").to(device)
    inp = torch.randn(shape, dtype=dtype, device=device)
    if inp.numel() > 0:
        inp.reshape(-1)[::17] = float("nan")
    return inp


def _out_input_fn(shape, dtype, device):
    inp = _make_input(shape, dtype, device)
    out = torch.empty((), dtype=dtype, device=device)
    yield (inp, {"out": out})


def _resolve_native():
    return torch.ops.aten.nanmedian.out


DTYPES = NANMEDIAN_DTYPES

_inputs = _out_input_fn

globals().update(export_contract("nanmedian_out", DTYPES, _resolve_native, _inputs))
