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


"""conj_physical: PyTorch baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_conj_physical.py:91 (test_conj_physical)
Native expression: torch.conj_physical
Constructor state was explicitly lowered; explicit source set_shapes.
Core source parameters and filters are preserved.
"""

import torch

from _kernelgen import export_contract

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _input_fn(shape, dtype, device):
    if dtype.is_complex:
        float_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        real = torch.randn(shape, dtype=float_dtype, device=device)
        imag = torch.randn(shape, dtype=float_dtype, device=device)
        input_tensor = torch.complex(real, imag).to(dtype)
    elif dtype in FP8_DTYPES:
        input_tensor = torch.randn(shape, dtype=torch.float32, device=device).to(dtype)
    elif dtype.is_floating_point:
        input_tensor = torch.randn(shape, dtype=dtype, device=device)
    else:
        input_tensor = torch.randn(shape, device=device).to(dtype)
    yield (input_tensor,)


DTYPES = [torch.complex64]


def _resolve_native():
    return torch.conj_physical


_inputs = _input_fn

globals().update(export_contract("conj_physical", DTYPES, _resolve_native, _inputs))
