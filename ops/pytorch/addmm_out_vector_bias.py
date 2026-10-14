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


"""addmm_out_vector_bias: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_addmm.py:188 (test_addmm_out_vector_bias)
Native expression: torch.addmm
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml addmm_out_vector_bias or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_fn_vector_bias(b, m, n, k, dtype, device, b_column_major):
    mat1 = torch.randn((m, k), dtype=dtype, device=device)
    if b_column_major:
        mat2 = torch.randn((n, k), dtype=dtype, device=device).t()
    else:
        mat2 = torch.randn((k, n), dtype=dtype, device=device)
    bias = torch.randn((n,), dtype=dtype, device=device)
    yield (bias, mat1, mat2)


def _input_fn_vector_bias_out(b, m, n, k, dtype, device, b_column_major):
    for bias, mat1, mat2 in _input_fn_vector_bias(
        b, m, n, k, dtype, device, b_column_major
    ):
        out = torch.empty((m, n), dtype=dtype, device=device)
        yield (bias, mat1, mat2, {"out": out})


def _input_iter(shape, dtype, device):
    for b, m, n, k in [shape]:
        yield from _input_fn_vector_bias_out(b, m, n, k, dtype, device, True)


def _resolve_native():
    return torch.addmm


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(
    export_contract("addmm_out_vector_bias", DTYPES, _resolve_native, _inputs)
)
