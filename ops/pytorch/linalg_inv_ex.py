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


"""linalg_inv_ex: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_linalg_inv_ex.py:63 (test_linalg_inv_ex)
Native expression: torch.linalg.inv_ex
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

VENDOR = "nvidia"

_TEST_DTYPES = [torch.float32, torch.float64] if VENDOR == "nvidia" else [torch.float32]


def _input_iter(shape, dtype, device):
    for inp_shape in [shape]:
        inp_shape = tuple(inp_shape)
        n = inp_shape[-1]
        A = torch.randn(inp_shape, dtype=dtype, device="cpu")
        for i in range(n):
            A[..., i, i] = A[..., i, i] + n
        A = A.to(device)
        yield (A, {})


def _resolve_native():
    return torch.linalg.inv_ex


DTYPES = _TEST_DTYPES

_inputs = _input_iter

globals().update(export_contract("linalg_inv_ex", DTYPES, _resolve_native, _inputs))
