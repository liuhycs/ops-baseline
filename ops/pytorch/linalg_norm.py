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


"""linalg_norm: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_linalg_norm.py:104 (test_linalg_norm)
Native expression: torch.linalg.norm
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: upstream fixed get_input_iter loop; first tensor shape identifies geometry.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract

VENDOR = "nvidia"

DTYPES = [torch.float32]
DTYPES.extend([torch.float64])  # Pinned NVIDIA capabilities.

SHAPE = [
    (1024,),
    (65536,),
    (64, 64),
    (256, 512),
    (2, 2048),
    (4, 32, 64),
    (8, 4, 32, 32),
    (1024, 1024),
]

_VECTOR_ORDS = (2, 1, 0, float("inf"))

_MATRIX_ORDS = ("fro", 1, -1, float("inf"), float("-inf"))

_SVD_ORDS = (2, -2, "nuc")

_BATCH_DIMS = [(-2, -1), (0, 2), 1]


def _svd_ok(shape):
    """GEMS SVD kernel limits: k = min(m, n) ∈ [2, 512], max(m, n) ≤ 2048."""
    k, rows = (min(shape[-2], shape[-1]), max(shape[-2], shape[-1]))
    return k >= 2 and k <= 512 and (rows <= 2048)


def _input_iter(shape, dtype, device):
    for shape in SHAPE:
        if dtype == torch.float64:
            inp = torch.randn(shape, dtype=dtype, device=device)
        else:
            inp = generate_tensor_input(shape, dtype, device)
        if len(shape) == 1:
            for ord_val in _VECTOR_ORDS:
                yield (inp.clone(), ord_val)
        elif len(shape) == 2:
            if dtype in (torch.float16, torch.bfloat16):
                return
            if VENDOR == "ascend":
                if _svd_ok(shape):
                    for ord_val in _SVD_ORDS:
                        yield (inp.clone(), ord_val)
                continue
            for ord_val in _MATRIX_ORDS:
                yield (inp.clone(), ord_val)
            if _svd_ok(shape):
                yield (inp.clone(), "nuc")
        else:
            if dtype not in (torch.float16, torch.bfloat16):
                yield (inp.clone(), None)
            if VENDOR == "ascend":
                yield (inp.clone(), 2, _BATCH_DIMS[2])
                if dtype not in (torch.float16, torch.bfloat16):
                    yield (inp.clone(), 2, _BATCH_DIMS[0])
                    if _svd_ok(shape):
                        yield (inp.clone(), "nuc", _BATCH_DIMS[0])
                continue
            for ord_val, dim in ((1, _BATCH_DIMS[1]), (2, _BATCH_DIMS[2])):
                yield (inp.clone(), ord_val, dim)
            if dtype not in (torch.float16, torch.bfloat16):
                yield (inp.clone(), 2, _BATCH_DIMS[0])
                if _svd_ok(shape):
                    yield (inp.clone(), "nuc", _BATCH_DIMS[0])


def _resolve_native():
    return torch.linalg.norm


DTYPES = DTYPES

_inputs = _input_iter

globals().update(export_contract("linalg_norm", DTYPES, _resolve_native, _inputs))
