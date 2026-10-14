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


"""tril: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_tril.py:71 (test_tril_extreme_diagonal)
Native expression: torch.tril
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml tril or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def _tril_extreme_diagonal_input_fn(shape, dtype, device):
    inp = generate_tensor_input(shape, dtype, device)
    n = shape[-1]
    for diagonal in (-n, -(n // 2), -1, 0, 1, n // 2, n):
        yield (inp, diagonal)


def _resolve_native():
    return torch.tril


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _tril_extreme_diagonal_input_fn

globals().update(export_contract("tril", DTYPES, _resolve_native, _inputs))
