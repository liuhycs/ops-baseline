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


"""inverse: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_inverse.py:64 (test_inverse)
Native expression: torch.inverse
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

INVERSE_DTYPES = [torch.float32] + ([torch.float64] if True else [])


def _well_conditioned(shape, dtype, device):
    n = shape[-1]
    eye = torch.eye(n, dtype=dtype, device=device)
    if len(shape) > 2:
        eye = eye.expand(shape).contiguous()
    return eye + 0.05 * torch.randn(shape, dtype=dtype, device=device)


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        A = _well_conditioned(shape, dtype, device)
        yield (A,)


def _resolve_native():
    return torch.inverse


DTYPES = INVERSE_DTYPES

_inputs = _input_iter

globals().update(export_contract("inverse", DTYPES, _resolve_native, _inputs))
