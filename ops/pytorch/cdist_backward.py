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


"""cdist_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_cdist_backward.py:45 (test_cdist_backward)
Native expression: torch.ops.aten._cdist_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml cdist_backward or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        batch, n1, dim = shape
        n2 = n1 // 2 + 1
        x1 = torch.randn(shape, dtype=dtype, device=device)
        x2 = torch.randn(batch, n2, dim, dtype=dtype, device=device)
        cdist = torch.cdist(x1, x2, p=2.0)
        grad = torch.randn(batch, n1, n2, dtype=dtype, device=device)
        yield (grad, x1, x2, 2.0, cdist)


def _resolve_native():
    return torch.ops.aten._cdist_backward


DTYPES = [torch.float32]

_inputs = _input_iter

globals().update(export_contract("cdist_backward", DTYPES, _resolve_native, _inputs))
