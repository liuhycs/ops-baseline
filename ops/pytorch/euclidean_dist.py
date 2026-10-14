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


"""euclidean_dist: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_euclidean_dist.py:43 (test_euclidean_dist)
Native expression: torch._euclidean_dist
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    for (n1, d), (n2, d2) in [shape]:
        x1 = torch.randn(n1, d, dtype=dtype, device=device)
        x2 = torch.randn(n2, d2, dtype=dtype, device=device)
        yield (x1, x2)


def _resolve_native():
    return torch._euclidean_dist


DTYPES = [torch.float32]

_inputs = _input_iter

globals().update(export_contract("euclidean_dist", DTYPES, _resolve_native, _inputs))
