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


"""igamma: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_igamma.py:46 (test_igamma)
Native expression: torch.igamma
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml Benchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def igamma_input_fn(shape, dtype, device):
    inp1 = torch.rand(shape, dtype=dtype, device=device) + 0.1
    inp2 = torch.rand(shape, dtype=dtype, device=device) + 0.1
    yield (inp1, inp2)


def _resolve_native():
    return torch.igamma


DTYPES = [torch.float32]

_inputs = igamma_input_fn

globals().update(export_contract("igamma", DTYPES, _resolve_native, _inputs))
