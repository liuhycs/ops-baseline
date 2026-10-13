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


"""mean: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_mean.py:23 (test_mean)
Native expression: torch.mean
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml UnaryReductionBenchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import standard_inputs, export_contract


def _resolve_native():
    return torch.mean


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = lambda shape, dtype, device: standard_inputs(
    "reduction", shape, dtype, device
)

globals().update(export_contract("mean", DTYPES, _resolve_native, _inputs))
