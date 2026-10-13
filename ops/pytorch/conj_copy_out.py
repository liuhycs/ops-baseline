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


"""conj_copy_out: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_conj_copy.py:35 (test_conj_copy_out)
Native expression: torch.ops.aten._conj_copy.out
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml Benchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import standard_inputs, export_contract


def _resolve_native():
    return torch.ops.aten._conj_copy.out


DTYPES = [torch.complex64]

_inputs = lambda shape, dtype, device: standard_inputs(
    "unary_out", shape, dtype, device
)

globals().update(export_contract("conj_copy_out", DTYPES, _resolve_native, _inputs))
