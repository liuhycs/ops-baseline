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


"""weight_norm_interface: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_weight_norm_interface.py:47 (test_weight_norm_interface)
Native expression: torch._weight_norm
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml GenericBenchmarkExcluse1D or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

vendor_name = "nvidia"


def weight_norm_input_fn(shape, dtype, device):
    v = torch.randn(shape, dtype=dtype, device=device)
    if vendor_name in ["cambricon", "enflame"]:
        g = torch.randn(shape[:1] + (1,) * (len(shape) - 1), dtype=dtype, device=device)
    else:
        g = torch.randn(shape, dtype=dtype, device=device)
    yield (v, g, 0)


def _resolve_native():
    return torch._weight_norm


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = weight_norm_input_fn

globals().update(
    export_contract("weight_norm_interface", DTYPES, _resolve_native, _inputs)
)
