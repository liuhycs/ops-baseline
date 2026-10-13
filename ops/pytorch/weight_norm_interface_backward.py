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


"""weight_norm_interface_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_weight_norm_interface.py:71 (test_weight_norm_interface_backward)
Native expression: torch.ops.aten._weight_norm_interface_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml GenericBenchmarkExcluse1D or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def weight_norm_interface_backward_input_fn(shape, dtype, device):
    dim = 0
    w_grad = torch.randn(shape, dtype=dtype, device=device)
    saved_v = torch.randn(shape, dtype=dtype, device=device)
    saved_g = torch.randn(shape[dim], dtype=dtype, device=device)
    saved_norms = torch.randn(shape[dim], dtype=dtype, device=device)
    yield (w_grad, saved_v, saved_g, saved_norms, dim)


def _resolve_native():
    return torch.ops.aten._weight_norm_interface_backward


DTYPES = [torch.float32]

_inputs = weight_norm_interface_backward_input_fn

globals().update(
    export_contract("weight_norm_interface_backward", DTYPES, _resolve_native, _inputs)
)
