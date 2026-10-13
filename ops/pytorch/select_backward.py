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


"""select_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_select_backward.py:77 (test_select_backward)
Native expression: torch.ops.aten.select_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml Benchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_fn(shape, dtype, device):
    ndim = len(shape)
    for dim in [0, ndim // 2, -1]:
        actual_dim = dim + ndim if dim < 0 else dim
        if actual_dim >= ndim:
            continue
        dim_size = shape[actual_dim]
        index = dim_size // 2
        grad_shape = list(shape)
        grad_shape.pop(actual_dim)
        grad_output = torch.randn(grad_shape, dtype=dtype, device=device)
        yield (grad_output, shape, actual_dim, index)


def _resolve_native():
    return torch.ops.aten.select_backward


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_fn

globals().update(export_contract("select_backward", DTYPES, _resolve_native, _inputs))
