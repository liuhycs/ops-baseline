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


"""replication_pad2d_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_replication_pad2d_backward.py:57 (test_replication_pad2d_backward)
Native expression: torch.ops.aten.replication_pad2d_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    for shape, padding in [shape]:
        pad_left, pad_right, pad_top, pad_bottom = padding
        x = torch.randn(shape, dtype=dtype, device=device)
        if x.ndim == 4:
            N, C, H, W = x.shape
            grad_shape = (N, C, H + pad_top + pad_bottom, W + pad_left + pad_right)
        else:
            C, H, W = x.shape
            grad_shape = (C, H + pad_top + pad_bottom, W + pad_left + pad_right)
        grad_output = torch.ones(grad_shape, dtype=dtype, device=device)
        yield (grad_output, x, padding)


def _resolve_native():
    return torch.ops.aten.replication_pad2d_backward


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(
    export_contract("replication_pad2d_backward", DTYPES, _resolve_native, _inputs)
)
