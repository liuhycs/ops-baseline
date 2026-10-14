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


"""replication_pad1d_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_replication_pad1d_backward.py:51 (test_replication_pad1d_backward)
Native expression: torch.ops.aten.replication_pad1d_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        if len(shape) == 2:
            B, W = shape
        else:
            B, W = (1, shape[0])
        padding = (1, 2)
        x = torch.randn(B, W, dtype=dtype, device=device)
        padded = torch.ops.aten.replication_pad1d(x, padding)
        W_out = padded.shape[-1]
        grad = torch.ones(B, W_out, dtype=dtype, device=device)
        yield (grad, x, padding)


def _resolve_native():
    return torch.ops.aten.replication_pad1d_backward


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(
    export_contract("replication_pad1d_backward", DTYPES, _resolve_native, _inputs)
)
