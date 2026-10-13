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


"""nested_from_padded_tensor: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_nested_from_padded_tensor.py:69 (test_nested_from_padded_tensor)
Native expression: _native_op
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

from torch.nested._internal.nested_tensor import _nt_view_dummy

import numpy as np

_DUMMY = _nt_view_dummy()


def _native_op(padded, offsets, ragged_idx):
    return torch.ops.aten._nested_from_padded_tensor.default(
        padded, offsets, _DUMMY, ragged_idx, None, None, None
    )


def _input_iter(shape, dtype, device):
    for B, T, D in [shape]:
        np.random.seed(42)
        lengths = np.random.randint(1, T + 1, size=B)
        offsets = torch.tensor(
            [0] + np.cumsum(lengths).tolist(), dtype=torch.int32, device=device
        )
        padded = torch.randn(B, T, D, dtype=dtype, device=device)
        yield (padded, offsets, 1)


def _resolve_native():
    return _native_op


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(
    export_contract("nested_from_padded_tensor", DTYPES, _resolve_native, _inputs)
)
