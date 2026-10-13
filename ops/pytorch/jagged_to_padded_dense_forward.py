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


"""jagged_to_padded_dense_forward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_jagged_to_padded_dense_forward.py:56 (test_jagged_to_padded_dense_forward)
Native expression: torch.ops.aten._jagged_to_padded_dense_forward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

import numpy as np


def _input_iter(shape, dtype, device):
    for batch_size, max_length in [shape]:
        np.random.seed(42)
        seq_lengths = np.random.randint(1, max_length + 1, size=batch_size).tolist()
        offsets = [0] + list(np.cumsum(seq_lengths).astype(int).tolist())
        offsets = torch.tensor(offsets, device=device, dtype=torch.int64)
        total_length = sum(seq_lengths)
        values = torch.randn(total_length, dtype=dtype, device=device)
        yield (values, [offsets], [max_length], 0.0)


def _resolve_native():
    return torch.ops.aten._jagged_to_padded_dense_forward


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(
    export_contract("jagged_to_padded_dense_forward", DTYPES, _resolve_native, _inputs)
)
