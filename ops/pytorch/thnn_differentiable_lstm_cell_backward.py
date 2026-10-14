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


"""thnn_differentiable_lstm_cell_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_thnn_differentiable_lstm_cell_backward.py:84 (test_thnn_differentiable_lstm_cell_backward)
Native expression: torch.ops.aten._thnn_differentiable_lstm_cell_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

DTYPES = list([torch.float16, torch.float32, torch.bfloat16])
DTYPES.extend([torch.float64])  # Pinned NVIDIA capabilities.


def _input_iter(shape, dtype, device):
    for batch_size, hidden_size, has_bias in [shape]:
        grad_hy = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
        grad_cy = torch.randn_like(grad_hy)
        input_gates = torch.randn(
            batch_size, 4 * hidden_size, dtype=dtype, device=device
        )
        hidden_gates = torch.randn_like(input_gates)
        cx = torch.randn_like(grad_hy)
        cy = torch.randn_like(grad_hy)
        if has_bias:
            input_bias = torch.randn(4 * hidden_size, dtype=dtype, device=device)
            hidden_bias = torch.randn_like(input_bias)
        else:
            input_bias = None
            hidden_bias = None
        yield (
            grad_hy,
            grad_cy,
            input_gates,
            hidden_gates,
            input_bias,
            hidden_bias,
            cx,
            cy,
        )


def _resolve_native():
    return torch.ops.aten._thnn_differentiable_lstm_cell_backward


DTYPES = DTYPES

_inputs = _input_iter

globals().update(
    export_contract(
        "thnn_differentiable_lstm_cell_backward", DTYPES, _resolve_native, _inputs
    )
)
