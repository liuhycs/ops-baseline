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


"""gru: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_gru.py:145 (test_gru)
Native expression: torch.gru
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

VENDOR = "nvidia"

if VENDOR == "ascend":
    DTYPES = [torch.float16]
else:
    DTYPES = [torch.float32]
    DTYPES.append(torch.float64)


def gru_input_fn(shape, dtype, device):
    (
        batch_size,
        seq_len,
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        batch_first,
        has_biases,
    ) = shape
    gru = torch.nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=has_biases,
        bidirectional=bidirectional,
        batch_first=batch_first,
    ).to(device=device, dtype=dtype)
    gru.flatten_parameters()
    input_shape = (
        (batch_size, seq_len, input_size)
        if batch_first
        else (seq_len, batch_size, input_size)
    )
    input = torch.randn(input_shape, dtype=dtype, device=device)
    num_directions = 2 if bidirectional else 1
    state_shape = (num_layers * num_directions, batch_size, hidden_size)
    h0 = torch.randn(state_shape, dtype=dtype, device=device)
    params = tuple(gru._flat_weights)
    yield (
        input,
        h0,
        params,
        has_biases,
        num_layers,
        0.0,
        False,
        bidirectional,
        batch_first,
    )


def _resolve_native():
    return torch.gru


DTYPES = DTYPES

_inputs = gru_input_fn

globals().update(export_contract("gru", DTYPES, _resolve_native, _inputs))
