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


"""lstm: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_lstm.py:53 (test_lstm)
Native expression: torch.lstm
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def lstm_input_fn(shape, dtype, device):
    batch_size, seq_len, input_size, hidden_size, num_layers, bidirectional = shape
    has_biases = True
    batch_first = True
    lstm = torch.nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=has_biases,
        bidirectional=bidirectional,
        batch_first=batch_first,
    ).to(device=device, dtype=dtype)
    input = torch.randn(batch_size, seq_len, input_size, dtype=dtype, device=device)
    num_directions = 2 if bidirectional else 1
    state_shape = (num_layers * num_directions, batch_size, hidden_size)
    h0 = torch.randn(state_shape, dtype=dtype, device=device)
    c0 = torch.randn(state_shape, dtype=dtype, device=device)
    params = tuple(lstm._flat_weights)
    yield (
        input,
        (h0, c0),
        params,
        has_biases,
        num_layers,
        0.0,
        False,
        bidirectional,
        batch_first,
    )


def _resolve_native():
    return torch.lstm


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = lstm_input_fn

globals().update(export_contract("lstm", DTYPES, _resolve_native, _inputs))
