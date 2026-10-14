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


"""rnn_relu: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_rnn_relu.py:56 (test_rnn_relu)
Native expression: torch.rnn_relu
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: upstream fixed get_input_iter loop; first tensor shape identifies geometry.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def rnn_relu_input_fn(shape, dtype, device):
    seq_len, batch_size, input_size = shape
    hidden_size = input_size
    inp = torch.randn(seq_len, batch_size, input_size, dtype=dtype, device=device)
    hx = torch.randn(1, batch_size, hidden_size, dtype=dtype, device=device)
    rnn = torch.nn.RNN(input_size, hidden_size, 1, nonlinearity="relu")
    rnn = rnn.to(dtype=dtype, device=device)
    params = tuple(rnn._flat_weights)
    yield (
        inp,
        {
            "hx": hx,
            "params": params,
            "has_biases": True,
            "num_layers": 1,
            "dropout": 0.0,
            "train": False,
            "bidirectional": False,
            "batch_first": False,
        },
    )


def _input_iter(shape, dtype, device):
    shapes = [(16, 4, 32), (32, 8, 64), (64, 16, 128)]
    for shape in shapes:
        yield from rnn_relu_input_fn(shape, dtype, device)


def _resolve_native():
    return torch.rnn_relu


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(export_contract("rnn_relu", DTYPES, _resolve_native, _inputs))
