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


"""mkldnn_rnn_layer: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_mkldnn_rnn_layer.py:89 (test_mkldnn_rnn_layer)
Native expression: _torch_lstm_layer_baseline
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: upstream fixed get_input_iter loop; first tensor shape identifies geometry.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

_MODE = 2

_NUM_LAYERS = 1

_BIDIRECTIONAL = False

_BATCH_FIRST = False


def mkldnn_rnn_layer_input_fn(shape, dtype, device):
    seq_len, batch_size, input_size = shape
    hidden_size = input_size
    inp = torch.randn(seq_len, batch_size, input_size, dtype=dtype, device=device)
    lstm = torch.nn.LSTM(input_size, hidden_size, 1).to(dtype=dtype, device=device)
    w_ih = lstm.weight_ih_l0
    w_hh = lstm.weight_hh_l0
    b_ih = lstm.bias_ih_l0
    b_hh = lstm.bias_hh_l0
    hx = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    cx = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    yield (
        inp,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        hx,
        cx,
        False,
        [],
        _MODE,
        hidden_size,
        _NUM_LAYERS,
        True,
        _BIDIRECTIONAL,
        _BATCH_FIRST,
        False,
    )


def _torch_lstm_layer_baseline(inp, w_ih, w_hh, b_ih, b_hh, hx, cx, *rest):
    """Same-device LSTM baseline.

    The aten mkldnn_rnn_layer (oneDNN) is CPU-only, so it cannot serve as an
    on-device torch baseline. Reproduce the identical single-layer LSTM with
    native PyTorch on the input device instead (gate order i, f, g, o)."""
    reverse = rest[0]
    seq_len = inp.shape[0]
    h = hx
    c = cx
    steps = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
    outputs = [None] * seq_len
    for t in steps:
        gates = torch.addmm(b_ih, inp[t], w_ih.t()) + torch.addmm(b_hh, h, w_hh.t())
        i_g, f_g, g_g, o_g = gates.chunk(4, dim=1)
        i_g = torch.sigmoid(i_g)
        f_g = torch.sigmoid(f_g)
        g_g = torch.tanh(g_g)
        o_g = torch.sigmoid(o_g)
        c = f_g * c + i_g * g_g
        h = o_g * torch.tanh(c)
        outputs[t] = h
    return (torch.stack(outputs, dim=0), h, c)


def _input_iter(shape, dtype, device):
    shapes = [(16, 4, 32), (32, 8, 64), (64, 16, 128)]
    for shape in shapes:
        yield from mkldnn_rnn_layer_input_fn(shape, dtype, device)


def _resolve_native():
    return _torch_lstm_layer_baseline


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(export_contract("mkldnn_rnn_layer", DTYPES, _resolve_native, _inputs))
