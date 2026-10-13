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


"""lstm_cell: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_lstm_cell.py:63 (test_lstm_cell)
Native expression: torch.ops.aten.lstm_cell
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import (
    BenchmarkCasePlan,
    generate_tensor_input,
    case_inputs,
    export_contract,
)


def _case_fn(shape, dtype):
    del dtype
    batch, input_size, hidden = shape
    yield BenchmarkCasePlan(
        shape={"input": (batch, input_size), "hidden": (batch, hidden)},
        params={"bias": True},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    batch, input_size, hidden = plan.builder_args[0]
    gates = 4 * hidden
    inp = generate_tensor_input((batch, input_size), dtype, device)
    w_ih = generate_tensor_input((gates, input_size), dtype, device)
    w_hh = generate_tensor_input((gates, hidden), dtype, device)
    b_ih = generate_tensor_input((gates,), dtype, device)
    b_hh = generate_tensor_input((gates,), dtype, device)
    h0 = generate_tensor_input((batch, hidden), dtype, device)
    c0 = generate_tensor_input((batch, hidden), dtype, device)
    return (inp, [h0, c0], w_ih, w_hh, b_ih, b_hh, {})


def _resolve_native():
    return torch.ops.aten.lstm_cell


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(export_contract("lstm_cell", DTYPES, _resolve_native, _inputs))
