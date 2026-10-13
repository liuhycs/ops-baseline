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


"""gru_cell: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_gru_cell.py:66 (test_gru_cell)
Native expression: torch.ops.aten.gru_cell
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
    batch, input_size, hidden_size = shape
    yield BenchmarkCasePlan(
        shape={
            "input": (batch, input_size),
            "hx": (batch, hidden_size),
            "w_ih": (3 * hidden_size, input_size),
            "w_hh": (3 * hidden_size, hidden_size),
        },
        params={"bias": True},
        builder_args=(batch, input_size, hidden_size),
    )


def _build_inputs_fn(plan, dtype, device):
    batch, input_size, hidden_size = plan.builder_args
    inp = generate_tensor_input((batch, input_size), dtype, device)
    hx = generate_tensor_input((batch, hidden_size), dtype, device)
    w_ih = generate_tensor_input((3 * hidden_size, input_size), dtype, device)
    w_hh = generate_tensor_input((3 * hidden_size, hidden_size), dtype, device)
    b_ih = generate_tensor_input((3 * hidden_size,), dtype, device)
    b_hh = generate_tensor_input((3 * hidden_size,), dtype, device)
    return (inp, hx, w_ih, w_hh, b_ih, b_hh, {})


def _resolve_native():
    return torch.ops.aten.gru_cell


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(export_contract("gru_cell", DTYPES, _resolve_native, _inputs))
