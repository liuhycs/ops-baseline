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


"""_aminmax_out: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_aminmax.py:101 (test__aminmax_out)
Native expression: torch.ops.aten._aminmax.out
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml UnaryReductionBenchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def _aminmax_out_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    min_out = torch.empty((), dtype=cur_dtype, device=device)
    max_out = torch.empty((), dtype=cur_dtype, device=device)
    yield (inp, {"out0": min_out, "out1": max_out})


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        yield from _aminmax_out_input_fn(shape, dtype, device)


def _resolve_native():
    return torch.ops.aten._aminmax.out


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(export_contract("_aminmax_out", DTYPES, _resolve_native, _inputs))
