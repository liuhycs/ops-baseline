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


"""log_softmax_out: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_log_softmax.py:60 (test_log_softmax_out)
Native expression: torch._log_softmax
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml GenericBenchmarkExcluse1D or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def log_softmax_out_input_fn(shape, dtype, device):
    inp = generate_tensor_input(shape, dtype, device)
    out = torch.empty_like(inp)
    if len(shape) > 1:
        yield (inp, 1, False, {"out": out})
    else:
        yield (inp, 0, False, {"out": out})


def _resolve_native():
    return torch._log_softmax


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = log_softmax_out_input_fn

globals().update(export_contract("log_softmax_out", DTYPES, _resolve_native, _inputs))
