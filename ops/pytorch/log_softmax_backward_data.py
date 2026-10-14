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


"""log_softmax_backward_data: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_log_softmax.py:80 (test_log_softmax_backward_data)
Native expression: torch._log_softmax_backward_data
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml GenericBenchmark2DOnly or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def log_softmax_backward_data_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    output = torch._log_softmax(inp, dim=-1, half_to_float=False)
    grad_output = torch.randn_like(output)
    yield (grad_output, output, -1, dtype)


def _resolve_native():
    return torch._log_softmax_backward_data


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = log_softmax_backward_data_input_fn

globals().update(
    export_contract("log_softmax_backward_data", DTYPES, _resolve_native, _inputs)
)
