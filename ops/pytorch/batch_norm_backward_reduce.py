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


"""batch_norm_backward_reduce: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_batch_norm_backward_reduce_perf.py:53 (test_batch_norm_backward_reduce)
Native expression: torch.batch_norm_backward_reduce
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml NormBenchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def batch_norm_backward_reduce_input_fn(shape, dtype, device):
    C = shape[1]
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    inp = torch.randn(shape, dtype=dtype, device=device)
    mean = torch.randn(C, dtype=torch.float32, device=device)
    invstd = torch.randn(C, dtype=torch.float32, device=device).abs() + 0.1
    weight = torch.randn(C, dtype=dtype, device=device)
    yield (grad_output, inp, mean, invstd, weight, True, True, True)


def _resolve_native():
    return torch.batch_norm_backward_reduce


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = batch_norm_backward_reduce_input_fn

globals().update(
    export_contract("batch_norm_backward_reduce", DTYPES, _resolve_native, _inputs)
)
