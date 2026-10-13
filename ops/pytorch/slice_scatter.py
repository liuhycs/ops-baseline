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


"""slice_scatter: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_slice_scatter.py:73 (test_slice_scatter)
Native expression: torch.slice_scatter
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml TensorSelectBenchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_fn(shape, dtype, device):
    dim = 0 if len(shape) == 1 else 1
    start = 0
    end = shape[dim]
    step = 2
    inp = torch.randn(shape, dtype=dtype, device=device)
    range = end - start
    valid_shape = list(inp.shape)
    if end < start:
        range = 0
    elif end - start > valid_shape[dim]:
        range = valid_shape[dim]
        start = 0
        end = valid_shape[dim]
    valid_shape[dim] = (range + (step - 1)) // step
    src = torch.randn(valid_shape, dtype=dtype, device=device)
    yield (inp, src, dim, start, end, step)


def _resolve_native():
    return torch.slice_scatter


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_fn

globals().update(export_contract("slice_scatter", DTYPES, _resolve_native, _inputs))
