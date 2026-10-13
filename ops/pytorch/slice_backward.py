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


"""slice_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_slice_backward.py:89 (test_slice_backward)
Native expression: torch.ops.aten.slice_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml Benchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_fn(shape, dtype, device):
    dim = 0 if len(shape) == 1 else 1
    start = 0
    end = shape[dim]
    step = 2
    size = shape[dim]
    start = start % size
    end = end % (size + 1)
    if end < start:
        end, start = (start, end)
    elif end == start:
        end = size
    slice_len = (end - start + step - 1) // step
    valid_shape = list(shape)
    valid_shape[dim] = slice_len
    grad_output = torch.randn(valid_shape, dtype=dtype, device=device)
    yield (grad_output, shape, dim, start, end, step)


def _resolve_native():
    return torch.ops.aten.slice_backward


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_fn

globals().update(export_contract("slice_backward", DTYPES, _resolve_native, _inputs))
