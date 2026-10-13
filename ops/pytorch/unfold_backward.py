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


"""unfold_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_unfold.py:56 (test_unfold_backward)
Native expression: torch.ops.aten.unfold_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_fn(config, dtype, device):
    input_sizes, dim, size, step = config
    d = dim % len(input_sizes)
    num_windows = (input_sizes[d] - size) // step + 1
    grad_shape = (
        list(input_sizes[:d]) + [num_windows] + list(input_sizes[d + 1 :]) + [size]
    )
    grad_in = torch.randn(grad_shape, dtype=dtype, device=device)
    yield (grad_in, list(input_sizes), dim, size, step)


def _input_iter(shape, dtype, device):
    for config in [shape]:
        yield from _input_fn(config, dtype, device)


def _resolve_native():
    return torch.ops.aten.unfold_backward


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(export_contract("unfold_backward", DTYPES, _resolve_native, _inputs))
