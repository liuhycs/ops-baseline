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


"""max_unpool3d: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_max_unpool3d.py:45 (test_max_unpool3d)
Native expression: max_unpool3d_ref
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def max_unpool3d_ref(input, indices, kernel_size, stride, padding, output_size):
    return torch.nn.functional.max_unpool3d(
        input,
        indices,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_size=output_size,
    )


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        x = torch.randn(shape, dtype=dtype, device=device)
        pool = torch.nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        output, indices = pool(x)
        yield (output, indices, 2, 2, 0, shape[2:])


def _resolve_native():
    return max_unpool3d_ref


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(export_contract("max_unpool3d", DTYPES, _resolve_native, _inputs))
