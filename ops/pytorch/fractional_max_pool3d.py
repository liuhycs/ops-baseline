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


"""fractional_max_pool3d: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_fractional_max_pool3d.py:62 (test_fractional_max_pool3d)
Native expression: torch.nn.functional.fractional_max_pool3d
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def fractional_max_pool3d_input_fn(shape, dtype, device):
    inp = generate_tensor_input(shape, dtype, device)
    yield (
        inp,
        {
            "kernel_size": 2,
            "output_size": (shape[2] // 2, shape[3] // 2, shape[4] // 2),
        },
    )


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        yield from fractional_max_pool3d_input_fn(shape, dtype, device)


def _resolve_native():
    return torch.nn.functional.fractional_max_pool3d


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(
    export_contract("fractional_max_pool3d", DTYPES, _resolve_native, _inputs)
)
