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


"""max_pool1d_with_indices: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_max_pool1d_with_indices.py:58 (test_max_pool1d_with_indices)
Native expression: torch.nn.functional.max_pool1d_with_indices
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: upstream fixed get_input_iter loop; first tensor shape identifies geometry.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def max_pool1d_input_fn(shape, dtype, device):
    inp = generate_tensor_input(shape, dtype, device)
    yield (
        inp,
        {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "dilation": 1,
            "ceil_mode": False,
        },
    )


def _input_iter(shape, dtype, device):
    shapes_3d = [
        (16, 64, 1024),
        (32, 128, 512),
        (64, 256, 256),
        (8, 32, 4096),
        (128, 512, 128),
    ]
    for shape in shapes_3d:
        yield from max_pool1d_input_fn(shape, dtype, device)


def _resolve_native():
    return torch.nn.functional.max_pool1d_with_indices


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(
    export_contract("max_pool1d_with_indices", DTYPES, _resolve_native, _inputs)
)
