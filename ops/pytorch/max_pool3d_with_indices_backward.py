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


"""max_pool3d_with_indices_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_max_pool3d_with_indices_backward.py:81 (test_max_pool3d_with_indices_backward)
Native expression: torch.ops.aten.max_pool3d_with_indices_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: upstream fixed get_input_iter loop; first tensor shape identifies geometry.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def max_pool3d_backward_input_fn(shape, dtype, device):
    """Generate inputs for max_pool3d_with_indices_backward benchmark.

    Performs a forward pass to obtain indices, then yields the backward inputs.
    """
    inp = generate_tensor_input(shape, dtype, device)
    params = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "ceil_mode": False,
    }
    output, indices = torch.nn.functional.max_pool3d(inp, return_indices=True, **params)
    grad_output = torch.randn_like(output)
    yield (
        grad_output,
        inp,
        params["kernel_size"],
        params["stride"],
        params["padding"],
        params["dilation"],
        params["ceil_mode"],
        indices,
    )


def _input_iter(shape, dtype, device):
    shapes_5d = [
        (4, 3, 16, 56, 56),
        (8, 64, 8, 28, 28),
        (16, 128, 4, 14, 14),
        (32, 256, 2, 7, 7),
    ]
    for shape in shapes_5d:
        yield from max_pool3d_backward_input_fn(shape, dtype, device)


def _resolve_native():
    return torch.ops.aten.max_pool3d_with_indices_backward


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(
    export_contract(
        "max_pool3d_with_indices_backward", DTYPES, _resolve_native, _inputs
    )
)
