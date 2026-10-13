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


"""nll_loss_nd_forward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_nll_loss_nd.py:64 (test_nll_loss_nd_forward)
Native expression: torch.nn.functional.nll_loss
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: upstream fixed get_input_iter loop; first tensor shape identifies geometry.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def nll_loss_nd_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    inp = torch.nn.functional.log_softmax(inp, dim=1)
    target_shape = list(shape)
    del target_shape[1]
    C = shape[1]
    target = torch.randint(0, C, target_shape, dtype=torch.long, device=device)
    yield (inp, target)


def _input_iter(shape, dtype, device):
    shapes = [
        (64, 64),
        (256, 256),
        (10000, 65536),
        (32, 128, 512),
        (64, 64, 4, 8),
        (256, 256, 4, 8),
        (4096, 4096, 4, 8),
        (64, 64, 8, 4, 8),
    ]
    for shape in shapes:
        yield from nll_loss_nd_input_fn(shape, dtype, device)


def _resolve_native():
    return torch.nn.functional.nll_loss


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(
    export_contract("nll_loss_nd_forward", DTYPES, _resolve_native, _inputs)
)
