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


"""bitwise_right_shift: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_bitwise_right_shift.py:40 (test_bitwise_right_shift)
Native expression: torch.bitwise_right_shift
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml Benchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        inp1 = generate_tensor_input(shape, dtype, device)
        shift_amount = torch.randint(0, 8, shape, dtype=dtype, device="cpu").to(device)
        yield (inp1, shift_amount)


def _resolve_native():
    return torch.bitwise_right_shift


DTYPES = [torch.int16, torch.int32]

_inputs = _input_iter

globals().update(
    export_contract("bitwise_right_shift", DTYPES, _resolve_native, _inputs)
)
