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


"""type_as: PyTorch baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_type_as.py:222 (test_type_as)
Native expression: torch.ops.aten.type_as
Constructor state was explicitly lowered; source TYPE_AS_SHAPES and default _OTHER_DTYPE pairing.
Core source parameters and filters are preserved.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def _input_iter(shape, dtype, device):
    target = torch.float16 if dtype == torch.float32 else torch.float32
    inp = generate_tensor_input(shape, dtype, device)
    other = torch.zeros((1,), dtype=target, device=device)
    yield (inp, other)


DTYPES = [torch.float16, torch.float32, torch.bfloat16]


def _resolve_native():
    return torch.ops.aten.type_as


_inputs = _input_iter

globals().update(export_contract("type_as", DTYPES, _resolve_native, _inputs))
