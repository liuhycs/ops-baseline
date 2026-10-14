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


"""cat: PyTorch baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_cat.py:65 (test_cat)
Native expression: torch.cat
Constructor state was explicitly lowered; benchmark/core_shapes.yaml Benchmark; source MAX_ELEMENTS=536870912.
Core source parameters and filters are preserved.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def _input_fn(shape, dtype, device):
    inp1 = generate_tensor_input(shape, dtype, device)
    inp2 = generate_tensor_input(shape, dtype, device)
    inp3 = generate_tensor_input(shape, dtype, device)
    yield ([inp1, inp2, inp3], {"dim": 0})


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        yield from _input_fn(shape, dtype, device)


DTYPES = [torch.float16, torch.float32, torch.bfloat16] + [torch.int16, torch.int32]


def _resolve_native():
    return torch.cat


_inputs = _input_iter

globals().update(export_contract("cat", DTYPES, _resolve_native, _inputs))
