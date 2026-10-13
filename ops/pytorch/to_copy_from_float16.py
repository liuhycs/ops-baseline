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


"""to_copy_from_float16: PyTorch baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_to_copy.py:98 (test_to_copy)
Native expression: torch.ops.aten._to_copy
Constructor state was explicitly lowered; source YAML default shapes and all source/destination loop pairs; source dtype in operator identity.
Core source parameters and filters are preserved.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    src_dtype = torch.float16
    if src_dtype.is_floating_point:
        inp = torch.randn(shape, dtype=src_dtype, device=device)
    else:
        inp = torch.randint(-100, 100, shape, dtype=src_dtype, device=device)
    yield (inp, {"dtype": dtype})


DTYPES = [
    torch.bfloat16,
    torch.float64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
]


def _resolve_native():
    return torch.ops.aten._to_copy


_inputs = _input_iter

globals().update(
    export_contract("to_copy_from_float16", DTYPES, _resolve_native, _inputs)
)
