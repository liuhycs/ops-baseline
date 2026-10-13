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


"""argsort: PyTorch baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_argsort.py:203 (test_argsort)
Native expression: torch.argsort
Constructor state was explicitly lowered; benchmark/core_shapes.yaml GenericBenchmark2DOnly.
Core source parameters and filters are preserved.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def _input_fn(shape, dtype, device):
    if dtype in (torch.int8, torch.uint8):
        low, high = (-128, 128) if dtype == torch.int8 else (0, 256)
        inp = torch.randint(low, high, shape, dtype=dtype, device="cpu").to(device)
    elif dtype == torch.int64:
        inp = torch.randint(-(2**60), 2**60, shape, dtype=dtype, device="cpu").to(
            device
        )
    else:
        inp = generate_tensor_input(shape, dtype, device)
    yield (inp, {"dim": -1, "descending": False})


DTYPES = (
    [torch.int16, torch.int32]
    + [torch.int8, torch.uint8, torch.int64]
    + [torch.float16, torch.float32, torch.bfloat16]
)


def _resolve_native():
    return torch.argsort


_inputs = _input_fn

globals().update(export_contract("argsort", DTYPES, _resolve_native, _inputs))
