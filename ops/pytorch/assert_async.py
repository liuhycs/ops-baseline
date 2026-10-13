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


"""assert_async: PyTorch baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_assert_async.py:56 (test_assert_async)
Native expression: torch._assert_async
Constructor state was explicitly lowered; explicit source set_shapes.
Core source parameters and filters are preserved.
"""

import torch

from _kernelgen import export_contract


def _input_fn(shape, dtype, device):
    if dtype == torch.bool:
        tensor = torch.ones(shape, dtype=dtype, device=device)
    else:
        tensor = torch.ones(shape, dtype=dtype, device=device)
    msg = "Benchmark assert_async"
    yield (tensor, msg)


DTYPES = [torch.bool, torch.int32, torch.float32, torch.float16, torch.bfloat16]


def _resolve_native():
    return torch._assert_async


_inputs = _input_fn

globals().update(export_contract("assert_async", DTYPES, _resolve_native, _inputs))
