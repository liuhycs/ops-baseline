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


"""cumprod: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_cumprod.py:63 (test_cumprod)
Native expression: torch_cumprod
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml GenericBenchmark2DOnly or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

CUMPROD_DTYPES = (
    [torch.float16, torch.float32, torch.bfloat16]
    + [torch.bool]
    + [torch.int16, torch.int32]
    + [torch.int8, torch.uint8, torch.int64]
)


def _make_input(shape, dtype, device):
    if dtype in [torch.float16, torch.float32, torch.bfloat16]:
        return torch.empty(shape, dtype=dtype, device=device).uniform_(0.99, 1.01)
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.int8, device="cpu").to(
            device, dtype=dtype
        )
    if dtype is torch.uint8:
        return torch.randint(0, 4, shape, dtype=dtype, device="cpu").to(device)
    return torch.randint(-3, 4, shape, dtype=dtype, device="cpu").to(device)


def input_fn(shape, dtype, device):
    inp = _make_input(shape, dtype, device)
    yield (inp, 1)


def torch_cumprod(inp, dim):
    if "nvidia" == "ascend" and inp.dtype is torch.bool:
        return torch.cumprod(inp.to(torch.uint8), dim)
    return torch.cumprod(inp, dim)


def _resolve_native():
    return torch_cumprod


DTYPES = CUMPROD_DTYPES

_inputs = input_fn

globals().update(export_contract("cumprod", DTYPES, _resolve_native, _inputs))
