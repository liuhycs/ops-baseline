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


"""contiguous: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_contiguous.py:35 (test_contiguous)
Native expression: torch.Tensor.contiguous
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml Benchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_fn(shape, dtype, device):
    if dtype in [torch.float16, torch.float32, torch.bfloat16]:
        inp = torch.randn(shape, dtype=dtype, device=device)
    else:
        inp = torch.randint(
            low=-10000, high=10000, size=shape, dtype=dtype, device="cpu"
        ).to(device)
    inp = inp[::2]
    yield (inp,)


def _resolve_native():
    return torch.Tensor.contiguous


DTYPES = [torch.float16, torch.float32, torch.bfloat16] + [torch.int16, torch.int32]

_inputs = _input_fn

globals().update(export_contract("contiguous", DTYPES, _resolve_native, _inputs))
