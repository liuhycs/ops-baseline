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


"""any_dims: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_any.py:62 (test_any_dims)
Native expression: torch.any
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml GenericBenchmarkExcluse1D or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def any_dims_input_fn(shape, dtype, device):
    if dtype == torch.bool:
        inp = torch.randint(0, 2, shape, dtype=dtype, device=device)
    else:
        inp = torch.randn(shape, dtype=dtype, device=device)
    yield (inp, {"dim": [0, 1]})


def _resolve_native():
    return torch.any


DTYPES = [torch.float16, torch.float32, torch.bfloat16] + [torch.bool]

_inputs = any_dims_input_fn

globals().update(export_contract("any_dims", DTYPES, _resolve_native, _inputs))
