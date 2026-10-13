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


"""nansum: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_nansum.py:39 (test_benchmark_nansum)
Native expression: torch.nansum
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml UnaryReductionBenchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

DTYPES = [torch.float16, torch.float32, torch.bfloat16] + [torch.int8, torch.uint8]


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        if dtype.is_floating_point:
            x = torch.randn(shape, dtype=dtype, device=device) * 10
            mask = torch.rand(shape, device=device) > 0.7
            x[mask] = float("nan")
        else:
            info = torch.iinfo(dtype)
            x = torch.randint(
                info.min, info.max, shape, dtype=torch.int64, device="cpu"
            ).to(device, dtype)
        ndim = x.ndim
        yield (x,)
        if ndim >= 2:
            yield (x, -1)
            yield (x, 0)
        if ndim >= 3:
            yield (x, 1)


def _resolve_native():
    return torch.nansum


DTYPES = DTYPES

_inputs = _input_iter

globals().update(export_contract("nansum", DTYPES, _resolve_native, _inputs))
