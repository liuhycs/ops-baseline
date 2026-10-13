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


"""nanmean_out: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_nanmean.py:61 (test_benchmark_nanmean_out)
Native expression: torch.nanmean
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml UnaryReductionBenchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        x = torch.randn(shape, dtype=dtype, device=device) * 10
        x[torch.rand(shape, device=device) > 0.7] = float("nan")
        yield (x, {"out": torch.empty((), dtype=dtype, device=device)})
        if x.ndim >= 2:
            for dim in (-1, 0):
                out_shape = torch.nanmean(x, dim=dim).shape
                out = torch.empty(out_shape, dtype=dtype, device=device)
                yield (x, dim, {"out": out})
        if x.ndim >= 3:
            out_shape = torch.nanmean(x, dim=1).shape
            out = torch.empty(out_shape, dtype=dtype, device=device)
            yield (x, 1, {"out": out})


def _resolve_native():
    return torch.nanmean


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(export_contract("nanmean_out", DTYPES, _resolve_native, _inputs))
