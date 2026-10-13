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


"""router_gemm: PyTorch baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_router_gemm.py:87 (test_perf_router_gemm)
Native expression: torch_router_gemm
Constructor state was explicitly lowered; explicit source set_shapes.
Core source parameters and filters are preserved.
"""

import torch

from _kernelgen import export_contract


def torch_router_gemm(x, weight):
    return torch.mm(x, weight.t()).to(torch.float32)


def _input_iter(shape, dtype, device):
    for m, n, k in [shape]:
        x = torch.randn((m, k), dtype=torch.bfloat16, device=device)
        weight = torch.randn((n, k), dtype=torch.bfloat16, device=device)
        yield (x, weight)


DTYPES = [torch.bfloat16]


def _resolve_native():
    return torch_router_gemm


_inputs = _input_iter

globals().update(export_contract("router_gemm", DTYPES, _resolve_native, _inputs))
