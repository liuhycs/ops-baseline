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


"""rwkv_ka_fusion: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_rwkv_ka_fusion.py:56 (test_rwkv_ka_fusion)
Native expression: torch_rwkv_ka
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml RWKVBenchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def rwkv_ka_fusion_input_fn(shape, dtype, device):
    T = shape[0]
    H = 8
    N = 64
    C = H * N
    k = torch.randn(T, C, dtype=dtype, device=device)
    kk = torch.randn(C, dtype=dtype, device=device)
    a = torch.randn(T, C, dtype=dtype, device=device)
    ka = torch.randn(C, dtype=dtype, device=device)
    yield (k, kk, a, ka, H, N)


def torch_rwkv_ka(k, kk, a, ka, H, N):
    T, C = k.shape
    assert (
        C == H * N and kk.shape == (C,) and (a.shape == (T, C)) and (ka.shape == (C,))
    )
    o_kk = torch.nn.functional.normalize((k * kk).view(T, H, N), dim=-1, p=2.0).view(
        T, H * N
    )
    o_k = k * (1 + (a - 1) * ka)
    o_kka = o_kk * a
    return (o_k, o_kk, o_kka)


def _resolve_native():
    return torch_rwkv_ka


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = rwkv_ka_fusion_input_fn

globals().update(export_contract("rwkv_ka_fusion", DTYPES, _resolve_native, _inputs))
