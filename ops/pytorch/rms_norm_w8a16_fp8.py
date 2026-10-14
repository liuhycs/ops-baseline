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


"""rms_norm_w8a16_fp8: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_rms_norm_w8a16_fp8.py:96 (test_rms_norm_w8a16_fp8)
Native expression: _torch_rms_norm
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

GROUP_SIZE = 128


def _torch_rms_norm(x, shape, weight_fp8, weight_scale, weight_ref):
    return torch.nn.functional.rms_norm(x, shape, weight_ref, eps=1e-05)


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        _, n = shape
        x = torch.randn(shape, dtype=dtype, device=device)
        weight = torch.randn(n, dtype=dtype, device=device)
        grouped = weight.float().reshape(-1, GROUP_SIZE)
        scales = (grouped.abs().amax(-1) / 448.0).clamp(min=1e-08).to(dtype)
        weight_fp8 = (
            (grouped / scales.float()[:, None])
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
            .reshape(n)
        )
        weight_ref = (
            (weight_fp8.float().reshape(-1, GROUP_SIZE) * scales.float()[:, None])
            .reshape(n)
            .to(dtype)
        )
        yield (x, (n,), weight_fp8, scales, weight_ref)


def _resolve_native():
    return _torch_rms_norm


DTYPES = [torch.bfloat16]

_inputs = _input_iter

globals().update(
    export_contract("rms_norm_w8a16_fp8", DTYPES, _resolve_native, _inputs)
)
