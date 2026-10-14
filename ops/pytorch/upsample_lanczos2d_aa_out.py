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


"""upsample_lanczos2d_aa_out: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_upsample_lanczos2d_aa.py:133 (test_upsample_lanczos2d_aa_out)
Native expression: _composite_reference_out
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

_WEIGHT_CACHE = {}


def _weight_matrix(input_size, output_size, device):
    key = (input_size, output_size, str(device))
    if key in _WEIGHT_CACHE:
        return _WEIGHT_CACHE[key]
    scale = input_size / output_size
    support = 3.0 * scale if scale >= 1.0 else 3.0
    invscale = 1.0 / scale if scale >= 1.0 else 1.0
    output_index = torch.arange(output_size, device=device, dtype=torch.float32)
    input_index = torch.arange(input_size, device=device, dtype=torch.float32)
    center = scale * (output_index + 0.5)
    index_min = (center - support + 0.5).to(torch.int64).clamp_min(0)
    index_max = (center + support + 0.5).to(torch.int64).clamp_max(input_size)
    distance = (input_index[None, :] - center[:, None] + 0.5) * invscale
    valid = (
        (input_index[None, :] >= index_min[:, None])
        & (input_index[None, :] < index_max[:, None])
        & (distance.abs() < 3.0)
    )
    weights = torch.where(valid, torch.sinc(distance) * torch.sinc(distance / 3), 0)
    weights /= weights.sum(dim=1, keepdim=True)
    _WEIGHT_CACHE[key] = weights
    return weights


def _composite_reference(
    input, output_size, align_corners=False, scales_h=None, scales_w=None
):
    output_h, output_w = output_size
    weights_w = _weight_matrix(input.shape[-1], output_w, input.device)
    weights_h = _weight_matrix(input.shape[-2], output_h, input.device)
    value = torch.einsum("nchi,oi->ncho", input.float(), weights_w).to(input.dtype)
    return torch.einsum("nciw,oi->ncow", value.float(), weights_h).to(input.dtype)


def _composite_reference_out(
    input, output_size, align_corners=False, scales_h=None, scales_w=None, *, out
):
    out.copy_(
        _composite_reference(input, output_size, align_corners, scales_h, scales_w)
    )
    return out


def _scale(shape):
    return 2.0 if shape[-2] < 64 else 0.5


def _input_fn_out(shape, dtype, device):
    scale = _scale(shape)
    output_size = (int(shape[-2] * scale), int(shape[-1] * scale))
    input = torch.randn(shape, device=device, dtype=dtype)
    out = torch.empty((*shape[:2], *output_size), device=device, dtype=dtype)
    yield (input, output_size, False, None, None, {"out": out})


def _resolve_native():
    return _composite_reference_out


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_fn_out

globals().update(
    export_contract("upsample_lanczos2d_aa_out", DTYPES, _resolve_native, _inputs)
)
