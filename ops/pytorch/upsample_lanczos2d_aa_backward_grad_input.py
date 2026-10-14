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


"""upsample_lanczos2d_aa_backward_grad_input: PyTorch baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_upsample_lanczos2d_aa_backward.py:168 (test_upsample_lanczos2d_aa_backward_grad_input)
Native expression: _composite_reference_out
Constructor state was explicitly lowered; fixed source constructor _configs.
Core source parameters and filters are preserved.
"""

import torch

from _kernelgen import export_contract

import math

from collections import OrderedDict


def _input_iter(shape, dtype, device):
    n, c, ih, iw, oh, ow = shape
    grad = torch.randn((n, c, oh, ow), dtype=dtype, device=device)
    out = torch.empty((n, c, ih, iw), dtype=dtype, device=device)
    yield (grad, (oh, ow), (n, c, ih, iw), False, None, None, {"grad_input": out})


_WEIGHT_CACHE = OrderedDict()


def _weight_matrix(input_size, output_size, device):
    key = (input_size, output_size, str(device))
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        _WEIGHT_CACHE.move_to_end(key)
        return cached
    scale = input_size / output_size
    support = 3.0 * scale if scale >= 1.0 else 3.0
    invscale = 1.0 / scale if scale >= 1.0 else 1.0
    output_index = torch.arange(output_size, dtype=torch.float32)
    input_index = torch.arange(input_size, dtype=torch.float32)
    center = scale * (output_index + 0.5)
    index_min = (center - support + 0.5).to(torch.int64).clamp_min(0)
    index_max = (center + support + 0.5).to(torch.int64).clamp_max(input_size)
    distance = (input_index[None, :] - center[:, None] + 0.5) * invscale
    pix = math.pi * distance
    sinc = torch.where(distance == 0.0, 1.0, torch.sin(pix) / pix)
    sinc_three = torch.where(distance == 0.0, 1.0, torch.sin(pix / 3.0) / (pix / 3.0))
    valid = (
        (input_index[None, :] >= index_min[:, None])
        & (input_index[None, :] < index_max[:, None])
        & (distance.abs() < 3.0)
    )
    weight = torch.where(valid, sinc * sinc_three, 0.0)
    weight /= weight.sum(dim=1, keepdim=True)
    weight = weight.to(device)
    if len(_WEIGHT_CACHE) >= 32:
        _WEIGHT_CACHE.popitem(last=False)
    _WEIGHT_CACHE[key] = weight
    return weight


def _composite_reference(
    grad_output,
    output_size,
    input_size,
    align_corners=False,
    scales_h=None,
    scales_w=None,
):
    n, c, input_h, input_w = input_size
    output_h, output_w = output_size
    weight_w = _weight_matrix(input_w, output_w, grad_output.device)
    weight_h = _weight_matrix(input_h, output_h, grad_output.device)
    temp = torch.einsum("ncho,oi->nchi", grad_output.float(), weight_w)
    result = torch.einsum("oh,ncow->nchw", weight_h, temp)
    return result.to(grad_output.dtype)


def _composite_reference_out(
    grad_output,
    output_size,
    input_size,
    align_corners=False,
    scales_h=None,
    scales_w=None,
    *,
    grad_input,
):
    grad_input.copy_(
        _composite_reference(
            grad_output, output_size, input_size, align_corners, scales_h, scales_w
        )
    )
    return grad_input


DTYPES = [torch.float16, torch.float32]


def _resolve_native():
    return _composite_reference_out


_inputs = _input_iter

globals().update(
    export_contract(
        "upsample_lanczos2d_aa_backward_grad_input", DTYPES, _resolve_native, _inputs
    )
)
