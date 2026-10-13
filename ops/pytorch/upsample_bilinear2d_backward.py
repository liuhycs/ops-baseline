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


"""upsample_bilinear2d_backward: PyTorch baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_upsample_bilinear2d_backward.py:200 (test_upsample_bilinear2d_backward)
Native expression: torch.ops.aten.upsample_bilinear2d_backward.default
Constructor state was explicitly lowered; source CASES, original layouts/scales and both align_corners values.
Core source parameters and filters are preserved.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    label, input_size, output_size, sh, sw, layout = shape
    n, c, ih, iw = input_size
    oh, ow = output_size
    for align_corners in (False, True):
        grad = torch.randn(
            (n, c, oh, ow * 2 if layout == "strided" else ow),
            dtype=dtype,
            device=device,
        )
        if layout == "strided":
            grad = grad[..., ::2]
        if layout == "channels_last":
            grad = grad.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
        yield (grad, output_size, input_size, align_corners, sh, sw)


DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16]


def _resolve_native():
    return torch.ops.aten.upsample_bilinear2d_backward.default


_inputs = _input_iter

globals().update(
    export_contract("upsample_bilinear2d_backward", DTYPES, _resolve_native, _inputs)
)
