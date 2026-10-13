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


"""reflection_pad3d_out: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_reflection_pad3d.py:88 (test_reflection_pad3d_out)
Native expression: torch.ops.aten.reflection_pad3d.out
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_fn_out(config, dtype, device):
    shape, padding = config
    x = torch.randn(shape, dtype=dtype, device=device)
    pad_l, pad_r, pad_t, pad_b, pad_f, pad_ba = padding
    D_out = x.shape[-3] + pad_f + pad_ba
    H_out = x.shape[-2] + pad_t + pad_b
    W_out = x.shape[-1] + pad_l + pad_r
    out_shape = (*x.shape[:-3], D_out, H_out, W_out)
    out = torch.empty(out_shape, dtype=dtype, device=device)
    yield (x, list(padding), {"out": out})


def _input_iter(shape, dtype, device):
    for config in [shape]:
        yield from _input_fn_out(config, dtype, device)


def _resolve_native():
    return torch.ops.aten.reflection_pad3d.out


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(
    export_contract("reflection_pad3d_out", DTYPES, _resolve_native, _inputs)
)
