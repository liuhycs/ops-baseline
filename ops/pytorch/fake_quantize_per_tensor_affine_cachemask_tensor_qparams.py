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


"""fake_quantize_per_tensor_affine_cachemask_tensor_qparams: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.py:42 (test_fake_quantize_per_tensor_affine_cachemask_tensor_qparams)
Native expression: torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml Benchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

QUANT_MIN = -128

QUANT_MAX = 127


def _fake_quant_input_fn(shape, cur_dtype, device):
    inp = torch.randn(shape, dtype=cur_dtype, device=device)
    scale = torch.tensor(0.1, dtype=torch.float32, device=device)
    zero_point = torch.tensor(0, dtype=torch.int32, device=device)
    fake_quant_enabled = torch.tensor(1, dtype=torch.int64, device=device)
    yield (inp, scale, zero_point, fake_quant_enabled, QUANT_MIN, QUANT_MAX)


def _resolve_native():
    return torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _fake_quant_input_fn

globals().update(
    export_contract(
        "fake_quantize_per_tensor_affine_cachemask_tensor_qparams",
        DTYPES,
        _resolve_native,
        _inputs,
    )
)
