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


"""fused_add_rms_norm: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_fused_add_rms_norm.py:43 (test_fused_add_rms_norm)
Native expression: torch_op
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml GenericBenchmarkExcluse1D or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    residual = torch.randn(shape, dtype=dtype, device=device)
    layer_shape = (shape[-1],)
    weight = torch.randn(layer_shape, dtype=dtype, device=device)
    yield (inp, residual, layer_shape, weight, 1e-05)


def torch_op(x, residual, layer_shape, weight, eps):
    x = x + residual
    variance = x.pow(2).mean(-1, keepdim=True)
    hidden_states = x * torch.rsqrt(variance + eps)
    return weight * hidden_states


def _resolve_native():
    return torch_op


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_fn

globals().update(
    export_contract("fused_add_rms_norm", DTYPES, _resolve_native, _inputs)
)
