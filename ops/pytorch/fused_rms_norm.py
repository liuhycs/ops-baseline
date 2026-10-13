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


"""fused_rms_norm: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_fused_rms_norm.py:44 (test_fused_rms_norm)
Native expression: _torch_fused_rms_norm
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml GenericBenchmark2DOnly or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _torch_fused_rms_norm(x, normalized_shape, weight=None, eps=1e-05):
    """Reference implementation for comparison."""
    upcast_x = x.to(torch.float32)
    variance = upcast_x.pow(2).mean(-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    hidden_states = upcast_x * inv_rms
    hidden_states = hidden_states.to(x.dtype)
    if weight is not None:
        return (weight * hidden_states, inv_rms.squeeze(-1))
    return (hidden_states, inv_rms.squeeze(-1))


def fused_rms_norm_input_fn(shape, dtype, device):
    M, N = shape
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn(N, dtype=dtype, device=device)
    yield (inp, (N,), weight)


def _resolve_native():
    return _torch_fused_rms_norm


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = fused_rms_norm_input_fn

globals().update(export_contract("fused_rms_norm", DTYPES, _resolve_native, _inputs))
