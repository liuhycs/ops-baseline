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


"""weight_norm_differentiable_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_weight_norm_differentiable_backward.py:46 (test_weight_norm_differentiable_backward_benchmark)
Native expression: torch.ops.aten._weight_norm_differentiable_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    for shape, dim in [shape]:
        broadcast_shape = [1] * len(shape)
        broadcast_shape[dim] = shape[dim]
        grad_w = torch.randn(shape, dtype=dtype, device=device)
        saved_v = torch.randn(shape, dtype=dtype, device=device)
        saved_g = torch.randn(broadcast_shape, dtype=dtype, device=device)
        _, saved_norms = torch._weight_norm_interface(saved_v, saved_g, dim)
        # Torch 2.8 CUDA returns fp32 saved norms for fp16 inputs, while its
        # meta implementation returns fp16. Match CUDA metadata without
        # changing the real CUDA input construction.
        if device == "meta" and dtype == torch.float16:
            saved_norms = saved_norms.to(torch.float32)
        yield (grad_w, saved_v, saved_g, saved_norms, dim)


def _resolve_native():
    return torch.ops.aten._weight_norm_differentiable_backward


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(
    export_contract(
        "weight_norm_differentiable_backward", DTYPES, _resolve_native, _inputs
    )
)
