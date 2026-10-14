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


"""fused_adam: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_fused_adam.py:95 (test_fused_adam)
Native expression: torch_op
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def fused_adam_input_fn(shape, dtype, device):
    param = torch.randn(shape, dtype=dtype, device=device)
    grad = torch.randn(shape, dtype=dtype, device=device)
    exp_avg = torch.zeros(shape, dtype=dtype, device=device)
    exp_avg_sq = torch.zeros(shape, dtype=dtype, device=device)
    max_exp_avg_sq = torch.zeros(shape, dtype=dtype, device=device)
    state_step = torch.tensor([1], dtype=torch.long, device=device)
    yield (param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, state_step)


def torch_op(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, state_step):
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-08
    step = state_step.item()
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad
    corrected_exp_avg = exp_avg / bias_correction1
    corrected_exp_avg_sq = exp_avg_sq / bias_correction2
    param = param - lr * corrected_exp_avg / (torch.sqrt(corrected_exp_avg_sq) + eps)
    return param


def _resolve_native():
    return torch_op


DTYPES = [torch.float32]

_inputs = fused_adam_input_fn

globals().update(export_contract("fused_adam", DTYPES, _resolve_native, _inputs))
