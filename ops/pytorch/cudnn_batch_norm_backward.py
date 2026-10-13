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


"""cudnn_batch_norm_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_cudnn_batch_norm_backward.py:95 (test_cudnn_batch_norm_backward)
Native expression: torch.ops.aten.cudnn_batch_norm_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml NormBenchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def batchnorm_input_fn(shape, dtype, device):
    C = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn((C,), dtype=dtype, device=device)
    bias = torch.randn((C,), dtype=dtype, device=device)
    running_mean = None
    running_var = None
    training = True
    momentum = 0.1
    eps = 1e-05
    cudnn_enabled = True
    yield (
        inp,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
        cudnn_enabled,
    )


def cudnn_batch_norm_backward_input_fn(shape, dtype, device):
    for forward_args in batchnorm_input_fn(shape, dtype, device):
        inp, weight, bias, running_mean, running_var, training, _, eps, _ = forward_args
        grad_output = torch.randn_like(inp)
        channels = weight.shape[0] if weight is not None else inp.shape[1]
        if weight is None:
            weight = torch.ones(channels, dtype=dtype, device=device)
        if bias is None:
            bias = torch.ones(channels, dtype=dtype, device=device)
        inp_f32 = inp.to(torch.float32)
        weight_f32 = weight.to(torch.float32)
        bias_f32 = bias.to(torch.float32)
        out, save_mean, save_var, reserve = torch.ops.aten.cudnn_batch_norm(
            inp_f32, weight_f32, bias_f32, None, None, training, eps, False
        )
        save_mean = save_mean.to(dtype)
        save_var = save_var.to(dtype)
        reserve = reserve.to(dtype)
        yield (
            inp,
            grad_output,
            weight,
            running_mean,
            running_var,
            save_mean,
            save_var,
            eps,
            reserve,
        )


def _resolve_native():
    return torch.ops.aten.cudnn_batch_norm_backward


DTYPES = [torch.float32]

_inputs = cudnn_batch_norm_backward_input_fn

globals().update(
    export_contract("cudnn_batch_norm_backward", DTYPES, _resolve_native, _inputs)
)
