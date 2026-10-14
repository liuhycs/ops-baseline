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


"""native_batch_norm_legit_no_training: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test__native_batch_norm_legit_no_training.py:59 (test_native_batch_norm_legit_no_training)
Native expression: torch_native_batch_norm_legit_no_training
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml NormBenchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

import torch.nn.functional as F


def native_batch_norm_legit_no_training_input_fn(shape, dtype, device):
    C = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn((C,), dtype=dtype, device=device)
    bias = torch.randn((C,), dtype=dtype, device=device)
    running_mean = torch.randn((C,), dtype=dtype, device=device)
    running_var = torch.abs(torch.randn((C,), dtype=dtype, device=device)) + 0.1
    momentum = 0.1
    eps = 1e-05
    yield (inp, weight, bias, running_mean, running_var, momentum, eps)


def torch_native_batch_norm_legit_no_training(
    inp, weight, bias, running_mean, running_var, momentum, eps
):
    return F.batch_norm(
        inp, running_mean, running_var, weight, bias, training=False, eps=eps
    )


def _resolve_native():
    return torch_native_batch_norm_legit_no_training


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = native_batch_norm_legit_no_training_input_fn

globals().update(
    export_contract(
        "native_batch_norm_legit_no_training", DTYPES, _resolve_native, _inputs
    )
)
