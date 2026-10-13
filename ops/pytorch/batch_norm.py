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


"""batch_norm: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_batch_norm.py:78 (test_batch_norm)
Native expression: torch.batch_norm
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml NormBenchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def input_fn(shape, dtype, device):
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


def _resolve_native():
    return torch.batch_norm


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = input_fn

globals().update(export_contract("batch_norm", DTYPES, _resolve_native, _inputs))
