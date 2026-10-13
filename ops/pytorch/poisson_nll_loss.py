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


"""poisson_nll_loss: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_poisson_nll_loss.py:45 (test_poisson_nll_loss)
Native expression: _torch_poisson_nll_loss
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_fn(shape, dtype, device):
    input = torch.randn(shape, dtype=dtype, device=device)
    target = torch.randint(0, 5, shape, device=device).to(dtype)
    yield (input, target)


def _torch_poisson_nll_loss(input, target):
    return torch.ops.aten.poisson_nll_loss(input, target, True, False, 1e-08, 1)


def _resolve_native():
    return _torch_poisson_nll_loss


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_fn

globals().update(export_contract("poisson_nll_loss", DTYPES, _resolve_native, _inputs))
