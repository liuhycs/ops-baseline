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


"""matmul_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_matmul_backward.py:64 (test_matmul_backward)
Native expression: matmul_backward_baseline
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def matmul_backward_baseline(grad, self, other, mask):
    """PyTorch baseline using component operations.

    matmul_backward computes gradients for: out = self @ other
    grad_self = grad @ other.T
    grad_other = self.T @ grad
    """
    grad_self = None
    grad_other = None
    if mask[0]:
        grad_self = torch.matmul(grad, other.transpose(-2, -1))
    if mask[1]:
        grad_other = torch.matmul(self.transpose(-2, -1), grad)
    return (grad_self, grad_other)


def _input_iter(shape, dtype, device):
    for m, k, n in [shape]:
        self_t = torch.randn(m, k, dtype=dtype, device=device)
        other_t = torch.randn(k, n, dtype=dtype, device=device)
        grad = torch.randn(m, n, dtype=dtype, device=device)
        yield (grad, self_t, other_t, [True, True])


def _resolve_native():
    return matmul_backward_baseline


DTYPES = [torch.float32, torch.float16]

_inputs = _input_iter

globals().update(export_contract("matmul_backward", DTYPES, _resolve_native, _inputs))
