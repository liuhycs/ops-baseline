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


"""sparse_semi_structured_linear: PyTorch baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_sparse_semi_structured_linear.py:99 (test_sparse_semi_structured_linear)
Native expression: _native
Constructor state was explicitly lowered; source set_shapes; both use_bias constructors.
Core source parameters and filters are preserved.
"""

import torch

from _kernelgen import export_contract


def _native(inp, weight, meta, bias=None):
    return (
        _torch_ref(inp, weight, meta)
        if bias is None
        else _torch_ref_with_bias(inp, weight, meta, bias)
    )


def _input_iter(shape, dtype, device):
    m, k = shape
    for use_bias in (False, True):
        inp = torch.randn((m, k), dtype=dtype, device=device)
        weight, meta = _build_2_4_weight(k, k, dtype, device)
        if use_bias:
            yield (inp, weight, meta, torch.randn((k,), dtype=dtype, device=device))
        else:
            yield (inp, weight, meta)


def _build_2_4_weight(N, K, dtype, device):
    """Build a dense 2:4 weight and its bool meta (N, K//4).

    Each row carries an independent 2:4 pattern, matching the kernel semantics.
    """
    K4 = K // 4
    base_w = torch.randn(N, K, dtype=dtype, device=device)
    choice = torch.randint(0, 2, (N, K4), device=device).bool()
    w = torch.zeros(N, K, dtype=dtype, device=device)
    for g in range(K4):
        keep = choice[:, g]
        w[:, 4 * g] = torch.where(
            keep, base_w[:, 4 * g], torch.zeros_like(keep, dtype=dtype)
        )
        w[:, 4 * g + 1] = torch.where(
            keep, base_w[:, 4 * g + 1], torch.zeros_like(keep, dtype=dtype)
        )
        w[:, 4 * g + 2] = torch.where(
            ~keep, base_w[:, 4 * g + 2], torch.zeros_like(keep, dtype=dtype)
        )
        w[:, 4 * g + 3] = torch.where(
            ~keep, base_w[:, 4 * g + 3], torch.zeros_like(keep, dtype=dtype)
        )
    return (w, choice.to(torch.int8))


def _torch_ref(input, weight, meta):
    """Reference: dense matmul with the 2:4 meta applied to the weight."""
    K4 = weight.shape[1] // 4
    choice = meta.to(torch.bool)
    w = weight.view(weight.shape[0], K4, 4)
    keep = torch.cat(
        [
            choice.unsqueeze(2),
            choice.unsqueeze(2),
            (~choice).unsqueeze(2),
            (~choice).unsqueeze(2),
        ],
        dim=2,
    )
    masked = torch.where(keep, w, torch.zeros_like(w)).reshape(weight.shape[0], -1)
    return torch.matmul(input, masked.t())


def _torch_ref_with_bias(input, weight, meta, bias):
    return _torch_ref(input, weight, meta) + bias


DTYPES = [torch.float16, torch.float32, torch.bfloat16]


def _resolve_native():
    return _native


_inputs = _input_iter

globals().update(
    export_contract("sparse_semi_structured_linear", DTYPES, _resolve_native, _inputs)
)
