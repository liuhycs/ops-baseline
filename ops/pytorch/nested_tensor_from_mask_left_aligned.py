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


"""nested_tensor_from_mask_left_aligned: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_nested_tensor_from_mask_left_aligned.py:52 (test_nested_tensor_from_mask_left_aligned)
Native expression: torch._nested_tensor_from_mask_left_aligned
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: upstream fixed get_input_iter loop; first tensor shape identifies geometry.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    shapes = [(8, 128), (64, 256), (512, 512), (2048, 1024)]
    for N, L in shapes:
        D = 64
        t = torch.randn(N, L, D, dtype=dtype, device=device)
        lengths = torch.randint(0, L + 1, (N,), device=device)
        mask = torch.arange(L, device=device)[None, :] < lengths[:, None]
        yield (t, mask)


def _resolve_native():
    return torch._nested_tensor_from_mask_left_aligned


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(
    export_contract(
        "nested_tensor_from_mask_left_aligned", DTYPES, _resolve_native, _inputs
    )
)
