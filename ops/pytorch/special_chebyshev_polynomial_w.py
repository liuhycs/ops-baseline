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


"""special_chebyshev_polynomial_w: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_special_chebyshev_polynomial_w.py:70 (test_special_chebyshev_polynomial_w)
Native expression: torch.special.chebyshev_polynomial_w
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def _input_iter(shape, dtype, device):
    n_val = 3
    for shape in [shape]:
        x = generate_tensor_input(shape, dtype, device)
        x = x * 2 - 1
        n = torch.tensor(n_val, dtype=torch.int64, device=device)
        yield (x, n)


def _resolve_native():
    return torch.special.chebyshev_polynomial_w


DTYPES = [torch.float32]

_inputs = _input_iter

globals().update(
    export_contract("special_chebyshev_polynomial_w", DTYPES, _resolve_native, _inputs)
)
