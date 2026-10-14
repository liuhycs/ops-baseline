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


"""linalg_multi_dot: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_linalg_multi_dot.py:77 (test_linalg_multi_dot_benchmark)
Native expression: torch.ops.aten.linalg_multi_dot.default
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    for dimensions in [shape]:
        tensors = [
            torch.randn(
                dimensions[index], dimensions[index + 1], dtype=dtype, device=device
            )
            for index in range(len(dimensions) - 1)
        ]
        yield (tensors,)


def _resolve_native():
    return torch.ops.aten.linalg_multi_dot.default


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(export_contract("linalg_multi_dot", DTYPES, _resolve_native, _inputs))
