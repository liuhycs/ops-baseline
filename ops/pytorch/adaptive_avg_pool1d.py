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


"""adaptive_avg_pool1d: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_adaptive_avg_pool1d.py:49 (test_perf_adaptive_avg_pool1d)
Native expression: torch.ops.aten.adaptive_avg_pool1d
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: upstream fixed get_input_iter loop; first tensor shape identifies geometry.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def adaptive_avg_pool1d_input_fn(shape, dtype, device):
    inp = generate_tensor_input(shape, dtype, device)
    yield (inp, [7])


def _input_iter(shape, dtype, device):
    shapes_3d = [(4, 3, 32), (8, 64, 56), (16, 128, 28), (1, 64, 224), (4, 128, 112)]
    for shape in shapes_3d:
        yield from adaptive_avg_pool1d_input_fn(shape, dtype, device)


def _resolve_native():
    return torch.ops.aten.adaptive_avg_pool1d


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(
    export_contract("adaptive_avg_pool1d", DTYPES, _resolve_native, _inputs)
)
