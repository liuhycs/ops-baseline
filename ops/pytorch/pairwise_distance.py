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


"""pairwise_distance: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_pairwise_distance.py:53 (test_pairwise_distance)
Native expression: safe_pairwise_distance
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml GenericBenchmark2DOnly or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def pairwise_distance_input_fn(shape, dtype, device):
    inp1 = generate_tensor_input(shape, dtype, device)
    inp2 = generate_tensor_input(shape, dtype, device)
    for p in (float("-inf"), float("inf"), 0.0, 1.0, 2.0, 6.6):
        yield (inp1, inp2, {"p": p})


safe_pairwise_distance = torch.pairwise_distance


def _resolve_native():
    return safe_pairwise_distance


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = pairwise_distance_input_fn

globals().update(export_contract("pairwise_distance", DTYPES, _resolve_native, _inputs))
