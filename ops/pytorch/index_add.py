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


"""index_add: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_index_add.py:83 (test_index_add)
Native expression: torch.index_add
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml TensorSelectBenchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def unpack_index_add_case(case):
    if isinstance(case[0], (list, tuple)):
        return (tuple(case[0]), case[1])
    shape = tuple(case)
    return (shape, 0 if len(shape) == 1 else 1)


def index_add_input_fn(case, dtype, device):
    shape, dim = unpack_index_add_case(case)
    inp = torch.randn(shape, dtype=dtype, device=device)
    src_shape = list(inp.shape)
    index_max = src_shape[dim]
    index_len = index_max // 2 if index_max >= 2 else 1
    index = torch.randperm(index_len, device=device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=device)
    yield (inp, dim, index, src)


def _resolve_native():
    return torch.index_add


DTYPES = [torch.float16, torch.bfloat16, torch.float32]

_inputs = index_add_input_fn

globals().update(export_contract("index_add", DTYPES, _resolve_native, _inputs))
