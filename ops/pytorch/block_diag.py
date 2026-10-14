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


"""block_diag: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_block_diag.py:41 (test_block_diag)
Native expression: _torch_block_diag
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _torch_block_diag(tensors):
    """Wrapper to call torch.block_diag with a list of tensors."""
    return torch.block_diag(*tensors)


def _input_iter(shape, dtype, device):
    for num_blocks, block_size in [shape]:
        blocks = [
            torch.randn((block_size, block_size), dtype=dtype, device=device)
            for _ in range(num_blocks)
        ]
        yield (blocks,)


def _resolve_native():
    return _torch_block_diag


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(export_contract("block_diag", DTYPES, _resolve_native, _inputs))
