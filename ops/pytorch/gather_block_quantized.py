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


"""gather_block_quantized: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_gather_block_quantized.py:65 (test_gather_block_quantized)
Native expression: _aten_gather_block_quantized
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _aten_gather_block_quantized(quantized_data, scales, indices, block_size):
    """Reference composed from native aten ops (no native block-quantized
    gather exists in PyTorch): expand per-block scales to element level,
    then dequantize."""
    scale_expanded = scales.repeat_interleave(block_size)
    scale_expanded = scale_expanded[: quantized_data.numel()]
    return quantized_data.float() * scale_expanded


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        n_elements = shape[0]
        block_size = 128
        n_blocks = (n_elements + block_size - 1) // block_size
        quantized_data = torch.randint(
            -100, 100, shape, dtype=torch.int8, device=device
        )
        scales = torch.rand(n_blocks, dtype=torch.float32, device=device) * 2 + 0.5
        yield (quantized_data, scales, None, block_size)


def _resolve_native():
    return _aten_gather_block_quantized


DTYPES = [torch.float32]

_inputs = _input_iter

globals().update(
    export_contract("gather_block_quantized", DTYPES, _resolve_native, _inputs)
)
