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


"""pad_sequence: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_pad_sequence.py:57 (test_pad_sequence)
Native expression: torch.nn.utils.rnn.pad_sequence
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: upstream fixed get_input_iter loop; first tensor shape identifies geometry.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def _input_iter(shape, dtype, device):
    cases = [
        [(64, 8), (32, 8)],
        [(128, 16), (256, 16), (100, 16), (200, 16)],
        [(64, 8), (32, 8), (48, 8), (16, 8), (40, 8), (56, 8), (24, 8), (8, 8)],
        [(512, 768), (256, 768), (400, 768), (128, 768)],
        [(1024, 1024), (768, 1024), (512, 1024), (256, 1024)],
        [(2048, 4096), (1024, 4096), (1536, 4096), (512, 4096)],
        [
            (512, 512),
            (384, 512),
            (256, 512),
            (128, 512),
            (448, 512),
            (320, 512),
            (192, 512),
            (64, 512),
            (400, 512),
            (300, 512),
            (200, 512),
            (100, 512),
            (256, 512),
            (128, 512),
            (96, 512),
            (32, 512),
        ],
        [(512, 1024)] * 32,
    ]
    for shapes in cases:
        sequences = [generate_tensor_input(shape, dtype, device) for shape in shapes]
        yield (sequences, {"batch_first": False, "padding_value": 0.0})
        yield (sequences, {"batch_first": True, "padding_value": 0.0})


def _resolve_native():
    return torch.nn.utils.rnn.pad_sequence


DTYPES = [torch.float32, torch.float16, torch.bfloat16]

_inputs = _input_iter

globals().update(export_contract("pad_sequence", DTYPES, _resolve_native, _inputs))
