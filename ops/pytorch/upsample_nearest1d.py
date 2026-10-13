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


"""upsample_nearest1d: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_upsample_nearest1d.py:49 (test_upsample_nearest1d)
Native expression: torch._C._nn.upsample_nearest1d
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml UpsampleBenchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_fn(shape, dtype, device):
    batch, channel, height, width = shape
    length = height * width
    input = torch.randn((batch, channel, length), device=device, dtype=dtype)
    scale_factors = 2
    output_size = int(length * scale_factors)
    yield ({"input": input, "output_size": (output_size,), "scales": None},)


def _resolve_native():
    return torch._C._nn.upsample_nearest1d


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_fn

globals().update(
    export_contract("upsample_nearest1d", DTYPES, _resolve_native, _inputs)
)
