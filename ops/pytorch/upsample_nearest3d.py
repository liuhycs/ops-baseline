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


"""upsample_nearest3d: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_upsample_nearest3d.py:60 (test_upsample_nearest3d)
Native expression: torch._C._nn.upsample_nearest3d
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml UpsampleBenchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def upsample_nearest3d_input_fn(shape, dtype, device):
    batch, channel, height, width = shape
    depth = 4
    width = width // 4
    new_height = height // depth
    real_shape = (batch, channel, depth, new_height, width)
    input = torch.randn(size=real_shape, device=device, dtype=dtype)
    scale_factors = (2.0, 2.0, 2.0)
    output_size = (
        int(depth * scale_factors[0]),
        int(new_height * scale_factors[1]),
        int(width * scale_factors[2]),
    )
    yield (
        {
            "input": input,
            "output_size": output_size,
            "scales_d": None,
            "scales_h": None,
            "scales_w": None,
        },
    )


def _resolve_native():
    return torch._C._nn.upsample_nearest3d


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = upsample_nearest3d_input_fn

globals().update(
    export_contract("upsample_nearest3d", DTYPES, _resolve_native, _inputs)
)
