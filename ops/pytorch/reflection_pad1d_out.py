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


"""reflection_pad1d_out: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_reflection_pad1d.py:83 (test_reflection_pad1d_out)
Native expression: torch.ops.aten.reflection_pad1d.out
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_fn_out(config, dtype, device):
    shape, padding = config
    x = torch.randn(shape, dtype=dtype, device=device)
    out_shape = list(shape)
    out_shape[-1] = out_shape[-1] + padding[0] + padding[1]
    out = torch.empty(tuple(out_shape), dtype=dtype, device=device)
    yield (x, list(padding), {"out": out})


def _input_iter(shape, dtype, device):
    for config in [shape]:
        yield from _input_fn_out(config, dtype, device)


def _resolve_native():
    return torch.ops.aten.reflection_pad1d.out


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(
    export_contract("reflection_pad1d_out", DTYPES, _resolve_native, _inputs)
)
