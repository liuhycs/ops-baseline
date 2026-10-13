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


"""clamp_tensor: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_clamp.py:81 (test_clamp_tensor)
Native expression: torch.clamp
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml Benchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def _clamp_tensor_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    mini = generate_tensor_input(shape, cur_dtype, device)
    maxi = generate_tensor_input(shape, cur_dtype, device)
    yield (inp, {"min": mini, "max": maxi})


def _resolve_native():
    return torch.clamp


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _clamp_tensor_input_fn

globals().update(export_contract("clamp_tensor", DTYPES, _resolve_native, _inputs))
