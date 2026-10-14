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


"""repeat_interleave_tensor: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_repeat_interleave.py:111 (test_repeat_interleave_tensor)
Native expression: torch.repeat_interleave
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml repeat_interleave_tensor or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract, repeat_interleave_repeats


def repeat_interleave_tensor_input_fn(shape, dtype, device):
    repeats = repeat_interleave_repeats(shape, dtype, device)
    yield (repeats,)


def _resolve_native():
    return torch.repeat_interleave


DTYPES = [torch.int32]

_inputs = repeat_interleave_tensor_input_fn

globals().update(
    export_contract("repeat_interleave_tensor", DTYPES, _resolve_native, _inputs)
)
