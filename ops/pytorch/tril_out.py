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


"""tril_out: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_tril.py:86 (test_tril_out_transposed)
Native expression: torch.tril
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml GenericBenchmarkExcluse1D or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import generate_tensor_input, export_contract


def _tril_out_transposed_input_fn(shape, dtype, device):
    inp = generate_tensor_input(shape, dtype, device)
    out_shape = (*shape[:-2], shape[-1], shape[-2])
    out = torch.empty(out_shape, dtype=dtype, device=device).transpose(-2, -1)
    yield (inp, 0, {"out": out})


def _resolve_native():
    return torch.tril


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _tril_out_transposed_input_fn

globals().update(export_contract("tril_out", DTYPES, _resolve_native, _inputs))
