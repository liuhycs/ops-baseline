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


"""slice_copy_out: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_slice_copy.py:104 (test_slice_copy_out)
Native expression: torch_op
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml GenericBenchmark2DOnly or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _slice_params(shape, dim):
    """Slice the middle half of ``dim`` with step 2."""
    dim_size = shape[dim]
    start = dim_size // 4
    end = dim_size - dim_size // 4
    step = 2
    return (start, end, step)


_BENCH_DIMS = (1, 0)


def torch_op(inp, dim, start, end, step, out=None):
    return torch.ops.aten.slice_copy.Tensor_out(inp, dim, start, end, step, out=out)


def _input_fn_out(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    for dim in _BENCH_DIMS:
        start, end, step = _slice_params(shape, dim)
        out_shape = list(shape)
        out_shape[dim] = max(0, (end - start + step - 1) // step)
        out = torch.empty(out_shape, dtype=dtype, device=device)
        yield (inp, dim, start, end, step, {"out": out})


def _resolve_native():
    return torch_op


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_fn_out

globals().update(export_contract("slice_copy_out", DTYPES, _resolve_native, _inputs))
