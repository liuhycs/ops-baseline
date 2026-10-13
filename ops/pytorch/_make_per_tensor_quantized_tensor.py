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


"""_make_per_tensor_quantized_tensor: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test__make_per_tensor_quantized_tensor.py:113 (test__make_per_tensor_quantized_tensor)
Native expression: torch.ops.aten._make_per_tensor_quantized_tensor
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import BenchmarkCasePlan, case_inputs, export_contract

MAKE_PERTENSOR_INPUT_DTYPES = [torch.uint8, torch.int8, torch.int32]

SCALE = 0.1

ZERO_POINT = 0


def _make_input(shape, dtype, device):
    info = torch.iinfo(dtype)
    return torch.randint(info.min, info.max, shape, dtype=dtype, device=device)


def _case_fn(shape, dtype):
    del dtype
    yield BenchmarkCasePlan(
        shape={"input": shape},
        params={"scale": SCALE, "zero_point": ZERO_POINT},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = _make_input(shape, dtype, device)
    return (
        inp,
        {"scale": plan.params["scale"], "zero_point": plan.params["zero_point"]},
    )


def _resolve_native():
    return torch.ops.aten._make_per_tensor_quantized_tensor


DTYPES = MAKE_PERTENSOR_INPUT_DTYPES

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(
    export_contract(
        "_make_per_tensor_quantized_tensor", DTYPES, _resolve_native, _inputs
    )
)
