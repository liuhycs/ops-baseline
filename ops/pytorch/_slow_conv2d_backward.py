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


"""_slow_conv2d_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test__slow_conv2d_backward.py:91 (test__slow_conv2d_backward)
Native expression: torch.ops.aten._slow_conv2d_backward.output_mask
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import (
    BenchmarkCasePlan,
    generate_tensor_input,
    case_inputs,
    export_contract,
)

_FULL_MASK = (True, True, True)


def _case_fn(shape, dtype):
    del dtype
    yield BenchmarkCasePlan(
        shape={"input": shape},
        params={"output_mask": _FULL_MASK},
        builder_args=(shape, 0),
    )


def _build_inputs_fn(plan, dtype, device):
    n, c_in, h, w, c_out, k_h, k_w, stride, padding = plan.builder_args[0]
    h_out = (h + 2 * padding - k_h) // stride + 1
    w_out = (w + 2 * padding - k_w) // stride + 1
    grad_output = generate_tensor_input((n, c_out, h_out, w_out), dtype, device)
    inp = generate_tensor_input((n, c_in, h, w), dtype, device)
    weight = generate_tensor_input((c_out, c_in, k_h, k_w), dtype, device)
    return (
        grad_output,
        inp,
        weight,
        (k_h, k_w),
        (stride, stride),
        (padding, padding),
        plan.params["output_mask"],
    )


def _resolve_native():
    return torch.ops.aten._slow_conv2d_backward.output_mask


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(
    export_contract("_slow_conv2d_backward", DTYPES, _resolve_native, _inputs)
)
