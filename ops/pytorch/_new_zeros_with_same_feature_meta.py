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


"""_new_zeros_with_same_feature_meta: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test__new_zeros_with_same_feature_meta.py:71 (test__new_zeros_with_same_feature_meta)
Native expression: torch.ops.aten._new_zeros_with_same_feature_meta
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


def _case_fn(shape, dtype):
    del dtype
    self_shape, other_shape, self_num_batch_dims = shape
    yield BenchmarkCasePlan(
        shape={"self": self_shape, "other": other_shape},
        params={"self_num_batch_dims": self_num_batch_dims},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    self_shape, other_shape, self_num_batch_dims = plan.builder_args[0]
    self_inp = generate_tensor_input(self_shape, dtype, device)
    other_inp = generate_tensor_input(other_shape, dtype, device)
    return (self_inp, other_inp, {"self_num_batch_dims": self_num_batch_dims})


def _resolve_native():
    return torch.ops.aten._new_zeros_with_same_feature_meta


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(
    export_contract(
        "_new_zeros_with_same_feature_meta", DTYPES, _resolve_native, _inputs
    )
)
