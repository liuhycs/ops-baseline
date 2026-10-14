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


"""_cudnn_init_dropout_state: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test__cudnn_init_dropout_state.py:85 (test__cudnn_init_dropout_state)
Native expression: torch.ops.aten._cudnn_init_dropout_state
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import BenchmarkCasePlan, case_inputs, export_contract

_STATE_SHAPE = {"state": "cudnn_dropout_rng_state"}


def _case_fn(descriptor, dtype):
    del dtype
    dropout, train, seed = descriptor
    yield BenchmarkCasePlan(
        shape=dict(_STATE_SHAPE),
        params={"dropout": dropout, "train": train, "seed": seed},
        builder_args=(dropout, train, seed),
    )


def _build_inputs_fn(plan, dtype, device):
    dropout, train, seed = plan.builder_args
    return (dropout, train, seed, {"dtype": dtype, "device": device})


def _resolve_native():
    return torch.ops.aten._cudnn_init_dropout_state


DTYPES = [torch.uint8]

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(
    export_contract("_cudnn_init_dropout_state", DTYPES, _resolve_native, _inputs)
)
