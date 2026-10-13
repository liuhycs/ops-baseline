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


"""_cudnn_ctc_loss: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test__cudnn_ctc_loss.py:147 (test__cudnn_ctc_loss)
Native expression: torch.ops.aten._cudnn_ctc_loss
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

_CASE_VARIANTS = (("cycle", False), ("repeat", False), ("cycle", True))


def _validate_descriptor(descriptor):
    """Validate and return a (T, N, C, S) ctc-loss descriptor.

    bool is an int subclass, so it is rejected explicitly. C == 1 is a native
    BAD_PARAM and would divide by zero in the target builder. S > T is allowed:
    native accepts it, and repeated labels can need more than S frames.
    """
    if not isinstance(descriptor, (tuple, list)) or len(descriptor) != 4:
        raise ValueError(f"ctc-loss descriptor must be (T, N, C, S): {descriptor!r}")
    for name, value in zip(("T", "N", "C", "S"), descriptor):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"descriptor {name} must be an int, got {value!r}")
    time_steps, batch, num_classes, target_length = descriptor
    if time_steps < 1:
        raise ValueError(f"T must be >= 1, got {time_steps}")
    if batch < 1:
        raise ValueError(f"N must be >= 1, got {batch}")
    if num_classes < 2:
        raise ValueError(
            f"C must be >= 2 (C == 1 is a native BAD_PARAM), got {num_classes}"
        )
    if target_length < 0:
        raise ValueError(f"S must be >= 0, got {target_length}")
    return (time_steps, batch, num_classes, target_length)


def _case_fn(shape, dtype):
    del dtype
    time_steps, batch, num_classes, target_length = _validate_descriptor(shape)
    for pattern, deterministic in _CASE_VARIANTS:
        yield BenchmarkCasePlan(
            shape={"input": (time_steps, batch, num_classes)},
            params={
                "target_length": target_length,
                "pattern": pattern,
                "deterministic": deterministic,
                "blank": 0,
                "zero_infinity": True,
            },
            builder_args=(
                time_steps,
                batch,
                num_classes,
                target_length,
                pattern,
                deterministic,
            ),
        )


def _build_inputs_fn(plan, dtype, device):
    time_steps, batch, num_classes, target_length, pattern, deterministic = (
        plan.builder_args
    )
    log_probs = generate_tensor_input((time_steps, batch, num_classes), dtype, device)
    total = target_length * batch
    if pattern == "cycle":
        targets = torch.arange(total, dtype=torch.int32) % (num_classes - 1) + 1
    else:
        targets = torch.ones(total, dtype=torch.int32)
    return (
        log_probs,
        targets,
        {
            "input_lengths": [time_steps] * batch,
            "target_lengths": [target_length] * batch,
            "blank": plan.params["blank"],
            "deterministic": plan.params["deterministic"],
            "zero_infinity": plan.params["zero_infinity"],
        },
    )


def _resolve_native():
    return torch.ops.aten._cudnn_ctc_loss


DTYPES = [torch.float32]

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(export_contract("_cudnn_ctc_loss", DTYPES, _resolve_native, _inputs))
