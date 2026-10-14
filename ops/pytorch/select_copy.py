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


"""select_copy: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_select_copy.py:157 (test_select_copy)
Native expression: torch.ops.aten.select_copy
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

_BARE_AXES = {}

_DTYPES = [
    dtype
    for dtype in [torch.float16, torch.float32, torch.bfloat16]
    if dtype != torch.bfloat16 or True
]


def _as_int(value, name):
    """Numeric view of dim/index, used only to validate the request. bool is a
    legal int argument for this schema, so it is accepted here."""
    if isinstance(value, bool):
        return int(value)
    if not isinstance(value, int):
        raise ValueError(f"select_copy {name} must be an integer, got {value!r}")
    return value


def _checked_shape(shape):
    """Reject metadata no select can be built from, before any allocation."""
    if not isinstance(shape, (tuple, list)) or not shape:
        raise ValueError(f"select_copy needs a rank >= 1 shape, got {shape!r}")
    for extent in shape:
        if isinstance(extent, bool) or not isinstance(extent, int) or extent < 0:
            raise ValueError(
                f"shape extents must be non-negative integers, got {shape!r}"
            )
    return tuple(shape)


def _validated_axis(shape, dim, index):
    """Validate one axis request against normalized coordinates and return the
    caller's dim/index unchanged, so an explicit negative or bool request is the
    value the operator actually receives instead of a rewritten positive one."""
    norm_dim = _as_int(dim, "dim")
    norm_index = _as_int(index, "index")
    norm_dim = norm_dim + len(shape) if norm_dim < 0 else norm_dim
    if not 0 <= norm_dim < len(shape):
        raise ValueError(f"dim {dim!r} is out of range for shape {shape}")
    extent = shape[norm_dim]
    norm_index = norm_index + extent if norm_index < 0 else norm_index
    if not 0 <= norm_index < extent:
        raise ValueError(f"index {index!r} is out of range for dim {dim!r} of {shape}")
    return (dim, index)


def _derived_axis(shape):
    """Axis for a caller-supplied bare shape that is not a table entry. Only a
    nonempty axis can be indexed, so an all-zero shape is rejected before any
    allocation instead of yielding a plan that cannot be built."""
    candidates = [axis for axis, extent in enumerate(shape) if extent > 0]
    if not candidates:
        raise ValueError(f"select_copy needs a nonempty axis, got shape {shape}")
    axis = max(candidates, key=lambda candidate: shape[candidate])
    return (axis, shape[axis] // 2)


def _case_axes(entry):
    """Resolve one shape entry to (shape, [(dim, index), ...])."""
    if (
        isinstance(entry, (tuple, list))
        and entry
        and isinstance(entry[0], (tuple, list))
    ):
        if len(entry) != 3:
            raise ValueError(
                f"select_copy shape entries are (shape, dim, index), got {entry!r}"
            )
        shape = _checked_shape(entry[0])
        return (shape, [_validated_axis(shape, entry[1], entry[2])])
    shape = _checked_shape(entry)
    if shape in _BARE_AXES:
        axes = [_validated_axis(shape, dim, index) for dim, index in _BARE_AXES[shape]]
        return (shape, axes)
    return (shape, [_derived_axis(shape)])


def _case_fn(entry, dtype):
    del dtype
    shape, axes = _case_axes(entry)
    for dim, index in axes:
        yield BenchmarkCasePlan(
            shape={"input": shape},
            params={"dim": dim, "index": index},
            builder_args=(shape, dim, index),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, dim, index = plan.builder_args
    inp = generate_tensor_input(shape, dtype, device)
    return (inp, dim, index)


def _resolve_native():
    return torch.ops.aten.select_copy


DTYPES = _DTYPES

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(export_contract("select_copy", DTYPES, _resolve_native, _inputs))
