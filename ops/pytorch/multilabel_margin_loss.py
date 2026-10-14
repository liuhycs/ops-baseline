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


"""multilabel_margin_loss: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_multilabel_margin_loss.py:131 (test_multilabel_margin_loss)
Native expression: torch.ops.aten.multilabel_margin_loss
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import BenchmarkCasePlan, case_inputs, export_contract

_REDUCTIONS = [0, 1, 2]

_DTYPES = [torch.float32, torch.float16]
_DTYPES.extend([torch.bfloat16, torch.float64])  # Pinned NVIDIA capabilities.


def _descriptor_dims(shape):
    if isinstance(shape, (list, tuple)):
        dims = shape
    elif isinstance(shape, dict):
        dims = shape.get("input")
    else:
        raise ValueError(
            f"descriptor must be a list of extents or a dict with 'input', got {shape!r}"
        )
    if not isinstance(dims, (list, tuple)):
        raise ValueError(f"descriptor must be a list of extents, got {dims!r}")
    if any(
        (isinstance(extent, bool) or not isinstance(extent, int) for extent in dims)
    ):
        raise ValueError(f"descriptor extents must be ints, got {tuple(dims)}")
    if len(dims) > 2:
        raise ValueError(
            f"multilabel_margin_loss expects a rank<=2 descriptor, got {dims}"
        )
    if any((extent < 0 for extent in dims)):
        raise ValueError(f"negative extent in descriptor {dims}")
    if dims and dims[-1] == 0:
        raise ValueError(f"descriptor {dims} has an empty class axis")
    return tuple(dims)


def _n_classes(dims):
    return dims[-1] if dims else 1


def _active_classes(n_classes, reduction):
    if reduction == 0:
        return 1
    if reduction == 1:
        return max(1, n_classes // 2)
    return n_classes


def _case_fn(shape, dtype):
    del dtype
    dims = _descriptor_dims(shape)
    n_classes = _n_classes(dims)
    for reduction in _REDUCTIONS:
        active = _active_classes(n_classes, reduction)
        yield BenchmarkCasePlan(
            shape={"input": list(dims), "target": list(dims)},
            params={"reduction": reduction, "target_active": active},
            builder_args=(dims, reduction, active),
        )


def _make_target(dims, active, device):
    target = torch.full(dims, -1, dtype=torch.int64, device=device)
    if target.numel() == 0:
        return target
    n_classes = dims[-1] if dims else 1
    rows = (
        target.reshape(1, n_classes)
        if target.dim() < 2
        else target.reshape(-1, n_classes)
    )
    n_active = max(0, min(active, n_classes))
    if n_active:
        ids = torch.arange(
            n_classes - n_active, n_classes, dtype=torch.int64, device=device
        ).flip(0)
        rows[..., :n_active] = ids
    return target


def _build_inputs_fn(plan, dtype, device):
    dims, reduction, active = plan.builder_args
    inp = torch.randn(dims, dtype=dtype, device=device)
    target = _make_target(dims, active, device)
    return (inp, target, reduction)


def _resolve_native():
    return torch.ops.aten.multilabel_margin_loss


DTYPES = _DTYPES

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(
    export_contract("multilabel_margin_loss", DTYPES, _resolve_native, _inputs)
)
