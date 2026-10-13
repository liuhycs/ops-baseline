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


"""_sparse_semi_structured_tile: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test__sparse_semi_structured_tile.py:135 (test__sparse_semi_structured_tile)
Native expression: torch.ops.aten._sparse_semi_structured_tile
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

SSST_DTYPES = [torch.float16] + ([torch.bfloat16] if True else [])

SSST_CALL_FORMS = [
    (None, True),
    (None, False),
    ("", True),
    ("", False),
    ("largest_values_greedy", True),
    ("largest_values_greedy", False),
    ("largest_abs_values_greedy", True),
    ("largest_abs_values_greedy", False),
]


def _validate_shape(shape):
    """Reject custom descriptors that are not valid workloads of this operator.

    The kernel sparsifies 2-D tiles, needs positive extents and fails for a zero
    extent ('CUDA error: invalid configuration argument'), and its result is only
    reproducible at 64-aligned rows and columns, so a descriptor outside that
    geometry cannot be timed meaningfully. An invalid descriptor is reported
    rather than silently replaced by a default workload.
    """
    if not isinstance(shape, (tuple, list)):
        raise ValueError(
            "descriptor %r is not a shape sequence (tuple/list), got %s"
            % (shape, type(shape).__name__)
        )
    if len(shape) != 2:
        raise ValueError(
            "descriptor %r has rank %d; this operator sparsifies 2-D tensors only"
            % (tuple(shape), len(shape))
        )
    rows, cols = shape
    for name, extent in (("rows", rows), ("cols", cols)):
        if isinstance(extent, bool) or not isinstance(extent, int):
            raise ValueError(
                "%s extent must be an int, got %r in %r" % (name, extent, tuple(shape))
            )
        if extent <= 0:
            raise ValueError(
                "%s extent must be positive, got %r in %r"
                % (name, extent, tuple(shape))
            )
        if extent % 64:
            raise ValueError(
                "%s must be a multiple of 64 for a reproducible workload, got %d in %r"
                % (name, extent, tuple(shape))
            )
    return (rows, cols)


def _case_fn(shape, dtype):
    del dtype
    rows, cols = _validate_shape(shape)
    for algorithm, use_cutlass in SSST_CALL_FORMS:
        if not use_cutlass and (rows % 128 or cols % 128):
            continue
        kwargs = {"use_cutlass": use_cutlass}
        if algorithm is not None:
            kwargs["algorithm"] = algorithm
        yield BenchmarkCasePlan(
            shape={"input": tuple(shape)},
            params={
                "algorithm": "omitted" if algorithm is None else algorithm,
                "use_cutlass": use_cutlass,
            },
            builder_args=(tuple(shape), dict(kwargs)),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, kwargs = plan.builder_args
    inp = generate_tensor_input(shape, dtype, device)
    return (inp, dict(kwargs))


def _resolve_native():
    return torch.ops.aten._sparse_semi_structured_tile


DTYPES = SSST_DTYPES

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(
    export_contract("_sparse_semi_structured_tile", DTYPES, _resolve_native, _inputs)
)
