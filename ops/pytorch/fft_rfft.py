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


"""fft_rfft: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_fft_rfft.py:175 (test_fft_rfft)
Native expression: torch.ops.aten.fft_rfft
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import BenchmarkCasePlan, case_inputs, export_contract

_RFFT_DTYPES = [torch.float32]
_RFFT_DTYPES.extend([torch.float64])  # Pinned NVIDIA capabilities.


def _is_power_of_two(value):
    return value > 0 and value & value - 1 == 0


def _next_power_of_two(value):
    return 1 << (value - 1).bit_length()


def _call_kwargs(n, dim, norm):
    kwargs = {}
    if n is not None:
        kwargs["n"] = n
    if dim != -1:
        kwargs["dim"] = dim
    if norm is not None:
        kwargs["norm"] = norm
    return kwargs


def _call_params(n, dim, norm):
    return {
        "n": "default" if n is None else n,
        "dim": "default" if dim == -1 else dim,
        "norm": "default" if norm is None else norm,
    }


def _validate_shape(shape):
    sizes = tuple(shape) if isinstance(shape, (tuple, list)) else ()
    if len(sizes) < 1:
        raise ValueError(
            "fft_rfft needs a descriptor of rank >= 1; a 0-dim input is rejected."
        )
    for size in sizes:
        if isinstance(size, bool) or not isinstance(size, int):
            raise ValueError(f"fft_rfft shape entries must be integers, got {size!r}.")
        if size < 1:
            raise ValueError(
                f"fft_rfft cannot transform a zero-length axis, got {sizes}."
            )
    return sizes


def _validate_call(sizes, n, dim, norm):
    if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n < 1):
        raise ValueError(f"fft_rfft requires an integer n >= 1, got {n!r}.")
    if (
        isinstance(dim, bool)
        or not isinstance(dim, int)
        or (not -len(sizes) <= dim < len(sizes))
    ):
        raise ValueError(f"fft_rfft dim {dim!r} is out of range for rank {len(sizes)}.")
    if norm not in (None, "backward", "forward", "ortho"):
        raise ValueError(
            f"fft_rfft norm must be None/backward/forward/ortho, got {norm!r}."
        )


def _plan_rows(shape):
    natural = shape[-1]
    return [
        (None, -1, None),
        (max(1, natural // 2), -1, "ortho"),
        (_next_power_of_two(natural), -1, "forward"),
        (natural + 1, -1, None),
        (4, 0, "ortho"),
    ]


def _transform_length(sizes, n, dim):
    return sizes[dim] if n is None else n


def _case_fn(shape, dtype):
    sizes = _validate_shape(shape)
    for n, dim, norm in _plan_rows(sizes):
        _validate_call(sizes, n, dim, norm)
        if dtype is torch.float16 and (
            not _is_power_of_two(_transform_length(sizes, n, dim))
        ):
            continue
        yield BenchmarkCasePlan(
            shape={"input": sizes},
            params=_call_params(n, dim, norm),
            builder_args=(sizes, n, dim, norm),
        )


def _make_input(shape, dtype, device):
    return torch.randn(shape, dtype=dtype, device=device)


def _build_inputs_fn(plan, dtype, device):
    shape, n, dim, norm = plan.builder_args
    inp = _make_input(shape, dtype, device)
    return (inp, _call_kwargs(n, dim, norm))


def _resolve_native():
    return torch.ops.aten.fft_rfft


DTYPES = _RFFT_DTYPES

_inputs = lambda shape, dtype, device: case_inputs(
    _case_fn, _build_inputs_fn, shape, dtype, device
)

globals().update(export_contract("fft_rfft", DTYPES, _resolve_native, _inputs))
