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


"""batch_norm_impl_index_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_batch_norm_impl_index_backward.py:67 (test_batch_norm_impl_index_backward)
Native expression: torch.ops.aten._batch_norm_impl_index_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml NormBenchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _batch_norm_impl_index_backward_input_fn(shape, dtype, device):
    """Yield argument tuples matching the ``_batch_norm_impl_index_backward``
    schema: ``(impl_index, input, grad_output, weight, running_mean,
    running_var, save_mean, save_var_transform, train, eps, output_mask,
    reservedSpace)``.

    The native backend (``impl_index == 0``) is exercised here because it
    supports every floating dtype, mirroring the real CUDA dispatcher's
    ``!train``-or-``impl_index == 0`` fallback path.
    """
    C = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    grad_output = torch.randn_like(inp)
    weight = torch.randn((C,), dtype=dtype, device=device)
    running_mean = torch.zeros((C,), dtype=dtype, device=device)
    running_var = torch.ones((C,), dtype=dtype, device=device)
    save_mean = torch.randn((C,), dtype=torch.float32, device=device)
    save_var = torch.abs(torch.randn((C,), dtype=torch.float32, device=device)) + 0.1
    impl_index = 0
    train = True
    eps = 1e-05
    output_mask = [True, True, True]
    reserved_space = torch.empty(0, dtype=torch.uint8, device=device)
    yield (
        impl_index,
        inp,
        grad_output,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_var,
        train,
        eps,
        output_mask,
        reserved_space,
    )


def _resolve_native():
    return torch.ops.aten._batch_norm_impl_index_backward


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _batch_norm_impl_index_backward_input_fn

globals().update(
    export_contract("batch_norm_impl_index_backward", DTYPES, _resolve_native, _inputs)
)
