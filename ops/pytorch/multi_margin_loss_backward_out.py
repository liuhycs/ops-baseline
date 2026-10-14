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


"""multi_margin_loss_backward_out: PyTorch baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_multi_margin_loss.py:168 (test_multi_margin_loss_backward_out)
Native expression: torch.ops.aten.multi_margin_loss_backward.grad_input
Constructor state was explicitly lowered; source loss DEFAULT_SHAPES; all 2x2x3 p/weighted/reduction combinations.
Core source parameters and filters are preserved.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    n, c = shape
    for p in (1, 2):
        for weighted in (False, True):
            for reduction in (0, 1, 2):
                inp = torch.randn((n, c), dtype=dtype, device=device)
                target = torch.randint(0, c, (n,), dtype=torch.int64, device=device)
                weight = (
                    torch.randn((c,), dtype=dtype, device=device) if weighted else None
                )
                args = (inp, target, p, 1.0, weight, reduction)
                output_shape = (n,) if reduction == 0 else ()
                grad = torch.randn(output_shape, dtype=dtype, device=device)
                args = (grad, *args)
                out = (
                    torch.empty_like(inp)
                    if True
                    else torch.empty(output_shape, dtype=dtype, device=device)
                )
                yield (*args, {"grad_input" if True else "out": out})


DTYPES = [torch.float16, torch.float32, torch.bfloat16] + [torch.float64]


def _resolve_native():
    return torch.ops.aten.multi_margin_loss_backward.grad_input


_inputs = _input_iter

globals().update(
    export_contract("multi_margin_loss_backward_out", DTYPES, _resolve_native, _inputs)
)
