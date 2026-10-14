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


"""_ctc_loss_out: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test__ctc_loss.py:82 (test_perf__ctc_loss_out)
Native expression: torch.ops.aten._ctc_loss.out
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml _ctc_loss_out or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

CTC_DTYPES = [torch.float32, torch.float64]


def ctc_loss_internal_input_fn(shape, dtype, device):
    t_steps, batch, classes, max_target = shape
    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = raw.log_softmax(-1).to(dtype)
    targets = torch.randint(
        1, classes, (batch, max_target), dtype=torch.long, device=device
    )
    input_lengths = [t_steps] * batch
    target_lengths = [max_target] * batch
    yield (log_probs, targets, input_lengths, target_lengths, {"blank": 0})


def ctc_loss_internal_out_input_fn(shape, dtype, device):
    """Same inputs as the base variant, plus the two preallocated outputs."""
    t_steps, batch, classes, max_target = shape
    for item in ctc_loss_internal_input_fn(shape, dtype, device):
        *args, kwargs = item
        out0 = torch.empty(batch, dtype=dtype, device=device)
        out1 = torch.empty(
            batch, t_steps, 2 * max_target + 1, dtype=dtype, device=device
        )
        yield (*args, {**kwargs, "out0": out0, "out1": out1})


def _resolve_native():
    return torch.ops.aten._ctc_loss.out


DTYPES = CTC_DTYPES

_inputs = ctc_loss_internal_out_input_fn

globals().update(export_contract("_ctc_loss_out", DTYPES, _resolve_native, _inputs))
