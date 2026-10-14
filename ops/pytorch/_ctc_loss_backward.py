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


"""_ctc_loss_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test__ctc_loss_backward.py:75 (test_perf__ctc_loss_backward)
Native expression: torch.ops.aten._ctc_loss_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml _ctc_loss_backward or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract

BACKWARD_DTYPES = [torch.float32, torch.float64]


def ctc_loss_backward_input_fn(shape, dtype, device):
    """Yield backward inputs, running the forward pass once to obtain them.

    _ctc_loss_backward consumes neg_log_likelihood and log_alpha from the
    forward pass, so the forward is run here in setup rather than inside the
    timed region.
    """
    t_steps, batch, classes, max_target = shape
    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = raw.log_softmax(-1).to(dtype)
    targets = torch.randint(
        1, classes, (batch, max_target), dtype=torch.long, device=device
    )
    input_lengths = [t_steps] * batch
    target_lengths = [max_target] * batch
    neg_log_likelihood, log_alpha = torch.ops.aten._ctc_loss(
        log_probs, targets, input_lengths, target_lengths
    )
    grad_output = torch.randn_like(neg_log_likelihood)
    yield (
        grad_output,
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        neg_log_likelihood,
        log_alpha,
        {"blank": 0},
    )


def _resolve_native():
    return torch.ops.aten._ctc_loss_backward


DTYPES = BACKWARD_DTYPES

_inputs = ctc_loss_backward_input_fn

globals().update(
    export_contract("_ctc_loss_backward", DTYPES, _resolve_native, _inputs)
)
