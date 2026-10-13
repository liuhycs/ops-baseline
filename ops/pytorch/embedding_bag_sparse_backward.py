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


"""embedding_bag_sparse_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_embedding_bag_sparse_backward.py:88 (test_embedding_bag_sparse_backward)
Native expression: torch.ops.aten._embedding_bag_sparse_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    for num_bags, embedding_dim, num_weights, samples_per_bag in [shape]:
        num_samples = num_bags * samples_per_bag
        weight = torch.randn(num_weights, embedding_dim, dtype=dtype, device=device)
        indices = torch.randint(
            0, num_weights, (num_samples,), dtype=torch.long, device=device
        )
        offsets = torch.arange(
            0, num_samples + 1, samples_per_bag, dtype=torch.long, device=device
        )[:num_bags]
        output, offset2bag, bag_size, maximum_indices = torch.ops.aten._embedding_bag(
            weight, indices, offsets, False, 0, True, None, False, -1
        )
        grad = torch.randn_like(output)
        yield (
            grad,
            indices,
            offsets,
            offset2bag,
            bag_size,
            num_weights,
            False,
            0,
            None,
            -1,
        )


def _resolve_native():
    return torch.ops.aten._embedding_bag_sparse_backward


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(
    export_contract("embedding_bag_sparse_backward", DTYPES, _resolve_native, _inputs)
)
