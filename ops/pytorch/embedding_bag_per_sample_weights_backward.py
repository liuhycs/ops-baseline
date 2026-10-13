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


"""embedding_bag_per_sample_weights_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_embedding_bag_per_sample_weights_backward.py:48 (test_embedding_bag_per_sample_weights_backward)
Native expression: torch.ops.aten._embedding_bag_per_sample_weights_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    for Batch, num_bags, bag_size, embedding_dim in [shape]:
        num_samples = Batch * bag_size
        num_embeddings = 64
        grad = torch.randn(num_bags, embedding_dim, dtype=dtype, device=device)
        weight = torch.randn(num_embeddings, embedding_dim, dtype=dtype, device=device)
        indices = torch.randint(
            0, num_embeddings, (num_samples,), device=device, dtype=torch.long
        )
        offsets = torch.tensor(
            [i * bag_size for i in range(num_bags + 1)], device=device, dtype=torch.long
        )
        offset2bag = torch.tensor(
            [i // bag_size for i in range(num_samples)], device=device, dtype=torch.long
        )
        yield (grad, weight, indices, offsets, offset2bag, 0, -1)


def _resolve_native():
    return torch.ops.aten._embedding_bag_per_sample_weights_backward


DTYPES = [torch.float32]

_inputs = _input_iter

globals().update(
    export_contract(
        "embedding_bag_per_sample_weights_backward", DTYPES, _resolve_native, _inputs
    )
)
