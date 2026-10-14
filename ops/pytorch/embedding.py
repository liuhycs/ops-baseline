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


"""embedding: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_embedding.py:59 (test_embedding)
Native expression: torch.nn.functional.embedding
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml EmbeddingBenchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def embedding_input_fn(shape, dtype, device):
    num_embeddings, embedding_dim = shape
    indices = torch.randint(0, num_embeddings, (num_embeddings,), device=device)
    weight = torch.randn((num_embeddings, embedding_dim), device=device, dtype=dtype)
    yield ({"input": indices, "weight": weight},)


def _resolve_native():
    return torch.nn.functional.embedding


DTYPES = [torch.float32, torch.float16]

_inputs = embedding_input_fn

globals().update(export_contract("embedding", DTYPES, _resolve_native, _inputs))
