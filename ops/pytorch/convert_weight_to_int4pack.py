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


"""convert_weight_to_int4pack: PyTorch baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_convert_weight_to_int4pack.py:134 (test_convert_weight_to_int4pack)
Native expression: torch._convert_weight_to_int4pack
Constructor state was explicitly lowered; explicit source set_shapes.
Core source parameters and filters are preserved.
"""

import torch

from _kernelgen import export_contract

INNER_K_TILES = (2, 4, 8)


def _convert_weight_input_fn(shape, device):
    """Yield paired inputs for native and gems from one int4 weight source.

    Native aten::_convert_weight_to_int4pack takes uint8 [n, k/2] (two int4
    values per byte) and produces a Marlin-style tiled int32 tensor; the
    FlagGems implementation takes int32 [n, k] (one int4 value per element)
    and produces byte-pair-packed uint8 [n, k/2]. Both inputs are derived
    from the same logical int4 weight [n, k] so the comparison is meaningful,
    and the conversion happens here, outside the timed region.
    """
    n, k_half = shape
    k = k_half * 2
    int4 = torch.randint(0, 16, (n, k), dtype=torch.int32, device=device)
    base_input = (int4[:, 1::2] << 4 | int4[:, 0::2]).to(torch.uint8)
    gems_input = int4
    for innerKTiles in INNER_K_TILES:
        yield ((base_input, innerKTiles), (gems_input, innerKTiles))


def _input_iter(shape, dtype, device):
    # Source _run_metric passes only the native half of the paired factory.
    for native_args, _gems_args in _convert_weight_input_fn(shape, device):
        yield native_args


DTYPES = [torch.uint8]


def _resolve_native():
    return torch._convert_weight_to_int4pack


_inputs = _input_iter

globals().update(
    export_contract("convert_weight_to_int4pack", DTYPES, _resolve_native, _inputs)
)
