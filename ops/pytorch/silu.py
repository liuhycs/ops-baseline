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

"""silu baseline（方案 B，pytorch）。

native：torch.nn.functional.silu(input, inplace=False) -> Tensor。
    源码核对：torch 2.8 的 torch/nn/functional.py:2358（`def silu(input, inplace=False)`），
    非 inplace 分支调用 torch._C._nn.silu(input)，即 CUDA 上的 aten silu kernel。
    单张张量入参、另分配输出，不原地写回 -> IS_INPLACE=False。

算子清单来源：FlagGems conf/operators.yaml（id=silu，labels=[aten, pointwise,
    nn.functional]，stage=beta 5.3）。

输入构造复刻 FlagGems benchmark/test_silu.py 的 test_silu：
    UnaryPointwiseBenchmark(op_name="silu", torch_op=torch.nn.functional.silu,
                            dtypes=consts.FLOAT_DTYPES)
    该基准类经 benchmark/utils.py generate_tensor_input 构造输入：
        inp = torch.randn(shape, dtype=dtype, device=device)
    调用形态 silu(inp)（单位置参数，inplace 缺省 False）。

shape 来源：FlagGems benchmark/core_shapes.yaml 的「UnaryPointwiseBenchmark」类条目
    （test_silu.py 与 op_name="silu" 均未在 yaml 中单列，按 set_shapes 的
    op_name -> 基准类名 回退命中该类条目）：
        shapes = [[1073741824], [64, 64], [4096, 4096], [64, 512, 512], [1024, 1024, 1024]]
"""

import torch

OP_NAME = "silu"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]  # consts.FLOAT_DTYPES
IS_INPLACE = False

# core_shapes.yaml「UnaryPointwiseBenchmark」的 shapes，原样保留（含 1D 项）。
_SHAPES = [
    (1073741824,),
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
]


def native():
    """解析 torch.nn.functional.silu；解析不到返回 None。"""
    try:
        from torch.nn.functional import silu as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = list(binding["shape"])
    inp = torch.randn(shape, dtype=dtype, device=device)
    return (inp,), {}


def key_shape(binding):
    return list(binding["shape"])
