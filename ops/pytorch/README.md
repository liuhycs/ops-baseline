# ops/pytorch

存放 PyTorch 原生算子基准模块（方案 B）。目录结构与契约同 `ops/vllm`
（见 `ops/vllm/__init__.py`）：每个算子一个 `ops/pytorch/<OP_NAME>.py`，导出
`OP_NAME / DTYPES / IS_INPLACE / native() / grid() / build_inputs() / key_shape()`
（复杂算子再加 `config()`）。

与 `ops/vllm`、`ops/sglang` 的区别在于 **native 来源**：这里 `native()` 绑定的是
PyTorch 官方算子（`torch.*` / `torch.nn.functional.*` / `torch.ops.aten.*`），
即 FlagGems 的对照基线侧，而非某个推理框架的自定义 kernel。

算子清单与每个算子的 native 坐标 / shape / 输入构造，以 FlagGems 仓库为来源：

- 清单：`FlagGems/conf/operators.yaml`（`id` / `for` / `labels` / `stage`）
- benchmark：`FlagGems/benchmark/test_<op>.py`（`torch_op=` 即 native，
  `input_fn` 即 `build_inputs` 的依据，`dtypes=` 即 `DTYPES`）
- shape：`FlagGems/benchmark/core_shapes.yaml`（按 `op_name` → 基准类名 → 默认值
  三级回退，见 `benchmark/base.py:set_shapes`）

采集时用 `--repo pytorch` 指向本目录：

    python3 tools/collect_baseline_nvidia.py --repo pytorch [--ops ...] [--device N]

缺省 `--repo vllm` 走 `ops/vllm`。
