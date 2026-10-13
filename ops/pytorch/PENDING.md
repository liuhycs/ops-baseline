# kernelgen-dev 待处理项

来源固定为 FlagGems `kernelgen-dev` 的 `9d57a23b4386bd669152a935fe88a96db907da7c`。
逐条原生调用、文件、测试函数、行号及跳过原因见 [coverage.json](coverage.json) 的 `pending`；它也记录已纳入算子的 dtype 排除项。记录数包含同名算子的多个 benchmark 变体，不能当作未覆盖的唯一算子数。

本次先保留能独立构造、具有 CUDA 计算 kernel、可重复调用且不会修改位置输入的接口。需要用户决定或后续专门适配的项目包括：

| 类别 | 本次处理 | 后续需要决定 |
|---|---|---|
| 原地 / optimizer / 状态更新 | 暂不纳入。当前采集器反复复用实参，直接运行可能测到累积状态、溢出或改变后的输入 | 是否增加每次调用前恢复输入、恢复过程不计时的机制 |
| 自动求导 backward benchmark | 暂不将 `is_backward=True` 的前向 callable 作为反向基线；具有明确原生 backward API 的模块可以纳入 | 是否采集 autograd 全路径，如何准备/重用计算图 |
| 自定义类状态、参数化测试、稀疏/量化/nested 输入 | 可独立确定的部分已纳入；剩余逐条记录输入/状态依赖和无法提取之处 | 先支持哪些输入计划、布局或变体；不要凭接口名猜输入 |
| 仅主机 / 视图 / 元数据操作 | 没有观察到 CUDA 计算 kernel 的接口暂不纳入 | 是否需要独立 CPU/调度耗时口径，以及是否保留 native no-op |
| 外部实现依赖 | 依赖 FlagGems、Triton、Transformer Engine 等、无法独立绑定 PyTorch 的部分暂不纳入 | 是否扩展原生来源，或为这些算子明确 PyTorch 复合参考 |
| 没有明确 benchmark 几何 | 保留算子 ID 和来源，未用正确性测试的大量边界用例替代性能档位 | 代表性性能 shape 和实际调用变体 |
| native / dtype 不可用 | 当前 Torch/CUDA 不支持的项及 dtype 单独记录 | 是否针对较新 Torch、其他 NVIDIA 架构另建覆盖 |

具体例子：`grouped_mm` 在当前 H20/Torch 上触发设备断言，已经隔离并排除；`unsafe_masked_index_put_accumulate` 的上游输入使用 int32 mask，而 meta 校验要求 bool，因此暂不把仅能发出 CUDA kernel 当成语义正确的证据。`svd` 仅保留实际可运行的 float32；`_cast_Int` 的 int32 输入是 native no-op，因此该 dtype 暂不纳入。

新增集合本身只复用现有采集器，没有顺带修改计时、NCU 指标、算子发现或 vLLM/SGLang 模块。
