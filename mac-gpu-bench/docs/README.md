# 文档索引

`macgpu` —— Mac GPU Compute & Bandwidth Benchmark。
本仓库文档分四层：**spec**（测什么、公式口径）、**design**（怎么实现、产物契约）、
**guides**（怎么用）、**process**（怎么验证）。**文档状态以本页为准。**

> 一致性约束：`make docs` 会校验本页是否收录了所有文档、文档里的内核名/JSON 字段是否与
> 实现一致、是否残留旧名字。**改代码前先改对应文档，改完跑 `make docs`。**

| 文档 | 状态 | 内容 | 何时需要改 |
|---|---|---|---|
| [spec/10-methodology.md](spec/10-methodology.md) | current | 测什么：指标定义、公式、通路与精度的对应关系、指标矩阵（内核 ↔ 计数） | 新增/删除测量项、改计数口径 |
| [design/20-architecture.md](design/20-architecture.md) | current | 怎么实现：模块划分、数据流、计时与校准、防优化手段 | 改模块边界/内核结构/计时方式 |
| [design/21-output-contract.md](design/21-output-contract.md) | current | 产物契约：`output/` 目录、JSON 字段、报告结构、版本策略 | 改目录/字段/报告章节 |
| [guides/30-quickstart.md](guides/30-quickstart.md) | current | 怎么用：安装、命令、结果解读、MCP 接入 | 改 CLI/参数/输出 |
| [process/40-validation.md](process/40-validation.md) | current | 怎么验证：selfcheck 覆盖范围、三个已踩过的坑、验收清单 | 改自检项/验收标准 |
| [records/50-m1-baseline.md](records/50-m1-baseline.md) | record | M1 / 8 核 GPU / 16 GB 的一次基线读数（历史记录，不随代码更新） | 仅追加新记录 |
| [records/51-mlx-crosscheck.md](records/51-mlx-crosscheck.md) | record | 用 oMLX 真实 4-bit 模型校核 roofline 预测，给出效率系数（历史记录） | 仅追加新记录 |

## 术语

- **path（通路）**：GPU 执行单元的种类。`mma` = 矩阵指令（`simdgroup_matrix`），`fma` = 普通着色器 ALU。
- **peak（峰值）**：寄存器内循环、无访存的合成负载，代表该通路的上限，不等于任何模型的实际速度。
- **MAC**：一次乘加。1 MAC = 2 FLOP。跨精度比较时用 MAC/s 比用 FLOPS 更公平。
