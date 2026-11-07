# 文档索引

> 文档状态以本页为准。**当前有效**文档描述 HEAD 版本行为；**归档**文档仅用于追溯，
> 其中的命令/路径可能已失效。校对本仓库文档：`make docs`。

## 当前文档

| 文档 | 内容 | 维护时机 |
|---|---|---|
| [TEST_PLAN.md](TEST_PLAN.md) | 测试方案设计：性能测试范围、样本数、口径、验收 | 测试范围/样本/指标变化 |
| [USER_GUIDE.md](USER_GUIDE.md) | 使用说明：Web/CLI/配置/测试集/排障 | CLI/配置/安装变化 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 架构与实现设计：分层、数据流、环境采集、扩展点 | 目录/模块/依赖变化 |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | 源码目录与文件清单（自动生成，勿手改） | `make docs-write` |
| [METHODOLOGY.md](METHODOLOGY.md) | 指标公式、输入上下文、缓存治理、环境采集 | 指标/缓存/样本语义变化 |
| [RESULTS.md](RESULTS.md) | 结果数据管理：bundle 结构、批次、index.json、analysis/、Excel 总结 | 产物/索引/Excel 口径变化 |
| [DATASETS.md](DATASETS.md) | 数据集来源、字段、许可、校验 | 数据集/下载变化 |
| [FEASIBILITY.md](FEASIBILITY.md) | 可行性、运行时估算、风险 | 性能/规模假设变化 |

## 历史归档

| 文档 | 说明 |
|---|---|
| [archive/DESIGN.md](archive/DESIGN.md) | 重构前方案设计，命令/目录已过时 |
| [archive/AUDIT.md](archive/AUDIT.md) | 独立审计记录（当时版本） |

## 防漂移机制

1. `make docs` 运行 `docs_links + docs_contract`：链接、模块、路径、CLI flag、根启动器
   （`./modelbench`，本仓库无 `[project.scripts]` console script）、生成的结构清单必须与代码一致。
2. `pytest test/test_docs_contract.py` 在测试阶段再次校验。
3. 修改 `src/modelbench` 目录结构后运行 `make docs-write` 重新生成
   [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)。
4. 改 CLI/配置/产物/schema 时，同一次提交更新本页列出的对应文档；否则门禁失败。

## 版本口径

- `SCHEMA_VERSION=2.1`：性能测试 run bundle（`overall` 按 `model × context_length`
  分行 / cells / samples + 环境快照）。旧 1.x 的 accuracy/上下文矩阵 run 不再生成；
  旧 2.0 run 仍可读（`overall` 无档位字段，Web 标“全部*”，不与 2.1 口径混比）。
