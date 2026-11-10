# 文档索引

> **状态口径**：`current` = 描述当前实现/契约，改代码必须先改它；`generated` = 自动生成
> 勿手改；`archived` = 仅追溯，命令/路径可能失效。**文档状态以本页为准。**
> 本仓库文档按编号分层：`spec/`（测什么/依据）、`design/`（怎么实现/契约）、
> `guides/`（怎么用）、`process/`（怎么验证/治理）、`records/`（时间性记录）、
> `archive/`（历史）。校对本仓库文档：`make docs`。

## 规范 spec/（测什么、依据与口径）

| 文档 | 状态 | 内容 | 维护时机 |
|---|---|---|---|
| [11-performance-testset.md](spec/11-performance-testset.md) | current v1.2 | 性能样本集设计：固定一套 8 场景 × 1K/8K/32K × 10、token 口径、抽样与复现；32K target 32,768 不特判 | 性能样本/构造/口径变化 |
| [12-accuracy-testset.md](spec/12-accuracy-testset.md) | current v1.1 | 精度样本集设计：固定一套 8 场景选型、抽样、5 类判题、构造/移植（隐藏离线资产） | 精度样本/判题/数据构造变化 |
| [13-testset-portability.md](spec/13-testset-portability.md) | current v1.0 | **固定样本契约 vs 测试系统职责** + 跨设备/跨系统比较准入（同 testset SHA、actual 记录、±10% 带、协议参数） | 样本交付/移植/比较口径变化 |
| [20-methodology.md](spec/20-methodology.md) | current | 测量方法学：指标公式、TG TPS 有效性、actual ISL 记录与 ±10% 入带聚合、缓存治理、环境采集、可复现性 | 指标/缓存/样本语义变化 |
| [21-test-plan.md](spec/21-test-plan.md) | current v2.3 | 测试方案：固定一套样本、actual 记录、±10% 入带聚合、32K 不特判、TG TPS 有效性、ci95、执行流程、AC-1…AC-11 | 测试范围/样本/指标/验收变化 |

## 设计 design/（怎么实现、对外契约）

| 文档 | 状态 | 内容 | 维护时机 |
|---|---|---|---|
| [30-architecture.md](design/30-architecture.md) | current v1.1 | 架构与实现设计（4+1 视图）：分层、端口/适配器、API 流程、安全边界、技术债 | 目录/模块/依赖/API/部署变化 |
| [31-output-contract.md](design/31-output-contract.md) | current | `output/` 现状契约：schema 3.0 平铺布局、7 个 run 条目、`index.json`、二次分析、verify/gc/rebuild/pack、保留与迁移 | 产物/索引/schema/目录结构变化 |
| [33-project-structure.md](design/33-project-structure.md) | generated | 源码目录与文件清单（`make docs-write` 生成，勿手改） | `make docs-write` |
| [34-feasibility.md](design/34-feasibility.md) | current v2.0 | 可行性：oMLX 能力实测、缓存 25× 证据、运行时/磁盘预算 | 性能/规模假设变化 |

## 操作 guides/（怎么用）

| 文档 | 状态 | 内容 | 维护时机 |
|---|---|---|---|
| [40-install.md](guides/40-install.md) | current | 安装与依赖：macOS/Windows 支持矩阵、Python/venv、镜像/代理、extras、验证、排障、重建 | 安装方式/依赖/启动器变化 |
| [41-user-guide.md](guides/41-user-guide.md) | current | 使用说明：Web/CLI/配置/测试集/结果解读/排障（含双锁、`verify/rebuild/gc/pack`、`analyze --paired`） | CLI/配置/安装变化 |

## 过程 process/（验证与治理）

| 文档 | 状态 | 内容 | 维护时机 |
|---|---|---|---|
| [50-validation-plan.md](process/50-validation-plan.md) | current v1.2 | 测试系统 V&V：对象/方法（M1–M10）/环境矩阵/验收判据（V-1…V-13，含 V-12 固定样本/V-13 样本量充分性）/再验证触发/证据留存/角色 | 验证方法、判据或再验证触发变化 |

## 记录 records/

| 文档 | 状态 | 内容 |
|---|---|---|
| [records/README.md](records/README.md) | records | 时间性记录目录说明（评审/变更/迁移/发布），**非规范、非证据** |
| [records/2026-11-10-vv-round.md](records/2026-11-10-vv-round.md) | 记录 v1.0 | 首篇 V&V 执行证据：gate --live 11/11、8 legacy run verify 复算、T-061/T-062 整合轮复验 |

> 根目录 `reports/` 是评审/审计/实验记录（gitignored，**不是测试证据**），其索引为
> `reports/README.md`（纯文本路径，不做 Markdown 链接）；长期结论应进入本目录或 `docs/`。

## 历史归档 archive/

| 文档 | 状态 | 说明 |
|---|---|---|
| [archive/DESIGN.md](archive/DESIGN.md) | archived | 重构前方案设计，命令/目录已过时 |
| [archive/AUDIT.md](archive/AUDIT.md) | archived | 独立审计记录（当时版本） |
| [archive/OUTPUT_LAYOUT.md](archive/OUTPUT_LAYOUT.md) | archived · superseded_by [design/31-output-contract.md](design/31-output-contract.md) | `output/` 重构的设计论证/实施记录；现状契约以 design/31 为准 |

## 防漂移机制

1. `make docs` 运行 `docs_links + docs_contract`：相对链接；`docs/**/*.md`（archive 除外）、
   `harness/prompts/*.md`、`dataset/README.md`、`tool/README.md` 中的模块、仓库路径、
   `make` 目标、`./modelbench <cmd> --flag`、根启动器 `./modelbench`、生成的结构清单，
   以及**已删除产物名黑名单**（白名单：`design/31-output-contract.md`、
   `archive/OUTPUT_LAYOUT.md` 历史段、`archive/` 与 V&V 政策文档）。
2. `pytest test/test_docs_contract.py` 在测试阶段再次校验（prompts/子目录纳入 + 黑名单检测）。
3. 修改 `src/modelbench` 目录结构后运行 `make docs-write` 重新生成
   [33-project-structure.md](design/33-project-structure.md)。
4. 改 CLI/配置/产物/schema/测量口径时，同一次提交更新本页列出的对应文档；否则门禁失败。
5. 测量语义/有效性口径变化时，同步 [50-validation-plan.md](process/50-validation-plan.md)
   的判据与再验证触发。

## 版本口径

- `SCHEMA_VERSION=3.0`：performance run bundle（`records.jsonl` + `run.json` 唯一环境快照 +
  `samples.csv`/`aggregates.json`/`report.md`/`MANIFEST.json`/`logs/`；prompt 正文按
  `request.prompt_sha256` 外置到 `output/blobs/prompts/`）。
- `INDEX_SCHEMA_VERSION=2.0`：仅 `output/index.json` 的格式（只存相对路径，含扁平 `runs[]`
  与分组 `batches[]`），与 bundle 版本独立演进、可单独重建。
- 输出布局：`runs/` 一层平铺（`runs/<run_id>/`，目录名即 `run_id`），批次元数据集中在
  `output/batches/<batch_id>.json`；`output/` 根 7 个条目。
- 3.0 与旧 2.x bundle **不兼容**：旧 bundle 已归档到 `output/archive/legacy-v2/`，不参与新索引，
  运行时读取侧不保留兼容分支；新迁移器写 `schema_version="3.0"` + `sample_meta.migrated_from_schema`，
  旧迁移件允许 2.1 legacy（详见 [31-output-contract.md](design/31-output-contract.md) §10）。
- 目录与产物规范见 [31-output-contract.md](design/31-output-contract.md)。
