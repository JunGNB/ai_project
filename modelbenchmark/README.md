# 本地大模型性能基准测试系统

对本地 oMLX 服务中的模型做端到端黑盒**性能测试**：按测试集场景（1K / 8K / 32K 分桶）
逐样本采集 **TTFT、TG TPS、峰值内存、输入上下文大小**，按
`(模型, 场景, 数据集, 档位)` 形成 cell，并保留逐样本结果；
每次 run 自动采集硬件与 oMLX 配置快照。提供本地 Web 控制台与 CLI。

> 0.3 起旧的多策略测试（perf/accuracy/context/custom）与上下文矩阵已删除；
> 当前只有性能基准测试。准确性判分代码保留但**不参与运行**。

## 快速开始

```bash
cd "/Users/huawei/Desktop/模型管家相关资料/modelbenchmark"

# 0) 自检（服务、配置、数据集、测试集）
./modelbench validate

# 1) Web 控制台（推荐；启动后自动打开默认浏览器）
./modelbench web          # http://127.0.0.1:8765
#    ./modelbench web --no-open              # 不自动打开浏览器
#    ./modelbench web --idle-timeout 1800    # 无请求 30 分钟后自动停止

# 2) CLI：每个场景取 5 个样本（默认 8 场景共 40 请求，快速对比）
./modelbench run --models minicpm5-1b qwen3.5-9b --samples 5

# 查看结果目录索引 / 重建索引
./modelbench runs
./modelbench index

# 3) 默认测试集全部样本（耗时最长）
./modelbench run --models minicpm5-1b

# 4) 只看计划
./modelbench run --models minicpm5-1b --samples 5 --dry-run

# 5) 结果对比
./modelbench analyze output/<run_a> output/<run_b>
```

> 源码根目录为 `src/`（Python 标准库已有 `code` 模块，避免包名冲突）；其余按
> `dataset/`、`output/`、`tool/`、`src/`、`test/` 组织。

## 环境与可移植性

- `.venv/` 是**项目内**的隐藏目录（Finder 默认不显示，`ls -a` 可见），不提交 git；
  它只保存 Python 解释器和依赖；**唯一入口是根目录的 `./modelbench`**（`setup`/`web`/`run` 等子命令）。
- **不要把 `.venv/` 复制到其他路径或机器**：脚本 shebang、`pyvenv.cfg` 和 Python 链接
  都写死了当前绝对路径。到新位置后重建环境：

```bash
./modelbench setup     # 或 make setup
# 等价于：创建 .venv → 安装依赖（默认清华镜像）
./modelbench validate --offline --allow-missing
```

- 可移植的是：源码、`pyproject.toml`、`setup.cfg`、配置、测试、文档；`dataset/`（原始
  文件、测试集、tokenizer 缓存）与 `output/` 不会随安装包分发，需要复制或重新生成。
- 运行还依赖外部 oMLX 服务（目前仅 macOS）和本地模型；换机器需确认 oMLX 地址/API key。
- 镜像不可用时：`UV_DEFAULT_INDEX=https://pypi.org/simple ./modelbench setup`。

## 测试内容

| 项 | 规则 |
|---|---|
| 测试范围 | `模型 × 测试集场景`；串行测量；默认 8 个场景覆盖 1K/8K/32K 三档（480 measured 请求 / 24 cells） |
| 样本数 | `perf.samples_per_dataset`（Web/CLI `--samples`）；留空 = 每个场景全部可用样本 |
| 上下文筛选 | `perf.context_windows`（Web 多选 / CLI `--context-windows`）；空=全部；1K 为 0～1.5K，其他 N 为 N±0.5K；只筛样本，聚合始终按档位分别求均值（不跨档位合并） |
| 预热 | 模型级默认 1 次（`warmup_per_model=1`，加载后短请求，丢弃）；per-cell `warmup_requests=0`；`--warmup N` 仅诊断 |
| 性能指标 | 逐样本 TTFT、TG TPS、TPOT、E2E、峰值 CPU/RSS、输入上下文 token/字符数；默认 `decode_guard` 追加长输出指令，避免 TG TPS 因 1-token EOS 为空 |
| 结果输出 | 每模型一个 `output/<批次>_performance_<模型>/` bundle：按上下文档位的整体平均值 + cells + 逐样本；批次分组在 output 下的 batches 目录与 `output/index.json`；每批次自动生成 `output/result_analysis/<批次>_performance_<模型>/summary.xlsx` |
| 环境快照 | 硬件/OS/oMLX 版本/全局设置/模型生效设置/加速 feature flags |
| 模型加载 | oMLX 默认配置；仅 MoE 模型开启 `moe_expert_offload_enabled` |

测试集位于 `dataset/test/`（默认 `performance`，可自行新增目录）；性能样本由
`tool/dataset` 预构造为 `dataset/test/performance/testset.jsonl`（480 条；默认 8 场景 480 measured 请求 / 24 cells），
快速子集 `performance_10` 为 240 条严格子集；场景 `chinese_long_context` 是中文长文
（原 `multilingual` 更名）。`dataset/test/accuracy/`（6,589 条 = 250×6 + 889 + 4,200）是**离线资产**：
`status=offline-asset`、runner 未接入、judge 仅契约，默认不参与运行。
默认口径（模型级预热 1 次、per-cell 预热 0、`max_tokens=50`、`repeats=1`、逐请求冷缓存）与 vLLM / oMLX 官方榜不同，
**不可直接横比**。
投机解码不属于默认基准（需非默认加速开关，单列专项）。
配置见 `src/modelbench/resources/config/default.yaml`（`perf`/`run`/`datasets`）与
`models.yaml`（模型清单，仅 MoE 模型带加载设置）。

## 目录

```
dataset/raw/    # 上游原始文件（只读；download_manifest.json 记 url/size/sha256）
dataset/test/   # 测试集：performance/、performance_10/（预构造性能样本）、accuracy/（离线资产，不默认参与）
dataset/.cache/ # tokenizer 计数缓存（可删，重跑重建）
output/         # 运行结果（每个 run 一个 bundle）
src/modelbench/
  domain/       # 纯领域：models/metrics/scenarios/evaluators
  application/  # 编排：ports/plan/benchmark
  adapters/     # IO：config/datasets/omlx/storage/host
  interfaces/   # 交付：cli/web/analysis
  resources/    # YAML 配置（随包分发）
modelbench        # 唯一入口：setup/web/run/…（.venv 由 setup 创建）
tool/           # 工程外部工具（dataset download；不被运行时导入）
test/           # DT 测试
harness/        # AI 工程门禁/任务板
docs/           # 方案/使用/架构/数据集
```

## 文档

- [docs/README.md](docs/README.md)：**文档索引与防漂移机制**（先看这里）
- [docs/TEST_PLAN.md](docs/TEST_PLAN.md)：测试方案设计（范围、样本、口径、验收）
- [docs/USER_GUIDE.md](docs/USER_GUIDE.md)：使用说明（Web/CLI/配置/排查）
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)：架构与实现设计
- [docs/METHODOLOGY.md](docs/METHODOLOGY.md)：指标公式、输入上下文、缓存治理
- [docs/RESULTS.md](docs/RESULTS.md)：结果数据管理（index.json / analysis/ / 分析用法）
- [docs/DATASETS.md](docs/DATASETS.md)：数据集来源、大小、许可
- [docs/archive/](docs/archive/)：历史设计/审计归档（重构前，命令已过时）

## 质量门禁

```bash
.venv/bin/python -m harness.gate        # 导入/编译/测试/validate/数据完整性/文档契约/密钥/卫生
pytest test/                            # 单元 + fake-server 集成测试
.venv/bin/python -m harness.gate --live # 额外真实 smoke
make build                              # wheel 输出到 output/build/dist
```

构建临时文件（setuptools build/egg-info）统一放在 `output/` 下的 build 子目录，源码目录保持干净。

AI 工作流与硬约束见 [AGENTS.md](AGENTS.md) 与 [harness/README.md](harness/README.md)。
