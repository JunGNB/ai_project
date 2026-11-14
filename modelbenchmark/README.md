# 本地大模型基准测试系统（性能 + 精度双方法）

对本地 oMLX 服务中的模型做端到端黑盒基准，支持**两种测试方法**（共用同一套测量内核与结果布局）：

- **性能（`method=performance`）**：按测试集场景（1K / 8K / 32K 分桶）逐样本采集
  **TTFT、TG TPS、峰值内存、输入上下文大小**，按 `(模型, 场景, 数据集, 档位)` 形成 cell，
  不判分（`score` 恒 null）。
- **精度（`method=accuracy`）**：8 场景 × 250 = 2,000 条固定样本，`temperature=0` / 冷缓存 /
  无 decode guard / `repeats=1`，用 5 类**确定性判题器**逐样本判分，输出场景分、8 场景宏平均、
  语言/数据集剖面，并支持 bf16 vs 量化的配对 Δ（McNemar/Holm）。

每次 run 自动采集硬件与 oMLX 配置快照，保留逐样本证据；提供本地 Web 控制台与 CLI。
两种方法**不混算、不合成综合分**，用 `run_id` 的方法段（`__performance__`/`__accuracy__`）区分。

## 新机器安装

把仓库 clone 到新机器后，先建环境再运行（Python ≥3.10，推荐 3.12；需要 git）：

```bash
# macOS / Linux
cd /path/to/modelbenchmark
./modelbench setup && ./modelbench validate --offline
```

```bat
REM Windows（cmd）
cd /d D:\path\to\modelbenchmark
.\modelbench.cmd setup && .\modelbench.cmd validate --offline
```

- 一键脚本会在项目内创建 `.venv` 并安装 `.[dev]`；也可手动
  `python3.12 -m venv .venv` + `pip install -e ".[dev]"`（Windows 用 `py -3.12` 与
  `.venv\Scripts\python.exe -m pip`）。
- **精度测试需要额外安装 `.[eval]`**（判题依赖 `nltk`/`langdetect`/`absl-py`/`immutabledict`/
  `jieba`/`rouge`）：`.venv/bin/pip install -e ".[eval]"`；缺失时精度 run 会显式告警并把相关
  样本记为 `judge_error`（不静默给分），见 [docs/spec/22-accuracy-test-plan.md](docs/spec/22-accuracy-test-plan.md) §3.2。
- 完整说明（支持矩阵、镜像/代理、可选 extras、安装验证、常见问题、重建）见
  [docs/guides/40-install.md](docs/guides/40-install.md)。
- live benchmark 还需要**可达的 oMLX 服务**；Windows/Linux 上可先用离线 validate、
  数据集构建与 Web/CLI 工具链，live 需连接一台可达的 oMLX 实例。

## 快速开始

```bash
cd /path/to/modelbenchmark

# 0) 自检（服务、配置、数据集、测试集）
./modelbench validate

# 1) Web 控制台（推荐；启动后自动打开默认浏览器）
./modelbench web          # http://127.0.0.1:8765
#    ./modelbench web --no-open              # 不自动打开浏览器
#    ./modelbench web --idle-timeout 1800    # 无请求 30 分钟后自动停止

# 2) CLI：性能 —— 每个场景取 5 个样本（8 场景共 40 请求，快速对比）
./modelbench run --models minicpm5-1b qwen3.5-9b --samples 5

# 2b) CLI：精度 —— 全量 2,000 样本（约 3h/模型；先构建 accuracy testset.jsonl）
./modelbench run --method accuracy --models minicpm5-1b
./modelbench run --method accuracy --models minicpm5-1b --samples 5   # 冒烟 8×5=40 条

# 查看结果目录索引 / 重建索引 / 校验证据链
./modelbench runs
./modelbench index
./modelbench verify

# 3) 默认测试集全部样本（耗时最长）
./modelbench run --models minicpm5-1b

# 4) 只看计划（两种方法）
./modelbench run --models minicpm5-1b --samples 5 --dry-run
./modelbench run --method accuracy --models minicpm5-1b --samples 5 --dry-run

# 5) 结果对比（run 目录或单段 run_id）：性能用默认，精度加 --method accuracy
./modelbench analyze <run_id_a> <run_id_b>
./modelbench analyze <run_id_bf16> <run_id_quant> --paired --method accuracy
```

> 源码根目录为 `src/`（Python 标准库已有 `code` 模块，避免包名冲突）；其余按
> `dataset/`、`output/`、`tool/`、`src/`、`test/` 组织。

## 环境与可移植性

- `.venv/` 是**项目内**的隐藏目录（Finder 默认不显示，`ls -a` 可见），不提交 git；
  它只保存 Python 解释器和依赖；**唯一入口是根目录的启动器**：macOS/Linux 为
  `./modelbench`，Windows 为 `.\modelbench.cmd`（PowerShell 也可 `.\modelbench.ps1`），
  均通过 `setup`/`web`/`run` 等子命令工作（本仓库没有 console script）。
- **不要把 `.venv/` 复制到其他路径或机器**：脚本 shebang、`pyvenv.cfg` 和 Python 链接
  都写死了当前绝对路径。到新位置后重建环境：

```bash
./modelbench setup     # 或 make setup
# 等价于：创建 .venv → 安装依赖（macOS/Linux 默认清华镜像；Windows 默认官方 PyPI，
# 可用 PIP_INDEX_URL/UV_DEFAULT_INDEX 换源）
./modelbench validate --offline --allow-missing
```

```bat
REM Windows（cmd）；PowerShell 把 .cmd 换成 .ps1
.\modelbench.cmd setup
.\modelbench.cmd validate --offline --allow-missing
```

> PowerShell 若提示禁止运行脚本（执行策略），先在当前窗口执行
> `Set-ExecutionPolicy -Scope Process Bypass`，或改用 `.\modelbench.cmd`。

- 可移植的是：源码、`pyproject.toml`、`setup.cfg`、配置、测试、文档；`dataset/`（原始
  文件、测试集、tokenizer 缓存）与 `output/` 不会随安装包分发，需要复制或重新生成。
- 运行还依赖外部 oMLX 服务（桌面应用目前仅 macOS）和本地模型；换机器需确认 oMLX
  地址/API key（Windows 上把 `server.base_url` 指向可达实例即可做 live 测试）。
- 镜像不可用时：`UV_DEFAULT_INDEX=https://pypi.org/simple ./modelbench setup`。
  更多安装细节（手动 venv、代理、排障）见 [docs/guides/40-install.md](docs/guides/40-install.md)。

## 测试内容（概览）

两种方法共用测量内核（冷缓存 fail-closed、模型默认配置、双锁串行、逐样本证据、环境快照），
但口径、判分与产物形状不同：

- **性能**：对每个 `模型 × 场景 × 上下文档位`（cell）串行测量逐样本 TTFT / TG TPS / TPOT /
  峰值内存 / 输入上下文大小；默认 8 场景 × 1K/8K/32K × 10 = 240 measured 请求 / 24 cells；
  不判分；固定一套 testset 全模型复用，actual ISL 按 target ±10% 入带聚合（不跳过样本）。
- **精度**：8 场景 × 250 = 2,000 条固定样本；`temperature=0` / cold / 无 decode guard /
  `repeats=1`；5 类确定性判题（`mc_letter`/`ifeval`/`normalized_match`/`code_exec`/
  `tool_call`，无 LLM judge、无网络）；输出场景分、8 场景宏平均、语言/数据集剖面与配对 Δ；
  全量约 3h/模型（长上下文约 2–2.5h）。

**口径与判据的单一事实源**（本 README 只留概览，细节以链接为准）：

- 测量口径/指标定义：[docs/spec/20-methodology.md](docs/spec/20-methodology.md)
- 性能测试方案与验收：[docs/spec/21-test-plan.md](docs/spec/21-test-plan.md)
- 精度测试方案与验收：[docs/spec/22-accuracy-test-plan.md](docs/spec/22-accuracy-test-plan.md)
- 双方法架构/分派：[docs/design/32-method-architecture.md](docs/design/32-method-architecture.md)
- 测试集设计与 ±10% 带推导：[docs/spec/11-performance-testset.md](docs/spec/11-performance-testset.md)
  与 [docs/spec/12-accuracy-testset.md](docs/spec/12-accuracy-testset.md)
- `output/` 产物契约（两种方法形状）：[docs/design/31-output-contract.md](docs/design/31-output-contract.md)
- 数据根（raw/testset/许可）：[dataset/README.md](dataset/README.md)
- 配置：`src/modelbench/resources/config/default.yaml`（`method`/`perf`/`accuracy`/`run`/`datasets`）与
  `models.yaml`（模型清单，仅 MoE 模型带加载设置）

> `dataset/test/accuracy/`（2,000 条）已接入精度方法（`method=accuracy`，独立 run），
> 不参与性能 run。默认口径与 vLLM / oMLX 官方榜不同，**不可直接横比**；精度分数也不得与
> 性能指标合并。

## 目录

```
dataset/raw/    # 上游原始文件（只读；download_manifest.json 记 url/size/sha256）
dataset/test/   # 测试集：performance/（240 条）、accuracy/（2,000 条，method=accuracy）
dataset/.cache/ # tokenizer 计数缓存（可删，重跑重建）
output/         # 运行结果与证据：runs/<run_id>/（平铺 bundle）、batches/（批次元数据）、
                # blobs/prompts/（prompt 池）、exports/（Excel）、
                # archive/（归档，GC 引用根）、.state/、index.json
src/modelbench/
  domain/       # 纯领域：models/metrics/accuracy_metrics/method/scenarios/judges
  application/  # 编排：ports/methods/plan/benchmark/accuracy
  adapters/     # IO：config/datasets/omlx/storage/host
  interfaces/   # 交付：cli/web/analysis
  resources/    # YAML 配置（随包分发）
modelbench        # 唯一入口：setup/web/run/…（.venv 由 setup 创建）
tool/           # 工程外部工具（dataset 构造/fetch/校验；不被运行时导入）
test/           # DT 测试
harness/        # AI 工程门禁/任务板
docs/           # 文档：spec/（规范）、design/（设计/契约）、guides/（操作）、
                # process/（V&V）、records/（时间性记录）、archive/（历史）
```

## 文档

- [docs/README.md](docs/README.md)：**文档索引与防漂移机制**（先看这里）
- 规范：[docs/spec/21-test-plan.md](docs/spec/21-test-plan.md)（性能测试方案与验收）、
  [docs/spec/22-accuracy-test-plan.md](docs/spec/22-accuracy-test-plan.md)（精度测试方案与验收）、
  [docs/spec/20-methodology.md](docs/spec/20-methodology.md)（指标/缓存/token/判分口径）、
  [docs/spec/11-performance-testset.md](docs/spec/11-performance-testset.md)、
  [docs/spec/12-accuracy-testset.md](docs/spec/12-accuracy-testset.md)（样本集设计）、
  [docs/spec/13-testset-portability.md](docs/spec/13-testset-portability.md)（跨设备/跨系统比较准入）、
  [dataset/README.md](dataset/README.md)（数据根：来源/许可/数据流）
- 设计：[docs/design/30-architecture.md](docs/design/30-architecture.md)（架构与实现）、
  [docs/design/31-output-contract.md](docs/design/31-output-contract.md)（`output/` 契约/schema 3.0）、
  [docs/design/32-method-architecture.md](docs/design/32-method-architecture.md)（双方法架构/分派）、
  [docs/design/34-feasibility.md](docs/design/34-feasibility.md)（可行性/预算）、
  [docs/design/33-project-structure.md](docs/design/33-project-structure.md)（自动生成的源码清单）
- 操作：[docs/guides/40-install.md](docs/guides/40-install.md)（安装）、
  [docs/guides/41-user-guide.md](docs/guides/41-user-guide.md)（Web/CLI 使用）
- 验证：[docs/process/50-validation-plan.md](docs/process/50-validation-plan.md)（测试系统 V&V）
- 记录/归档：[docs/records/](docs/records/)（时间性记录）、
  [docs/archive/](docs/archive/)（历史设计/审计/OUTPUT_LAYOUT，已归档）

## 质量门禁

```bash
.venv/bin/python -m harness.gate        # 导入/编译/测试/validate/数据完整性/文档契约/密钥/卫生
pytest test/                            # 单元 + fake-server 集成测试
.venv/bin/python -m harness.gate --live # 额外真实 smoke
make build                              # wheel 输出到 output/build/dist
```

构建临时文件（setuptools build/egg-info）统一放在 `output/` 下的 build 子目录，源码目录保持干净。

AI 工作流与硬约束见 [AGENTS.md](AGENTS.md) 与 [harness/README.md](harness/README.md)。
