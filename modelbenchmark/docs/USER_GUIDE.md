# 使用说明

> 先看 [测试方案](TEST_PLAN.md) 了解口径；改代码/重构看 [架构设计](ARCHITECTURE.md)。
>
> 0.3 起旧策略（perf/accuracy/context/custom）与上下文矩阵已删除；当前只有
> **性能基准测试**：按测试集逐数据集取样本，逐样本测 TTFT / TG TPS / 峰值内存 /
> 输入上下文大小，对外按上下文档位（1K/8K/32K）**分别**展示平均值（不跨档位合并），
> 同时保留逐样本结果。

## 1. 环境与安装

- macOS + oMLX `0.7.3.dev2`，服务 `http://127.0.0.1:8000`（OpenAI 兼容 + admin API）。
- Python ≥3.10，项目自带 `.venv`。
- API key（任选其一）：`export OMLX_API_KEY=...`、`src/modelbench/resources/config/default.yaml` 的
  `server.api_key`、或自动读取 `~/.omlx/settings.json`。

```bash
cd "/Users/huawei/Desktop/模型管家相关资料/modelbenchmark"
./modelbench validate        # 服务/配置/数据集/测试集自检
```

重建环境（一般不需要）：

```bash
UV_CACHE_DIR=$PWD/.uv-cache UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
  uv pip install --python .venv/bin/python -e ".[dev]"
```

## 1.1 可移植性与 `.venv`

`.venv/` 只是项目内虚拟环境；唯一入口是根目录的 `./modelbench`，不是系统命令。
`.venv` 是隐藏目录（Finder 用 `Cmd+Shift+.` 显示），且不提交 git。

**不要跨路径/跨机器复制 `.venv`**：脚本 shebang、`pyvenv.cfg`、Python 链接都是绝对路径。
新位置重新初始化：

```bash
./modelbench setup     # 或 make setup
./modelbench validate --offline --allow-missing
```

数据（`dataset/`）与结果（`output/`）不随包安装，需复制或重新生成（`tool/dataset`）；oMLX 服务与模型
需在目标机器可用。默认使用清华 PyPI 镜像，可覆盖：
`UV_DEFAULT_INDEX=https://pypi.org/simple ./modelbench setup`。

## 2. 目录

```
dataset/
  raw/            # 上游原始文件（只读）
  test/           # 测试集：performance/（默认）、performance_10/（快速子集）、accuracy/（离线资产，不默认参与）、自定义
output/           # 运行结果（每个 run 一个目录）
src/modelbench/
  domain/         # 数据契约、指标、场景、判分器（无 IO；判分当前不参与运行）
  application/    # Orchestrator、测试集应用、后端端口
  adapters/       # config/datasets/omlx/storage/host（外部 IO）
  interfaces/     # cli / web / analysis
  resources/      # default/models/scenarios YAML
modelbench          # 环境初始化入口
tool/             # 工程外部工具（download/；不被运行时导入）
test/             # DT 测试
docs/             # 方案/使用/架构
```

## 3. 快速开始（Web）

> 源码包为 `src/modelbench/`（标准 src 布局，避免与标准库 `code` 冲突），
> 源码模块为 `modelbench.*`；日常使用 `./modelbench` / `./modelbench web`。

```bash
./modelbench web        # 启动后自动在默认浏览器打开 http://127.0.0.1:8765
./modelbench web --no-open        # 不自动打开浏览器
./modelbench web --idle-timeout 120    # 页面关闭约 2 分钟后自动停止（有任务运行时不会停）
```

> 页面打开期间每 30 秒会发送心跳；配合 `--idle-timeout`，关闭浏览器标签后服务会在
> 超时且无运行任务时自动退出（建议超时值大于 60 秒）。默认不启用（便于后台 API 使用），
> 仍可用 `Ctrl+C` 停止。

> **关闭浏览器标签不会停止服务**：Web 控制台是一个本地 HTTP 服务，浏览器只是客户端；
> 关闭标签页后服务仍在后台（正在跑的 benchmark 也会继续）。要停止服务请在终端按 `Ctrl+C`，
> 或使用 `--idle-timeout` 让服务在长时间无请求且无运行任务时自动退出。

1. **测试集**列表来自 `dataset/test/<name>/`（默认 `performance`：8 场景 / 480 measured / 24 cells；
   另有 `performance_10` 快速子集 240 条；`accuracy/` 无 `testset.yaml`、是离线资产，不出现在可选列表）。
   勾选测试集即表示运行该测试集下的**全部场景**；右侧「打开目录」可在 Finder 中打开 `dataset/test/`。
   界面不再展开场景/数据集行，也不提供逐数据集样本数（统一用下方全局「每个场景样本数」）。
2. 勾选**模型**（可多选，串行测；默认第一个；提供全选/清空）。模型卡片右上角有
   **「刷新模型」** 与 **「同步清单」**：
   - **刷新模型**：实时检索 oMLX 的 `GET /v1/models/status`，服务端已不存在的清单条目置灰并标注
     「服务端不存在」（不可勾选）；服务端有但 `models.yaml` 未登记的模型提示「未登记」。
     oMLX 不可达时只显示错误，不会把清单条目误判为失效。
   - **同步清单**：把差异**文本级**写回 `src/modelbench/resources/config/models.yaml`——
     新增条目自动生成 `alias`，失效条目被移除，文件头注释与已有 `alias`/`notes`/`load`
     设置原样保留；写入是原子的，写前会重新解析校验，有任务在跑时拒绝同步（409）。
   - **开始测试**前还会再校验一次：勾选的模型若不在服务端清单内会直接报错（HTTP 400），
     不会再产生 0 记录的空 run。
3. 可选高级项：全局样本数、缓存模式（默认 cold）、思考模式、生成长度、重复轮数、
   **预热开关**（默认开：每个模型加载后发 1 条丢弃短请求，仿 oMLX QUICK，触发 JIT/Metal 编译；
   关闭后首个 measured 样本会包含冷启动，属非默认口径，结果不可与默认口径混算）与 **上下文窗口**（可多选，默认不选）。
   上下文档位只筛选样本，且界面**只列出数据集实际存在的档位**（`performance` 为 1K/8K/32K；
   1K 接受 0～1.5K token，其他 N 接受 N±0.5K token）；不选则测全部档位；cell 仍按场景×数据集×档位聚合。
4. 点“开始测试”，运行状态区会显示：**模型 已完成/总数、测试项 已完成/总数、
   已测样本 已完成/预计、当前测试项内进度**以及进度条；完成后在“历史运行”点开查看
   **按档位整体平均值**、数据集汇总、逐样本结果、环境与配置、报告。
5. 历史运行行右侧可直接打开**原始输出文件**、在 Finder 中打开 run 目录。

> 样本选择：界面只提供全局「每个场景样本数」（空 = 全部）。预构造样本按
> `(context_length, lang)` 分层抽取，小样本量也会覆盖 1K/8K/32K；legacy 数据集
> 仍按 sample_id 取前 N。逐样本浏览/手选样本仍可通过 API
> `GET /api/testsets/<name>/scenarios/<scenario>/samples` 与 `POST /api/run` 的
> `scenario_samples`/`sample_ids` 使用（界面已按简洁模式移除入口）。
>
> 控制台仅监听 127.0.0.1、无鉴权，不要暴露到公网。长任务可关闭页面，后台仍会继续
> （进程不退出时），刷新页面可看到状态。

## 4. 快速开始（CLI）

```bash
# 默认测试集、全部样本（最完整，耗时长）
./modelbench run --models minicpm5-1b
# 多模型：一次启动，但输出按模型拆分为
# output/<batch_id>_performance_<model>/，批次信息在 output/batches/
./modelbench run --models minicpm5-1b qwen3.5-9b --samples 5

# 每个场景取 5 个样本做快速对比（8 场景共 40 请求）
./modelbench run --models minicpm5-1b qwen3.5-9b --samples 5

# 指定测试集 + 更长的生成窗口
./modelbench run --testset performance --models minicpm5-1b --samples 5 --max-tokens 256

# 只测 1K 与 8K 上下文附近的样本（1K: 0-1.5K；8K: 7.5-8.5K），每项 5 个样本
./modelbench run --models minicpm5-1b --samples 5 --context-windows 1024 8192 --warmup 2

# 只打印计划
./modelbench run --models minicpm5-1b --samples 5 --dry-run
```

`run` 关键参数：

| 参数 | 说明 |
|---|---|
| `--testset NAME` | `dataset/test/` 下的测试集，默认 `performance` |
| `--models a b` | 模型 id 或 alias，可多选 |
| `--scenarios a b` | 只测指定场景（场景=数据集+参数） |
| `--samples N` | 每个场景取 N 个样本；不填/配置为 null = 全部 |
| `--max-tokens N` | 统一生成长度/OSL（默认 **50**，Web「生成长度」也可配置）。达到上限的样本 `finish_reason=length`，**不丢弃**：TTFT/内存/吞吐照常聚合，TG TPS 按实际生成 token 计算 |
| `--repeats N` | 每个样本测量轮数（默认 1） |
| `--context-windows TOKENS...` | legacy/诊断用：只测落在这些窗口的样本（并集；1024=1K 接受 0～1536，其他 N 接受 N−512～N+512）。新数据集请用场景/bucket 选样，不要混用 |
| `--warmup N` | 可选诊断项：每个 cell 预热（丢弃）次数，默认 0；模型级预热默认自动执行 1 次（`run.warmup_per_model=1`，Web 用「预热」开关控制）。Web 界面不暴露 per-cell 次数 |
| `--cache-mode cold\|warm\|none` | 默认 cold；warm/none 仅诊断用 |
| `--thinking-mode disabled\|auto\|enabled` | 默认 disabled |
| `--output-dir DIR` | 默认 `output` |
| `--run-id ID` / `--overwrite` | 指定批次前缀 / 覆盖已存在的模型目录 |
| `--max-requests N` | 测量请求上限（warmup 不计入） |
| `--dry-run` | 只看计划，不发请求 |

退出码：0 成功，1 失败，2 无 API key，3 oMLX 不可达/鉴权失败，130 中断。

其他命令：

```bash
./modelbench validate --offline     # 离线校验（配置/数据集解析；完整 SHA-256 由 tool/dataset/verify.py 校验）
./modelbench hardware               # 硬件/oMLX/依赖版本/模型配置
./modelbench list --what testsets   # 也可 scenarios/datasets/models/all
./modelbench download --dataset all # 下载/校验数据集
./modelbench runs                   # 按批次分组列出历史结果（batch -> 模型）
./modelbench export-analysis <batch_id>   # 重新生成该批次的 Excel 总结
./modelbench export-analysis --all        # 为所有批次重建 Excel
./modelbench runs --output-dir DIR  # 指定结果根目录
./modelbench index                  # 重建 output/index.json（含 batches）
./modelbench analyze output/<run_a> output/<run_b> \
  --metrics ttft_ms gen_tps peak_memory_gb input_context_tokens   # 结果对比
```

## 5. 配置

| 文件 | 作用 |
|---|---|
| `src/modelbench/resources/config/default.yaml` | 服务地址/key、run 执行参数、perf 测试参数、目录 |
| `src/modelbench/resources/config/models.yaml` | 模型清单 + MoE 模型的 `load.settings` |
| `src/modelbench/resources/config/scenarios.yaml` | 场景→数据集、dataset_params |

### 5.1 性能测试参数

```yaml
perf:
  samples_per_dataset: null   # null = 每个场景全部可用样本；填整数 = 每个场景取前 N 个（字段名为历史遗留，实际按 scenario 生效）
  max_tokens: 50              # 统一生成长度/OSL（默认 50，可配置；触顶样本不丢弃）

run:
  repeats: 1                  # 每个样本测量轮数
  cache_mode: cold            # 冷缓存测量（推荐）
  isolate_models: true        # 切换模型前卸载其它模型
  restore_state: true         # 结束后卸载本次加载的模型
```

### 5.2 模型加载（默认配置）

模型按 oMLX 下载后的默认配置加载。**只允许** MoE 模型显式开启 `moe_expert_offload_enabled`：

```yaml
  - id: qwen3.6-35B-A3B
    load:
      optional: true            # 设置失败只记 warning，按默认配置继续
      settings:
        moe_expert_offload_enabled: true
```

- 其他字段（`turboquant_kv_enabled`、`dflash_enabled`、`mtp_enabled` 等）会被配置校验拒绝；
  性能测试不做任何加速/压缩的额外配置。
- 每次 run 会在 `omlx.json` 里记录模型的实际生效设置与 `feature_flags`，可确认是否处于默认配置。
- oMLX 的模型设置是持久化的：若历史 run 留下了 `turboquant_kv_enabled` / `dflash_enabled` 等
  非默认值，runner 会在加载前自动复位为 false（仅 MoE offload 允许为 true），复位动作见
  `output/<run_id>/logs/run.log`。`omlx.json` 的 `upload_projection` 会把本轮本地参数投影成
  oMLX 社区榜上传载荷的形状（只落盘、不上传）。
- 非 MoE 模型不要配置 `load.settings`（保持 `{}`）。
- MoE offload 的**常驻专家比例**采用 oMLX 默认值：`moe_expert_offload_resident_fraction = 0.25`
  （每层约 25% 的 experts 常驻，其余按需从 checkpoint 流式读取，解码变慢但精度不变）。
  基准**不写该字段**（`models.yaml` 白名单只允许 `moe_expert_offload_enabled`），run 会沿用模型
  当前的持久值；若在 oMLX 后台把某个 MoE 模型改成 0.5 等非默认值，该 run 属于非默认口径，
  结果中必须注明。实际生效值可在 `omlx.json` 对应模型的 settings 里核对。

## 6. 自定义测试集

```bash
mkdir -p dataset/test/my_test
cat > dataset/test/my_test/testset.yaml <<'YAML'
name: my_test
title: 我的测试集
scenarios:
  math_logic:
    dataset: custom
    dataset_params:
      path: dataset/test/my_test/my.jsonl
YAML
```

JSONL 字段：`question`、`answer`、`answer_type`（`numeric|mc|exact|...`）、可选
`choices`/`context`/`tools`。放入后 Web 下拉自动出现，或 `--testset my_test`。
性能测试只用 `question/context` 构建 prompt；`answer` 不参与判分（当前不判分）。

## 7. 结果解读

- 输出目录按「时间 + 测试方法 + 模型」分开：同一批多模型测试会生成多个
  `output/<batch_id>_performance_<model>/`，每个 bundle 内部仍是本模型的独立 records/summary。
  批次关系见 `output/batches/<batch_id>.json` 与 `output/index.json` 的 `batches`。
- `summary.overall[]`：每个**模型 × 上下文档位**一行的整体平均值（`context_length`
  字段区分 1K/8K/32K；TTFT、TG TPS、峰值内存、输入上下文大小等）。**不要跨档位
  再求平均**；旧 schema 2.0 run 无档位字段，Web 会标记“全部*”且不与新口径混比。
- **不可与官方榜横比**：默认模型级预热 1 次（`warmup_per_model=1`，per-cell 预热 0）、`max_tokens=50`、`repeats=1` 且逐请求冷缓存，
  与 vLLM / oMLX 官方 benchmark 的预热、生成长度、批处理口径不同；本系统数字只适合在
  **同口径、同 run 配置**下做模型间相对比较。需要更长生成时用 `--max-tokens`，但须与默认结果分开报告。
- `summary.cells[]`：每个 `model × scenario × context_length × dataset` 的分布
  （mean/median/std/p95；默认 8 场景 24 cells）。
- `summary.samples[]` 与 `samples.csv` 带 `context_length` 档位列（legacy 为 null）。
- `summary.samples[]` / `samples.csv`：**逐样本结果**，含 `input_context_tokens`、
  `input_context_chars`、TTFT、TG TPS、峰值内存、`finish_reason`/`length_capped`、状态。
- `summary.csv`：整体平均值表（每模型×档位一行，含 `context_length`、`n_length_capped`）；
  `cells.csv`：数据集粒度表。
- **输出触顶（`finish_reason=length`，达到 `max_tokens`）不是丢弃**：样本照常参与
  TTFT/内存/TG TPS 平均；`gen_tps/tpot` 用实际 `completion_tokens` 计算到停止前。
  报告/Excel/Web 的“触顶数”只是透明计数；仅 `completion_tokens < 16` 不参与 TG TPS
  统计（TTFT/内存仍统计），报告显示 `mean (n/有效数)`。
- `records.jsonl`：逐请求完整证据（请求、响应、指标、缓存状态）。
- `report.md`：人读摘要；`manifest.json`：配置/硬件/oMLX/模型/git commit；
  `hardware.json`/`omlx.json`：环境与配置快照。
- `analysis/`：`run.json`、`samples_long.csv`（长表）、`environment.json` 等二次分析投影；
  `output/index.json` 是全部 run 的目录索引（详见 [RESULTS.md](RESULTS.md)）。
- 峰值内存是 oMLX 服务进程树 `phys_footprint` 峰值（GB/GiB，含 Metal 分配），不是系统总内存；
  与 oMLX 自带 benchmark 的 `PeakMem`（`mlx.get_peak_memory()` 的 Metal 峰值）不是同一指标，
  后者通常低约 0.4–0.9 GB（经验值，本仓库未系统实测），不能逐位对比。
- 报告/Web 的环境页还给出每个模型的"进程内存基线 → 加载后"：基线是模型加载前的 oMLX 进程
  物理内存（会随进程历史/缓存保留在约 0.9～3.0 GB 间变化，不是固定值），加载差值才是本次模型
  及引擎缓冲的增量；测试系统自身（modelbench 进程）通常只占 100～200 MB（普通数据集）。

### 7.1 为什么勾选了两个模型，结果只有一个？

模型是**串行**加载测试的；如果某个模型加载失败（例如 oMLX 内存上限不足、模型文件缺失），
该模型会被跳过并写入 `summary.skipped`（同时出现在 Web 详情顶部警告与历史列表的「跳过 N」）。
其它模型的结果不受影响。可在 `output/<run_id>/logs/run.log` 查看完整失败原因；
在 Web 历史列表用 `已测/已选` 计数即可区分。
CLI `run` 启动前会对照 `/v1/models`：若所选模型不在服务端，会直接报错退出（exit 3），
不会跑到一半才发现 404；`./modelbench validate` 在线模式也会提示 `models.yaml` 与服务端的差异。

**最常见原因是模型清单过期**：Web 的可选模型来自 `models.yaml`，不会每次实时扫描模型目录。
若在 oMLX 后台删除/重命名了模型目录（例如把 `MiniCPM5-2B-MLX`、
`MiniCPM-V-4.7-1B-heretic` 换成官方 `MiniCPM-V-4.7-1B`），旧 id 仍会留在清单里，
加载时返回 `HTTP 404 Model not found` 并被记为 skipped。处理方式：

- Web：点模型卡片的 **「刷新模型」**，失效条目会标「服务端不存在」；再点 **「同步清单」**
  写回（确认框会列出新增/移除的 id）。
- CLI：`./modelbench validate`（在线）会对比 `models.yaml` 与 `GET /v1/models` 并告警
  （Web 的「刷新模型」使用同源、带加载状态的 `/v1/models/status`）。

2026-11-06 的 `web-20261106T111248Z-8d03` 批次即为此原因：两个已删除模型的 run 各
0 记录、0.6–0.7 s 结束，`summary.skipped` 里是 `HTTP 404: Model not found`。

### 7.2 内存不足（oMLX memory guard / HTTP 507）怎么排查？

先看**谁占用了内存**（本机实测示例）：

| 项 | 实测占用 |
|---|---|
| 系统总内存 / 可用 | 25.8 GB / 约 4.4 GB（已用 20.4 GB，swap 3.2 GB） |
| 其它 App（含 IDE/浏览器/DSH 等） | 占绝大部分；oMLX 动态上限因此只有 3.6–5.5 GB |
| oMLX 空载 / 加载 1B 模型 | 约 0.5 GB / 约 2.6 GB（模型权重 + 运行时 + KV） |
| 本测试系统（CLI run / Web 服务） | 典型 40–250 MB；长上下文数据集加载是例外（见下） |
| LongBench-v2 数据集加载 | 优化前峰值 **3.9 GB**；现在流式解析（约 0.6 GB 峰值），且 Web 样本浏览在一次性子进程中完成，Web 服务常驻仍约 37 MB |

结论：

- **模型切换本身没有泄露**：oMLX 日志会记录
  `Unloaded model: ..., freed=2.42GB, active_memory: 1.84KB (settled)`，
  卸载后 `current_model_memory=0`；本系统也会在 `restore_state` 时只卸载仍加载的模型。
- 失败通常是**系统整体可用内存不足** + 模型本身接近 oMLX 动态上限：例如 3.04 GB 的模型
  加上运行时/KV 后，动态上限只有 3.8 GB，就会触发
  `prefill_memory_exceeded`（HTTP 400）或加载失败（HTTP 507）。
- **并发 run 会互相驱逐模型**（旧版本问题）：两个 Web 任务同时跑时，oMLX 可能为加载 B
  而驱逐 A，A 的后续请求再重载失败（507）。现在 Web 任务已强制**串行排队**，同一实例
  不应再并发启动测试。
- 本系统 `memory_errors` 会写入 `summary.json`，并在 `warnings` 中给出一次明确提示；
  内存守卫拒绝不再盲目重试。

处理建议：

```bash
./modelbench hardware          # 查看可用内存 / swap / oMLX 动态 ceiling
```

1. 关闭不必要的 App（尤其是浏览器/IDE/其它本地模型服务），释放可用内存；
2. 在 oMLX 设置中适当提高 `memory_guard_tier`（safe → balanced → aggressive）；
3. 换更小模型，或先用 `--models` 单模型、`--samples` 限流；
4. 长上下文（`long_context`）单独跑；测试运行期间**不要**在 Web 打开长上下文样本浏览器
   （预览已在子进程隔离，但仍会短暂占用系统内存）；
5. 不需要手工确认并发：Web 同进程内已串行，CLI/Web 跨进程通过 `output/.benchmark.lock`
   （非阻塞文件锁）拦截，第二个 run 会快速失败并提示持有者 PID；如看到该错误，等待前一个 run
   结束或先停止它（AGENTS 不变量 10）。

## 8. 故障排查

| 现象 | 处理 |
|---|---|
| `validate` 报 oMLX 不可达 | 启动 oMLX；检查 `server.base_url` |
| 401 / invalid key | `export OMLX_API_KEY=...` 或检查 `~/.omlx/settings.json` |
| 模型设置 400 | 检查是否只配了 `moe_expert_offload_enabled`；非 MoE 模型不要配 |
| OOM / 加载失败（HTTP 507） | 看 `./modelbench hardware` 的可用内存与 oMLX ceiling；关闭其它 App、提高 memory_guard_tier、换小模型；MoE 开 offload |
| `prefill_memory_exceeded`（HTTP 400） | 系统可用内存低于模型+KV 需求；释放内存、提高 guard tier、减少 `--samples`/上下文；`summary.memory_errors` 会计数 |
| 回答为空/`finish_reason=length` | 增大 `--max-tokens`，或关闭思考模式 |
| 样本被跳过 | 预构造样本超过模型窗口会跳过并写入 `summary.skipped`（不截断）；legacy 样本才头尾截断并记录 `truncated` |
| `cache clear failed` | admin key 不可用；检查 oMLX 管理接口，cell 会中止以避免假冷数据 |
| `contaminated` | 冷模式仍命中缓存；系统重试后剔除，不应出现在聚合 |
| 端口被占用 | `--port 8766` |
| 历史结果太多 | 手动归档 `output/`，不要删除正在运行的目录 |

## 9. 数据与合规

数据集来源/许可见 [DATASETS.md](DATASETS.md) 与 [dataset/README.md](../dataset/README.md)；
`dataset/raw/`、`dataset/test/**/testset.jsonl`、`output/` 是证据目录，不要手改或提交 git；
精度数据集（6,589 行 = 250×6 + 889 + 4,200）是**离线资产**（`status=offline-asset`、runner 未接入、judge 仅契约），
保留在磁盘但默认不参与运行；性能场景名 `chinese_long_context` 是中文长文（原 `multilingual` 更名），
不要与精度侧 42 语种（Global-MMLU 42 × 100 配对题）的 `multilingual` 混淆。
