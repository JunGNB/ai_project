# 架构与实现设计

> 面向维护者与 AI Agent。测试口径见 [TEST_PLAN.md](TEST_PLAN.md)，操作见
> [USER_GUIDE.md](USER_GUIDE.md)，历史方案推导见 [archive/DESIGN.md](archive/DESIGN.md)。

## 1. 设计目标

- 统一测量：`模型 × 测试集数据集` 下逐样本采集性能（TTFT / TG TPS / 峰值内存 / 输入上下文大小），
  冷缓存可复现。
- 可替换的应用后端：当前 oMLX；后续 Ollama/LM Studio 只替换 `src/modelbench/adapters/`。
- 配置与执行分离：`perf` 配置只决定样本数与生成长度，runner 只负责执行与记录。
- 默认模型配置：除 MoE 模型的 `moe_expert_offload_enabled` 外，不改动 oMLX 模型设置。
- 可被 AI 持续演进：确定性门禁 + 任务板 + 硬契约（见根目录 `AGENTS.md`）。

> **版本说明**：0.3 起删除旧策略层（`strategies.yaml`、`StrategyDef`、`apply_strategy`、
> 上下文长度矩阵）。当前唯一测试形式是性能基准；准确性判分代码保留但不参与运行。
> `dataset/test/accuracy/testset.jsonl`（6,589 行）是**离线数据资产**（`status=offline-asset`，
> runner 未接入、judge 仅契约），不是可运行测试。

## 2. 目录与分层

```
src/modelbench/                 # 可安装包（src 布局）
  __init__.py                   # 版本
  paths.py                      # 资源/项目根路径解析
  domain/                       # 纯领域：无网络/文件 IO
    models.py                   # 全部数据契约（Sample/RequestSpec/SampleRecord/...）
    metrics.py                  # 指标公式 + cell/overall 聚合 + 逐样本投影
    context_windows.py          # 上下文窗口筛选（1K: 0~1.5K；其他 N: N±0.5K）
    scenarios/                  # 场景适配（含新增的 ChineseLongContextScenario；计分当前不调用）
    evaluators/                 # 纯函数判分器（当前不参与运行）
  application/                  # 用例编排（依赖 domain 的契约）
    ports.py                    # AppBackend 协议 + ModelInfo + BackendError
    plan.py                     # 测试集应用（场景/数据集覆盖）+ 计划概览
    batch.py                    # BatchRunner：一次动作 -> 每模型一个 bundle
    benchmark.py                # Orchestrator：模型生命周期、逐样本请求、记录
  adapters/                     # 外部世界实现（IO）
    config/                     # YAML 配置加载、测试集
    datasets/                   # 11 个 DatasetAdapter + registry + 下载 helper（默认只用 perf_jsonl）
    omlx/                       # client / cache / backend / environment（应用在测）
    storage/                    # RunWriter/catalog/excel_report：bundle、索引、Excel 总结
    host/                       # hardware 档案、CPU/RSS/phys_footprint 采样
  interfaces/                   # 交付层
    cli.py                      # 命令行
    web/                        # 本地控制台（server/jobs/static）
    analysis/                   # JSON 结果对比（compare + CLI）
  resources/config/             # 随包分发的 YAML（default/models/scenarios）
modelbench                        # 唯一启动入口（setup/web/run/…）
tool/dataset/                   # 测试集构造/子集/校验（build_perf.py、build_acc.py、verify.py）
tool/download/                  # 工程外部下载工具（不被运行时导入）
test/                           # DT 测试（unit + fake-server integration）
dataset/raw/                    # 上游原始文件（只读）
dataset/test/<name>/            # 测试集：testset.yaml + 预构造/自定义样本
dataset/test/performance/       # 性能样本（testset.jsonl + testset_10.jsonl，tool/dataset 生成）
dataset/test/accuracy/          # 精度离线资产（testset.jsonl + manifest.json；无 testset.yaml、不接入 run）
output/<run_id>/                # 运行结果 bundle
```

分层依赖（只允许向下）：

```
interfaces  →  application  →  domain
                    ↑
               adapters（实现 application.ports 的协议，被 interfaces 组装）
```

- `domain/` 不 import `application/`、`adapters/`、`interfaces/`；可独立单测。
- `application/` 负责编排，不直接拼接 HTTP/YAML/文件格式。
- `adapters/` 封装 oMLX、数据集、存储、主机观测；新增后端只改这里。
- `interfaces/` 只做输入解析与展示；Web Job 复用 `application.benchmark`，不复制测量逻辑。
- `resources/config/` 是数据不是代码；通过 `src/modelbench/paths.py` 定位。

> 完整目录清单（自动生成，勿手改）：[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)。
>
> **命名说明**：源码根用 `src/`，包名 `modelbench`。Python 标准库已有 `code` 模块，
> 若包名为 `code` 会导致安装后的 console script/导入被标准库遮蔽；`src/modelbench`
> 是标准 src 布局，也避免 `src` 本身作为包名的歧义。

## 3. 核心数据流

```
Web / CLI
  → modelbench.adapters.config.settings.load_config()
  → modelbench.application.plan.apply_testset()      # 可选：覆盖场景数据集
  → modelbench.application.batch.BatchRunner.run()
       逐模型创建独立 bundle：output/<batch_id>_<method>_<model>/
       内部 Orchestrator: isolate → MoE 设置 → 加载 → 逐数据集/样本（native context, cold cache）
       批次元数据: output/batches/<batch_id>.json + index.batches
  → RunWriter.finalize() → output/<run_id>/{records,summary,cells,samples,report,...}
```

### 3.1 模型生命周期（关键）

`Orchestrator._ensure_model_ready`：

1. 读取 `/v1/models/status`（是否已加载、上下文窗口）；
2. `isolate_models=true`：卸载其它模型；
3. 读取模型当前生效 settings，复位历史遗留的加速开关（TurboQuant/DFlash/MTP/SpecPrefill/ANE
   → false）；若模型配置了 `load.settings`（仅允许 `moe_expert_offload_enabled`）则一并应用：
   卸载已加载实例 → `PUT /admin/api/models/<id>/settings` → 重新加载（保证 engine 级参数生效）；
   失败按 `optional` 中止或记 warning；
4. `wait_until_loaded`；采集该模型 admin 返回的 settings/feature flags；
5. run 结束 `restore_state=true` 卸载本次加载的模型并清缓存。

## 4. 数据契约（`src/modelbench/domain/models.py`）

| 对象 | 作用 |
|---|---|
| `Sample` | 数据集→场景：`sample_id/dataset/scenario/question/choices/context/answer/answer_type/tools/metadata` |
| `RequestSpec` | 场景→客户端：messages/tools/max_tokens/`chat_template_kwargs` + 上下文元数据 |
| `StreamResult`/`Usage` | 客户端流式结果：文本/推理/工具调用 + 时间戳 + usage（含 cached_tokens） |
| `SampleRecord` | 最小审计单元：请求、响应、指标、缓存状态、模型快照、phase/status |
| `PerfMetrics` | 包含 TTFT/TG TPS/CPU/RSS，以及 `input_context_tokens/input_context_chars` |
| `CellAggregate` | `model×scenario×context_length×dataset` 聚合：perf 分布、n_valid、stability |
| `OverallAggregate` | `model×context_length` 平均值（对外展示的主结果；1K/8K/32K 分行，不跨档位合并） |
| `RunManifest` | 配置快照、硬件、oMLX、模型状态、git commit、时间 |
| `SCHEMA_VERSION` | 当前 `2.1`（`overall` 增加 `context_length`）；破坏性字段变更需升版本（见技术债） |

不变量：

- 冷模式 measured 记录必须 `cleared_before_request=true` 且 `cached_tokens==0`；缺失/命中
  视为污染，不进入聚合；
- warmup（若显式配置）不计分、不进聚合；`gen_tps/tpot` 仅 `completion_tokens≥16` 输出；
- 输出达到 `max_tokens`（`finish_reason=length`）的样本**不丢弃**：TTFT/内存/吞吐照常聚合，
  `gen_tps/tpot` 按实际生成 token 计算；`n_length_capped` 记录触顶样本数供审计；
- 性能模式默认在每条请求末尾追加 decode guard 长输出指令（`perf.decode_guard`，
  原文写入 `record.request`/`context_meta`）：它是追加的一条 user 消息（183 字符，
  实测约 +39 prompt token），只保证 TG TPS 有解码窗口；档位/声明 tokens 不变，但
  `input_context_tokens` 会包含该协议开销；关闭仅用于诊断或历史口径复现；
- 性能模式 `score` 恒为 null（不调用判分器）；缓存模式不混算；
- 每个 measured 记录必须带 `input_context_tokens`（缺失时回落 `request.actual_prompt_tokens`）；
- 预构造样本（`Sample.metadata.prebuilt`）在 `SampleRecord.context_length` 上记录声明的档位
  （1024/8192/32768），并驱动 cell 分组；legacy 样本无该字段时为 `None`。

## 5. 性能测试配置与测试集

- `perf` 配置（`src/modelbench/resources/config/default.yaml`）：
  - `samples_per_dataset`：每个数据集取多少样本；`null` = 全部可用样本；
  - `max_tokens`：统一生成长度/OSL（默认 50，CLI/Web 可配置；达到上限的样本标记 `finish_reason=length` 但**不丢弃**，TG TPS 按实际生成 token 计算）。
- `run` 配置：缓存/预热/重试/模型隔离/repeats（默认 1）等执行参数。
- 测试集 = `dataset/test/<name>/testset.yaml`（默认 `performance`），可把场景指向
  `custom` JSONL；`apply_testset()` 只做场景 `dataset` / `dataset_params` 覆盖。
- 默认 8 个场景都指向 `perf_jsonl` 适配器，通过 `dataset_params.scenario` 选择切片
  （含已更名的 `chinese_long_context`；精度侧的 `multilingual` 不在默认 run 中）；
  样本来自 `dataset/test/performance/testset.jsonl`（480 条），`params.buckets` 可按
  1k/8k/32k 过滤，`limit` 限制样本数。适配器流式解析，`plan()` 用缓存索引给出
  `(scenario, dataset, context_length, n)` 估算。
- 抽样：预构造样本按 `(context_length, lang)` 分层、最大余数法分配 N、组内按 `sample_id`
  取前 k（`stratified_sample`，跨 run 可复现）；legacy 样本才 `sample_id` 排序取前 N
  （`deterministic_sample`）。
- **没有上下文长度矩阵**：不按目标长度填充/对齐；但预构造样本自带 1K/8K/32K 档位，
  `context_length` 作为 **cell 维度**（与聚合键一致），每个
  `(scenario, dataset, context_length)` 单独成 cell；没有档位的 legacy 样本仍为 `null`。
- **cell 执行粒度**：runner 在模型加载后先发 1 条模型级短预热请求
  （`run.warmup_per_model=1`，仿 oMLX QUICK，丢弃且不记录），再把每个场景的选中样本按
  `(sample.dataset, sample.metadata.context_length)` 分组，逐个 group 调用一次 cell
  （per-cell `warmup_requests=0`），`cell_total = 模型数 × 总 group 数`，与
  `aggregate_records` 的 cell key 完全一致。

## 6. 环境采集

`src/modelbench/adapters/omlx/environment.py` 在 run 开始时投影 oMLX 的本地参数：

- 硬件：`device-info`（chip/chip_variant/memory_gb/gpu_cores）+ `utils/hardware` 等价值；
- oMLX：版本、base_url、全局设置（密钥脱敏）；
- 模型：admin 模型目录（engine/model_type/quantization/上下文/大小）；
- 设置：每模型实际生效 settings 的 oMLX 上传白名单投影；
- 特性：由 settings 推导的 `feature_flags`（默认配置应为空）。

写入 `hardware.json` / `omlx.json` / `models.json` 与 `manifest.json` 快照，
使结果可回答"在哪台机器、什么配置、有没有开加速"。

## 7. Web 控制台

`src/modelbench/interfaces/web/server.py`（stdlib `ThreadingHTTPServer`，仅回环）：

- `GET /api/overview`：服务器状态、模型（含 `on_server` 实时校验标记与 `model_sync`
  差异摘要）、测试集列表（`dataset/test/<name>` 目录、标题、
  场景数与样本/测试项总数）、perf 默认值、
  历史 run（来自 `output/index.json` 索引）、catalog 信息；
- `GET /api/models`：实时检索 oMLX `GET /v1/models/status`，返回配置清单与
  `model_sync`（`configured_missing`/`server_only`/`in_sync`/`models_path`）；
  服务端不可达时返回 `error` 且不把任何模型判为失效；
- `POST /api/models/sync`：`{apply?: bool}`。`apply=false` 只预览差异；
  `apply=true` 把差异文本级写回 `models.yaml`（保留注释、自动 alias、原子写入、
  写前重新解析校验），有运行中任务时返回 409；
- `GET /api/testsets/<testset>/scenarios/<scenario>/samples`：按需加载样本列表
  （id/问题摘要/上下文大小），供 Web 勾选具体样本；
- `POST /api/run`：`{models, testset, scenarios, samples?, scenario_samples?,
  sample_ids?, max_tokens?, repeats?, cache_mode?, thinking_mode?}`；提交前用实时清单
  校验模型存在，缺失模型直接 400（避免 404 空 run）→ `JobManager`；
- `GET /api/jobs`、`GET /api/jobs/<id>`：任务列表与单个任务状态/当前数据集/记录数/
  日志尾，以及 `plan`/`progress`（模型 done/total、测试项 done/total、样本 done/total、
  当前项进度、百分比）；页面加载时用 `GET /api/jobs` 恢复运行中任务（刷新不丢状态）；
  `POST .../cancel` 协作取消；
- `POST /api/estimate`：不入队，按当前模型/场景/样本数/上下文窗口返回 `plan_counts`
  （cells_total/samples_total/repeats），Web `#estimate` 与后端计划同源；
- `GET /api/runs`, `GET /api/runs/<id>`：读取 summary/manifest/omlx/report 与文件清单；
- `GET /api/runs/<id>/file?path=...`：安全读取 run 内文件（文本内联、大文件/原始记录下载）；
- `POST /api/runs/<id>/reveal`、`POST /api/output/reveal`、`POST /api/testsets/reveal`：
  在 Finder 中打开 run / output / `dataset/test` 目录；
- `GET /api/index` / `POST /api/index/rebuild`：读取/重建 `output/index.json`；
- 静态页面 `src/modelbench/interfaces/web/static/{index.html,style.css,app.js}`。
- 绑定安全：`main()` 默认只接受回环主机（127.0.0.1/localhost/::1），非回环需显式
  `--allow-external`（控制台无鉴权）。
- 启动器：`./modelbench web` 默认在服务绑定后自动打开默认浏览器（`--no-open` 关闭）；
  关闭浏览器标签不会停止服务（HTTP 服务独立运行），`Ctrl+C` 或
  `--idle-timeout SECONDS`（无请求且无运行任务时自动退出）用于停止。

Job 线程复用 `Orchestrator(progress_cb, cancel_event)`，不复制测量逻辑；取消通过
`RunCancelled` 在模型/数据集/样本/请求边界检查，仍会执行 finally 清理并写 summary。
Web 的 `#estimate` 使用 `/api/overview` 下发的每场景 dataset plan（缓存索引，不重复扫描
JSONL），默认显示 480 次测量 / 24 个测试项。
**同一 oMLX 实例上的 Web job 通过进程内锁串行执行**（否则两个 run 会互相驱逐模型）；
CLI/Web 跨进程由 `application/locking.py` 的 `<output>/.benchmark.lock` 非阻塞文件锁拦截，
第二个 run 快速失败并提示持有者 PID；
`perf_jsonl` 以 UTF-8 逐行流式解析约 27.5 MB 性能样本，Web 样本浏览按需加载并在进程内缓存
预览元数据；估算只读取缓存索引，不把全量 dict 常驻。精度 testset（约 76 MB / 6,589 行）
不被默认 run 读取，只有离线工具/未来接入时才解析。

## 8. 结果分析

`src/modelbench/interfaces/analysis/compare.py` 读取多个 `summary.json`，按
`(model, context_length)` 对齐 `overall` 平均值输出 Markdown 对比表（不同档位不合并）；
命令行 `./modelbench analyze`。
`src/modelbench/adapters/storage/writer.py` 负责单模型 run 的
`summary.json/summary.csv/cells.csv/samples.csv/report.md` 与 `analysis/` 投影；
`src/modelbench/application/batch.py` 把一次多模型测试拆成每模型一个 bundle，
并写批次元数据（`output/batches/<batch_id>.json`）与 Excel 总结
（`src/modelbench/adapters/storage/excel_report.py` → `output/result_analysis/.../summary.xlsx`）；
`src/modelbench/adapters/storage/catalog.py` 维护 `output/index.json` 目录索引
（写入/重建/安全文件解析），`./modelbench runs` / `./modelbench index` 为 CLI 入口。
详细的数据管理约定见 [RESULTS.md](RESULTS.md)。

## 9. 扩展点

| 扩展 | 做法 |
|---|---|
| 新数据集 | 实现 `DatasetAdapter.ensure_downloaded/load`，在 `datasets/registry.py` 注册 |
| 新场景 | 实现 `ScenarioAdapter.build_request/score`，注册到 `scenarios/registry.py` |
| 新判分（离线） | `evaluators/` 纯函数，输入 `Sample + RawResponse`（当前不接入 run） |
| 新应用后端 | 实现 `src/modelbench/application/ports.py` 的 `AppBackend`，提供 chat/tokenize/load/cache/settings |
| 新测试集 | 在 `dataset/test/` 建目录 + `testset.yaml` + 数据文件（`datasets.testset_dir` 可改根） |
| 新模型 | `src/modelbench/resources/config/models.yaml` 增加 id/alias（MoE 才加 load.settings） |

## 10. 可复现与质量

- 冷缓存逐请求清理 + `cached_tokens` 校验；污染剔除；
- 固定抽样 `deterministic_sample`、可配置 seed、记录 manifest（含 git commit）；
- 门禁：`python -m harness.gate`（导入/编译/测试/validate/文档链接/密钥/仓库卫生，
  `--live` 额外跑真实 smoke）；`pytest test/` 覆盖 fake server 全流程。

## 11. 技术债与演进

| 优先级 | 事项 | 说明 |
|---|---|---|
| P1 | Schema 迁移器 | `SCHEMA_VERSION=2.1` 无升级/迁移工具，旧 run 兼容需补齐（Web 已对 2.0 的 `overall` 降级显示“全部*”） |
| P1 | 后端完全解耦 | runner 仍直接使用 OMLXClient 类型；改为 AppBackend 注入 |
| P1 | 环境字段对齐 | oMLX 上传白名单会随版本变化，需同步维护 allowlist |
| P2 | 报告可视化 | 目前 Web 展示表格，缺曲线图（TTFT/上下文长度散点） |
| P2 | 并发/吞吐 | 串行测量，不覆盖并发服务能力 |
| P2 | 准确性回归 | 判分器保留但未接入；若未来恢复，需要独立 run 模式和 CI 口径 |
