# AGENTS.md — AI Agent 工作契约

本仓库是 oMLX 本地模型**性能**基准测试系统（逐样本性能 + 环境快照 + 本地 Web 控制台）。
任何 AI 在改代码前必须读本文件、[docs/design/30-architecture.md](docs/design/30-architecture.md)、
[docs/spec/21-test-plan.md](docs/spec/21-test-plan.md)、[docs/spec/20-methodology.md](docs/spec/20-methodology.md)。

## 1. 环境与命令

```bash
cd <clone 目录>                          # 路径含空格/中文时加引号
PY=.venv/bin/python                     # Windows: .venv\Scripts\python.exe
$PY -m harness.gate                 # 提交前门禁（必须 PASS）
./modelbench validate --offline          # 配置/数据集/测试集
$PY -m pytest test/                 # DT 测试
./modelbench web           # 本地控制台 :8765
```

> 跨平台：macOS/Linux 用 `./modelbench`；Windows 用 `modelbench.cmd`（cmd.exe）或
> `.\modelbench.ps1`（PowerShell），命令集相同。安装/依赖见 [docs/guides/40-install.md](docs/guides/40-install.md)，
> `dataset/raw` 初始化与测试集构建见 [dataset/README.md](dataset/README.md)。

## 2. 硬不变量（违反即无效）

1. **冷缓存**：measured 请求前清 hot+SSD，`cached_tokens==0`；缺失/命中=污染，不进聚合；
   清理失败必须中止 cell。
2. **warmup/污染/错误** 不参与聚合；缓存模式不混算。
3. `gen_tps/tpot` 仅在**可信解码窗口**输出：`completion_tokens≥16` 且解码窗口 ≥50ms、
   `finish_reason≠tool_calls`、值 ≤10,000；被抑制时置 null 并在 `metrics.gen_tps_invalid`
   记原因（`tool_calls`/`decode_window_too_short`/`missing_timestamps`/`invalid_timestamps`/
   `implausible_tps`），聚合给 `n_gen_tps_invalid`。性能模式 **不判分**，
   `SampleRecord.score` 恒为 null（判分器代码保留但不接入 run）。
4. `dataset/raw/`、`dataset/test/**/testset.jsonl`、`dataset/.cache/`、`output/` 是证据/输入
   目录：只读、不提交 git；数据问题走 `tool/dataset/{performance,accuracy}` 重建。测试集 YAML
   （`dataset/test/<name>/testset.yaml`）随仓库提交。
5. 新数据集/场景/判分器必须走 `src/modelbench/domain/` 的抽象类与注册表；不要在 runner 堆 `if scenario`。
6. 新应用后端必须满足 `src/modelbench/application/ports.py` 的**结构契约**：
   `backend_contract_errors(backend) == []`（`REQUIRED_BACKEND_METHODS` 鸭子类型检查，
   不强制继承 ABC；`test_core` 对内置后端全方法断言），并补齐 Orchestrator 实际使用的
   端口外能力（见 ARCHITECTURE §3.5/§8.2）；不要把 Ollama 等逻辑塞进 oMLX 模块。
7. **策略层已删除**：禁止重新引入 `strategies.yaml`、`StrategyDef`、`apply_strategy`
   或上下文长度矩阵；测试计划只由 `perf` 配置（样本数/生成长度）、测试集与模型决定。
8. **模型默认配置**：除 MoE 模型的 `moe_expert_offload_enabled` 外，不得写入任何 oMLX
   模型设置；TurboQuant / DFlash / MTP / SpecPrefill / ANE 等加速配置一律禁止。
   设置经 admin API 应用并重载；失败按 `optional` 显式处理，禁止静默忽略。
9. 不静默降级：跳过/失败/降级进 `status`、`run.json.outcome.skipped/warnings` 或日志；
    oMLX 内存守卫拒绝（prefill_memory_exceeded / HTTP 507）不得盲目重试，需记录并告警。
10. **单实例串行（双锁）**：同一 oMLX 实例上禁止并发 benchmark job；Web JobManager 必须串行排队，
    跨进程必须先取 **endpoint 锁**（系统临时目录
    `modelbench-locks/<sha256(normalized base_url)[:16]>.lock`，normalize = scheme+host+port）
    再取 **output 根锁**（`<output>/.state/benchmark.lock`）。同一 `base_url` 即使不同 output
    根也互斥；不同 `base_url` 可并发。锁不可用/被占用时快速失败，禁止绕过。
11. 测试是契约：不删/弱化测试；语义变更加回归用例；门禁不通过不得声称完成。
12. **文档是契约**：改 CLI/路径/配置/结构必须同步文档；`docs_contract` 会校验 `docs/**/*.md`、
    `harness/prompts/*.md`、模块、路径、flag、根启动器 `./modelbench`、`docs/design/33-project-structure.md`
    与**已删除产物名黑名单**（白名单见 `docs/process/50-validation-plan.md` V-7）；结构变化后运行
    `make docs-write`。
13. **逐样本可追溯 + 固定样本口径**：每个 selected 样本必须有一条 measured 记录（或进
    `run.json.outcome.skipped` 说明原因），记录含 `input_context_tokens`
    （`tokenized_prompt_tokens` 为 actual 值）、TTFT/TG TPS/峰值内存与
    `context_meta.isl_tolerance`；聚合只能由这些记录复算，不得只存平均值。
    **一套固定 testset 全设备/全模型复用**（不做 per-model 构造、不做运行时裁剪/补齐，
    32K target 32,768 不特判）；**actual ISL 不触发跳过**：所有能测样本都 measured，均值只
    使用 actual 在 target ±10%（1K 921–1,126；8K 7,372–9,011；32K 29,491–36,044）的入带
    样本，出带样本保留但不进均值（`n_isl_in_band`/`n_isl_out_of_band`，in-band 不足标
    `insufficient_samples`）；跨系统比较见 `docs/spec/13-testset-portability.md`。

## 3. 工作流

```bash
$PY -m harness.taskctl list --status pending
$PY -m harness.taskctl claim T-00X --owner "$AGENT_NAME"
# 先写/更新测试 → 实现 → 文档
$PY -m harness.gate                      # 必须跑完整 10 项；--only 子集不能用于 complete --gate
$PY -m harness.taskctl complete T-00X --gate --note "变更/测试/证据"
git add -A && git commit -m "T-00X: <summary>"
```

`complete --gate` 校验的是**完整默认门禁**（`imports/compile/tests/validate/
dataset_integrity/perf_plan/docs_links/docs_contract/secrets/hygiene`）；用 `--only` 生成的
子集报告会被拒绝并提示缺失检查，需要重跑完整 `harness.gate`。

小修（typo/注释）可直接改但仍需跑门禁；行为/配置/schema 变更必须先建任务。

## 4. 代码地图

| 改什么 | 位置 |
|---|---|
| 指标/聚合（overall/cell/逐样本） | `src/modelbench/domain/metrics.py` |
| 执行/缓存/重试/取消/样本选择 | `src/modelbench/application/benchmark.py`、`src/modelbench/adapters/omlx/cache.py` |
| SSE/admin/模型生命周期 | `src/modelbench/adapters/omlx/{client,backend}.py` |
| 环境/配置采集（硬件 + oMLX） | `src/modelbench/adapters/omlx/environment.py`、`src/modelbench/adapters/host/hardware.py` |
| 场景/prompt 构建 | `src/modelbench/domain/scenarios/` |
| 数据集/下载 | `src/modelbench/adapters/datasets/`、`tool/download/` |
| 判分（当前不接入 run） | `src/modelbench/domain/evaluators/` |
| 配置/测试集 | `src/modelbench/adapters/config/{settings,testsets}.py`、`src/modelbench/resources/config/`、`src/modelbench/application/plan.py` |
| Web（目录树/样本勾选/文件导航） | `src/modelbench/interfaces/web/` |
| 结果目录索引与二次分析 | `src/modelbench/adapters/storage/{catalog,writer,blobstore,tools,excel_report,layout}.py`、`docs/design/31-output-contract.md` |
| 结果分析 | `src/modelbench/interfaces/analysis/` |
| 单实例双锁 | `src/modelbench/application/locking.py`（output 根锁 + endpoint 锁） |
| 输入大小容差/窗口判定 | `src/modelbench/domain/context_tolerance.py`、`application/benchmark.py` |
| 门禁/任务板 | `harness/` |

## 5. 测试要求

- 新逻辑覆盖边界（空/缺失/歧义/超时/失败）。
- 改 runner/cache/writer：扩展 `test/test_integration.py` fake-server 用例
  （逐样本记录、输入上下文、环境文件、无判分）。
- 改配置/测试集/样本选择：更新 `test/test_perf.py` 的矩阵/抽样断言。
- 改 Web/CLI：至少验证 `/api/overview`、`--dry-run`、命令退出码。
- 真实性能结论必须来自独立 run_id，并在任务 note 引用。

## 6. 多 Agent 与验证

- 一个任务同时一个 owner；`scope` 是写作用域，冲突先完成/重开任务。
- 实现者只改 scope；验证者只读（`/tmp` 或报告），不得自写自判。
- 验证者用 [harness/prompts/verifier.md](harness/prompts/verifier.md)，重点复算
  `run.json.outcome` / `aggregates.json` / `records.jsonl` / `samples.csv` 的一致性、冷缓存字段、
  样本数与 skipped 对账、TG TPS 有效性（`n_gen_tps_invalid`）、actual ISL 记录与
  ±10% 入带聚合（`isl_band`/`n_isl_in_band`/`n_isl_out_of_band`/`insufficient_samples`）、
  固定 testset SHA、双锁互斥与模型加载设置是否真实默认/MoE-only；V&V 判据见 [docs/process/50-validation-plan.md](docs/process/50-validation-plan.md)。

## 7. 完成标准

- [ ] `acceptance` 全部满足；
- [ ] `pytest test/` 全绿、`harness.gate` PASS；
- [ ] 测量/产物变更时 `./modelbench verify` 通过（含 gen_tps/ISL 统计一致）；
- [ ] 文档同步（按 `docs/README.md` 索引：`spec/20-methodology`、`spec/21-test-plan`、
      `design/30-architecture`、`design/31-output-contract`、`guides/41-user-guide` 等视变更，
      涉及验证口径时同步 `docs/process/50-validation-plan.md`）；
- [ ] `taskctl complete --gate`、git commit（不含 dataset 证据/output）。

## 8. 需要人类确认

删除/重写 `dataset/test/performance/testset.jsonl`、`dataset/raw/` 或 `output/`；
修改测量语义（缓存/指标/schema/样本选择）；下载 >1GB 数据；
改硬件/系统/启动停止 oMLX。阻塞时：
`$PY -m harness.taskctl block T-00X --reason "..."`。
