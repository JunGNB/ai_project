# AGENTS.md — AI Agent 工作契约

本仓库是 oMLX 本地模型**性能**基准测试系统（逐样本性能 + 环境快照 + 本地 Web 控制台）。
任何 AI 在改代码前必须读本文件、[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)、
[docs/TEST_PLAN.md](docs/TEST_PLAN.md)、[docs/METHODOLOGY.md](docs/METHODOLOGY.md)。

## 1. 环境与命令

```bash
cd "/Users/huawei/Desktop/模型管家相关资料/modelbenchmark"
PY=.venv/bin/python
$PY -m harness.gate                 # 提交前门禁（必须 PASS）
./modelbench validate --offline          # 配置/数据集/测试集
$PY -m pytest test/                 # DT 测试
./modelbench web           # 本地控制台 :8765
```

## 2. 硬不变量（违反即无效）

1. **冷缓存**：measured 请求前清 hot+SSD，`cached_tokens==0`；缺失/命中=污染，不进聚合；
   清理失败必须中止 cell。
2. **warmup/污染/错误** 不参与聚合；缓存模式不混算。
3. `gen_tps/tpot` 仅在 `completion_tokens≥16` 输出；性能模式 **不判分**，
   `SampleRecord.score` 恒为 null（判分器代码保留但不接入 run）。
4. `dataset/raw/`、`dataset/test/**/testset.jsonl`、`dataset/.cache/`、`output/` 是证据/输入
   目录：只读、不提交 git；数据问题走 `tool/dataset` 重建。测试集 YAML
   （`dataset/test/<name>/testset.yaml`）随仓库提交。
5. 新数据集/场景/判分器必须走 `src/modelbench/domain/` 的抽象类与注册表；不要在 runner 堆 `if scenario`。
6. 新应用后端实现 `src/modelbench/application/ports.py` 的 `AppBackend`；不要把 Ollama 等逻辑塞进 oMLX 模块。
7. **策略层已删除**：禁止重新引入 `strategies.yaml`、`StrategyDef`、`apply_strategy`
   或上下文长度矩阵；测试计划只由 `perf` 配置（样本数/生成长度）、测试集与模型决定。
8. **模型默认配置**：除 MoE 模型的 `moe_expert_offload_enabled` 外，不得写入任何 oMLX
   模型设置；TurboQuant / DFlash / MTP / SpecPrefill / ANE 等加速配置一律禁止。
   设置经 admin API 应用并重载；失败按 `optional` 显式处理，禁止静默忽略。
9. 不静默降级：跳过/失败/降级进 `status`、`summary.skipped/warnings` 或日志；
    oMLX 内存守卫拒绝（prefill_memory_exceeded / HTTP 507）不得盲目重试，需记录并告警。
10. **单实例串行**：同一 oMLX 实例上禁止并发 benchmark job；Web JobManager 必须串行排队，
    否则两个 run 会互相驱逐模型、污染结果。
11. 测试是契约：不删/弱化测试；语义变更加回归用例；门禁不通过不得声称完成。
12. **文档是契约**：改 CLI/路径/配置/结构必须同步文档；`docs_contract` 会校验模块、路径、
    flag、根启动器 `./modelbench` 与 `docs/PROJECT_STRUCTURE.md`；结构变化后运行 `make docs-write`。
13. **逐样本可追溯**：每个 selected 样本必须有一条 measured 记录（或进 `summary.skipped`
    说明原因），记录含 `input_context_tokens`、TTFT/TG TPS/峰值内存；聚合只能由这些记录复算，
    不得只存平均值。

## 3. 工作流

```bash
$PY -m harness.taskctl list --status pending
$PY -m harness.taskctl claim T-00X --owner "$AGENT_NAME"
# 先写/更新测试 → 实现 → 文档
$PY -m harness.gate
$PY -m harness.taskctl complete T-00X --gate --note "变更/测试/证据"
git add -A && git commit -m "T-00X: <summary>"
```

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
| 结果目录索引与二次分析 | `src/modelbench/adapters/storage/{catalog,writer,excel_report}.py`、`docs/RESULTS.md` |
| 结果分析 | `src/modelbench/interfaces/analysis/` |
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
  summary/overall/samples、冷缓存字段、样本数与抽样、模型加载设置是否真实默认/MoE-only。

## 7. 完成标准

- [ ] `acceptance` 全部满足；
- [ ] `pytest test/` 全绿、`harness.gate` PASS；
- [ ] 文档同步（TEST_PLAN/METHODOLOGY/USER_GUIDE/ARCHITECTURE 视变更）；
- [ ] `taskctl complete --gate`、git commit（不含 dataset 证据/output）。

## 8. 需要人类确认

删除/重写 `dataset/test/performance/testset.jsonl`、`dataset/raw/` 或 `output/`；
修改测量语义（缓存/指标/schema/样本选择）；下载 >1GB 数据；
改硬件/系统/启动停止 oMLX。阻塞时：
`$PY -m harness.taskctl block T-00X --reason "..."`。
