# AGENTS.md — AI Agent 工作契约

本仓库是 oMLX 本地模型**双方法**基准测试系统：`method=performance`（逐样本性能 + 环境快照）
与 `method=accuracy`（逐样本确定性判题），共用测量内核、双锁与 `output/` 布局，附本地 Web 控制台。
任何 AI 在改代码前必须读本文件、[docs/design/30-architecture.md](docs/design/30-architecture.md)、
[docs/design/32-method-architecture.md](docs/design/32-method-architecture.md)、
[docs/spec/21-test-plan.md](docs/spec/21-test-plan.md)、
[docs/spec/22-accuracy-test-plan.md](docs/spec/22-accuracy-test-plan.md)、
[docs/spec/20-methodology.md](docs/spec/20-methodology.md)。

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
   `implausible_tps`），聚合给 `n_gen_tps_invalid`。**判分按方法**：性能模式**不判分**，
   `SampleRecord.score` 恒为 null（evaluator 域不接入 run）；精度模式逐样本**确定性判分**
   （`score` 非空；`judge_error` 计 0 且留在分母，`format_failure` 显式计数，不静默剔除）。
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
    **样本设计固定**（同一 target/抽样/构造算法全设备、全模型复用，不运行时换样本）；
    **长度按目标模型 tokenizer 严格对齐**：声明 `strict_input_tokens: true` 的性能集，样本正文按通用
    参考 tokenizer 超量构建（2× 档位），runner 发送前用目标模型 tokenizer 把 body 裁剪/平铺（任务块
    独立成消息，不被裁），以服务端 `usage.prompt_tokens` 为口径对账到 1,024/4,096/32,768（容差 0；
    `usage - /v1/tokenize` 的常量偏移在 warmup 学习），使实际计费输入长度严格等于档位目标；
    **不做 ±10% 入带**。等值由对齐保证，不因 token 不等而 skip；仅 tokenizer 不可用/无法对齐/窗口
    不足才写 `run.json.outcome.skipped`（原因含 `runtime_alignment`）。strict 集照常追加 decode guard
    （先追加再整体对齐；固定 50-token 解码窗口由测试集声明、CLI/Web 可覆盖，逐样本记
    `decode_guard_applied=true` 与 `context_meta.prompt_alignment`，run 级 warning 说明运行期对齐）。
    未声明 strict 的历史/精度口径保留原带内聚合。
    跨系统比较见 `docs/spec/13-testset-portability.md`。
14. **精度判题（method=accuracy）**：判题只用 `src/modelbench/domain/judges/` 的 5 类
   **确定性**判题器——**无 LLM judge、无网络、不读外部状态**；判题异常/依赖缺失转
   `status=judge_error`（计 0 分且**留在分母**，不静默剔除），空/无法解析记 `format_failure`；
   判题器只读样本的 `judge` 契约 + 模型响应。
15. **精度运行协议固定**：`cache_mode=cold`、`temperature=0`、`top_p=1`、**无 decode guard**、
   `repeats=1`、传输错误重试 1 次、生成长度取逐行 `judge.max_new_tokens`、默认
   `thinking_mode=disabled`；违反由 `MethodProfile.configure` 直接拒绝，不得静默换口径。
   **独立结果**：精度必须用独立 `run_id`（`__accuracy__` 段），不与性能 run 混算/合成综合分。
16. **精度可比性**：两次精度 run 只有**同一 `testset.jsonl` SHA-256 + 同一
   `judge_contract_version`**（`batches/<batch_id>.json` 的 `inputs.testset`）才可配对比较；
   `--samples N` 的冒烟 run 不得与全量 run 比分数；样本不齐/判题族不一致必须显式报错
   （见 `docs/spec/13-testset-portability.md` §4.2）。

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
| 指标/聚合（overall/cell/逐样本，性能） | `src/modelbench/domain/metrics.py` |
| 执行/缓存/重试/取消/样本选择 | `src/modelbench/application/benchmark.py`、`src/modelbench/adapters/omlx/cache.py` |
| 测试方法分派/协议/方法卡片 | `src/modelbench/application/methods.py`、`src/modelbench/domain/method.py` |
| 精度编排/窗口/判题落地 | `src/modelbench/application/accuracy.py` |
| 精度判题族（单一事实源） | `src/modelbench/domain/judges/` |
| 精度聚合/CI/配对统计 | `src/modelbench/domain/accuracy_metrics.py` |
| SSE/admin/模型生命周期 | `src/modelbench/adapters/omlx/{client,backend}.py` |
| 环境/配置采集（硬件 + oMLX） | `src/modelbench/adapters/omlx/environment.py`、`src/modelbench/adapters/host/hardware.py` |
| 场景/prompt 构建 | `src/modelbench/domain/scenarios/` |
| 数据集/下载（含精度 testset 读取） | `src/modelbench/adapters/datasets/`（`perf_jsonl`/`accuracy_jsonl`）、`tool/download/` |
| 判分（性能不接入；精度走 judges） | `src/modelbench/domain/evaluators/`（离线能力）、`src/modelbench/domain/judges/`（精度运行） |
| 配置/测试集（含 `AccuracyConfig`/method） | `src/modelbench/adapters/config/{settings,testsets}.py`、`src/modelbench/resources/config/`、`src/modelbench/application/plan.py` |
| Web（方法卡片/目录树/样本勾选/文件导航） | `src/modelbench/interfaces/web/` |
| 结果目录索引与二次分析 | `src/modelbench/adapters/storage/{catalog,writer,accuracy_output,blobstore,tools,excel_report,layout}.py`、`docs/design/31-output-contract.md` |
| 结果分析（性能对比 / 精度配对） | `src/modelbench/interfaces/analysis/` |
| 单实例双锁 | `src/modelbench/application/locking.py`（output 根锁 + endpoint 锁） |
| 输入大小容差/窗口判定 | `src/modelbench/domain/context_tolerance.py`、`application/benchmark.py` |
| 门禁/任务板 | `harness/` |

## 5. 测试要求

- 新逻辑覆盖边界（空/缺失/歧义/超时/失败）。
- 改 runner/cache/writer：扩展 `test/test_integration.py` fake-server 用例
  （逐样本记录、输入上下文、环境文件、无判分）。
- 改精度链路：`test/test_accuracy_run.py`（fake-server e2e：方法/run_id/判分/冷缓存/超窗/
  `judge_error`/plan）、`test/test_accuracy_metrics.py`（聚合/CI/宏平均/配对复算）、
  `test/test_accuracy_adapter.py`、`test/test_accuracy_dataset.py`、
  `test/test_accuracy_analysis.py`、`test/test_judges.py`（判题确定性 + gold 自检）。
- 改配置/测试集/样本选择：更新 `test/test_perf.py` 的矩阵/抽样断言；精度选择断言见
  `test_accuracy_adapter`/`test_accuracy_run`。
- 改 Web/CLI：至少验证 `/api/overview`（含 `methods`）、`--dry-run`（两种方法）、命令退出码、
  `run --method`/`analyze --method` 的冲突与配对报错。
- 真实性能/精度结论必须来自独立 run_id（`__performance__`/`__accuracy__`），并在任务 note 引用。

## 6. 多 Agent 与验证

- 一个任务同时一个 owner；`scope` 是写作用域，冲突先完成/重开任务。
- 实现者只改 scope；验证者只读（`/tmp` 或报告），不得自写自判。
- 验证者用 [harness/prompts/verifier.md](harness/prompts/verifier.md)，重点复算
  `run.json.outcome` / `aggregates.json` / `records.jsonl` / `samples.csv` 的一致性、冷缓存字段、
  样本数与 skipped 对账、TG TPS 有效性（`n_gen_tps_invalid`）、actual ISL 记录与严格对齐
  （`strict_input_tokens` 集复算每条 `usage.prompt_tokens == bucket`（计费口径）与
  `context_meta.prompt_alignment` 证据、`isl_tolerance` 的 low==high==target、skip 原因仅
  tokenizer/对齐/窗口、`decode_guard_applied=true`；非 strict 集复算
  ±10% 入带聚合 `isl_band`/`n_isl_in_band`/`n_isl_out_of_band`）、
  固定 testset SHA、双锁互斥与模型加载设置是否真实默认/MoE-only；**精度 run** 另复算判题
  确定性/gold 自检（V-14）、精度聚合与协议（V-15）、样本 SHA/配对可比性（V-16）、双方法隔离
  （V-17）；V&V 判据见 [docs/process/50-validation-plan.md](docs/process/50-validation-plan.md)。

## 7. 完成标准

- [ ] `acceptance` 全部满足；
- [ ] `pytest test/` 全绿、`harness.gate` PASS；
- [ ] 测量/产物变更时 `./modelbench verify` 通过（含 gen_tps/ISL 统计一致；精度 run 按精度分支）；
- [ ] 方法变更时 `./modelbench run --method accuracy --dry-run` 与
      `./modelbench run --method performance --dry-run` 均正确，且精度 fake-server e2e
      （`test/test_accuracy_run.py`）通过；
- [ ] 文档同步（按 `docs/README.md` 索引：`spec/20-methodology`、`spec/21-test-plan`、
      `spec/22-accuracy-test-plan`、`design/30-architecture`、`design/31-output-contract`、
      `design/32-method-architecture`、`guides/40-install`、`guides/41-user-guide` 等视变更，
      涉及验证口径时同步 `docs/process/50-validation-plan.md`）；
- [ ] `taskctl complete --gate`、git commit（不含 dataset 证据/output）。

## 8. 需要人类确认

删除/重写 `dataset/test/performance/testset.jsonl`、`dataset/raw/` 或 `output/`；
修改测量语义（缓存/指标/schema/样本选择）；下载 >1GB 数据；
改硬件/系统/启动停止 oMLX。阻塞时：
`$PY -m harness.taskctl block T-00X --reason "..."`。
