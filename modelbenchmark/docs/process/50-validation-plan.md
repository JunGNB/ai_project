# 测试系统验证与确认计划（V&V）

> **状态**：current ｜ **版本**：v1.2（2026-11-10；v1.2 固定样本/actual 记录/±10% 入带聚合，
> 新增 V-12/V-13）｜ **适用**：`modelbench` 0.3.x
> （bundle schema `3.0` / index schema `2.0` / 平铺 `output/runs/<run_id>/`）｜
> **读者**：维护者、独立验证者（verifier）、发布负责人、AI Agent ｜
> **范围**：**被测对象是本基准测试系统自身**（runner/缓存/指标/落盘/索引/CLI/Web），
> 不是被测模型；模型性能口径见 [21-test-plan.md](../spec/21-test-plan.md) 与
> [20-methodology.md](../spec/20-methodology.md)。验收标准与 TEST_PLAN §7 AC-1…AC-11 同源。

## 1. 目的

回答三个问题，并给出可执行的判据与证据：

1. 系统是否**按设计测量**（冷缓存、token 口径、TG TPS 有效性、样本可追溯）？
2. 系统是否**如实记录**（逐样本权威、聚合可复算、环境快照、完整性校验）？
3. 变更后是否**仍然成立**（回归证据、再验证触发、责任人与独立性）？

## 2. 验证对象与风险分级

| 对象 | 主要风险 | 级别 |
|---|---|---|
| `application/benchmark.py`（执行/缓存/重试/取消/窗口判定） | 污染/少测/静默截断/口径漂移 | P0 |
| `domain/metrics.py`（指标/聚合/TG TPS 有效性） | 非法 TPS、跨档合并、覆盖率缺失 | P0 |
| `domain/context_tolerance.py` + tokenize（actual ISL） | 容差边界错、失败静默 | P0 |
| `adapters/storage/{writer,blobstore,catalog,tools}.py` | 证据链断裂、索引/verify 失真 | P0 |
| `application/locking.py`（双锁） | 并发 run 互相驱逐模型 | P0 |
| `interfaces/cli.py`、`interfaces/web/*` | 计划/展示与事实不符 | P1 |
| `tool/dataset/*`（测试集构建） | 样本/声明 token 不可复现 | P1 |
| 文档与文档契约 | 文档承诺与实现不一致 | P1 |

## 3. 验证方法（按成本从低到高）

| # | 方法 | 覆盖 | 命令/证据 |
|---|---|---|---|
| M1 | 静态与文档契约 | 模块/路径/CLI flag/结构清单/旧产物名/prompts/链接 | `python -m harness.docs_contract`、`harness.gate --only docs_links,docs_contract` |
| M2 | 单元 + fake-server 集成 | 执行/缓存/重试/取消/窗口判定/指标/有效性标记 | `pytest test/`（含 `test_integration.py` fake server） |
| M3 | golden bundle 复算 | 聚合与 outcome 计数可由 records 复算；`verify`/`rebuild` 幂等 | 由 `records.jsonl` 独立重算 `aggregates.json`；`./modelbench verify` |
| M4 | 属性/不变量断言 | 冷缓存全绿、`n_selected==n_measured+len(skipped)`、gen_tps 边界、ISL 计数 | 扫描任意历史 run（见 §5 判据） |
| M5 | 端到端 smoke | 真实 oMLX 一次小样本 run，检查记录/环境/锁/索引 | `harness.gate --live`（需服务，不进 CI） |
| M6 | 对抗性只读复核 | "测试通过但结论不可信"的独立复算 | 由 verifier 按 [../../harness/prompts/verifier.md](../../harness/prompts/verifier.md) 执行 |
| M7 | 负例/故障注入 | 清理失败、tokenize 失败、超窗、非法 TPS、双锁占用 | fake server 用例 + 手工注入（记录在任务 note） |
| M8 | Web 安全负例 | 写端点 Origin/Host 校验：跨站 Origin → 403、同源/无 Origin → 放行 | `pytest test/test_web_console.py` + curl 手工验证（记录命令/响应码） |
| M9 | 后端结构契约 | `AppBackend` 必需方法齐全且可调用（鸭子类型） | `pytest test/test_core.py -k backend_implements_application_port_contract`；`backend_contract_errors(OMLXBackend()) == []` |
| M10 | 迁移记录 schema | 新迁移件 records `schema_version=3.0` 且有 `sample_meta.migrated_from_schema`；旧迁移件允许 2.1 legacy | 检查 `tool/migrate/legacy_v2_to_v3.py` 输出 + 抽读 `records.jsonl` |

## 4. 环境矩阵

| 环境 | 能做什么 | 不做什么 |
|---|---|---|
| macOS + oMLX（live 主平台） | 全量：run/缓存/内存/CPU/Web/live smoke | — |
| Linux CI（`harness.gate --ci`） | imports/compile/tests/perf_plan/docs/secrets/hygiene；数据集缺失显式 skipped | 不跑 live smoke、不校验数据 SHA |
| Windows（工具链/离线） | 离线 validate、数据集构建、Web/结果工具；CI `gate-windows` job（`ci.yml`，`harness.gate --ci` 同套检查） | 无本机 oMLX（除非连可达实例） |

## 5. 验收判据（可量化）

通用判据直接引用 TEST_PLAN §7 的 AC-1…AC-11：selected 可追溯 100%；`status=ok` ≥95%；
非法 `gen_tps`（>10,000 或 <0）为 0；覆盖率（`n_valid`/`gen_tps_n`/`n_length_capped`/
`n_gen_tps_invalid`）显式展示；**同一套 testset/SHA**（无 per-model 构造、无运行时裁剪补齐）；
每条 measured 记录有 `tokenized_prompt_tokens`/`isl_tolerance`；均值只使用 actual 在
target ±10%（所有档位对称，区间见 [20-methodology.md](../spec/20-methodology.md) §5）的入带样本，
出带保留但不进均值（`n_isl_in_band`/`n_isl_out_of_band`）；in-band 不足标记
`insufficient_samples`（入带 <8，见 `MIN_IN_BAND_SAMPLES`）；32K target 32,768 不特判；tokenize 失败回退
`declared + max_tokens + 128 + decode_guard_reserve`（默认 242）并 warning；
聚合可由 records 复算；噪声与口径标注（`repeats`/CV/unstable）；环境字段完整。

V&V 自身的附加判据：

| # | 判据 | 通过条件 |
|---|---|---|
| V-1 | 冷缓存证据 | 所有进入聚合的记录满足 `cache.cleared_before_request == true` 且 `cache.cached_tokens == 0`；污染/错误记录在 `records.jsonl` 可查且不进聚合 |
| V-2 | TG TPS 有效性 | 聚合/导出中无 `gen_tps > 10,000` 或 `<0`；`gen_tps` 为 null 的记录必有 `metrics.gen_tps_invalid` 原因（枚举见 [20-methodology.md](../spec/20-methodology.md) §2）或 `completion_tokens<16`；`n_gen_tps_invalid` 与逐条记录一致；`verify` 的 `gen_tps_invalid` 统计与记录一致，旧 run 非法值计入 `legacy_gen_tps_invalid` 且不导致失败，同一条记录同时有 `gen_tps` 与 `gen_tps_invalid` 必须报 issue |
| V-3 | actual ISL 记录与 ±10% 入带聚合 | 每条**预构造 measured** 记录有 `request.tokenized_prompt_tokens`、`context_meta.isl_tolerance{target,low,high,actual,ok}` 与 `context_meta.window_check_basis`；`samples.csv` 在 `actual_prompt_tokens` 后有 `isl_band` 列；`overall/cells` 只对入带样本求均值，出带样本仍保留在 records/samples.csv 但不进均值；`n_isl_in_band`/`n_isl_out_of_band` 与逐条复算一致（`verify` 的 `isl_in_band`/`isl_out_of_tolerance` 同义对账）；`run.json.outcome` 不含 `n_isl_skipped`；tokenize 失败回退"声明+242"、`window_check_basis=declared` 且 warning |
| V-4 | 窗口跳过 | 超窗样本只出现在 `run.json.outcome.skipped`（含原因），不出现 `truncated=true` 的预构造行；32K not-testable 数在报告可见 |
| V-5 | 双锁互斥 | **先 endpoint 后 output** 获取；锁文件公式见 [30-architecture.md](../design/30-architecture.md) §4.3（单一事实源）；同一 `base_url` + 不同 `output` 根并发时第二个 run 快速失败（提示锁路径与 holder PID）；同一 output 根也互斥；不同 `base_url` 可并发 |
| V-6 | 完整性 | `./modelbench verify` 三类对账全过（另加 V-2/V-3 的统计一致性）；`blob_refs.missing == 0`；`MANIFEST.json` 每个文件大小/sha256 可复核 |
| V-7 | 文档契约 | `docs_contract` 覆盖 `docs/**/*.md` 与 `harness/prompts/*.md`；旧产物名（`summary.json`/`hardware.json`/`omlx.json`/`models.json`/`summary.csv`/`cells.csv`/`result_analysis`/`output/<run_id>`/`output/.benchmark.lock`）仅允许出现在 `harness/docs_contract.py` `DEAD_ARTIFACT_WHITELIST` 的显式白名单文件（`design/31-output-contract.md` 历史段、`archive/OUTPUT_LAYOUT.md`、本文件），`docs/archive/` 整体豁免 |
| V-8 | 可复现性 | 同一 `--samples` 连续两次 run 的样本集合一致；`rebuild` 后 `aggregates.json` 语义等价；`index` 重建内容一致（除 `generated_at`） |
| V-9 | Web 写端点同源防护 | 所有 `POST`（estimate/run/models sync/jobs cancel/runs reveal/output reveal/testsets reveal/index rebuild）在 `do_POST` 最前面校验 Origin：带 Origin 时必须为 `http`/`https` 且 netloc == `Host`（含端口），否则 403；无 Origin 的 curl/CLI 请求放行；**这不等于鉴权**，`--allow-external` 仍不得用于不可信网络 |
| V-10 | 后端结构契约 | `REQUIRED_BACKEND_METHODS` 全部存在且可调用：`backend_contract_errors(OMLXBackend()) == []`；`test_core` 的 `test_omlx_backend_implements_application_port_contract` 通过；鸭子类型检查不要求继承 ABC（Orchestrator 端口外能力缺口仍见 ARCHITECTURE §8.2/T-004） |
| V-11 | 迁移记录版本 | 新迁移件每条 record `schema_version=3.0` 且 `sample_meta.migrated_from_schema` 非空；旧迁移件允许 2.1（历史不改写），不得据单条 record 版本误判 bundle 布局 |

| V-12 | 固定样本/SHA | 所有 run 引用同一 `testset.jsonl` SHA + manifest；不存在 per-model 构造/运行时裁剪补齐；跨设备/跨系统比较符合 [../spec/13-testset-portability.md](../spec/13-testset-portability.md) 的准入条件 |
| V-13 | 样本量充分性 | 每个 cell/overall 报告 `n_isl_in_band`；in-band 样本不足以支撑均值时显式标记 `insufficient_samples`，不得静默当作有效结论 |

## 6. 再验证触发条件（任一命中必须重跑 §3 M2 + M3，M5 视风险）

1. 修改测量语义（缓存/指标/gen_tps 有效性/ISL 入带与窗口判定/schema/样本选择）；
2. `SCHEMA_VERSION` / `INDEX_SCHEMA_VERSION` / 输出布局变更；
3. oMLX 升级（settings 白名单、`usage` 字段、缓存管理端点、tokenize 行为可能变化）；
4. 模型/tokenizer/量化版本变化，或更换参考 tokenizer；
5. 数据集重建（`testset.jsonl`/manifest SHA 变化）；
6. 依赖升级（httpx/pydantic/psutil/openpyxl 等）或 Python 版本变化；
7. Web 路由/安全校验（Origin/Host）、`application/ports.py` 结构契约、迁移器输出 schema 变更；
8. 任何 P0 缺陷修复后。

## 7. 证据留存

| 证据 | 位置 | 保留 |
|---|---|---|
| 门禁报告 | `harness/reports/gate-<UTC>.json` / `latest.json` | 每次门禁（gitignored） |
| 测试输出 | `pytest` 输出 / CI 日志 | 每次提交 |
| 端到端 run | `output/runs/<run_id>/`（7 条目 + MANIFEST） | 按 RESULTS §7.5 保留/归档/打包 |
| 校验结果 | `./modelbench verify --json` 输出 | 附在任务 note 或评审报告 |
| 独立复核 | 评审报告（如 `reports/review/*.md`） | 与任务/commit 关联 |
| 任务记录 | `harness/tasks/` 下 T-XXX.json 的 `notes`（含 run_id 与命令） | 随任务板（本地，不提交） |

## 8. 角色与独立性

| 角色 | 职责 | 边界 |
|---|---|---|
| Implementer | 实现 + 自测 + 跑门禁 + 提供证据 | 不得自判 PASS |
| Verifier | **只读**独立复算（§3 M6），按清单输出 PASS/FAIL 与 file:line 证据 | 不改仓库；临时脚本写 `/tmp` |
| Lead / 发布负责人 | 汇总证据、判断是否可发布、维护 CHANGELOG | 对 P0 例外签字 |
| AI Agent | 按 AGENTS 硬不变量执行；门禁/verify 不通过不得声称完成 | 见 `AGENTS.md` §2/§7 |

## 9. 已知局限（如实说明）

1. fake-server 只能证明协议/编排正确，不能证明真实 oMLX 的时序/内存行为；
2. live smoke 单次小样本（`--samples 1`）不构成性能结论，只验证链路；
3. 本计划不覆盖"模型能力"验证（判分器未接入 run）；
4. 10 样本/格的统计稳定性不足以支撑 p95/p99 断言（见 TEST_PLAN §8 第 10 条）；
5. 双锁依赖 OS 文件锁；不支持文件锁的平台上系统直接拒绝启动（fail-closed）；
6. actual ISL 不触发跳过：所有能测样本都 measured，均值只使用 ±10% 入带样本；出带样本保留
   但不进均值，in-band 不足会标记 `insufficient_samples`。跨设备/跨系统比较需按
   [../spec/13-testset-portability.md](../spec/13-testset-portability.md) 的准入条件（同 testset
   SHA、同 target/带、同协议参数、报告 actual 分布）。

## 10. 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-11-10 | v1.0 | 首版：T-047 文档 P0 整改；纳入 actual ISL 容差、TG TPS 有效性、双锁与 9 条 AC |
| 2026-11-10 | v1.1 | 新增 Web 同源防护（V-9）、后端结构契约（V-10）、迁移 schema（V-11）判据 |
| 2026-11-10 | v1.2 | 口径改为固定一套样本 + actual 记录 + ±10% 入带聚合；新增 V-12（固定样本/SHA）与 V-13（样本量充分性）；AC 对齐 AC-1…AC-11 |
| 2026-11-10 | v1.3 | `n_measured` 口径改为全部 measured 记录（error/timeout 入账）+ `n_ok` 判据（AC-2）；V-7 白名单表述对齐 `docs_contract.py` 实际清单 |
