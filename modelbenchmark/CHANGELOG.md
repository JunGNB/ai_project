# CHANGELOG

本文件记录面向使用者的显著变更（Lead 维护，职责见
[docs/process/50-validation-plan.md](docs/process/50-validation-plan.md) §8）。
口径：仅记"变了什么、为什么"；详细证据在 `docs/records/`，规范以 `docs/` 为准。
历史（0.3.0 之前）的任务级明细见 `harness/tasks/` 任务板与 `docs/archive/`。

## 2026-11-12（精度首轮全量结果审计）

### Fixed

- **`mc_letter` 判题优先级 bug**：旧实现先检查 refusal 正则（如 "not enough information"/
  "无法确定"）再解析明确答案，导致回答结尾写了 "The answer is B." / "答案是 C" 的样本被记为
  `format_failure`。现改为"明确答案优先，refusal 仅在没有答案承诺时生效"；新增回归测试。
  首轮 1500 条 mc_letter 中 3 条受影响（2 条由错变对，1 条由 format_failure 变为正常错），
  宏平均影响约 +0.05pp；**旧 bundle 仍为旧判题结果，正式引用前应重跑/重判**。
- **模型级 thinking 关闭策略**：打包的 mlx_vlm 0.7.10 在 MiniCPM-V-4.6 的
  `enable_thinking=false` 分支里丢掉了模板默认的空 `<think></think>` 块（只追加
  `<|im_start|>assistant\n`），使这个默认非思考模型开始开放 reasoning、耗尽
  `max_new_tokens`（首轮 215 条 content 为空/224 条格式失败）。新增
  `ModelSpec.thinking_disable`（`explicit_false` / `omit`），4.6 设为 `omit`；实际策略写入
  `request.context_meta.thinking_control`。修复后 8 样本真实冒烟 content 非空、
  `format_failure=0`；需重跑 4.6 全量后再引用分数。

### Analysis

- 首轮 3 个全量精度 run（MiniCPM-V-4.6 / V-4.7-1B / gemma-4-E4B-it-oQ4，各 2,000 条）审计：
  数据完整、testset SHA 一致；发现 MiniCPM-V-4.6 的 215 条 content 为空/224 条格式失败
  （reasoning 占满 `max_new_tokens`，`enable_thinking=false` 未生效），条件宏平均
  0.551→0.620；详见 [records/2026-11-12-accuracy-run-review.md](docs/records/2026-11-12-accuracy-run-review.md)。
- 当前 3 个 run 是不同模型，**不是同模型 bf16/量化配对**，不能据此计算量化损失；可用
  `gemma-4-E4B-it` 与 `gemma-4-E4B-it-oQ4` 在同一 testset 上补跑配对。

## 2026-11-11（精度测试方法 method=accuracy，WP-1…WP-6）

### Added

- **双测试方法**：新增 `method=accuracy`（8 场景 × 250 = 2,000 条固定样本，5 类确定性判题），
  与既有 `method=performance` 共用测量内核、双锁与 `output/` 布局；`run_id` 方法段
  （`__accuracy__`）、`run.json.method`、`index.json` 的 `method`/`mode` 标识方法。
- CLI `run --method performance|accuracy`、`analyze --method auto|performance|accuracy` 与
  `--noise-baseline-pp`；Web 首屏方法卡片、按方法过滤测试集、精度参数面板与结果标签页
  （场景得分/语言与数据集/逐样本判分/错误统计）。
- 判题单一事实源 `src/modelbench/domain/judges/`（5 族；`tool/system/judges.py` 保留为转发）；
  精度聚合/CI/配对统计 `src/modelbench/domain/accuracy_metrics.py`；精度产物投影
  `src/modelbench/adapters/storage/accuracy_output.py`（live writer 与 `rebuild` 共用）。
- 精度测试集接入：`dataset/test/accuracy/testset.yaml`（`method: accuracy`）+
  `manifest.json` `status=wired`/`runner=accuracy`/`hidden=false`；`testset.jsonl` 仍 gitignored。
- 文档：新增 [docs/spec/22-accuracy-test-plan.md](docs/spec/22-accuracy-test-plan.md)、
  [docs/design/32-method-architecture.md](docs/design/32-method-architecture.md)、
  [docs/records/2026-11-11-accuracy-method.md](docs/records/2026-11-11-accuracy-method.md)；
  同步 docs/README、spec/12、spec/13、spec/20、spec/21、design/30、design/31、guides/40、
  guides/41、process/50、dataset/README、AGENTS、README、harness/prompts/verifier.md；
  `make docs-write` 重生成 `docs/design/33-project-structure.md`。

### Changed

- 性能口径不变（`score` 仍恒 null、AC-1…AC-11 不变）；`samples.csv`/`aggregates.json`/
  `report.md` **按方法两套形状**（性能 36 列/按档位；精度 29 列判分表/按场景），
  `run.json.outcome` 精度侧追加判分计数并移除性能专用计数。
- `verify`/`rebuild`/Excel/Web/分析按 `run.json.method` 分派；旧 run 无 `method` 回退
  `performance`。

### Fixed

- **perf `verify` 的 `n_gen_tps_invalid` 复算口径**（T-077，预存缺陷）：TG TPS 被抑制时
  记录是 `gen_tps=null` + `metrics.gen_tps_invalid=<reason>`，旧 verify 只在 `gen_tps`
  非空时计无效，导致每个被抑制样本都报 `declared != detected`（真实 `output/` 历史 run
  报 11 != 0）。现以 reason 为权威计数标记，历史 `output/` 复验 0 issue（11 条被正确计数）。

### Validation

- `harness.docs_contract`、`docs_links`、`pytest test/test_docs_contract.py`、
  `test/test_web_console.py`、`test/test_entrypoints.py` 通过；精度链路测试
  `test_accuracy_run.py`/`test_accuracy_metrics.py`/`test_accuracy_adapter.py`/
  `test_accuracy_dataset.py`/`test_accuracy_analysis.py`/`test_judges.py`。
- 完整 `harness.gate` 与独立 V&V 由 T-075/T-076 门禁报告给出。
- **真实 oMLX 精度冒烟（T-078）**：MiniCPM5-1B 与 MiniCPM-V-4.7-1B 各跑
  `--method accuracy --samples 1`（8 场景 × 1 = 8 样本），cold/temp0/无 guard/逐行
  max_new_tokens 全部成立，`verify` 0 issue、`rebuild` 后 report 逐字节一致、
  `analyze --paired --method accuracy` 输出 Δ/McNemar/Holm，批次 Excel 三页签。
- **clean-env 判题依赖（T-078）**：屏蔽 `[eval]` 可选依赖后 `validate --offline` 明确
  warning（列出 absl/nltk/langdetect/immutabledict/jieba/rouge），ifeval/中文 F1 返回
  `judge_error`（不静默给分）；完整 pip 干净安装因网络超时未验证（见 T-078 note）。

## 2026-11-10（双口径聚合 + exports 自动生成，T-067/T-068）

### 平均值双口径（T-067）

- 所有平均值输出（`aggregates.json`/`report.md`/Excel/`Web`）同时给出两个口径：
  **默认列 = 全样本口径**（该行全部 valid measured 样本，含 ISL 出带；新增
  `overall/cells` 的 `perf_all` 字段）；**「（±10%过滤）」列 = 入带口径**（原
  `perf`，只统计 actual 在 target ±10% 带内的样本；跨 run/跨系统比较用这个，
  `insufficient_samples` 标记仍针对该口径）。两口径除 ISL 带过滤外规则一致
  （warmup/污染/错误剔除、TG TPS 可信窗口判定）。
- Excel 平均页签改为**默认列读 `perf_all`**（单一事实源；旧 run 无该字段时从
  records 现算兜底），入带列加「（±10%过滤）」后缀；T-064 时代的
  「全样本」后缀列移除（默认列即全样本）。
- `verify` 的聚合复算对账扩展到 `perf_all`（双口径都可由 `records.jsonl` 复算）。
- Web 整体/数据集表新增两列（默认 + （±10%过滤）），表头口径注释同步。

### exports 自动生成（T-068）

- `output/exports/<batch_id>.xlsx` 在**每个 batch 结束时自动生成**（无需再手动
  `export-analysis`）；`export-analysis` 保留为重建/补出命令。
- 生成尽力而为：失败只向 `batch.json.warnings` 与 stderr 告警（AGENTS #9），run
  本身不失败；`batch.json` 新增 `exports` 字段（相对路径），返回值带 `export_file`
  绝对路径（不落盘）。

## 2026-11-10（端到端评审后优化轮，T-053…T-060）

### 验证与档案

- **V&V 执行证据轮（T-053）**：`harness.gate --live` 11/11 PASS（含 live_smoke 真实链路
  run），`verify` 对 8 个 legacy run 复算 1920 条记录全部通过；首篇执行记录落
  `docs/records/2026-11-10-vv-round.md`；本 CHANGELOG 建立。
- 修复后的 ISL 超差跳过/TG TPS 有效性/双锁/Origin 校验语义首次在真实链路验证
  （live run 携带 `isl_tolerance` 元数据、`window_check_basis=tokenized`、冷缓存 0）。

### 测量语义修复（T-054）

- `outcome.n_measured` 改为**全部 measured 记录数**（含 error/timeout），新增 `n_ok`；
  `verify` 双语义对账（无 `n_ok` 的历史 run 按 ok-only 口径）。
  此前 error 样本落入对账盲区、AC-2 判据恒 100%。
- `metrics.input_context_tokens` 回落链补 `tokenized_prompt_tokens` 中间层
  （usage 缺失时优先目标模型实测量而非声明值）。
- 清理过时代码注释（`batch.py` 旧嵌套布局、`writer.py`、`ports.py` 旧路径）；
  ISL 开关 `enforce_isl_tolerance` 由 T-061 整体移除（元数据无条件记录）。

### ISL 语义重设计（T-061/T-062，另一会话；本会话复验）

- ISL 从"选择期跳过"改为**固定样本全测 + 聚合按 target ±10% 入带过滤**（对称带
  1K 921–1,126 / 8K 7,372–9,011 / 32K 29,491–36,044；`n_isl_in_band`/
  `n_isl_out_of_band`/`samples.csv.isl_band`/`insufficient_samples` 守卫）；
  AGENTS.md #13 与新增 `docs/spec/13-testset-portability.md` 同步。
- 本会话复验：`gate --live` 11/11 PASS，新语义在真实链路确认（live run
  `isl_tolerance{low:921,high:1126}`、`n_isl_in_band=2`、冷缓存 0）；
  `verify` 10 run / 1924 记录全过。详见 `docs/records/2026-11-10-vv-round.md`。

### 文档一致性修复（T-055/T-056）

- 30-architecture 图 4-3 双锁顺序修正（先 endpoint 后 output，与代码一致）；
  §2.10.3 补 `GET /api/runs/<id>/samples`；§3.4 verify 措辞对齐 31。
- 31-output-contract §7.1 补第 6 类对账（runs/ 残留目录检查）。
- 20-methodology 口径补齐（decode_window_ms 保留条件、guard 187 字符、工具 schema
  不重试、`run.max_requests`、非 macOS 内存回落、CI 口径等）。
- 40-install `eval` extra 补 `jieba`/`rouge`；41-user-guide 补 `--config/--host/
  --rebuild/--json/--no-checksum` 文档并修正 `--output-dir` 适用范围。
- spec/11 派生数字改"以 manifest 为准"（selection_report 路径、体量单位、
  17+tokenizer 口径、schema 示例）；spec/12 配额字段名与上游口径对齐 manifest。

### 精度试点重跑（T-060）

- `tool/system/pilot.py` 对 v1.1 testset 重跑（`MiniCPM-V-4.7-1B`，12 条/场景）；
  `math_logic` 0.917 触发天花板（>0.85），其余 7 场景在可用带；spec/12 §8 表已回写并
  保留 v1.0 历史注记；发现多语言场景平均生成 440 token（选择题长推理）。

### 结构精简（T-057/T-058）

- 删除 `docs/design/32-dataset-sources.md`（约 80% 与四方复写），唯一增量
  （数据流图、合并许可表）并入 `dataset/README.md`。
- 契约数字单一事实源阶段一：ISL 容差/gen_tps 条件/双锁公式/输出树/decode guard/
  32K 推导/MoE 清单收敛为"一处权威 + 各处链接"（约 −400 行，漂移面大幅收窄）。

### 工程（T-059）

- 新增 `test/test_migration.py`（V-11/M10 迁移 schema 自动化）；CI 增加
  windows-latest 离线门禁 job（50 计划 §4 环境矩阵）。

### 已知遗留

- git 提交策略与任务板 tracked 语义统一（按用户指示暂缓）。
- 精度集 v1.1 难度校准试点重跑（T-060，待 live 长任务窗口）。
