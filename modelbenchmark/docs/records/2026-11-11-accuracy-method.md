# 2026-11-11 — 精度方法落地与文档同步（method=accuracy）

> **状态**：records（非规范、非证据）｜ **日期**：2026-11-11 ｜ **任务**：T-075（文档实现，
> owner `accuracy-docs`）/ T-076（独立验证）｜ 结论一句话：**双方法 `performance|accuracy`
> 代码完成（WP-1…5）、文档契约同步完成（WP-6）**；真实 oMLX 全量精度 run 尚未执行，
> 端到端证据来自 fake-server 与判题自检。

## 1. 范围

- 新增测试方法 `accuracy`：8 场景 × 250 = 2,000 条固定样本，`temperature=0` / 冷缓存 /
  无 decode guard / `repeats=1` / 重试 1 次，5 类确定性判题（`mc_letter`/`ifeval`/
  `normalized_match`/`code_exec`/`tool_call`），输出场景分 / 8 场景宏平均 / 语言与数据集剖面 /
  配对 Δ（McNemar、配对 bootstrap、Holm）。
- 与既有 `performance` 共用测量内核（oMLX 驱动、模型生命周期、冷缓存、资源采样、双锁、
  环境快照、逐样本证据、bundle 布局），方法差异集中声明。
- 本次任务只改文档与生成物（`docs/**`、根文档、`AGENTS.md`、`CHANGELOG.md`、
  `dataset/README.md`、`harness/prompts/verifier.md`、`make docs-write` 生成物）；
  **未改 `src/`、`test/`、`tool/` 代码**。

## 2. 设计决策（已实现）

1. **共享 kernel + 方法 profile**：`application/methods.py` 的 `MethodProfile` 声明方法差异
   （orchestrator/writer/plan/fingerprint/固定协议/Web 参数元数据）；`BatchRunner` 只在组合根
   按 `get_method()` 分派，避免 runner/Web 散落 `if method == ...`。
2. **判题单一事实源**：5 个判题族迁入 `src/modelbench/domain/judges/`，
   `tool/system/judges.py` 保留为兼容转发；`evaluate()` 永不抛异常，异常转 `judge_error`。
3. **输出分派**：`AccuracyRunWriter` 只覆写 `finalize`；`aggregates.json`/`samples.csv`/
   `report.md` 由 `adapters/storage/accuracy_output.py` 的纯函数生成（与 `rebuild` 共用）；
   `verify`/`rebuild` 按 `run.json.method` 分派，未知方法报错。
4. **Web/CLI 双方法**：方法卡片 + 按方法过滤测试集 + 方法专属参数面板；结果视图按
   `summary.method`/`mode` 切换；CLI 新增 `run --method`、`analyze --method`/
   `--noise-baseline-pp`。
5. **协议强制**：`MethodProfile.configure` 对精度拒绝非 cold、`repeats>1`、非法
   `AccuracyConfig`（`ValueError`，不静默换口径）。
6. **测试集接入**：`dataset/test/accuracy/testset.yaml`（提交）+ `manifest.json`
   `status=wired`/`runner=accuracy`/`hidden=false`；`testset.jsonl` 仍 gitignored 可重建。

## 3. 测试与门禁证据

| 证据 | 命令/位置 | 结果 |
|---|---|---|
| 文档契约 | `python -m harness.docs_contract` | PASS（路径/模块/make/CLI flag/结构清单/旧产物名） |
| 文档链接 | `python -m harness.gate --only docs_links`（或 `make docs`） | PASS（相对链接） |
| 文档测试 | `pytest test/test_docs_contract.py -q` | PASS |
| Web/入口回归 | `pytest test/test_web_console.py test/test_entrypoints.py -q` | PASS |
| 结构清单 | `make docs-write` 重生成 `docs/design/33-project-structure.md` | 已重生成（含新模块） |
| 精度 e2e（fake server） | `test/test_accuracy_run.py` | 方法/run_id/判分/冷缓存/超窗/`judge_error`/plan 用例 |
| 精度聚合/配对 | `test/test_accuracy_metrics.py`、`test/test_accuracy_analysis.py` | 可复算/CI/宏平均/McNemar/Holm |
| 精度数据/读取 | `test/test_accuracy_adapter.py`、`test/test_accuracy_dataset.py` | 契约与 plan |
| 判题确定性 | `test/test_judges.py` + `tool/system/validate_judges.py` | gold 自检 PASS（依赖 `[eval]`） |
| 完整门禁 | `python -m harness.gate` | 见本次任务 T-075/T-076 门禁报告（Lead/验证者统一执行） |

> 门禁报告路径（运行后）：`harness/reports/gate-<UTC>.json` / `latest.json`（gitignored）。

## 4. 未验证 / 风险

1. **未做真实 oMLX 全量精度 run**：2,000 样本/模型约 3h（长上下文 2–2.5h），本次没有
   真实模型分数；`run --method accuracy` 的真实链路（模型加载/长上下文预填/超窗）
   仍需 live smoke 与全量 run 验证。
2. **端到端证据是 fake server**：证明协议/判分/产物/复算正确，不证明真实 oMLX 的时序与内存行为。
3. **判题依赖**：缺 `.[eval]` 时 ifeval/中文 F1 会 `judge_error`（显式告警，不静默）；
   本次已用"屏蔽可选依赖 + `validate --offline`/逐族 evaluate"复现该行为；完整 pip 干净
   安装因 PyPI 网络超时未完成（见 §4.5）。
4. **时间/内存预算**：全量精度 run 的长上下文场景对内存/时长压力大，Web 首次估算会读
   ~15 MB testset JSONL（之后缓存）。
5. **文案已清理**：`dataset/test/accuracy/manifest.json` 的 `description`/`deviations` 末条已与
   `tool/dataset/accuracy/build.py` 同步为 wired 表述；`status`/`runner`/`hidden` 三字段为
   wired/accuracy/false，`build.py --check` 校验三个字段与 `testset.yaml` 存在。
6. **执行单元标签**：精度按 scenario 分组（一个 cell/scenario），run.log/progress 的 cell 标签
   记为 `accuracy`；聚合仍按 `(model, scenario, dataset)`，不影响 `records[].score.detail` 与复算。

### 4.5 真实 oMLX 冒烟与 clean-env 依赖验证（T-078）

在真实 oMLX（0.7.3.dev2，M4 Pro 24GB）上运行全链路冒烟（独立 `/tmp` 输出根，不污染仓库
`output/`）：

```bash
./modelbench run --method accuracy --models MiniCPM5-1B --samples 1 \
  --output-dir /tmp/mb_acc_live --run-id acc-live-smoke-20261111
# -> 8 场景 × 1 = 8 measured，8 scored，cold/0 cached，temp=0/top_p=1，
#    max_tokens 分别等于各行 judge.max_new_tokens，context_meta 无 decode guard
#    macro_score=0.3910（n=1/场景，仅协议冒烟，不作精度结论）
./modelbench run --method accuracy --models MiniCPM-V-4.7-1B --samples 1 \
  --output-dir /tmp/mb_acc_live --run-id acc-live-smoke2-20261111
./modelbench analyze <run_a> <run_b> --paired --method accuracy --noise-baseline-pp 2.0
# -> Δ(A−B)/McNemar/Holm、unpaired=0/unscored=0/mismatch=0；n=1 场景 CI 为 "-"（符合期望）
```

- 两个真实 run 的 `run.json.config.testset.sha256` 同为 `502d827c…`（与
  `dataset/test/accuracy/manifest.json` 一致），`judge_dependencies` 为 dict 且
  `missing=[]`；`tools.verify_output` 两 run 均 **0 issue**；`rebuild` 后 `report.md`
  **逐字节一致**；批次 `exports/*.xlsx` 三页签（场景得分/逐样本判分/错误统计）可读。
- **clean-env 模拟**：用 meta-path 屏蔽 `absl/nltk/langdetect/immutabledict/jieba/rouge`
  后，`validate --offline` 输出明确 warning（列出缺失包）且退出 0；`ifeval`/中文 `f1_zh`
  返回 `judge_error=True`（0 分、保留分母），`mc_letter` 等无依赖族仍正常判分。
- 结论：协议/判题/产物/复算在真实 oMLX 上成立；**全量 2,000 条 × bf16/量化对比仍需正式
  run**（约 3h/模型），不作为本记录的精度结论。

## 5. 相关文档

- 方案：[spec/22-accuracy-test-plan.md](../spec/22-accuracy-test-plan.md)（精度）、
  [spec/21-test-plan.md](../spec/21-test-plan.md)（性能，边界）
- 设计：[design/32-method-architecture.md](../design/32-method-architecture.md)（双方法）、
  [design/31-output-contract.md](../design/31-output-contract.md)（产物形状）
- 规范：[spec/12-accuracy-testset.md](../spec/12-accuracy-testset.md)（样本集）、
  [spec/13-testset-portability.md](../spec/13-testset-portability.md)（比较准入）、
  [spec/20-methodology.md](../spec/20-methodology.md) §6（判分口径）
- 验证：[process/50-validation-plan.md](../process/50-validation-plan.md) V-14…V-17
