# 独立验证与问题修复记录

> **历史归档**：本文记录重构前的设计/审计，目录、命令与路径已过时，仅用于追溯；
> 当前设计见 [../ARCHITECTURE.md](../design/30-architecture.md)，测试方案见 [../TEST_PLAN.md](../spec/21-test-plan.md)。

本文件记录一次**独立只读审计**（审计者不修改项目文件、不重跑真实 benchmark）的结论，以及审计后 Lead 的修复与复验。审计对象为 commit `e04bc0e` 之前的实现，修复后代码见当前仓库 HEAD。

## 1. 审计执行内容

| # | 检查项 | 方法 | 结论 |
|---|---|---|---|
| 1 | 单元/集成测试 | `.venv/bin/python -m pytest test/` | PASS（审计时 240 passed） |
| 2 | 离线自检 | `.venv/bin/python -m bench.cli validate --offline` | PASS（9 数据集全部可加载） |
| 3 | 矩阵配置一致性 | `run --dry-run` 对照 `configs/matrix/*.yaml` | PASS（smoke 12 请求等） |
| 4 | run bundle 可复算 | 逐 cell 重算 accuracy/Wilson/TTFT/TP/summary.csv/report.md | PASS（22 cells，0 mismatch） |
| 5 | 冷缓存证据 | records.jsonl 逐条检查 `cleared_before_request`、`cached_tokens` | PASS（92/92 measured 干净） |
| 6 | 指标口径 | metrics.py 对照 METHODOLOGY 公式 | PASS（2 处低风险差异，已修） |
| 7 | 代码质量 | TODO/FIXME、密钥、硬编码路径、超时 | PASS |
| 8 | 判分器抽查 | MC/MATH/工具调用/EM-F1 边界输入 | PASS（2 处边界缺陷，已修） |
| 9 | 可复现性 | 两次独立 smoke 同名 cell 对比 | PASS（TTFT 偏差 ≤1.11%） |

## 2. 审计发现与修复状态

| 编号 | 级别 | 问题 | 状态 |
|---|---|---|---|
| 1 | 中 | `validate` 缺数据集仅告警、不校验 SHA，与文档承诺不符 | **已修**：缺数据集默认失败（`--allow-missing` 放宽）；按 `download_manifest.json` 校验大小+SHA-256（`--no-checksum` 关闭） |
| 2 | 中 | `restore_state` 声明但未实现，run 结束不卸载模型 | **已修**：记录本次 run 加载的模型，结束时卸载；仅卸载"本 run 加载"的模型，保留运行前已加载状态 |
| 3 | 中 | `fail_fast` 声明但未使用 | **已修**：模型加载/数据集加载/建请求/单请求 ERROR 均可立即中止 |
| 4 | 中 | 按上下文分级 repeats 仅文档承诺 | **已修**：`run.repeats_by_context`（默认 32K/64K → 2 轮），dry-run 展示每档实际轮数 |
| 5 | 中 | `full.yaml` 上限 6000 < 估算 11340，静默截断 | **已修**：上限 12000（分级 repeats 后估算 10080）；dry-run 与 run 启动时超限告警 |
| 6 | 低 | `#### -$1,234.50` 丢失负号 | **已修**：符号可在货币符号前后；货币符号全位置剥离 |
| 7 | 低 | MC 回退把选项回显 `A. 2 B. 4 C. 6 D. 8` 判成 D | **已修**：回退仅在候选字母唯一时生效，多个候选返回 `None` |
| 8 | 低 | `summary.skipped`/`duration_s` 未落盘 | **已修**：`writer.finalize(extra=...)` 在写盘前合并 |
| 9 | 低 | 冷缓存校验在服务端缺 `cached_tokens` 时 fail-open | **已修**：缺字段按污染处理（fail-closed） |
| 10 | 低 | `last_token_ts` 被 usage/finish 块延长，gen_tps 略低估 | **已修**：仅 token delta 更新解码窗口 |
| 11 | 低 | 文档陈旧（场景数、许可、git 快照承诺等） | **已修**：文档同步；补充 LICENSE(MIT) 与 git 仓库；manifest 记录 `git_commit`/`git_dirty` |

## 3. 修复后复验

- `pytest test/` → **246 passed**（新增 validate 校验、分级 repeats、fail_fast、restore_state、git 记录等回归用例）。
- `validate --offline` → 9/9 数据集通过，含 manifest SHA-256 校验。
- `run --matrix configs/matrix/smoke.yaml`（run `smoke-live-05`，commit `e04bc0e`）→ 18 records / 6 cells / 41.7s / skipped 0，measured 12/12 冷缓存干净。
- `demo-compare-03`（同 commit）→ 84 records / 12 cells / 359.9s / 错误 0 / 污染 0。

## 4. 仍存在的限制（如实说明）

1. **准确率样本量**：demo 每 cell 仅 6 个有效样本，Wilson 95%CI 宽约 ±0.3；比较模型准确率应使用 `--samples 20` 以上。
2. **长上下文准确率压力**：GSM8K 在 8K 下由中性填充构造，1B/9B 均降到 0；这是"填充后推理"压力结果，不能直接外推为模型真实长文能力。
3. **上下文档位覆盖**：本次只实测 1K/8K；32K/64K 需更长实际时间（系统已支持，按分级 repeats 控制）。
4. **Agentic 场景**：工具为本地模拟执行（单轮检索/固定工具集），不代表真实外部工具网络效应。
5. **无 CI**：本机项目未接入远程 CI；已提供 pytest/validate/dry-run 三段式本地门禁与 git 历史。
6. **oMLX 版本耦合**：管理 API（登录、清缓存、卸载）为 oMLX `0.7.3.dev2` 实测接口；升级后需回归 `probe`/缓存清理。
