# CHANGELOG

本文件记录面向使用者的显著变更（Lead 维护，职责见
[docs/process/50-validation-plan.md](docs/process/50-validation-plan.md) §8）。
口径：仅记"变了什么、为什么"；详细证据在 `docs/records/`，规范以 `docs/` 为准。
历史（0.3.0 之前）的任务级明细见 `harness/tasks/` 任务板与 `docs/archive/`。

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
