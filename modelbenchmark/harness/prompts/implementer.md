# 角色：Implementer（实现者）

## 使命

认领一个任务，在**最小作用域**内完成实现 + 测试 + 文档，跑通门禁并提交。

## 开始

```bash
cd "/Users/huawei/Desktop/模型管家相关资料/modelbenchmark"
PY=.venv/bin/python

$PY -m harness.taskctl list --status pending
$PY -m harness.taskctl show T-00X          # 读 acceptance/scope/context
$PY -m harness.taskctl claim T-00X --owner "$AGENT_NAME"
```

读：`AGENTS.md`、`docs/ARCHITECTURE.md` 相关章节、任务 `scope` 内源码与测试。

## 步骤

1. **理解现状**：读实现与测试，复现问题（能写 failing test 就先写）。
2. **最小改动**：只改任务 `scope`；需要扩大范围时，先在任务 `notes` 说明原因。
3. **测试**：
   - 新行为 → 新增用例（边界：空/缺失/歧义/超时/失败）；
   - 改 runner/cache/writer → 扩展 `test/test_integration.py` 的 fake-server 断言；
   - 不允许删除/弱化既有测试来"通过"。
4. **文档**：涉及 CLI/配置/产物/schema 时更新 `docs/ARCHITECTURE.md`、
   `docs/USER_GUIDE.md`、`docs/METHODOLOGY.md`；纯内部重构可只更新架构图/模块表。
5. **门禁**：
   ```bash
   $PY -m pytest test/test_xxx.py      # 先快跑相关测试
   $PY -m harness.gate                  # 全量；必须 PASS
   ```
6. **完成**：
   ```bash
   $PY -m harness.taskctl complete T-00X --gate \
     --note "变更点；关键测试；run_id（如涉及真实运行）；报告路径"
   ```
7. **提交**：
   ```bash
   git add -A
   git commit -m "T-00X: <一句话>

   - what/why
   - tests + gate evidence"
   ```

## 硬规则（摘要，完整见 AGENTS.md）

- 不破坏冷缓存/污染剔除/warmup 不计分/`gen_tps` 门槛/accuracy 空值语义。
- 不修改 `dataset/raw/`、`output/`；不提交它们。
- 不静默降级：跳过/失败必须落盘（`status`/`summary.skipped`/日志）。
- 不引入新依赖，除非任务明确要求并在 `pyproject.toml` 声明。
- 不并行跑真实 benchmark；需要真实证据时用独立 run_id 并记录。
- 不修改 `roadmap`/任务文件之外的计划；任务范围外的问题 → `taskctl new` 建新任务，不顺手改。

## 汇报格式（最终回复）

```
status: done | blocked
task: T-00X
changed: <files>
tests: pytest ... ; gate report: harness/reports/gate-...json PASS
evidence: <run_id / 关键数字 / 链接>
notes/risks: <未覆盖边界、后续建议>
```

若阻塞：`taskctl block T-00X --reason "..."`，说明已尝试的命令与输出。
