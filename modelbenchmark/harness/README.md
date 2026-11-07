# Harness：让 AI 持续演进这个基准测试系统

本目录是**工程侧 harness**，不是被测系统的一部分。它把"改代码"变成可验证、可审计、
可交接的闭环，使 AI Agent（或人）可以安全地长期维护项目：

```
        ┌─────────────┐     claim      ┌──────────────┐
        │ task board  │ ─────────────► │ implementer  │
        │ tasks/*.json│                │ (AI/human)   │
        └──────┬──────┘                └──────┬───────┘
               │ complete --gate              │ edits + tests
               ▼                              ▼
        ┌─────────────┐   reads report  ┌──────────────┐
        │  gate.py    │ ◄────────────── │ git working  │
        │ 10 checks   │                 │ tree         │
        └──────┬──────┘                 └──────┬───────┘
               │ reports/latest.json           │ independent audit
               ▼                              ▼
        ┌─────────────┐                ┌──────────────┐
        │  证据/门禁   │                │  verifier    │
        └─────────────┘                └──────────────┘
```

三块能力：

| 能力 | 实现 | 解决的问题 |
|---|---|---|
| **门禁** | `harness/gate.py` | 每次改动都可确定性验证，AI 不能"自我宣布完成" |
| **任务板** | `harness/taskctl.py` + `tasks/*.json` | 工作项、验收标准、状态、负责人、证据可追踪 |
| **角色契约** | `AGENTS.md` + `harness/prompts/*.md` | 规划/实现/验证分离，避免同一 Agent 既写又判 |

---

## 1. 快速命令

```bash
cd "$PROJ"

.venv/bin/python -m harness.gate            # 完整门禁（默认 10 项，离线，约 10–30 秒）
.venv/bin/python -m harness.gate --ci       # CI 模式：允许数据集缺失、跳过校验和与数据完整性检查
.venv/bin/python -m harness.gate --only tests,docs_links,docs_contract
.venv/bin/python -m harness.gate --live     # 额外跑真实 oMLX smoke（需要服务）
.venv/bin/python -m harness.gate --json     # 机器可读报告

.venv/bin/python -m harness.taskctl list
.venv/bin/python -m harness.taskctl show T-001
.venv/bin/python -m harness.taskctl claim T-001 --owner implementer-1
.venv/bin/python -m harness.taskctl complete T-001 --note "..." --gate
.venv/bin/python -m harness.taskctl journal --limit 10
```

`make check` / `make tasks` 是上述命令的简写。

---

## 2. 目录结构

```
harness/
  gate.py              # 质量门禁（10 个确定性检查 + 可选 live smoke）
  taskctl.py           # 任务板 CLI（JSON 文件 + 原子写 + journal）
  tasks/T-XXX-*.json   # 任务定义（tracked，评审对象）
  prompts/             # 角色提示词：planner / implementer / verifier
  reports/             # 门禁报告（gitignored；latest.json 供 AI 读取）
  state/journal.jsonl  # 状态迁移日志（gitignored）
AGENTS.md              # 根目录：AI Agent 的硬契约（必读）
Makefile               # check/test/smoke/tasks/docs
.github/workflows/ci.yml  # 远程 CI 镜像（gate --ci）
```

---

## 3. 门禁（gate）

### 3.1 检查项

| name | 内容 | 失败意味着 |
|---|---|---|
| `imports` | 导入 modelbench/harness；数据集 ≥11、场景 ≥8 | 包结构/注册表被破坏 |
| `compile` | `compileall src harness tool test` | 语法错误 |
| `tests` | `pytest test/`（单元 + fake-server 集成） | 行为回归 |
| `validate` | `modelbench.interfaces.cli validate --offline`（配置 + 数据集解析；仅对有 `download_manifest.json` 的 raw 做校验和；CI 用 `--allow-missing --no-checksum`） | 数据集/配置不可用 |
| `dataset_integrity` | `tool/dataset/verify.py`：结构 / manifest / 逐文件 SHA-256（`--ci` 且数据缺失时显式 skipped） | 数据集或溯源证据被破坏 |
| `perf_plan` | `run --dry-run` 可解析 perf 配置/测试集/模型并输出计划 | 性能测试计划或配置被破坏 |
| `docs_links` | 相对 Markdown 链接全部可解析；禁止绝对路径链接 | 文档失效/不可移植 |
| `docs_contract` | 文档中的模块/路径/CLI flag/根启动器存在；`docs/PROJECT_STRUCTURE.md` 未过期（gitignored 的生成物路径不要求存在） | 文档与代码/目录漂移 |
| `secrets` | tracked 文本中无 key/token/私钥字面量 | 有密钥泄漏风险 |
| `hygiene` | .gitignore 覆盖证据目录；无生成物被跟踪；打包入口存在 | 仓库卫生退化 |

> 修改 `src/modelbench` 目录后运行 `make docs-write`，否则 `docs_contract` 会失败。

> 旧的 `matrix_plans` golden 检查随上下文矩阵一起删除；新的 `perf_plan` 只验证
> 计划可被解析（样本数/数据集/模型），不锁定具体请求数。

### 3.2 报告

每次运行写入：

- `harness/reports/gate-<UTC>.json`（历史）
- `harness/reports/latest.json`（稳定路径，供 AI/任务板读取）

报告字段：`schema_version, generated_at, git_commit, git_dirty, python, platform, ci,
live, duration_ms, checks[{name, ok, detail, data, duration_ms}], summary{total, passed,
failed, ok}`。`complete --gate` 只检查 `summary.ok == true`。

### 3.3 增加检查

在 `gate.py` 写 `def check_xxx(ctx) -> Result`，加入 `CHECKS` 列表，并在
`harness/README.md` 上表补一行。要求：只读（除 `harness/reports/`）、<120s、确定性、
不依赖网络/真实模型（除非新增 opt-in 标记并在 `main` 参数中控制）。

---

## 4. 任务板（taskctl）

### 4.1 任务文件

`harness/tasks/T-001.json`（一个任务一个文件，git 可 diff）：

```json
{
  "id": "T-001",
  "title": "一句话目标",
  "status": "pending",            // pending | in_progress | blocked | done
  "priority": "P0",               // P0 | P1 | P2
  "owner": null,                  // 认领者标识
  "created_at": "...", "updated_at": "...",
  "scope": ["src/modelbench/application/benchmark.py", "test/"],
  "context": "为什么做、现状问题",
  "acceptance": ["可验证的验收条件 1", "python -m harness.gate 通过"],
  "notes": [],                    // 完成备注/阻塞原因（结构化）
  "links": ["docs/ARCHITECTURE.md#..."]
}
```

### 4.2 状态机

```
pending ──claim──► in_progress ──complete --gate──► done
                        │
                        └──block──► blocked ──reopen──► pending
done ──reopen──► pending
```

规则：

- `claim` 仅允许 `pending`；认领后 `owner` 必填（默认 `$AGENT_NAME` 或 `agent`）。
- `complete` 仅允许 `in_progress`，必须 `--note`；`--gate` 要求最新门禁通过。
- `block` 必须给具体原因；`reopen` 清空 owner。
- 每次变更 append 到 `harness/state/` 下的 journal（时间、动作、任务、owner、commit、gate 报告）。
- `taskctl check` 校验任务板一致性（in_progress 有 owner、done 有 note、blocked 有原因）。

### 4.3 创建任务

```bash
.venv/bin/python -m harness.taskctl new \
  --title "Add RULER long-context dataset" \
  --priority P1 \
  --scope src/datasets/ --scope src/modelbench/resources/config/scenarios.yaml \
  --acceptance "validate passes" --acceptance "pytest passes" \
  --context "LongBench-v2 不足以覆盖多跳长文推理" \
  --link docs/ARCHITECTURE.md
```

任务 ID 自动递增；新任务默认 `acceptance` 含"门禁通过"。

---

## 5. AI 全自动闭环

### 5.1 单轮循环（一个 Agent）

```bash
# 0) 读契约（每次会话开始时）
cat AGENTS.md
cat docs/ARCHITECTURE.md      # 架构与不变量
cat harness/prompts/implementer.md

# 1) 选任务
.venv/bin/python -m harness.taskctl list --status pending
.venv/bin/python -m harness.taskctl show T-00X
.venv/bin/python -m harness.taskctl claim T-00X --owner "$AGENT_NAME"

# 2) 实现 + 自测（先跑相关测试，再全量）
.venv/bin/python -m pytest test/test_xxx.py
.venv/bin/python -m harness.gate

# 3) 完成（必须引用门禁报告；建议同时 git commit）
.venv/bin/python -m harness.taskctl complete T-00X --gate --note "变更点/测试/门禁报告摘要"

# 4) 提交
git add -A && git commit -m "T-00X: ..."
```

### 5.2 多 Agent 分工（推荐）

| 角色 | 提示词 | 职责 | 写权限 |
|---|---|---|---|
| Planner | [prompts/planner.md](prompts/planner.md) | 读架构/审计/门禁，拆任务、定验收、排序 | 只写 `harness/tasks/` |
| Implementer | [prompts/implementer.md](prompts/implementer.md) | 认领任务、实现、补测试、跑 gate、提交 | 任务 `scope` 内 |
| Verifier | [prompts/verifier.md](prompts/verifier.md) | 只读独立审计：复算 bundle、核对冷缓存、检查文档承诺 vs 代码 | 只写 `/tmp` 与报告 |

**写作用域**：同一时刻一个任务只有一个 owner；若两个任务改同一文件，先完成一个再认领下一个，
或在任务 `scope` 中显式声明冲突。跨 Agent 交接必须通过任务 `notes` 与 git commit 消息，
不要依赖对话上下文。

---

## 6. 与 CI 的关系

- 本地：`harness.gate` 默认检查包含 `validate` 与 `dataset_integrity`（需要已下载数据集）。
- CI：`.github/workflows/ci.yml` 运行 `harness.gate --ci`（无数据集、无 oMLX、无校验和），
  仍覆盖 imports/compile/tests/perf_plan/docs_links/docs_contract/secrets/hygiene；
  `dataset_integrity` 报告为显式 skipped，launcher/数据依赖测试在缺少 `.venv` 或数据时跳过。
- live smoke 不进 CI（需要真机 oMLX），由人工或带 GPU 的自托管 runner 触发：
  `harness.gate --live`。

---

## 7. 设计取舍

1. **文件即状态**：任务用 JSON 文件而非数据库，AI 可直接读写、git 可 diff、冲突可见。
2. **确定性优先**：门禁不依赖网络、不依赖 LLM 判断；同一 commit 多次运行结果一致。
3. **证据而非声明**：`complete --gate` 读取通过报告；`--live` 会实际跑 smoke 并检查
   `records.jsonl` 的冷缓存字段，而不是相信日志。
4. **不引入重量级依赖**：只用标准库 + 项目既有依赖，降低 AI 环境差异。
5. **失败即信号**：gate 失败时 `detail` 给出可操作原因；任务 `blocked` 必须写具体阻塞条件。

---

## 8. 常见问题

| 现象 | 处理 |
|---|---|
| `validate` 失败（数据缺失） | 本地先 `download --dataset all`；CI 用 `--ci` |
| `perf_plan` 失败 | 检查 `default.yaml`/`models.yaml`/`scenarios.yaml` 与 `run --dry-run` |
| `hygiene` 报 tracked 生成物 | `git rm --cached <path>` 并在 `.gitignore` 加规则 |
| `complete --gate` 报 report missing | 先运行 `python -m harness.gate` 生成 `latest.json` |
| 任务 ID 冲突 | `taskctl new` 自动取最大编号 +1；不要手工改已有 ID |
| 想重置任务板 | 任务文件是 tracked 的，用 git 恢复；journal 可直接删除（gitignored） |
