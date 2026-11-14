# 精度测试方案（ACCURACY TEST PLAN）

> **状态**：current ｜ **版本**：v1.0（2026-11-11）｜
> **适用**：`modelbench` 0.3.x 双方法（method=`performance` / `accuracy`）｜
> **读者**：维护者、验证者、AI Agent ｜
> **范围**："精度测什么、怎么跑、怎么判、怎么统计、如何验收"；实现见
> [32-method-architecture.md](../design/32-method-architecture.md)，样本设计见
> [12-accuracy-testset.md](12-accuracy-testset.md)，性能侧见 [21-test-plan.md](21-test-plan.md)。
>
> 本文是**精度方法的唯一事实源**。性能方法的运行协议、指标与验收仍以
> [21-test-plan.md](21-test-plan.md) 为准；两者不混算、不互相引用分数。

## 1. 目标、对象与边界

**目标**：对同一组固定精度样本，逐样本调用 oMLX 的 chat completions，用**确定性判题器**得到
0/1 或 F1 分数，聚合出 8 场景分数与宏平均，并对同一批样本上的两个模型（如 bf16 与量化版本）
做配对比较，给出精度损失 Δ 与显著性。

**测试对象**：本地 oMLX 服务上的语言模型（默认小模型；精度样本集面向 1B 级模型校准）。
被测对象与性能测试**相同**：模型、oMLX 版本、硬件、默认配置。

**输入资产**：`dataset/test/accuracy/testset.jsonl`（2,000 行 = 8 场景 × 250；由
`tool/dataset/accuracy/build.py` 从 `dataset/raw` 确定性构建，gitignored）。每行自带
`messages`（完整 prompt）、`judge`（判题契约）、`input_tokens`、`language`、`dataset`、`meta`。

**非目标**：

1. 不与公开榜单（MMLU/HumanEval 等）横比：提示词、抽样口径、判题实现不同；
2. 不做 LLM judge / 人工评分 / 工具循环 / 多轮对话 / 图像理解；
3. 不测训练数据污染、安全对齐、毒性；
4. 精度分数**不得**与性能指标相乘/加权成"综合分"，也不与性能 run 混算。

**与性能测试的关系**：

| 维度 | 性能（performance） | 精度（accuracy） |
|---|---|---|
| 问题 | 多快、占多少内存 | 答得对不对 |
| 样本 | 8 场景 × 1K/8K/32K × 10 = 240 | 8 场景 × 250 = 2,000 |
| 单请求 | 统一 `perf.max_tokens=50` + decode guard | 每行 `judge.max_new_tokens`，**无** decode guard |
| 判分 | 不判分（`score=null`） | 5 类确定性判题，逐样本落分 |
| 聚合 | TTFT/TPS/内存分布 + ISL 双口径 | 场景得分/宏平均/配对 Δ/显著性 |
| 输出视图 | 整体/数据集/逐样本/环境/报告 | 场景得分/语言与数据集/逐样本判分/错误统计/环境/报告 |

## 2. 运行协议（固定口径）

| 项 | 约定 | 记录位置 |
|---|---|---|
| 输入 | 直接使用 jsonl 的 `messages`，**不重渲染、不截断、不填充** | `request.prompt_sha256` + blob |
| 解码 | `temperature=0`、`top_p=1`、不重复惩罚；`stream=true`、`include_usage=true` | `request.temperature`/`request.top_p`（`run.json.config.run` 记协议快照） |
| 生成长度 | 每行 `judge.max_new_tokens`（知识/多语 1,536；指令 1,024；长上下文 32/64/128；数学 512；编程 768；Agent 256；搜索 32） | `request.max_tokens` |
| 缓存 | **cold**（每个 measured 请求前清 hot+SSD，禁写缓存，`cached_tokens==0`）；warm/none 拒绝 | `records[].cache` |
| 思考模式 | 默认 `disabled`；按模型的 `ModelSpec.thinking_disable` 选择关闭方式：`explicit_false`（发送 `chat_template_kwargs.enable_thinking=false`，默认）或 `omit`（不发送 chat-template kwargs）。`auto/enabled` 仅诊断 | `run.json.config.run.thinking_mode` + `request.context_meta.thinking_control` |
| 重试 | 传输/HTTP 错误重试 **1** 次（线性退避）；重试仍失败记 `error`，该样本不计判分、单列错误率 | `records[].status/error` |
| 预热 | 模型级 `warmup_per_model=1`（丢弃、不判分）；per-cell 预热 0 | `run.json.config.run` |
| 轮数 | 固定 `repeats=1`（每样本一次响应对应一条判分记录）；重复性基线用独立 run | `run.json.config.run.repeats` |
| 超窗 | `tokenized(prompt) + max_new_tokens + 128 > model_context_window` → `skipped(over_window)`，**不截断** | `run.json.outcome.skipped` |
| 温度/种子 | 不使用随机种子（温度 0 已确定） | — |
| 判题失败 | 判题器异常/依赖缺失 → `status=judge_error`，计 0 分并计入错误率；响应原文保留 | `records[].score.detail` |
| 空响应/无法解析 | 该样本计 0 分，`format_failure=true`；格式失败率单独统计 | `records[].score` |

**为什么 cold**：精度结论必须是"同一输入得到同一输出"的确定性结果；prefix cache 命中会改变
数值路径（尤其量化内核）。精度 run 与性能 run 都使用冷缓存，但精度侧**不启用 decode guard**
（guard 会改变被评测的回答本身）。

**模型级 thinking 关闭策略（2026-11 实测）**：`MiniCPM-V-4.6` 本身是**默认非思考**模型，
但其运行时 processor（打包的 mlx_vlm 0.7.10）对 `enable_thinking=False` 有特殊分支：按
`add_generation_prompt=False` 渲染后只追加 `<|im_start|>assistant\n`，**丢掉了模型模板默认的
空 `<think>\n\n</think>\n\n` 块**；模型因此从开放 reasoning 开始，耗尽 `max_new_tokens`、
content 为空。不传该 kwargs 时走模板默认（空 think 块），不会思考。`ModelSpec.thinking_disable`
允许按模型选择 `explicit_false`（默认）或 `omit`；`MiniCPM-V-4.6` 使用 `omit`。每条请求在
`request.context_meta.thinking_control` 记录实际策略；若某模型 format_failure 异常高，应先
检查该字段、实际 prompt（`/v1/tokenize`）与 `reasoning_text`，不要把问题当成判题 bug。

## 3. 判题

### 3.1 判题族（恰好 5 个，全部确定性）

| 判题族 | 场景（行数） | 输入契约 | 判定 | 指标 |
|---|---|---|---|---|
| `mc_letter` | knowledge_qa 250 + multilingual 250 | `judge.answer`（A–D） | 优先末次 "answer is X / 答案是 X"，否则括号外独立字母 | 0/1 |
| `ifeval` | instruction_following 250 | `judge.instruction_id_list` + `kwargs` | Google 官方 checker，prompt-level strict | 0/1 |
| `normalized_match` | long_context 250 + search_agent 250 + math_logic 250 | `answers[]`、`metric`、`tolerance` | `f1` / `f1_zh` / `rouge_zh` / `numeric_exact` | F1∈[0,1] 或 0/1 |
| `code_exec` | coding 250 | `test_code`、`prompt_prefix`、`entry_point`、`timeout_s` | 子进程执行官方单测（6s、CPU/句柄/文件大小限制） | pass@1 |
| `tool_call` | general_agent 250 | `ground_truth`、`functions` | JSON/JSONL 解析；函数名 + 参数可接受值；多调用无序；irrelevance 要求空调用 | 0/1 |

判题实现单一事实源：`src/modelbench/domain/judges/`（由 `tool/system/judges.py` 迁移；
`tool/system/judges.py` 保留为兼容转发）。判题器**只读**样本的 `judge` 字段与模型响应，
不读网络/外部状态。gold 自检见 `tool/system/validate_judges.py`（1,750 正例 + negative controls）。

### 3.2 可选依赖与失败策略

`ifeval` 需要官方 checker（`absl-py`/`nltk`/`langdetect`/`immutabledict`）；中文 F1/ROUGE 需要
`jieba`/`rouge`。这些在 `pip install -e ".[eval]"` 中声明：

1. run 启动时检查**本次测试集 manifest 声明的判题族**的依赖（`domain/judges/missing_dependencies`，
   用 `importlib.util.find_spec` 探测、不 import 重模块；报告缺失的**模块名**：`absl`/`nltk`/
   `langdetect`/`immutabledict`/`jieba`/`rouge`）。传族名时按族保守判断（`normalized_match`
   会连带要求中文栈 `jieba`+`rouge`；如调用方持有 judge 契约，可传入契约把判断收窄到实际
   metric，例如纯英文 `f1` 不要求中文包）。缺失时写 `run.json.outcome.warnings`（含缺失模块名与
   `pip install -e ".[eval]"` 提示），对应样本 `status=judge_error`（不静默给分）；
2. `judge_error` 计 0 分并计入 `n_judge_error` 与错误率；报告显式列出；
3. 缺依赖时**不**把该场景排除出分母（否则会虚高），由错误率与 judge_error 计数暴露质量。

### 3.3 单样本契约（`records[].score`）

```jsonc
{
  "correct": true,              // 0/1 判题族；F1 场景为 (score >= 1.0)
  "score": 1.0,                 // 0/1 或 F1
  "detail": {
    "family": "mc_letter",      // 判题族
    "metric": "accuracy",       // accuracy | f1 | f1_zh | rouge_zh | numeric_exact
    "pass": true,
    "pred": "B",                // 解析出的预测（截断到 500 字符）
    "gold": "B",
    "reason": "letter_match",
    "format_failure": false,    // 响应为空/无法解析
    "judge_error": false,       // 判题器异常/依赖缺失
    "judge_latency_ms": 1.2,
    "max_new_tokens": 1536
  }
}
```

## 4. 指标定义

设场景 s 的行集合为 `R_s`（每行一个判分结果 `score ∈ [0,1]`，错误/判题失败按 0 计）。
**计数口径（单一事实源 `domain/accuracy_metrics.py`）**：`n_valid = n_ok + n_judge_error`
（产生了判分尝试的 measured 记录）；**`n_scored = n_valid`**——判题失败计 0 分但**仍留在
分母**（`n_scored = n_ok - n_judge_error` 会虚高分数，禁止）；`error`/`timeout` 不进分母、
单列；`n_missing_score` 统计 `status=ok` 但无 `score` 的契约违约行（计 0 并暴露）。

| 指标 | 公式 | 说明 |
|---|---|---|
| 场景分 `scenario_score` | `mean(score for r in R_s)` | 8 场景各自先算分（等权） |
| 场景准确率 | 0/1 判题族：`n_correct / n_scored`；F1 场景不适用（报 `score`） | 与 Wilson CI 一起 |
| 宏平均 `macro_score` | `mean(scenario_score for s in 8)` | **总分口径**，避免行数/场景权重影响 |
| 微平均 `micro_score` | `mean(score over all rows)` | 仅辅助，不作结论 |
| 覆盖/质量 | `n_samples`、`n_valid`、`n_ok`、`n_error`、`n_timeout`、`n_judge_error`、`n_format_failure`、`n_missing_score`、`n_scored`、`n_correct`、`n_scenarios_scored`、`n_skipped(over_window)` | 必须与分母对账（`n_ok + n_error + n_timeout + n_judge_error == n_measured`） |
| 95% CI | 0/1：Wilson score interval（`accuracy_ci95_*`）；score：`mean ± 1.96·std/√n`（`score_ci95_*`） | `n<2` 时 null |
| 语言剖面 | 按 `language` 分组的场景分/宏平均（`by_language`） | 单语 25 条只做剖面，不做排名 |
| 数据集剖面 | 按 `dataset` 分组（math=gsm8k/cmath，coding=humaneval/mbpp，long_context=7 任务） | 定位退化来源 |
| 格式失败率 | `n_format_failure / n_measured` | 单场景 >20% 报告标注"格式不适配" |

**配对比较（bf16 vs 量化，同一 testset SHA）**：

| 统计量 | 定义 | 用途 |
|---|---|---|
| Δ(pp) | `100 × (score_bf16 − score_quant)`（同场景） | 主结论：绝对精度损失 |
| 相对损失 | `Δ / score_bf16` | 跨场景可比 |
| McNemar | 2×2（bf16 对/量化错 vs 反之），精确二项检验 | 0/1 场景显著性 |
| Holm 校正 | 8 场景同时检验 | 控制族错误率 |
| 配对 bootstrap | 对 2,000 样本对重采样 10,000 次，取 Δ 的 2.5%/97.5% 分位 | F1 场景 Δ 的 95% CI |

**判定口径**：`p<0.05（Holm）`（0/1 场景）或配对 bootstrap CI 排除 0（F1 场景）才视为统计显著；
要标记"显著退化/提升"还需 `|Δpp|` 超过**噪声基线**。噪声基线由使用者用
`./modelbench analyze ... --noise-baseline-pp N` 显式给出（通常取 bf16 独立重复 run 的场景 Δ
波动；工具**不自动**推算 2× 基线）；缺省 0 时只报方向与显著性，不宣称退化。

## 5. 执行流程

```text
选择方法=accuracy / 模型 / 测试集（默认 accuracy）/ 场景 / 样本上限
  → 双锁（endpoint 锁 → output 根锁）
  → 逐模型：隔离其它模型 → 默认配置复位 → 加载 → 环境快照 → 模型级预热
  → 逐场景（8）：
        读取 accuracy_jsonl 场景切片 → 选择样本（默认 250；--samples 分层截断）
        → 每个样本：
             构造 RequestSpec（messages 原样、max_new_tokens=judge 值、温度 0）
             → tokenize 最终 messages（窗口判定；测量窗口外）
             → 清 hot+SSD 缓存 → 请求（禁写缓存）
             → 响应落原文 → 确定性判题 → score/format_failure/judge_error
             → 写 records.jsonl（含 score）+ prompt blob
  → 聚合：cells（model×scenario×dataset）+ overall（model×宏平均）+ 语言剖面
  → 写 aggregates.json / samples.csv / report.md / MANIFEST.json
  → batches/<batch_id>.json + index.json + exports/<batch>.xlsx
  → 释放双锁；恢复模型状态
```

**样本选择**：默认**全部 250/场景**（固定样本，不裁剪）；`--samples N` 仅用于冒烟/排期，
按 `meta.stratum` 做确定性分层最大余数分配（与性能侧同一实现），并在 `run.json` 记录
`n_selected` 与每场景选择数；不同 `N` 的 run 不得互相比较分数。`--sample-ids` 支持手选（诊断）。

## 6. 输出

bundle 布局与性能相同（`output/runs/<run_id>/`，`run_id=<batch>__accuracy__<model_slug>`），
但 `method=accuracy`，且 `aggregates.json` / `samples.csv` / `report.md` 为精度形状：

| 文件 | 精度侧内容 |
|---|---|
| `run.json` | `method=accuracy`；`config.accuracy`（样本选择：samples_per_scenario/samples_by_scenario/sample_ids）、`config.run`（固定协议快照：cold/repeats=1/max_retries=1/warmup_requests=0/thinking_mode）、`outcome` 含判分计数 |
| `records.jsonl` | 每样本请求/响应/判分；`score` 非空；`sample_meta` 含 language/upstream/judge |
| `aggregates.json` | `overall[]`（model 宏平均/微平均/CI/语言剖面）+ `cells[]`（model×scenario×dataset） |
| `samples.csv` | 逐样本判分表（score/correct/format_failure/judge_error/judge_family/…） |
| `report.md` | 场景得分表、宏平均、语言剖面、失败/错误统计、协议与样本 SHA |
| `MANIFEST.json` | 与其他方法同构（文件 + blob 引用） |

**独立 run_id**：精度必须使用独立 run（`__accuracy__`），不得与性能 run 混算；比较两个模型时
必须引用同一 `testset.jsonl` SHA-256（写入 `batch.json.inputs.testset.sha256`）与同一
`judge` 契约版本。

## 7. 验收标准（可量化、逐条可复核）

| # | 验收项 | 判据 |
|---|---|---|
| AC-A1 | 样本可追溯 100% | 每个 selected 样本恰有一条 measured 记录，或一条 `outcome.skipped`（含 reason）；`n_selected == n_measured + len(skipped)` |
| AC-A2 | 判分覆盖 | 所有 `status=ok` 记录都有非空 `score`；`status=judge_error` 计 0 且单列；`n_ok + n_error + n_timeout + n_judge_error == n_measured` |
| AC-A3 | 判题确定性 | 同一响应重复判分结果逐位一致（`test/test_judges.py` 单测 + `tool/system/validate_judges.py` gold 自检 PASS；判题器无网络、无 LLM、不读外部状态） |
| AC-A4 | 协议一致 | 每条记录 `request.temperature=0`、`top_p=1`、无 decode guard、`cache.mode=cold`、`cache.cached_tokens==0`（冷证据）；`request.max_tokens == judge.max_new_tokens`；`run.json.config.run.repeats=1` 且 `config.method="accuracy"` |
| AC-A5 | 聚合可复算 | `aggregates.json` 的 cells/overall 可由 records.jsonl 逐条复算（场景分/宏平均/微平均/`n_scored`/`n_correct`/`n_judge_error`/`n_format_failure`/`n_missing_score`/`n_scenarios_scored`/`by_language`/CI 一致，浮点末位允许差）；`cells` 键为 `(model, scenario, dataset)`，`overall` 每模型一行 |
| AC-A6 | 宏平均口径 | `macro_score` = 8 场景等权平均；每场景分母只含该场景 selected 样本；错误/判题失败按 0 计并单列 |
| AC-A7 | 错误可见 | 格式失败率、judge_error、error/timeout、over_window skipped、缺依赖 warning 全部落盘并在报告/Web 展示，不静默剔除 |
| AC-A8 | 同一样本集 | 所有设备/模型使用同一 `testset.jsonl`（SHA-256 与 manifest 一致）；不做 per-model 构造/裁剪 |
| AC-A9 | 不混算 | 精度 run 的 `score` 不进入性能聚合；性能 run 的 `score` 恒 null；两者 `method` 不同、目录/索引可区分 |
| AC-A10 | 配对分析 | 两个 accuracy run 可按同 `(scenario, sample_id)` 配对，输出 Δ、McNemar（0/1）与 Holm 校正；样本不齐/判题族不一致必须显式报错而不是静默丢弃 |

## 8. 风险与已知限制

1. **耗时**：2,000 样本 × 1 模型约 3h（长上下文约 2–2.5h，见
   [12-accuracy-testset.md](12-accuracy-testset.md) §6.5）；bf16 + 量化两次约 6h。冒烟用
   `--samples N`，但分数不得与全量结果混比。
2. **代码执行风险**：`code_exec` 在子进程执行模型生成代码（6s + 资源限制），**不是安全沙箱**；
   仅在受控测试机运行，判题器不访问网络。
3. **可选依赖**：缺 `[eval]` 依赖时 ifeval / 中文 F1 判题会 `judge_error`；必须按 §3.2 显式暴露。
4. **难度带/污染**：样本难度由参考模型试点校准；公开数据无法排除预训练污染。本方案的主结论是
   **同一模型 bf16 vs 量化的相对比较**，污染对两版近似同向，不影响主结论，但不得与公开榜横比。
5. **统计分辨率**：单场景 250 行约 ±6.2pp（独立近似）；双语配对/多语 25×10 行间不独立，正式报告
   用题级聚类 bootstrap，不要把 ±6.2pp 当配对场景的精确 CI。
6. **思考模式**：默认关闭，但关闭方式按模型选择（`ModelSpec.thinking_disable`：
   `explicit_false` / `omit`）。若某模型把 `enable_thinking` 的**存在**当成开启（如
   MiniCPM-V-4.6），reasoning 会耗尽 `max_new_tokens`，空响应/截断进入 `format_failure`；
   该类 run 不得与默认口径混算，需先按 §2 的模型级策略修复并记录
   `request.context_meta.thinking_control`。
7. **长上下文超窗**：模型窗口不足的行 `skipped(over_window)`，分母与错误率显式展示；不得截断后
   当作原样本判分。
8. **Web/CLI 预算**：全量 2,000 行/模型的单批耗时很长，Web 必须展示估算样本数与时间提示；
   同一 oMLX 实例仍由双锁串行。

## 9. 与性能测试的验收边界

性能侧 AC-1…AC-11（见 [21-test-plan.md](21-test-plan.md) §7）**不适用于**精度 run：精度不要求
`gen_tps` 有效性、ISL ±10% 入带、`n_gen_tps_invalid` 等指标。反之，精度 AC-A1…AC-A10 不要求
性能 run 满足。两者共同的硬不变量仍是：冷缓存 fail-closed、逐样本可追溯、不静默降级、
单实例双锁、固定样本集、文档契约。
