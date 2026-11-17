# 性能测试样本集：场景、构造、抽样与有效性评估

> 范围：**只覆盖性能测试的样本选择与构造**（TTFT / TG TPS / TPOT / 峰值内存），不涉及判分与能力结论。
> 被测对象：本地小模型（1B–32B，单请求串行）。产物：**8 场景 × 1K/4K/32K × 10 = 240 条**样本、
> 24 个 cell；上游证据在 `dataset/raw/`，产物在 `dataset/test/performance/`，
> 构造器在 `tool/dataset/performance/`。
>
> 核心口径：**实际 prompt token 数必须精确等于 1,024 / 4,096 / 32,768（容差 0）**，以服务端
> `usage.prompt_tokens`（计费口径）为准。长度不是靠"选到差不多的样本"碰出来的，而是拆成
> **构造期材料超量** + **运行期按目标模型 tokenizer 裁剪/补齐**两段来保证；构造期负责
> "内容完整、证据在位、材料足够"，运行期负责"长度精确"。运行期不做长度断言、不因长度跳过。

---

## 0. 摘要

| # | 场景 | 输入主料（primary，必须完整） | 尾部填充（filler，可裁） | 任务 | 语言 |
|---|---|---|---|---|---|
| 1 | 知识问答 | HotpotQA gold 段落（1K/4K）；LongBench-Pro T3 中文整篇（32K） | distractor 段落 / 同族整篇 | 原题问答 | en / zh |
| 2 | 指令遵循 | LongBench v1 `vcsum` 中文会议记录 | 同族会议记录 | Multi-IF 真实约束（渲染为中文要求） | zh |
| 3 | 长上下文 | LongBench-Pro T4 英文整篇 | 同任务族整篇 | 汇总并给出要点 | en |
| 4 | 数学与逻辑 | MATH-500 完整题+解（示范） | 更多完整示范 | AIME 2026 题面 | en |
| 5 | 编程 | HumanEval 题+官方解（1K）；LongBench v1 `lcc` 代码上下文（4K/32K） | `repobench-p` 仓库上下文 | 读代码答问题 | en |
| 6 | 通用 Agent | BFCL v4 目标工具 schema | 去重工具目录 | 文本形式给出用户请求 | en |
| 7 | 搜索 Agent | HotpotQA gold 段落（1K/4K）；LongBench-Pro T1/T5 英文整篇（32K） | distractor / 同族整篇 | 检索/归因任务 | en |
| 8 | 多语言 | Global-MMLU 同语种题目+答案（示范） | 更多示范 | 最后一道选择题 | zh/en/fr/de/es/pt/ru/ja/ko/ar |

**六个关键决策**

1. **先决定"输入里必须有什么"，再决定长度**：性能测量只依赖 token 数，但样本必须仍是"该场景下
   真实可答的请求"。因此每个 target 拆成 `primary`（承载场景语义、必须完整进入 prompt）与
   `filler`（同一语料族的其它真实文本，只负责把长度补到档位）。裁剪永远只发生在 filler 尾部。
2. **构造期与运行期分工**：被测模型 tokenizer 各不相同（同一段中文，在 cl100k 与 MiniCPM 上可差
   2 倍）。构造期用**通用参考 tokenizer（cl100k_base）**把材料做到档位的 2 倍；运行期由 runner
   用**目标模型 tokenizer** 把 body 裁到精确档位（不足时平铺、最后空白补齐），并以服务端
   `usage.prompt_tokens` 对账。这样一套设计可服务所有模型，不需要"每个模型重建一份数据集"。
3. **primary 完整性按最坏 tokenizer 留余量**：`primary_model_tokens + task_tokens + 2 ≤ 档位`，
   其中 `primary_model_tokens` 取本地模型库 11 个 tokenizer 的最大值（无本地 tokenizer 时用
   文档化的膨胀系数：英文 1.25、中文 0.85）。构造期不满足的候选直接丢弃，不进入样本。
4. **不假设三档嵌套**：同一场景的 1K/4K/32K 是三个独立 target（不同主料/不同题），只共享
   "构造算法 + 抽样顺序"。避免"1K 是 32K 的前缀"这种在 B′ 下必然丢证据的结构。
5. **中文不可缺**：中文行 43/240（18%）：知识问答 32K、指令遵循全档、多语言含 zh；其余场景为英文。
6. **输出长度固定 50 token**：testset 声明 `max_tokens: 50`（TG TPS 的固定解码窗口），runner 照常
   追加长回答指令（decode guard），guard 计入 prompt 后再做对齐。

---

## 1. 目标、指标与约束

### 1.1 测什么

| 指标 | 定义 | 本文档的作用 |
|---|---|---|
| TTFT | 请求发出到首个非空 delta 的时间（不含 tokenize/清缓存） | 由 prompt token 数决定 → 长度必须准 |
| TG TPS | `(completion_tokens-1)/(t_last-t_first)`，仅可信解码窗口输出 | 需要 `completion_tokens ≥ 16`、窗口 ≥50ms；本文档用 **max_tokens=50** 固定窗口 |
| 峰值内存 | 请求窗口内 oMLX 进程树 RSS 峰值 | 由 KV/prefill 规模决定 → 长度档位是自变量 |

指标细节以 [20-methodology.md](20-methodology.md) 为准；本文档只负责"输入样本"。

### 1.2 硬约束

| # | 约束 |
|---|---|
| C1 | 覆盖 8 个场景：知识问答、指令遵循、长上下文、数学与逻辑、编程、通用 Agent、搜索 Agent、多语言 |
| C2 | 每个场景覆盖三档上下文：**1K = 1,024 / 4K = 4,096 / 32K = 32,768** prompt tokens |
| C3 | 每个 `(场景, 档位)` cell 10 条，共 240 条、24 cell |
| C4 | 实际 prompt token 数严格等于档位（容差 0，计费口径） |
| C5 | 不能全英文：中文至少覆盖 1 个完整场景 + 多语言含中文（实际 43/240） |
| C6 | 输出长度 >50 的要求由 `max_tokens=50` + 长回答指令满足；样本本身不依赖模型"愿意写长" |
| C7 | 数据源可公开下载、许可允许本地评测使用，可用 `fetch.py` 还原并核对 SHA-256 |
| C8 | 同一份设计（target/抽样/构造/任务块）对所有设备、所有模型复用；只允许运行期改编排长度，不允许换样本 |
| C9 | 运行期不因"长度不匹配"跳过；只有 tokenizer 不可用、无法对齐、窗口不足才 skip 并记录原因 |

### 1.3 非目标

- 不评质量/正确率：性能模式不判分（`score` 恒为 null）。样本的"正确答案"只用于构造期证据校验。
- 不做榜单同口径声明：与官方 leaderboard 的提示词、判分、长度口径均不同。
- 不比较场景之间或档位之间的"能力"；同一场景的多档只用于观察长度对 TTFT/内存的影响。

---

## 2. 设计原则

- **P1 设计固定**：target 集合、抽样顺序、任务块、filler 顺序全部由 `SEED=42` + 上游 id 的
  SHA-256 排序确定；同一设计在任何机器上重建出同一 `rows_sha256`（上游文件一致时）。
- **P2 长度与内容是两件事**：档位只规定 prompt token 数；内容只负责"该场景的真实请求形态"。
- **P3 primary + filler**：primary 必须完整（承载问题所依赖的全部证据），filler 只补长度。
- **P4 结构简单可审计**：每条样本 = `[primary 文本] + [filler 文本]`（body，一条 user 消息）+
  任务块（第二条 user 消息）；不做检索模拟、不做多轮、不注入 tools 字段。
- **P5 构造期验证，运行期对齐**：所有"内容是否正确/是否放得下"在构建时校验；运行期只做长度对齐。
- **P6 可复现**：所有上游文件有固定 URL + SHA-256；构造与抽样确定；`build.py --check` 可离线复核。

---

## 3. 场景定义与数据源选择

### 3.1 场景定义（"这条样本要模拟什么请求"）

| 场景 | 请求形态 | 输入里必须是真实内容的部分 |
|---|---|---|
| 知识问答 | 长文档 + 针对文档的问题 | 问题的答案所在段落（gold）必须完整 |
| 指令遵循 | 长材料 + 一组格式/内容约束 | 材料本身完整；约束来自真实 IF 数据集 |
| 长上下文 | 长文档 + 汇总任务 | 整篇文档（只要放得下） |
| 数学与逻辑 | 示例题解 + 目标题 | 示范完整（题+解），目标题完整 |
| 编程 | 代码上下文 + 问题 | 目标代码单元完整 |
| 通用 Agent | 工具目录 + 用户请求 | 目标工具的 schema 完整；请求完整 |
| 搜索 Agent | 多文档 + 检索/归因问题 | 答案所在段落（gold）完整 |
| 多语言 | 同语种示例 + 最后一道题 | 目标题完整；示例为同语种真实题 |

### 3.2 选型矩阵（按档位）

| 场景 | 1K | 4K | 32K | 语言 |
|---|---|---|---|---|
| 知识问答 | HotpotQA gold 段落 + distractors | HotpotQA gold 段落 + distractors | LB-Pro T3 中文整篇 + 同族 | en / zh |
| 指令遵循 | vcsum 会议记录 + 约束 | 同 | 同 | zh |
| 长上下文 | LB-Pro T4 英文整篇（8K band 段边界前缀） | LB-Pro T4 英文整篇（8K band） | LB-Pro T4 英文整篇（16K band） | en |
| 数学 | MATH-500 完整题解 + AIME 题 | 同（更多示例） | 同（最多示例） | en |
| 编程 | HumanEval 题 + 官方解 | lcc 代码上下文 | lcc + repobench-p | en |
| 通用 Agent | BFCL 目标工具 + 去重目录 | 同 | 同（目录更长） | en |
| 搜索 Agent | HotpotQA gold + distractors | 同 | LB-Pro T1/T5 英文整篇 | en |
| 多语言 | Global-MMLU 同语种示例 + 题 | 同 | 同 | 10 语种 |

### 3.3 逐场景论证（为什么是它，备选是什么）

**知识问答**。需要"答案在文档里、问题在文档后"的形态。1K/4K 用 HotpotQA：gold 段落 ~100–600
cl100k tokens 恰好落在 1K/4K 的 primary 预算内，且答案字符串可作为构造期证据校验；distractor
段落提供同分布 filler。32K 的 primary 预算约 25K cl100k tokens（英文最坏膨胀 1.25×），HotpotQA 单条文档不够，改用
LongBench-Pro T3（中文，原生 8K–256K，整篇 primary；任务块为原题；`answer` 是选项字母，故证据校验
退化为"整篇完整性"，记录 `evidence_check=complete-document`）。备选：MMLU-Pro/C-Eval（题干太短，
只能靠堆题，不模拟"长文档问答"）、LongBench v2 SD-QA（原生 ≥12K，1K/4K 需砍前缀，会丢证据，被 B′
否决）。

**指令遵循**。需要"长材料 + 多条硬约束"。vcsum 是中文会议记录（200 篇，0.6K–23K tokens），
天然适合做"读长文按格式输出"；约束来自 Multi-IF 中文 454 条真实 `instruction_id/kwargs`（如
"不要出现逗号""恰好 3 段"），渲染成中文要求后与材料拼接。备选：IFBench（HF 镜像 404，
GitHub 可用但无中文长文载体）、IFEval（英文、短 prompt，无长材料）。局限：Multi-IF 原始
responses 为空，约束不构成多轮历史——这里只取"约束集合"作为任务，不声称复现 Multi-IF 的多轮协议。

**长上下文**。LongBench-Pro T4 `Summarization & Synthesis` 是唯一"原生 8K–256K + 英文 + 有
汇总任务"的公开集；1K/4K 用 8K band 文档在段落边界截断（`cut_policy=paragraph`），32K 用 16K band
整篇（约 27K–36K cl100k tokens），filler 取同任务族整篇。备选：HELMET（data_v2 8.9GB，超 1GB
下载门槛）、∞Bench/NoLiMa（镜像不可得或需另配）。

**数学与逻辑**。AIME 2026（30 题，2026-10 发布）提供"当年新题"的时效性与难度；题面短，需要
示范把长度撑起来，MATH-500（500 题，含完整解）提供完整题解作 primary/示范——这是目前唯一
"题+解都完整、可公开下载"的组合。备选：GSM8K（题太短、解短）、Omni-MATH/OlympiadBench（镜像
可得但解答质量参差）、GPQA（gated）。说明：本场景的 primary 是"示范"而非"证据"，
`evidence_check=complete-document`，问题不依赖特定段落。

**编程**。1K 用 HumanEval 单题（题面 + 官方解完整，约 100–300 tokens）；4K/32K 用 LongBench v1
`lcc`（代码补全上下文，跨文件）与 `repobench-p`（仓库级上下文）拼出真实代码阅读负载，任务块让
模型"读代码回答问题"。备选：SWE-bench Verified / terminal-bench（需要 clone 仓库或执行代码，
破坏离线可复现；且单条上下文可达 100K+，与档位不匹配）、LongBench v2 Code Repository（原生 ≥26K，
1K/4K 只能砍前缀，会丢证据）。说明：本场景是**代码阅读**负载，不是代码生成——生成任务的输出
长度不可控，会污染 TG TPS 口径。

**通用 Agent**。BFCL v4（2026-04 发布）提供真实工具 schema（simple_python 400 + multiple 200，
每条 1–2 个工具，schema 100–250 tokens）。构造为"目标工具置顶 + 全库去重工具目录 + 用户请求"，
不通过 API `tools` 字段传工具（避免 `finish_reason=tool_calls` 使 TG TPS 失效），而是把工具目录
作为文本上下文。备选：ToolBench/xLAM（工具数量多但 schema 噪声大）、terminal-bench（偏 OS 任务、
需要执行环境）、τ²-bench（HF 镜像 404）。局限：这是"读工具目录做规划"的负载，不测真实函数调用协议。

**搜索 Agent**。需要"多文档 + 检索/归因"。1K/4K 用 HotpotQA distractors（10 段落里 2 段 gold，
正好模拟"在干扰项中定位"）；32K 用 LongBench-Pro T1 `Retrieval & Ranking` / T5 `Attribution &
Citation` 英文整篇（原生 8K–256K）。备选：BrowseComp / browsecomp-plus（query 短、证据在开放网络，
无法离线固定语料）、GAIA（gated、依赖外部工具）、Musique/2WikiMultiHopQA（可用，但 4K 档的整篇
context 仍会丢证据，被 B′ 否决作为主料）。局限：不模拟真实检索动作（无工具调用），只模拟"给定
候选文档做定位/归因"。

**多语言**。Global-MMLU 42 语种，取 10 个（zh/en/fr/de/es/pt/ru/ja/ko/ar），每语种 14,042 题。
每档 10 条 = 10 语种各 1 条；同语种题目+答案作示范（filler），最后一道题作任务。备选：MMMLU
（14 语种、无 fr/de 之外的扩展）、MGSM（数学多语，但题短且语言少）、Belebele（阅读理解多语，
但每语种题量少、语言覆盖与中文兼容性一般）。说明：这是"同语种 many-shot + 选择"负载，不测翻译
质量。

### 3.4 复用边界

- LongBench-Pro 出现在 3 个场景（知识问答 32K、长上下文全档、搜索 Agent 32K），但**任务族互斥**
  （T3 / T4 / T1+T5），行集不相交（manifest `cells[*].dataset` 可核对）。**不得跨场景比较**：
  同一模型在 T3 与 T4 上的差异混合了任务与上下文分布两个因素。
- 多语言 10 条样本各来自不同语种：**只能按语言看，不能把 10 条当作同一分布的 10 次重复**。

### 3.5 语言覆盖

中文 43 行：知识问答 32K（10）+ 指令遵循 30 + 多语言 zh（3）；英文 173 行；其余 8 语种各 3 行
（多语言）。满足"不能全英文、单场景可单语言"。

---

## 4. 长度口径与构造

### 4.1 token 口径

- **计数对象**：最终发送的 chat request 的 prompt tokens（含 chat template 与 decode guard），
  以服务端 `usage.prompt_tokens` 为准；`/v1/tokenize` 与其差一个常量偏移（本机实测 +2），
  由 runner 在 warmup 时学习并在记录里写 `prompt_alignment.server_prompt_offset`。
- **参考 tokenizer**：`cl100k_base`（tiktoken，通用、非任何被测模型）。它只用于"构造期量长度"，
  不用于声明"某模型的 token 数"。
- **档位目标**：1K = 1,024、4K = 4,096、32K = 32,768。

### 4.2 两阶段

| 阶段 | 由谁做 | 目标 | 失败处理 |
|---|---|---|---|
| 构造期 | `tool/dataset/performance/build.py` | 材料做到 2× 档位；primary 完整且在最坏 tokenizer 下放得进档位；证据在 primary 内 | 丢弃候选；不够 10 条则该 cell 构建失败（不静默降级） |
| 运行期 | `src/modelbench/application/prompt_align.py` | prompt token 精确 == 档位（计费口径） | tokenizer 不可用/无法对齐/窗口不足才 skip，写原因 |

### 4.3 超量材料

`content_goal(bucket) = 2 × 档位`（cl100k tokens）：1K→2,048；4K→8,192；32K→65,536。

- **为什么 2×**：实测同一中文文本在 cl100k 与 MiniCPM 上的 token 比约 0.48–0.59，英文约
  0.99–1.15。2× 保证即使被测模型 tokenizer "最省"（中文 0.48×）也仍有 ≥0.96 档位的材料可用，
  运行期只需裁剪；同时文件大小控制在 ~24MB（若按"保证不裁剪"设计需 ≥2.2×，文件会到 30–40MB）。
- **残余风险**：若某模型 tokenizer 比 cl100k 省 2 倍以上（理论可能，实际未见），运行期对齐器会
  用"整段 body 平铺"补足（见 4.6），仍能精确命中档位。

### 4.4 primary 完整性预算

- `primary_model_tokens + task_tokens + 2 ≤ 档位`（`TEMPLATE_MARGIN=2`）。
- `primary_model_tokens` 取本地模型库（`~/.omlx/models/**/tokenizer.json`，深度 ≤3）全部
  tokenizer 的最大计数；本地没有模型时用回退系数（英文 1.25、中文 0.85）。
- 参考侧预算：`primary_cl100k_tokens < 0.92 × content_goal`（给 filler 留空间）。
- 构建期 `content_tokens` 与 `content_goal` 允许 95%–110% 的边界容差（段落/句边界裁剪与 BPE 边界所致）；该值只决定材料是否足够，不进入测试：运行期会把正文精确对齐到档位。
- 典型上限（按英文 1.25 估算）：1K 档 primary ≤ ~700 cl100k tokens；4K ≤ ~3,100；32K ≤ ~26,000。
  超出上限、且 `cut_policy=document` 的候选直接丢弃；`paragraph`/`sentence`/`example` 策略的候选
  按边界缩短后保留（仅用于允许截断的场景）。
- 构造期把 `primary_tokens / primary_model_tokens / task_model_tokens / fits_bucket` 写入每行
  `meta`，`build.py --check` 复核。

### 4.5 filler 规则

- 与 primary 同族（同一语料/同一任务族）、去重（按 SHA-256 前缀）、按确定性顺序追加；
- 单条行 filler 字符预算 `content_goal × 12`（防止把整个语料族灌进一行）；
- filler 必须与 primary 不同（同 target 内按文本哈希去重）；
- filler 只补尾部：**裁剪只砍 filler**；primary 与任务块永不裁剪。

### 4.6 运行期对齐（不重复细节，只列算法与保证）

`align_messages(messages, target, count_fn, tokenizer_path)`：

1. 取最长 user 消息为 body（任务块是另一条消息，不参与裁剪）；
2. 若本地能找到目标模型 `tokenizer.json`：在 token 空间裁剪（`ids[:need]` + decode，去尾随
   `\ufffd`），不足时按整段平铺（`_tile`）后再裁；
3. 无本地 tokenizer 时用字符级二分 + 文本平铺；
4. 残差用"单字符多点插入"（尾部/头部/倒数第二位/中点）补齐，最多 48 轮；
5. 发送前用 `/v1/tokenize` 复核（≤8 轮），最终以 `usage.prompt_tokens` 对账；
6. 结果写 `context_meta.prompt_alignment`（`mode`/`server_calls`/`trim_chars`/`pad_tokens`/`iterations`/
   `server_prompt_offset`/`latency_ms`）。

**保证**：最终 `usage.prompt_tokens == 档位`；实测对齐耗时 0.2–0.5s/条（32K 档，本地 tokenizer）。
**不做**：运行期长度断言、±10% 误差带、"长度不等即 skip"。

### 4.7 构造期校验（`build.py --check`）

| # | 检查 |
|---|---|
| 1 | 24 cell × 10 行；id 唯一；`tokens == 档位`；无 `tools` 字段；两条 user 消息 |
| 2 | `content_tokens` 落在 `content_goal` 的 95%–110%（构建期边界容差；运行期对齐保证最终精确）；`body_tokens + task_tokens == content_tokens` |
| 3 | `body.startswith(primary)`；`primary_complete=true`；`evidence_in_primary`（适用时）为真 |
| 4 | `primary_model_tokens + task_model_tokens + 2 ≤ 档位`（`fits_bucket=true`） |
| 5 | 语言合法且与场景声明一致；中文行 ≥40 |
| 6 | 三档 target 独立（不要求嵌套）；同 target 的 `task_sha256` 在重跑间一致 |
| 7 | manifest `rows_sha256` 与 testset 一致；`reference_tokenizer` SHA 与本地一致 |

---

## 5. 抽样

### 5.1 目标数与分配

- 每 `(场景, 档位)` 10 条；8 场景 × 3 档 × 10 = 240 条。
- 多语言每档 10 条 = 10 语种各 1（不随机选语种，固定全覆盖）。

### 5.2 分层与均匀性

- 候选池先按 `stratum` 分层，再按最大余数法分配 10 个名额；`stratum` 定义见下表。
- 层内排序：`sha256(f"{SEED}|{upstream_id}")` 升序（确定性随机、与文件顺序无关）；
- 分层保证小层不被大层淹没，且跨 run/跨机器可复现。

| 场景 | stratum |
|---|---|
| 知识问答 | HotpotQA：`n_gold`（1/2+）；LB-Pro：`secondary_task` |
| 指令遵循 | `vcsum` 长度三分位 |
| 长上下文 | `secondary_task`（T4.1/T4.2） |
| 数学 | AIME `problem_idx` 分桶（30 题按序 3 段） |
| 编程 | 1K：`entry_point` 哈希；4K/32K：`lcc` 语言 / `repobench` 仓库 |
| 通用 Agent | `simple_python` / `multiple` |
| 搜索 Agent | HotpotQA：`n_gold`；LB-Pro：`secondary_task` |
| 多语言 | `language`（10 语种各 1） |

### 5.3 可复现

`SEED=42`；上游 id + 固定 URL；`manifest.rows_sha256` 记录最终文件的 SHA-256。抽样代码在
`tool/dataset/performance/sampling.py`（`deterministic_pick` / `allocate`），构造器与文档共用同一实现。

---

## 6. 数据来源

所有文件放在 `dataset/raw/`（gitignored），由 `tool/dataset/performance/fetch.py` 还原并核对 SHA-256；
来源与许可如下（字节数/SHA-256 前 16 位见 `reports/perf_design_2026/raw_provenance.json`）。

| 源 | 文件（`dataset/raw/` 下） | 上游（HF） | 许可 | 用途 |
|---|---|---|---|---|
| cl100k 词表 | `tokenizer/cl100k_base/cl100k_base.tiktoken` | openaipublic blob | MIT | 构造期量长度 |
| cl100k tokenizer.json | `tokenizer/cl100k_base/tokenizer.json` | 由 `derive_cl100k.py` 生成 | MIT | 构造期计数 |
| LongBench-Pro | `longbench_pro/longbench_pro.json` | `caskcsg/LongBench-Pro` | apache-2.0 | 知识 32K / 长上下文 / 搜索 32K |
| LongBench v1 归档 | `longbench_v1/data.zip` | `zai-org/LongBench` | apache-2.0 | 还原下列成员 |
| vcsum | `longbench_v1/data/vcsum.jsonl` | 同上（成员） | apache-2.0 | 指令遵循 |
| lcc | `longbench_v1/data/lcc.jsonl` | 同上（成员） | apache-2.0 | 编程 4K/32K |
| repobench-p | `longbench_v1/data/repobench-p.jsonl` | 同上（成员） | apache-2.0 | 编程 32K |
| HotpotQA | `hotpotqa/distractor/validation-00000-of-00001.parquet` | `hotpotqa/hotpot_qa` | cc-by-sa-4.0 | 知识 1K/4K、搜索 1K/4K |
| HumanEval | `humaneval/openai_humaneval/test-00000-of-00001.parquet` | `openai/openai_humaneval` | MIT | 编程 1K |
| Multi-IF | `multi_if/multiIF_20241018.csv` | `facebook/Multi-IF` | cc-by-nc-2.0 | 指令遵循约束 |
| AIME 2026 | `aime2026/data/train-00000-of-00001.parquet` | `MathArena/aime_2026` | cc-by-nc-sa-4.0 | 数学任务 |
| MATH-500 | `math_500/test.jsonl` | `HuggingFaceH4/MATH-500` | MIT | 数学示例 |
| BFCL v4 | `bfcl/BFCL_v4_simple_python.json` | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | apache-2.0 | 通用 Agent |
| Global-MMLU | `global_mmlu/{lang}/test.parquet` ×10 | `CohereLabs/Global-MMLU` | apache-2.0 | 多语言 |

还原：`python tool/dataset/performance/fetch.py --mirror https://hf-mirror.com --manifest reports/perf_design_2026/raw_provenance.json`
（zip 成员自动抽取；`--verify` 只读核对）。

---

## 7. 复现与接线

```bash
# 1) 还原上游（已存在时只校验）
python tool/dataset/performance/fetch.py --verify

# 2) 构造（约 2–15 分钟，取决于场景）
python tool/dataset/performance/build.py            # 全量 240 条
python tool/dataset/performance/build.py --scenarios long_context --dry-run   # 单场景调试

# 3) 离线复核（24 cell、token、primary/evidence、manifest）
python tool/dataset/performance/build.py --check

# 4) 校验测试集可被配置加载
./modelbench validate --offline
./modelbench run --method performance --testset performance --dry-run

# 5) 正式跑分（运行期自动对齐）
./modelbench run --method performance --testset performance --models <model-id>
```

产物：`dataset/test/performance/testset.jsonl`（gitignored，约 24MB）、`manifest.json`、
`testset.yaml`（tracked，`strict_input_tokens: true`、`max_tokens: 50`、`buckets: 1k/4k/32k`）。
`tool/dataset/performance/align.py` 保留为**可选预检**（构造一条精确长度的 testset 供人工核对，
不参与正式跑分；正式跑分的对齐在 runner 内）。

---

## 8. 数据集有效性评估

> 这一节是"这套样本能不能用来测性能"的自检，也是给使用者看的注意事项——不是"能测多准"的能力声明。

### 8.1 对性能指标的有效性

- **TTFT / prefill / 峰值内存** 只由最终 prompt token 数（及模型/后端）决定；内容来源不改变
  注意力计算量。因此 240 条样本在"长度精确 == 档位"的前提下，测出的 TTFT/内存差异可归因于
  模型/后端，而不是"这条样本恰好更长"。
- **TG TPS** 依赖解码窗口：`max_tokens=50` 固定输出上限；长回答指令（decode guard）保证模型
  不会在 10 个 token 内草草结束；若模型拒答/提前 EOS（<16 tokens），该样本的 `gen_tps` 为 null
  （原因记 `completion_too_short`），聚合给 `n_gen_tps_invalid`，不静默剔除。
- **样本内容对结果的影响**：内容只影响"模型是否愿意作答"，不影响 TTFT/内存的物理量；因此本集
  不用来比较回答质量，也不建议用同一批 run 做"能力"结论。

### 8.2 构造期已消除的失效模式

| 失效模式 | 处理 |
|---|---|
| 1K 档正文截断导致答案不在输入里 → 模型拒答/输出 <16 token | B′：gold/整篇 primary 必须完整；`evidence_in_primary` 校验；1K 档 primary 预算按最坏 tokenizer 预留 |
| 不同模型 tokenizer 不同 → 同一文件长度差 2 倍 | 构造 2× 超量 + 运行期按目标 tokenizer 裁剪；主料按"最坏 tokenizer"预算裁剪 |
| 三档嵌套导致低档"只剩文档开头" | 取消嵌套：每档独立 target，按档位选择可放下的 primary |
| filler 与 primary 重复 / filler 过长 | 去重 + 单行字符预算；裁剪只砍 filler |
| 多语言缺中文 | 10 语种固定含 zh；中文行 43/240 |
| 工具调用导致 TG TPS 无效 | 工具目录以文本形式进 prompt，不使用 `tools` 字段 |

### 8.3 已知局限（解读时必须带上）

1. **3 个场景共享 LongBench-Pro**：任务族互斥、行集不相交，但语料风格相近；**不得跨场景比较**。
2. **数学/多语言是 many-shot 构造**：读的是"示例+题"，不是单篇长文档；适合测长 prompt 的
   prefill/decode，但不模拟"长文档推理"。
3. **指令遵循没有 assistant 历史**：Multi-IF 的 responses 为空，本集只取约束集合，不是多轮对话协议。
4. **通用 Agent 不测函数调用协议**：工具目录是文本，模型不需要输出 JSON 调用；这是有意的
   （`tool_calls` 会让 TG TPS 抑制）。
5. **长度依赖目标模型 tokenizer**：若目标模型 tokenizer 文件不可得，运行期只能用字符二分（较慢、
   极端文本可能无法精确命中 → 该样本 skip，原因 `runtime_alignment`）。
6. **32K 档需要服务端窗口 ≥32,946**（32,768 + 50 + 128 safety）；窗口不足的模型整档 skip（窗口
   判定优先于对齐）。
7. **24MB 数据集 + 构造脚本**：干净环境需先 `fetch.py` 还原 2.6GB 上游；不联网无法重建。
8. **性能模式不判分**：本集不提供"正确答案"用于评分（构造期的 answer 只用于证据校验）；
   不要用性能 run 的响应做质量结论。
9. **对齐会改写 body 尾部**：`prompt_alignment.trim_chars/pad_tokens` 非 0 的样本，其发送文本与
   文件中的文本存在尾部差异（只影响 filler）；若要做输入内容研究，需同时记录该字段。

### 8.4 使用前自检清单

- [ ] `build.py --check` 通过（240 行、24 cell、primary/evidence 标志）
- [ ] `tool/dataset/performance/review.py` 输出与预期一致（primary 占比、任务块、证据计数；参考 `reports/perf_dataset_review/dataset_review.md`）
- [ ] `./modelbench validate --offline` 显示 240 samples / 8 selections
- [ ] `--dry-run` 显示 24 cells / 240 requests
- [ ] 目标模型 tokenizer 在 `~/.omlx/models/**` 可解析（或接受字符级对齐）
- [ ] 服务端 `context_window ≥ 32,946`（否则 32K 档会整档 skip）
- [ ] 首轮 run 的 `records.jsonl` 中 `usage.prompt_tokens == context_length` 100% 成立、`n_skipped=0`

---

## 9. 附录

### A. 行 schema（关键字段）

```json
{"id":"perf-knowledge_qa-1k-000","scenario":"knowledge_qa","bucket":"1k",
 "target_id":"tgt-knowledge_qa-1k-000","dataset":"hotpotqa",
 "tokens":1024,"messages_tokens":1024,"tools_tokens":0,
 "messages":[{"role":"user","content":"<primary + filler>"},{"role":"user","content":"<task>"}],
 "tools":[],"constructed":true,
 "meta":{"lang":"en","source":"hotpotqa","upstream_id":"...","target_tokens":1024,
   "content_tokens":2048,"body_tokens":1980,"task_tokens":68,"primary_units":2,
   "primary_tokens":512,"filler_units":9,"primary_share":0.25,
   "primary_model_tokens":560,"task_model_tokens":70,"fits_bucket":true,
   "evidence_in_primary":true,"primary_complete":true,"cut_policy":"document",
   "alignment":"primary+filler,oversize","runtime_alignment":"trim+pad",
   "oversize":2,"task_sha256":"...","body_sha256":"..."}}
```

### B. manifest 字段

`name/buckets/max_tokens/reference_tokenizer{path,sha256}/construction{policy,oversize,
content_tokens,template_margin_tokens,runtime_alignment}/validation{checks,tokenizer_probe,
evidence_rows}/sources/cells{rows,dataset}/stats{langs,task_tokens,content_chars}/rows_sha256`。

### C. testset.yaml

`name/title/description/scenarios{perf_jsonl + dataset_params{scenario,path}}/sampling{seed,
targets_per_scenario,method}/buckets{1k:1024,4k:4096,32k:32768}/max_tokens:50/
strict_input_tokens:true`（无 `osl` 字段）。

### D. 术语

`primary`：承载场景语义、必须完整进入 prompt 的文本；`filler`：同族真实文本，用于补长度、可裁；
`档位/bucket`：1024/4096/32768 三档目标长度；`设计指纹`：target 集合+抽样+构造+任务块的确定性
描述，跨模型不变；`运行期对齐`：runner 用目标模型 tokenizer 把 body 裁/补到精确档位的步骤。
