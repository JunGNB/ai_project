# 本地小模型性能测试：样本集设计与抽样方案

> 范围：**只覆盖性能测试的样本选择与构造**（TTFT / TG TPS / TPOT / 峰值内存），不涉及判分与能力结论。
> 被测对象：本地小模型（1B–8B 级，单请求串行）。产物：**8 场景 × 1K/8K/32K × 10 = 240 条**样本、
> 24 个 cell、8 个场景各 1 个主数据源；上游证据在 `dataset/raw/`，产物在 `dataset/test/performance/`，
> 构造器在 `tool/dataset/performance/`。
>
> 核心口径：**长度严格对齐**——每条样本的最终 prompt token 数必须精确等于 1024 / 8192 / 32768，
> 容差 0，不做误差带统计；运行前按被测模型自身 tokenizer 重建并逐条校验，不相等即跳过并记录。

---

## 0. 摘要

| # | 场景 | 数据源（本地证据） | 语言 | 1K/8K/32K 构造 | 每档 target |
|---|---|---|---|---|---:|
| 1 | 知识问答 | LongBench-Pro `T3 Evidence-Grounded QA`（中文 60 条，过滤后 13 条） | zh | 原生 32K–128K 文档前缀 + 原题任务块 | 10 |
| 2 | 指令遵循 | LongBench v1 `vcsum` 中文长文 + Multi-IF 中文真实约束 | zh | 同源会议记录按序拼接取前缀 + 约束集渲染 | 10 |
| 3 | 长上下文 | LongBench-Pro `T4 Summarization & Synthesis`（英文 60 条，过滤后 30 条） | en | 原生 32K–128K 文档前缀 + 汇总任务块 | 10 |
| 4 | 数学与逻辑 | AIME 2026 题面（30 题）+ MATH-500 完整解示范 | en | 同集满分解答 many-shot + 目标题块 | 10 |
| 5 | 编程 | LongBench v2 `Code Repository Understanding`（50 条） | en | 原生仓库代码文档前缀 + 原题任务块 | 10 |
| 6 | 通用 Agent | BFCL v4 `simple_python` + `multiple`（600 条） | en | 工具 schema 目录按序拼接 + 用户请求块 | 10 |
| 7 | 搜索 Agent | LongBench-Pro `T1 Retrieval & Ranking` + `T5 Attribution & Citation`（英文 58 条） | en | 原生多文档前缀 + 检索/归因任务块 | 10 |
| 8 | 多语言 | Global-MMLU（10 语种 × 14,042 题） | zh/en/fr/de/es/pt/ru/ja/ko/ar | 同语种题库 many-shot + 目标题块 | 10（每语种 1） |

**七个关键决策**

1. **按"prefill 形状"选源，而不是按榜单热度**：性能测试关心的是真实长上下文下 prefill/decoding 的
   时间与内存，所以优先选**原生就提供长文本**的语料（LongBench-Pro 8K–256K、LongBench v2 仓库文档）
   而不是把短题堆到 32K；确实需要堆叠时（数学/多语言/Agent 工具目录），堆叠单元必须是**真实样本**且
   同一单元集按确定性顺序取用。
2. **8 场景 8 个数据源载体、互不重叠为主**：LongBench-Pro 只出现在 3 个长上下文类场景（T3/T4/T1+T5
   三类任务族，行集不相交）；其余 5 个场景各用自己的载体。
3. **长度严格对齐、0 容差**：声明值即最终 prompt token 目标；构建用通用参考 tokenizer 精确命中，
   生产运行前用**目标模型 tokenizer** 重建同一设计并逐条 `/v1/tokenize` 校验；不相等即 skip，不做
   ±10% 入带、不做运行期裁剪。
4. **同 target 三档严格嵌套**：1K 档的上下文正文是 8K 档的前缀、8K 是 32K 的前缀；三档只有长度变化，
   任务块逐字节相同（`task_sha256` 校验）。
5. **中文不可缺**：中文场景 2 个（知识问答、指令遵循）+ 多语言含中文，共 63/240 行（26%）。
6. **输出长度固定 50 token**：`max_tokens=50` 是默认值，由 testset 声明并被 `apply_testset` 应用
   （CLI/Web 可覆盖）。TG TPS 就是这 50-token 固定解码窗口上的值；任务块要求详细回答，使窗口在
   多数样本上被填满（`finish_reason=length`），窗口内 token 数不相同的行为不会发生。
7. **确定性**：`seed=42` + `sha256(seed|id)` 排序 + 分层最大余数配额；同一份 raw 重跑逐字节一致，
   SHA-256 写入 manifest，抽样与构造的每一步都可复算。

---

## 1. 目标、约束与非目标

### 1.1 硬约束与落实

| # | 约束 | 本方案落实 |
|---|---|---|
| C1 | 覆盖 8 个场景 | 8 场景 × 10 target，无缺席（§3） |
| C2 | 每场景覆盖 1K/8K/32K | 每场景 3 档 × 10 target = 30 行，共 240 行、24 cell |
| C3 | **长度严格对齐**（硬性） | `tokens` 即 prompt token 目标；构建期精确命中（§4.3），运行前按目标 tokenizer 重建并逐条校验（§4.4），不相等 skip（§4.5） |
| C4 | 输出长度固定 50 token（默认，可配置） | testset `max_tokens: 50`；任务块要求详细回答以填满窗口；触顶样本照常聚合（§1.2） |
| C5 | 不能全英文 | 中文：知识问答、指令遵循（全中文）+ 多语言（zh 1/10 语种），共 63 行 |
| C6 | 同 target 三档可比 | 三档只有上下文长度变化，任务块逐字节一致、正文严格前缀（§4.3） |
| C7 | 可复现、可审计 | 固定 seed、确定性抽样、逐文件 SHA-256、manifest 记录抽样明细与 `rows_sha256` |
| C8 | 本地小模型、单请求串行 | 单 `chat/completions` 请求；不测并发/批处理/多轮工具循环 |

### 1.2 测量口径（与 [20-methodology.md](20-methodology.md) 一致，此处只列本样本集相关的部分）

| 指标 | 定义 | 有效条件 |
|---|---|---|
| TTFT | 请求发出到**首 token** 的时间；tokenize 与清缓存不计入窗口 | 冷缓存、`cached_tokens==0` |
| TG TPS / TPOT | 解码窗口内 `(C-1)/(t_last-t_first)` 与其倒数 | `C≥16`、解码窗口 ≥50ms、`finish_reason≠tool_calls`、值 ≤10,000，否则置 null 并记 `gen_tps_invalid` |
| 峰值内存 | 请求窗口内 oMLX 服务进程树 RSS 峰值 | 与环境快照同一次 run |
| 生成长度 | `max_tokens=50`（默认值；TG TPS 按该固定解码窗口计算） | 由 testset `max_tokens` 声明，`apply_testset` 写入；CLI `--max-tokens` / Web 可覆盖 |
| 重复次数 | 每样本 1 次 | `repeats=1`；warmup/污染/错误样本不进聚合 |

因为指标依赖"同一输入"才能跨模型比较，C3 的严格对齐是本方案的前提：任意两条 run 的同一 `cell`
若 prompt token 不同，TTFT 不可比。

### 1.3 非目标

- 不判分、不报告准确率；性能模式 `score` 恒为 null，不与任何公开榜横比；
- 不测并发吞吐、多轮工具调用、Web 浏览、代码仓库级修改、多模态；
- 精度测试是独立方案（[12-accuracy-testset.md](12-accuracy-testset.md)），与本集互不推导。

---

## 2. 业界数据源调研（性能视角）

### 2.1 筛选标准

性能样本的筛选与"能力评测选型"不同，本方案只用四条：

1. **能提供真实长上下文**：原生 ≥32K token 的文档/代码/多文档最好；否则堆叠单元必须是真实样本，
   且长度主要来自"内容量"而不是重复模板；
2. **负载形状真实**：prefill 的 token 分布（自然语言 / 代码 / 工具 JSON / 多语种文字）与真实推理
   请求接近，避免"同一句话重复 N 次"这类 tokenizer 与 attention 都过于规整的病态输入；
3. **离线可复现**：只需下载数据文件，不需要 clone 仓库、起容器、联网检索或执行代码；
4. **许可清晰、可引用**：有明确的 license 与稳定的 HF/GitHub 地址，能写进来源表。

### 2.2 调研方法与证据

- 检索日 **2026-11-14**；环境限制：`huggingface.co` 直连不可用、`web_search` 工具不可用，
  因此统一走 **hf-mirror.com**：数据集列表页（`?sort=trending|downloads` + 关键词）、
  `/api/datasets/<id>` 元数据（downloads / likes / `lastModified` / `createdAt` / license / language）、
  `/api/datasets/<id>/tree/main` 文件树、`/resolve/main/<path>` 下载。
- 每条候选都记录：**listing URL + HTTP 状态 + 抓取时间戳 + 下载量/点赞数 + lastModified**，
  快照落在 `reports/perf_research_2026/`（`listing_snapshots.md`、`candidates.md`、`evidence.json`）。
- 引用时只使用**实测到**的数字；镜像 404 的条目一律不写进"可选"（见 §2.5）。

### 2.3 候选与热度证据（仅列与选型相关的行）

| 候选 | 场景 | 语言 | 原生长度 | 许可 | 2026-11-14 热度（实测） |
|---|---|---|---|---|---|
| caskcsg/LongBench-Pro | 长文/检索类 | zh+en | 8K/16K/32K/64K/128K/256K，各 250 条 | apache-2.0 | 2,703 dl / lastModified 2026-01-07 |
| zai-org/LongBench-v2 | 仓库代码/多文档 | en | 文档级，最长 >1M | apache-2.0 | 63,559 dl / 65 likes / 2024-12-20 |
| zai-org/LongBench (v1) | 中文会议纪要 | zh+en | 单文档 0.6K–27K | apache-2.0 | 47,490 dl / 204 likes / 2024-12-18 |
| facebook/Multi-IF | 指令遵循 | 8 语种（含 zh） | 单轮短指令 | cc-by-nc-2.0 | 2,555 dl / 43 likes / 2024-10-30 |
| MathArena/aime_2026 | 数学 | en | 单题（无解答） | cc-by-nc-sa-4.0 | 32,002 dl / 98 likes / 2026-10-19 |
| HuggingFaceH4/MATH-500 | 数学示范 | en | 题+完整解 | mit | 215,327 dl / 357 likes / 2025-12-15 |
| gorilla-llm/Berkeley-Function-Calling-Leaderboard | Agent | en | 单轮/多轮工具 | apache-2.0 | 78,941 dl / 142 likes / 2026-04-29 |
| CohereLabs/Global-MMLU | 多语言 | 42 语种 | 单题 | apache-2.0 | 35,663 dl / 168 likes / 2025-08-14 |
| TIGER-Lab/MMLU-Pro | 知识（备选） | en | 单题（<1K） | mit | 413,614 dl / 601 likes / 2026-05-02 |
| princeton-nlp/HELMET | 长文（备选） | en | 8K–128K | mit | 2,832 dl / 2026-07-14，但数据包 8.9–11.3 GB |
| Tevatron/browsecomp-plus | 搜索（备选） | en | 检索语料 + 问题 | —— | 32,368 dl / 2025-12-20 |

### 2.4 三条筛选结论

1. **原生 32K 的文本评测集很稀少**：热门榜单（MMLU-Pro、GSM8K、C-Eval、CMMLU、HumanEval 等）
   原生都是"单题级"输入，直接放到 32K 只能用重复题目堆叠，形状不真实；能原生长上下文的集中在
   LongBench 系（v1/v2/Pro）、HELMET、∞Bench 一类。
2. **长上下文集的中文覆盖不均**：LongBench v2 全英文；LongBench v1 有中文任务但数据 2023 年；
   LongBench-Pro 是少数**中英各半、原生到 256K、且持续维护**的集，因此承担中文长文与英文长文两类角色。
3. **2026 的新集多集中在 Agent/执行类**（terminal-bench、SWE-bench Pro、gdpval、browsecomp 变体）：
   它们需要容器执行、仓库 clone 或联网检索，**离线单请求性能测试无法复现**，只能作为"不选"的证据，
   不能作为本方案主源。

### 2.5 明确排除（不可用/不适合）

| 候选 | 原因 |
|---|---|
| princeton-nlp/HELMET | 数据包 `data_v2.tar.gz` 8.86 GB / `data.tar.gz` 11.27 GB，超过仓库"下载 >1 GB 需人工确认"的约束，且单次下载成本远高于其余全部源之和 |
| GPQA / HLE / apex-agents / GAIA | HF 为 gated，需登录申请，无法在离线/CI 复现 |
| sierra-research/tau²-bench、allenai/IFBench、cais/mmlu_pro、openai/SimpleQA、BABILong、openai/browsecomp、THUDM/AgentBench、SciCode、multi-swe-bench | 实测 HF 404（镜像无此仓库），不能作为可核查依据 |
| SWE-bench Verified / terminal-bench | 需 clone 仓库、起容器、执行测试，且 prompt 不是"长上下文 prefill"形状 |
| Alibaba-NLP/Open-DeepResearch | tool 消息内容为空，不能作为检索结果上下文 |
| muset-ai/DeepResearch-Bench | 只有生成的报告与人工标注，无原始 query |

---

## 3. 场景与数据源选型

### 3.1 矩阵与论证

**场景 1｜知识问答（zh）→ LongBench-Pro T3 Evidence-Grounded QA**

- 场景要测什么：中文长文档（报告/学术/法规）作为背景，末尾一个需要跨段落整合的问题。这是本地模型
  最典型的"塞一篇长中文材料再提问"的负载。
- 主选依据：LongBench-Pro 的中文 T3 行是**原生长文档 + 证据定位问题**（T3.1 多文档整合、T3.2 单跳事实），
  原生 8K–256K；中文占比一半；评测量级与任务形状都匹配性能测试。
- 备选与落选：C-Eval/CMMLU（2025-07/2023-07，单题 <100 token，无法原生 32K）；MMLU-Pro（413K 下载量
  最高，但同样单题短，堆叠 32K 时是"题库重复"形状）；LongBench v1 中文任务（数据 2023 年、上下文上限
  27K，凑 32K 需拼接整篇文档，且缺少 2024–2026 的文本分布）。
- 池与配额：中文 T3 共 60 条，其中 native ∈ {32k,64k,128k} 且任务块 ≤400 token 的有 **13 条**
  （strata：T3.1 7 条、T3.2 6 条）→ 按分层最大余数取 10（5+5）。

**场景 2｜指令遵循（zh）→ vcsum 中文会议纪要 + Multi-IF 中文真实约束**

- 场景要测什么：长中文材料 + 多条显式格式/内容约束（"至少 3 个要点""结尾必须…""禁止出现…"）。
  这是"指令遵循"在长上下文下的真实形态：先读材料，再按约束写作。
- 主选依据：Multi-IF 提供**真实人工标注的约束集**（8 语种，中文 454 条，每条约 1–3 个约束，来自 18 个
  约束族）；vcsum 提供中文长文（会议纪要 0.6K–23K token）。两者组合成 document-grounded IF：
  约束来自真实标注、材料来自真实长文，二者都不需要在评测时临时编造。
- 备选与落选：IFEval（188K 下载，最热，但英文、prompt 短，堆到 32K 只能是同一指令模板重复，形状病态）；
  IFBench（HF 404，仅 GitHub 发布）；ComplexBench/CFBench（HF 404）。Multi-IF 的 `responses` 列在
  公开 CSV 中为空，所以只取约束（instruction_id + kwargs）渲染成中文要求，历史对话不含 assistant 轮，
  这一点在 §9 如实记录。
- 池与配额：vcsum 200 篇按长度分层取 10 篇；中文约束按 key 去重后确定性取 10 条，二者按固定顺序配对。

**场景 3｜长上下文（en）→ LongBench-Pro T4 Summarization & Synthesis**

- 场景要测什么：英文长文（多份材料的全局汇总或查询聚焦摘要）。长上下文场景与知识问答的区别是任务
  本身要求"覆盖全文/全局信息"，输入是长文档而不是"文档 + 定位问题"。
- 主选依据：T4 是原生长文摘要任务（T4.1 全局覆盖约束摘要、T4.2 查询聚焦摘要），英文 60 条、原生 8K–256K。
- 备选与落选：HELMET（8.9 GB 数据包，见 §2.5）；∞Bench（镜像 404）；LongBench v1 英文任务（2023 数据，
  且多数任务的原生长度落在 8K–20K 区间，需拼接才能到 32K）。
- 池与配额：英文 T4 共 60 条，native ∈ {32k,64k,128k} 且任务块 ≤400 token 的 **30 条**
  （T4.1 15、T4.2 15）→ 分层取 10（5+5）。

**场景 4｜数学与逻辑（en）→ AIME 2026 题面 + MATH-500 满分解 many-shot**

- 场景要测什么：英文数学推理。本地小模型的数学请求通常是"题目 + 若干已解示例（many-shot）"，
  输出是逐步推理，因此 prefill 由示例撑起来，decode 由推理撑起来。
- 主选依据：AIME 2026（30 题，2026-10 更新）提供**最新题面**且是干净的短题；MATH-500（500 题，
  含完整解，2025-12 更新）提供**高质量长解**作为 many-shot 示例。两者组合后，1K/8K/32K 三档的示例
  数量随长度自然增长，且示例内容真实、格式统一（`Problem:/Solution:`）。
- 备选与落选：GSM8K（1.1M 下载量最高，但题目与解都短、模板化严重，32K 需要上百条示例，重复感强）；
  AIME 2025/2024（题量 15–30，无解答，只能当目标题）；OlympiadBench（中英混合、题目长，示例格式不统一）。
- 池与配额：AIME 30 题确定性取 10；MATH-500 500 条按 `sha256(seed|unique_id)` 排序作为示例流。

**场景 5｜编程（en）→ LongBench v2 Code Repository Understanding**

- 场景要测什么：长代码上下文（仓库级文件/多文件片段）作为背景 + 一个与代码相关的问题。这是本地模型
  在 IDE/代码问答里最典型的 prefill 形状：**输入是大量代码，输出是短结论**。
- 主选依据：LongBench v2 的 Code Repository 子集（50 条）原生就是"仓库代码文档 + 问题/选项"，
  实测上下文中位数约 597K token（最长 4.6M），前缀裁剪到 1K/8K/32K 都是**真实代码前缀**。
- 备选与落选：SWE-bench Verified / SWE-bench Lite（2026-08 仍热度很高，但需要 clone 仓库、执行测试，
  且 prompt 是 issue 描述而不是代码上下文本身，属能力/流程评测）；LiveCodeBench、BigCodeBench、HumanEval、
  MBPP（单函数级 prompt，几十到几百 token，堆叠到 32K 会变成"同一文件重复"）。
- 池与配额：50 条代码仓库样本，确定性取 10。

**场景 6｜通用 Agent（en）→ BFCL v4 工具目录 + 用户请求**

- 场景要测什么：大工具库（MCP/函数目录）+ 用户请求的 prefill。真实 Agent 请求的输入里工具 schema
  往往占大头（几十个工具时可达数万 token），这是"工具目录 prefill"负载。
- 主选依据：BFCL v4（2026-04-29 发布，78,941 下载）的工具定义是真实函数 schema，`simple_python`
  400 条（1 工具）、`multiple` 200 条（2 工具），可组合成"目标工具置顶 + 去重工具目录"的真实形态。
- 与真实协议的差异（重要）：**工具以纯文本目录呈现，不走 API `tools` 字段**。原因有两点：
  其一，性能测试要的是 prefill 负载，`tools` 字段在部分后端会被模板/工具解析额外处理，跨后端不可比；
  其二，若模型实际返回 `tool_calls`，`finish_reason=tool_calls` 会让 TG TPS 失效（见 §1.2），
  而本集要测的是生成速度。工具目录文本用 `Tool: <name>` + JSON schema 渲染，逐字节可复现。
- 备选与落选：τ²-bench（HF 404）；ToolBench（旧，且需执行环境）；MCP-Universe / terminal-bench
  （需容器与真实工具执行，离线不可复现）；xlam-function-calling-60k（16,638 dl，单轮函数调用数据但
  无 schema 目录概念）。
- 池与配额：strata = {simple_python 400, multiple 200}，按最大余数取 10（3 simple + 7 multiple）。

**场景 7｜搜索 Agent（en）→ LongBench-Pro T1 Retrieval & Ranking + T5 Attribution & Citation**

- 场景要测什么：搜索/检索类 Agent 的上下文是"多份检索结果拼接 + 指令"，要求从大量材料里定位、
  排序或引用。本地模型在 RAG 场景下就是这种 prefill 形状。
- 主选依据：T1（全局连贯检索、关键片段检索）与 T5（整句/关键陈述归因）都是**原生长多文档**任务，
  英文各 60 条；把这两类放在一起，覆盖"排序"和"引用"两种检索输出形态。
- 备选与落选：browsecomp-plus（32,368 dl，但需要配套检索语料 + 真实浏览工具循环）、deepsearchqa
  （26,287 dl，同类）、HotpotQA（86,515 dl，但单跳短文档，32K 需堆叠）、GAIA（gated）。
- 池与配额：英文 T1+T5 共 120 条，native ∈ {32k,64k,128k} 且任务块 ≤400 token 的 **58 条**
  （T1.1 15、T1.2 15、T5.1 13、T5.2 15）→ 分层取 10（3/3/2/2）。

**场景 8｜多语言（10 语种）→ Global-MMLU**

- 场景要测什么：不同文字系统与 tokenizer 压缩率下的 prefill。多语言场景最关心的不是"答对"，而是
  **同一段语义在 zh/en/fr/de/es/pt/ru/ja/ko/ar 十种语言下的 token 数、TTFT 与内存差异**——这正是
  本地模型换 tokenizer/换语言后性能变化的来源。
- 主选依据：Global-MMLU 42 语种、每语种 14,042 题、题目长度适中、许可 apache-2.0；取 10 个语种覆盖
  拉丁（en/fr/de/es/pt）、西里尔（ru）、CJK（zh/ja/ko）、阿拉伯（ar）。
- 备选与落选：MMMLU（14 语种，题目更短、题量小）；Belebele（96 语种但单文档阅读题，长度固定）；
  mgsm（2026-05 更新，但只有数学一种任务，模板化）；EU-MMLU（2026-09，欧洲语种、无中文）。
- 池与配额：每语种 1 个 target（确定性抽 1 题），同一语种用题库 many-shot（示例含 `Answer:`）拼到
  目标长度，末题 = 目标题 + 该语种的回答指令。

### 3.2 复用与边界

- LongBench-Pro 被 3 个场景使用，但**任务族互不重叠**（T3 / T4 / T1+T5），行集两两不相交，`upstream_id`
  唯一，manifest 可核对；目的是用同一个高质量长文载体覆盖"问答 / 摘要 / 检索"三种 prefill 形状。
- 付出的代价：三个长文场景来自同一语料发布方，若该语料的文本分布有偏，这三个场景会共享同一偏差。
  因此**场景之间不做横向对比**（例如不比较"知识问答 vs 搜索 Agent 谁 TTFT 高"），只在同一场景内比较
  模型/设备/档位。该边界写入 §9。
- 每个场景只有 1 个主源，但 §3.1 记录备选与落选原因；建立"主源失效时的替换路径"（例如 LongBench-Pro
  不可得时，知识问答回退到 LongBench v1 中文任务，长上下文回退到 LongBench v1 英文摘要任务），替换时
  需保留同样的"原生 ≥32K 或真实单元堆叠 + 严格对齐"约束。

---

## 4. 长度口径与严格对齐

### 4.1 三档定义

| bucket | 目标 prompt token | 含义 |
|---|---:|---|
| 1k | 1,024 | 短请求，测 TTFT 基线 |
| 8k | 8,192 | 中长上下文，测 prefill 随长度增长 |
| 32k | 32,768 | 长上下文，测内存与 prefill 拐点 |

- `tokens` 字段 = **最终渲染后的 chat prompt token 数**，包含 chat template 与实际发送的 messages；
  不是"内容 token"、不是"字符数"。
- 记录上同时给出 `content_tokens`（内容 token）与 `template_reserve_tokens`（模板预留），
  三者关系：`tokens = content_tokens + template_reserve_tokens`（本集模板预留为 0，见 §4.4）。
- 容差 **0**：`tokens` 必须精确等于上表值，不存在"接近即可"。

### 4.2 为什么必须"参考 tokenizer 构建 + 运行前重建"

- 不同模型 tokenizer 对同一段中文的 token 数可差 2 倍以上（实测：中文短指令 MiniCPM/cl100k ≈ 0.40，
  中文长文 ≈ 0.47；英文数学文本 ≈ 0.93–1.03）。因此**不能用字符数或固定 token 预算近似**。
- 若直接用某个被测模型的 tokenizer 构建，则换模型即失效；若运行期再裁剪，会破坏"同一 target 三档
  嵌套"和"同一 cell 跨模型可比"，且裁剪发生在测量窗口内、无法离线复核。
- 业界通行做法（evalscope 等性能工具）是"按模型 tokenizer 精确构造/截断 prompt 长度"。本方案把它
  拆成两步：**构建期用通用参考 tokenizer 定稿设计**（`cl100k_base`，永不作为被测模型），
  **运行前用目标模型 tokenizer 重建同一设计并逐条校验**（§4.4）。这样仓库里始终有一份可审查的基线，
  而测量时又严格对齐目标模型。

### 4.3 构造算法（每个 target 一次定稿，三档复用）

1. **有序单元流**：target 定义 `units = [u1, u2, …]`（原生文档 / 满分解示例 / 工具 schema / 同语种题库），
   单元之间用 `\n\n` 连接；单元流总长必须 ≥36,000 token（保证 32K 档有足够上下文）。
2. **固定任务块**：target 定义 `task`（末尾问题 + 长回答要求），任务块 ≤400 token（保证 1K 档仍有
   ≥600 token 的上下文），三档逐字节相同。
3. **精确截断**：对每个 bucket，取单元流的前缀并微调到精确 `bucket - task_tokens` 个 token：
   - 先在 token 空间取 `stream_ids[:n]` 解码，再重新计数，按差值迭代收敛（≤40 轮）；
   - 若 tokenizer 边界效应导致计数跳过目标值（例如 32767 → 32769），依次更换 join 分隔符
     （`\n\n` / `\n` / 空格 / 空串 / 多重换行等，不同分隔符在 join 处消耗的 token 数不同）；
   - 仍不可达时，在正文前缀与任务块之间插入**可调 token 数的空白 pad**，并对正文前缀做 ±2 token 的
     微调；pad 属于 join 的一部分，不破坏正文前缀嵌套；
   - 解码前缀可能产生尾随半个 UTF-8 字符（`U+FFFD`），必须剥掉后重新计数，否则计数与多语种前缀校验
     都会出错。
4. **嵌套校验**：同一 target 的 1K 正文必须是 8K 正文的前缀、8K 必须是 32K 的前缀（按字符严格前缀，
   构建器记录 `body_sha256` 并逐条校验）。
5. **记账**：每行写 `tokens / messages_tokens / tools_tokens / meta.content_tokens / meta.task_sha256 /
   meta.body_sha256`；`validate_rows` 在构建阶段强制上述所有约束，任何一条不满足即构建失败。

### 4.4 运行前对齐（`tool/dataset/performance/align.py`）

```
# 用目标模型 tokenizer 重建同一批 240 行并逐条校验
python tool/dataset/performance/align.py \
    --tokenizer /path/to/model/tokenizer.json \
    --base-url http://127.0.0.1:8000 --model <model-id>
```

若 oMLX 开启了鉴权，`align.py` 依次取 `--api-key`、环境变量 `OMLX_API_KEY`、`modelbench` 配置中
`server` 的 key（与 runner 同一来源），不需要手动导出。

流程：

1. **探针测模板预留**：取一段固定内容，渲染成与样本相同的单条 user 消息，调用服务器
   `/v1/tokenize` 得到 prompt token 数，`reserve = prompt_tokens - content_tokens`（本集无 system
   prompt、无 tools，所以 reserve 主要由 chat template 决定，约 10–40 token）。
2. **重建**：以 `content = bucket - reserve` 为目标，按 §4.3 的同一单元流、同一顺序、同一任务块重建
   240 行；重建只改变"前缀长度"，不重新抽样、不换样本。
3. **逐条校验**：对每一行调用 `/v1/tokenize`（与 runner 发送时同一口径，含 chat template），要求
   `prompt_tokens == bucket`；有偏差时先用该行自身的 delta 单独修正该行（每行 ≤3 次），仍有偏差再按
   中位 delta 调整 reserve 整体重试（≤3 轮）；每行修正记入对齐报告。
4. **落盘**：写 `dataset/test/performance/testset.jsonl`（与基线同路径，被 gitignore），并把对齐报告写
   `dataset/.cache/performance/alignment-<slug>.json`（tokenizer SHA-256、探针结果、每轮 reserve、
   各档行数、`rows_sha256`、是否通过服务器校验）。
5. **失败即停**：任一行无法精确对齐则脚本非零退出，不生成 testset；不允许"带着偏差进入测量"。

离线模式（`--tokenizer … --reserve N`）只做本地预检（`content + reserve == bucket`），用于没有服务器
时的自查；正式测量必须用服务器模式完成逐条校验。

在本地 1B 模型 + 32K 窗口服务器上实测：探针 `reserve = 12`，第一轮即 240/240 行全部命中
（0 条逐行修正），产出 `rows_sha256 = cbff70f2…`；同一份 `testset.jsonl` 交给 `build.py --check`
时，因为 SHA 与 manifest 的参考基线不同，检查自动回落到对齐报告的 tokenizer + reserve 复核并通过。

### 4.5 运行期严格校验（不再有 ±10% 误差带）

- testset 在 yaml 声明 `strict_input_tokens: true` → `PerfConfig.strict_input_tokens`，
  由 `apply_testset` 应用；accuracy 集声明该字段直接报错（生成长度口径不同）。
- runner 对每条 prebuilt 样本在发送前调用 `client.tokenize_messages(messages, ...)`：
  - `tokenized == bucket` → 正常测量，`context_meta.isl_tolerance = {target, low=high=target, actual, ok=true}`；
  - 不相等或 tokenize 失败 → **该样本不测量**，写入 `run.json.outcome.skipped`（带 expected/got 原因），
    不中止整个 run，也不静默剔除；
  - 聚合只使用 measured 样本，并在结果里给出 `n_measured / n_skipped`；不再计算 `n_isl_in_band /
    n_isl_out_of_band` 之类的带内比例（严格模式下二者无意义）。
- **decode guard 与严格对齐的冲突处理**：guard 会追加一条 user 指令（约 35 token），使 prompt 超过
  bucket。因此 strict 集**不追加 decode guard**；`max_tokens=50` 固定解码窗口，任务块要求详细回答
  （≥80–150 词/≥100–120 字）使窗口被填满，`C<16` 或解码窗口 <50ms 的样本照常记 `gen_tps_invalid`
  但不影响 TTFT/内存。非 strict 集（accuracy 集等）行为不变。
  该行为在 run 级记 warning、逐样本记 `decode_guard_applied=false`，不静默。
- **生成长度口径**：`max_tokens=50` 是产品默认值，可由 testset 声明覆盖、也可用 CLI/Web 临时覆盖；
  TG TPS 就是这段固定解码窗口上的速度，窗口一致才能跨模型/跨 run 比较。窗口太短（很快的模型可能
  <50ms）会触发 `gen_tps_invalid`，此时用 `--max-tokens` 提高后重测即可，样本本身不需要重建。
- **窗口判定**：`bucket + max_tokens(50) + context_safety_margin_tokens(128) <= model_context_window`，
  即 32K 档要求模型窗口 ≥**32,946**；不满足则该样本不可测，记 skip（不做静默截断）。

### 4.6 参考实现对照：evalscope perf 1.15.0

对照目的：确认"通用 tokenizer 构建 + 运行前重建"是业界可行路线，并找出必须补强的地方。结论来自只读
阅读该版本 `perf/` 源码（约 1.6 万行），未执行其命令。

- **长度目标**：`--min/--max-prompt-length` 设了 `--tokenizer-path` 时单位为 token、否则为字符；真实文本
  插件另用 `--dataset-args {"target_input_len": N}` 逐样本给目标。random 插件把 min/max 当采样区间。
- **tokenizer 只在本机**：`AutoTokenizer` 指到本地目录或 Hub id，整个 perf 包不调用服务端 tokenize；
  `--tokenize-prompt` 是反向操作（客户端分词后把 token id 直接发 `/v1/completions`）。
- **不精确时**：`converge_to_token_len` 做 decode→re-encode 不动点（最多 11 轮）；真实文本插件对超长一律
  截断，对不足在 `cap` 模式下原样保留、在 `drop` 模式下整条丢弃；残余偏差只打一条 warning。
- **实际 ISL**：优先取服务端 `usage.prompt_tokens`，只用于报表均值/分位，**不与目标值核对**，也没有
  超差剔除或 0 容差断言。

本集采纳其中三点：用本地 tokenizer 构建、decode 后重新编码再计数（§4.3 第 3 条）、目标长度先扣模板
（§4.4 探针）。不采纳并补强两点：其 `cap` 模式会让短样本以真实长度进入压测，本集改为"重建到精确等值"；
其运行期不做等值校验，本集把校验放在发送前并在不等时 skip（§4.5），同时把服务端返回的
`usage.prompt_tokens` 记入 `records.jsonl`，作为事后可复核的证据。其 `--tokenize-prompt` 能消除服务端
重分词，但会绕开 chat 模板语义，本集不采用，仅记为未来可选档位。

---

## 5. 抽样方法与均匀性

### 5.1 总体流程

```
候选池（数据源全部行）
  → 场景过滤（语言 / 任务族 / 原生长度 / 任务块长度 / 去重 / 互斥）
  → 分层（见 §5.3）
  → 确定性排序（sha256(seed|id)）
  → 分层最大余数配额（每场景 10 个 target）
  → 每个 target 生成 3 档（1K/8K/32K）
  → 构建期校验（240 行、24 cell、严格嵌套、中文存在）
  → 运行前对齐（按目标模型 tokenizer 重建 + 逐条校验）
```

### 5.2 可复现性机制

- `seed = 42`；排序键 `sha256(f"{seed}|{id}")`，与文件行序无关；配额用最大余数法，逐层确定性展开。
- 抽样只依赖数据源内容与 seed，不依赖时间、机器、文件行序；同一份 raw 重跑得到同一批 target。
- `manifest.json` 记录每个场景的 `upstream_ids`（10 个）、tokenizer SHA-256、逐档行数、
  `rows_sha256`；`testset.jsonl` 的 SHA-256 与 manifest 一致才能进入测量（见 §6.4）。

### 5.3 分层键与配额

| 场景 | 分层键 | 候选池 | 取 10 的分配 |
|---|---|---|---|
| 知识问答 | LongBench-Pro `secondary_task`（T3.1/T3.2） | 13 | 5 + 5 |
| 指令遵循 | vcsum 文档长度层（10 篇）× Multi-IF 约束 key（1 条） | 200 篇 / 454 约束 | 固定顺序逐对配对 |
| 长上下文 | T4.1 / T4.2 | 30 | 5 + 5 |
| 数学与逻辑 | AIME 题号（30 题） | 30 | 确定性取 10 |
| 编程 | LongBench v2 仓库代码子集 | 50 | 确定性取 10 |
| 通用 Agent | BFCL simple_python / multiple | 400 / 200 | 3 + 7 |
| 搜索 Agent | T1.1 / T1.2 / T5.1 / T5.2 | 58 | 3/3/2/2 |
| 多语言 | 语种（10 个，每语种 1） | 10 × 14,042 | 每语种 1 |

### 5.4 过滤、去重与互斥规则

- 语言/任务族过滤：只保留场景指定的 language 与 primary_task；
- 原生长度过滤：LongBench-Pro 只取 `token_length ∈ {32k, 64k, 128k}`，保证 32K 档是**原生文档前缀**
  而不是拼接产物（1K/8K 档在同一原生文档上取前缀）；
- 任务块过滤：`task_tokens ≤ 400`，否则 1K 档留给上下文的空间不足；
- 单元流下限：`stream_tokens ≥ 36,000`，不足则按"同层优先、全体兜底"追加同源文档，直到达标；
- 去重：同一场景内 `upstream_id` 唯一；LongBench-Pro 三个场景的任务族两两不相交；
- 排除不完整样本：多语种题库要求 4 个选项齐全；BFCL 要求工具 schema 可解析；AIME 只取有 `answer`
  的题面（答案不参与性能测量，仅用于回溯）。

### 5.5 均匀性论证

- **长度维度**：每场景 3 档 × 10 条，每档 prompt token 精确相等（§4），档位内方差为 0；
- **场景维度**：8 场景各 10 个 target，24 个 cell 的样本数完全相同（10），不存在某格样本偏少；
- **来源维度**：LongBench-Pro 场景按 secondary_task 分层、多语言按语种分层、Agent 按工具数量分层，
  避免某一子域占满 10 条；小池场景（知识问答 13 选 10）分层后覆盖两个子域；
- **语言维度**：zh 63 / en 153 / 其他 8 语种各 3；中文集中在两个中文场景 + 多语言 zh 1 条；
- **随机性维度**：抽样对候选池是确定性的均匀覆盖（sha256 哈希打散后按层配额），不是"取前 10 条"。

### 5.6 为什么每格 10 条

性能指标是连续量（TTFT/内存 ms 级、TPS 小数级），10 条/格足以给出中位数与 IQR 并识别离群；
240 条 × 1 次重复 = 240 个测量请求、24 个 cell，单模型单机一轮在半小时量级（32K 档占大头）。
增加条数只提高精度边际，但会线性增加测量时间与内存波动暴露面，且 24 格结构不变。若需更高置信度，
用 `--samples` 由 runner 在每个 cell 内确定性取子集，而不是改测试集。

---

## 6. 样本规格与产物

### 6.1 文件布局

```
dataset/raw/                     # 上游原始数据（gitignored，fetch.py 可复原）
dataset/test/performance/
  testset.jsonl                  # 240 行（gitignored；基线构建或 align.py 产物）
  manifest.json                  # 抽样/构造/对齐证据（tracked）
  testset.yaml                   # 测试集声明（tracked）
dataset/.cache/performance/
  alignment-<slug>.json          # 目标模型对齐报告（gitignored）
tool/dataset/performance/
  build.py                       # 构建器（参考 tokenizer）
  align.py                       # 运行前按目标 tokenizer 重建 + 校验
  fetch.py                       # 下载/校验上游文件（--verify）
  derive_cl100k.py               # 从 .tiktoken 生成 cl100k tokenizer.json
  sources.py                     # 数据源清单（URL/许可/本地路径）
  sampling.py                    # 确定性抽样原语
```

### 6.2 单行 schema

```json
{"id":"perf-knowledge_qa-1k-000","scenario":"knowledge_qa","bucket":"1k",
 "target_id":"tgt-knowledge_qa-000","dataset":"longbench_pro",
 "tokens":1024,"messages_tokens":1024,"tools_tokens":0,
 "messages":[{"role":"user","content":"…(上下文 + 任务块)…"}],
 "tools":[], "constructed":false,
 "meta":{"lang":"zh","source":"longbench_pro:T3. Evidence-Grounded QA",
         "upstream_id":"…","target_tokens":1024,"content_tokens":1024,
         "template_reserve_tokens":0,"task_tokens":395,
         "stream_tokens":36002,"stream_units":3,
         "alignment":"prefix+token-cut","task_sha256":"…","body_sha256":"…"}}
```

| 字段 | 含义 |
|---|---|
| `id` | `perf-<scenario>-<bucket>-<NNN>`，NNN 为 target 序号 |
| `target_id` | 同一 target 的 3 档共用，且三档只有上下文长度不同 |
| `dataset` | 场景级稳定标签（保证 plan 是 24 cell；精确上游在 `meta.source`） |
| `tokens` / `messages_tokens` | 目标 prompt token（含 template）；运行时以服务器 `/v1/tokenize` 复核 |
| `tools_tokens` / `tools` | 本集恒为 0 / 空（工具以文本目录内嵌，理由见 §3.1 场景 6） |
| `constructed` | 上下文是否由多单元拼接（长文/示例/工具目录类为 true） |
| `meta.content_tokens` | 内容 token（不含 template） |
| `meta.task_tokens` | 任务块 token（≤400） |
| `meta.task_sha256` / `body_sha256` | 三档一致性与嵌套校验锚点 |
| `meta.lang` | 行语言（多语言为具体语种） |

### 6.3 manifest 关键字段

`name / generated_at / builder / seed / targets_per_scenario / buckets / max_tokens /
min_completion_tokens_required / tokenizer{path,sha256,vocab_size,role} /
construction{prompt_tokens, template_reserve_tokens, content_tokens, tolerance_tokens, method,
stream_min_tokens} / alignment{unit, tolerance_tokens, verified_by} / cells / langs /
stats{task_tokens_min/max, content_chars} / sources{scenario: dataset + upstream_ids} /
rows / rows_sha256`。

- `construction.tolerance_tokens = 0`、`alignment.tolerance_tokens = 0`：声明本集没有误差带；
- `tokenizer.role` 明确"参考 tokenizer，永不作为被测模型"；
- `alignment.verified_by` 指向 `align.py`：生产运行前由它按目标模型重建并校验。

### 6.4 testset.yaml

```yaml
name: performance
title: 性能测试（8 场景 × 1K/8K/32K × 10）
description: 严格对齐的 prompt token；strict 校验不通过即 skip；max_tokens=50
scenarios:
  knowledge_qa: {dataset: perf_jsonl, dataset_params: {scenario: knowledge_qa, path: dataset/test/performance/testset.jsonl}}
  # … 其余 7 个场景同构
sampling: {seed: 42, targets_per_scenario: 10, method: sha256(seed|upstream_id)+stratified largest remainder}
buckets: {1k: 1024, 8k: 8192, 32k: 32768}
strict_input_tokens: true     # 运行期严格校验，无 ±10% 带
max_tokens: 50                # 默认输出窗口；由 apply_testset 写入 perf.max_tokens，CLI/Web 可覆盖
```

校验：`python -m pytest test/test_perf_dataset.py`（行数/精确 token/嵌套/语言/manifest 一致性）、
`./modelbench validate --offline`（加载器抽样 240 条）、`./modelbench run --method performance --dry-run`
（24 cell / 240 请求）。

---

## 7. 数据来源、许可与下载

### 7.1 源文件清单（`dataset/raw/`，SHA-256 截断显示）

| source | 本地路径 | 字节数 | SHA-256（前 16） | 许可 | 用途 |
|---|---|---:|---|---|---|
| reference_tokenizer_blob | tokenizer/cl100k_base/cl100k_base.tiktoken | 1,681,126 | 223921b76ee99bde | MIT | 通用参考 tokenizer 原始词表 |
| reference_tokenizer | tokenizer/cl100k_base/tokenizer.json | 2,729,984 | 745dc2f8e2eb601a | MIT | 构建期唯一计数 tokenizer（非被测模型） |
| longbench_pro | longbench_pro/longbench_pro.json | 531,535,940 | 92ff05f6088e212d | apache-2.0 | 知识问答/长上下文/搜索 Agent |
| longbench_v2 | longbench_v2/data.json | 465,490,535 | 15d61c22d92c9690 | apache-2.0 | 编程（仓库代码 50 条） |
| vcsum | longbench_v1/data/vcsum.jsonl | 8,923,352 | d7412361c4187946 | apache-2.0 | 指令遵循（中文长文） |
| multi_if | multi_if/multiIF_20241018.csv | 7,402,992 | 6c588af031fe67b1 | cc-by-nc-2.0 | 指令遵循（中文真实约束） |
| aime_2026 | aime2026/data/train-00000-of-00001.parquet | 10,062 | ae72853349a46c5f | cc-by-nc-sa-4.0 | 数学目标题面（30 题） |
| math_500 | math_500/test.jsonl | 446,564 | 35dc41080a368085 | mit | 数学 many-shot 满分解 |
| bfcl_v4 | bfcl/BFCL_v4_simple_python.json | 283,274 | 82dd63ba502eb252 | apache-2.0 | 通用 Agent 工具 schema |
| global_mmlu (zh) | global_mmlu/zh/test.parquet | 3,618,771 | e1ba0c287b650043 | apache-2.0 | 多语言 zh |
| global_mmlu (en) | global_mmlu/en/test.parquet | 3,910,925 | 5baec81ba9724372 | apache-2.0 | 多语言 en |
| global_mmlu (fr) | global_mmlu/fr/test.parquet | 4,550,854 | e602201459c29c5d | apache-2.0 | 多语言 fr |
| global_mmlu (de) | global_mmlu/de/test.parquet | 4,543,635 | b58a0d5994e756da | apache-2.0 | 多语言 de |
| global_mmlu (es) | global_mmlu/es/test.parquet | 4,327,396 | efbcc4ce1cf14f2f | apache-2.0 | 多语言 es |
| global_mmlu (pt) | global_mmlu/pt/test.parquet | 4,211,041 | 15dda0dd7dbd8eeb | apache-2.0 | 多语言 pt |
| global_mmlu (ru) | global_mmlu/ru/test.parquet | 5,884,449 | 9037e1aaca6aed18 | apache-2.0 | 多语言 ru |
| global_mmlu (ja) | global_mmlu/ja/test.parquet | 4,363,616 | b56ad153180d4368 | apache-2.0 | 多语言 ja |
| global_mmlu (ko) | global_mmlu/ko/test.parquet | 4,273,255 | aa896380f066e8e9 | apache-2.0 | 多语言 ko |
| global_mmlu (ar) | global_mmlu/ar/test.parquet | 4,854,848 | e969c186f001c754 | apache-2.0 | 多语言 ar |

### 7.2 下载与校验

```bash
# 下载缺失文件（幂等，只补缺）+ 校验既有文件
.venv/bin/python tool/dataset/performance/fetch.py --verify
# 若 reference_tokenizer 缺失，fetch 会自动调用 derive_cl100k.py 从 .tiktoken 重建
```

- 镜像：`https://hf-mirror.com`（`--mirror` 可改）；下载器只依赖标准库，逐文件 SHA-256 校验。
- 上游地址：
  - LongBench-Pro `https://huggingface.co/datasets/caskcsg/LongBench-Pro`
  - LongBench v2 `https://huggingface.co/datasets/zai-org/LongBench-v2`
  - LongBench v1（vcsum）`https://huggingface.co/datasets/zai-org/LongBench`
  - Multi-IF `https://huggingface.co/datasets/facebook/Multi-IF`
  - AIME 2026 `https://huggingface.co/datasets/MathArena/aime_2026`
  - MATH-500 `https://huggingface.co/datasets/HuggingFaceH4/MATH-500`
  - BFCL `https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard`
  - Global-MMLU `https://huggingface.co/datasets/CohereLabs/Global-MMLU`
  - cl100k_base `https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken`
- 许可注意：Multi-IF（cc-by-nc-2.0）与 AIME 2026（cc-by-nc-sa-4.0）为非商用/相同方式共享许可，
  只用于本地性能测量，不随结果对外分发原文；其余为 apache-2.0 / mit。

---

## 8. 复现与接线

### 8.1 五步复现

```bash
# 1) 上游数据（幂等）
.venv/bin/python tool/dataset/performance/fetch.py --verify
# 2) 参考基线（通用 tokenizer，容差 0，240 行）
.venv/bin/python tool/dataset/performance/build.py
# 3) 运行前按目标模型 tokenizer 重建 + 逐条 /v1/tokenize 校验
.venv/bin/python tool/dataset/performance/align.py \
    --tokenizer <model>/tokenizer.json --base-url http://127.0.0.1:8000 --model <model-id>
# 4) 校验 + dry-run
./modelbench validate --offline
./modelbench run --method performance --dry-run
# 5) 正式测量
./modelbench run --method performance
```

### 8.2 与 runner 的接线

- `load_testsets()` 扫描 `dataset/test/*/testset.yaml`，本集 name = `performance`；
- `apply_testset()` 应用 `strict_input_tokens` 与 `max_tokens`（accuracy 集声明这两项直接报错；
  CLI `--max-tokens` 显式优先）；
- `perf_jsonl` 适配器按 scenario/bucket 读行，`tokens` 进 `declared_tokens`，messages 原样发送；
- runner 在 prebuilt 分支：tokenize → 窗口判定 → strict 等值校验（不等则 skip）→ 发送请求；
  `context_meta` 记录 `declared_tokens / actual_prompt_tokens / isl_tolerance(exact) / bucket /
  decode_guard_applied`；
- 结果记录逐样本保留 TTFT/TG TPS/峰值内存/`completion_tokens`/`gen_tps_invalid`，聚合由这些记录复算；
- 控制台同一语义：测试集卡片显示 `max_tokens` 与「严格对齐」徽标，运行表单用一句话提示先跑
  `align.py`；strict run 的结果页只展示有效样本口径（不再有 ISL 入带/出带与 ±10% 过滤列），
  被跳过的样本在顶部横幅列出。

### 8.3 质量门禁

- `python -m pytest test/test_perf_dataset.py test/test_perf.py test/test_integration.py`：
  行数/精确 token/嵌套/语言/manifest 一致性、strict 运行期行为、prebuilt 发送路径；
- `.venv/bin/python tool/dataset/performance/build.py --check`：行数/24 cell/每档 10/`tokens` 等值/
  SHA 与 manifest 一致/yaml 契约（strict、max_tokens、无 osl）；对 `align.py` 产物按对齐报告里的
  目标 tokenizer 复核；
- `.venv/bin/python -m harness.gate`：imports/compile/tests/validate/dataset_integrity/perf_plan/
  docs_links/docs_contract/secrets/hygiene 全项；
- `./modelbench validate --offline`：加载器逐个 dataset selection 统计样本数（本集 240）。

### 8.4 运行预算与窗口

- 请求数：240（8 场景 × 3 档 × 10），cell 数 24；每样本 1 次重复；
- 计算量集中在 prefill：32K 档 80 条、8K 档 80 条、1K 档 80 条；
- 内存峰值随 32K 档上升，需保证 `model_context_window ≥ 32,946`（32,768 + 50 生成长度 + 128 安全
  余量）；不满足时 32K 档样本会被 skip 并记录，不静默截断；
- 单实例串行（双锁），不同 `base_url` 可并行，同一 `base_url` 禁止并发。

---

## 9. 已知局限与风险

1. **三个长文场景共用一个上游发布方**（LongBench-Pro 的 T3/T4/T1+T5）。行集互不相交、任务族不同，
   但语料来源同一，若其文本分布有偏，这三个场景会共享偏差；因此禁止跨场景横比（§3.2）。
2. **数学与多语言是"题库 many-shot"而非原生单长文档**：长度来自示例数量，示例来自真实题目，
   但 prefill 的 token 分布比自然长文更模板化；这是"原生 32K 数学文本几乎不存在"的现实取舍。
3. **指令遵循是文档 + 约束的组合**：Multi-IF 的约束是真实标注，但历史对话不含 assistant 轮
   （公开 CSV 的 `responses` 为空），因此不是真正的多轮指令遵循；本集只取约束与长文材料。
4. **通用 Agent 的工具以文本目录呈现**（不走 `tools` 字段）：与真实工具调用协议有差异；换取的是
   prefill 负载可复现、不会因 `tool_calls` 使 TG TPS 失效。工具目录是真实 schema 的确定性拼接。
5. **严格对齐依赖目标模型 tokenizer**：若模型目录缺 `tokenizer.json`、服务器不支持 `/v1/tokenize`，
   `align.py` 无法完成校验，strict 集会全部 skip；这是显式的失败，不是降级。部署时应先跑 align。
6. **align 需要 `dataset/raw/` 可达**：运行前重建读原始数据（2.6 GB 量级，但只需相关文件）；纯远程
   部署需同步 raw 或预生成 testset 并保持 SHA 一致。
7. **中文占比 63/240**：满足"不能全英文"，但场景数上中文只覆盖 2 个中文场景 + 多语言 1 条；若后续
   需要中文编程/中文 Agent 场景，需另找源（现有源无原生中文长代码/工具目录）。
8. **Token 目标与模型窗口的耦合**：32K 档要求窗口 ≥32,946；小窗口模型（如 32K 窗口）会整档 skip，
   结果中只有 1K/8K 两档，聚合必须显式报告缺失档位。

---

## 10. 附录

### 附录 A：抽样统计（构建实测）

| 场景 | 候选池（过滤后） | 取 target | 上游 id 数 | 语言 |
|---|---:|---:|---:|---|
| knowledge_qa | 13 | 10 | 10 | zh |
| instruction_following | 200 篇 / 454 约束 | 10 | 20（文档+约束各 10） | zh |
| long_context | 30 | 10 | 10 | en |
| math_logic | 30 题 / 500 示范 | 10 | 10（+500 示范流） | en |
| coding | 50 | 10 | 10 | en |
| general_agent | 600（400+200） | 10 | 10 | en |
| search_agent | 58 | 10 | 10 | en |
| multilingual | 10 × 14,042 | 10 | 10 | 10 语种 |

行数 240、target 80、cell 24；语言分布 zh 63 / en 153 / fr,de,es,pt,ru,ja,ko,ar 各 3；
`task_tokens` 56–395；每档内容字符数（cl100k 口径）：1K 768–4,877、8K 5,722–42,196、
32K 22,556–169,256。

### 附录 B：构建实测摘要

- 构建耗时：约 2 分钟（Reference tokenizer，240 行，含 LongBench-Pro 531 MB 流式解析）；
- 产物：`testset.jsonl` 12.1 MB / 240 行；`manifest.json` 记录 `rows_sha256`；
- 校验：`validate_rows` 0 问题（240 行、24 cell、每档 10、三档嵌套、`tokens` 精确、中文存在）；
  `./modelbench validate --offline` 通过；`--dry-run` 输出 24 cell / 240 请求。
