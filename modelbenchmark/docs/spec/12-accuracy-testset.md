# 本地小模型精度测试：样本集设计与抽样方案

> 版本：v1.1（2026-11-10）
> 范围：**只覆盖精度（accuracy）测试的样本选择**，不涉及性能/时延测量。
> **固定一套隐藏离线样本、全设备/全模型复用**：不按模型重新构造/裁剪/补齐；不含性能 run 的
> 判分接入。样本契约与测试系统职责边界见 [13-testset-portability.md](13-testset-portability.md)。
> 被测对象：`openbmb/MiniCPM-V-4.6` 这类小模型（本机实测：24 层、hidden 1024、bf16 权重约 2.4 GB、最大上下文 262,144），用于 **bf16 原始模型 vs 量化模型** 的精度损失对比。
> 产物：8 个场景 × 250 条 = 2,000 条样本、5 类确定性判题器、可逐字节复现的抽样脚本与 manifest。

---

## 0. 摘要

| # | 场景 | 数据集（上游） | 行数 / 语言 | 判题族 | 主指标 |
|---|---|---|---:|---|---|
| 1 | 知识问答 | Global-MMLU | 250 = 125 题 × en/zh | `mc_letter` | Accuracy |
| 2 | 指令遵循 | IFEval | 250（en） | `ifeval` | Prompt-level strict |
| 3 | 长上下文 | LongBench v1 | 250 = en 150 + zh 100 | `normalized_match`（F1/中文 F1/ROUGE-L） | 平均 F1/ROUGE-L |
| 4 | 数学与逻辑 | GSM8K + CMATH | 250 = en 125 + zh 125 | `normalized_match`（数值） | Accuracy |
| 5 | 编程 | HumanEval + MBPP | 250 = 各 125（en） | `code_exec` | pass@1 |
| 6 | 通用 Agent | BFCL v4（11 个单轮类） | 250（en） | `tool_call` | Accuracy |
| 7 | 搜索 Agent | HotpotQA（distractor） | 250（en） | `normalized_match`（F1） | 平均 F1 |
| 8 | 多语言 | Global-MMLU | 250 = 25 题 × 10 语 | `mc_letter` | Accuracy |

关键决策：

1. **9 个上游数据集覆盖 8 个场景**：Global-MMLU 同时服务“知识问答”和“多语言”（一题两用），其余场景各 1–2 个数据集；
2. **判题族恰好 5 个**，全部确定性、无 LLM judge、无网络、无人工评分；
3. **中英双语**：en 1,400 行 / zh 350 行 / 其他语种 250 行；指令、编程、Agent、搜索四类官方数据本身只有英文，按需求允许“单场景单语言”；
4. **小模型难度校准**：8 个场景先用参考模型各跑 12 条试点，全部落在 15%–85% 可用带，排除了 MMLU-Pro、MATH-500、AIME、LiveCodeBench、LongBench v2 等对 1B 模型过难（接近随机）的选项；
5. **确定性抽样**：seed=42 + `sha256(seed|upstream_id)` 排序 + 分层最大余数配额，重跑逐字节一致（manifest 记录每个文件的 SHA-256）。

---

## 1. 目标、约束与非目标

### 1.1 硬约束（来自测试需求）

| # | 约束 | 本方案的落实 |
|---|---|---|
| C1 | 覆盖 8 场景，每场景样本量相同 | 8 × 250 = 2,000 行；每行恰好属于一个场景 |
| C2 | 不能全为单一语言 | en 1,400 / zh 350 / 其他 250；zh 覆盖知识、数学、长上下文 |
| C3 | 判题不依赖大模型，判题器 ≤ 5 种 | 5 个判题族：`mc_letter`、`ifeval`、`normalized_match`、`code_exec`、`tool_call` |
| C4 | 面向小模型，题目不能整体过难 | 参考模型 8 场景 × 12 条试点（v1.0 旧 testset，v1.1 需重跑，§8）全部落在 15%–85%；弃用超难题集（§3.3） |
| C5 | 数据集不宜过多，能复用则复用 | 9 个数据集覆盖 8 场景；Global-MMLU 复用 2 个场景 |
| C6 | 结果可复现、可审计 | 固定 seed；样本 id 与 `upstream_id` 一一对应；manifest 记录上游文件 size/SHA-256 |

### 1.2 设计原则

1. **能力对齐**：每个场景只选直接测该能力的任务，不用“能沾边”的代理任务；
2. **可程序判定**：判题只依赖参考答案、单测、官方 checker 或固定规则，任何主观项不进入精度分；
3. **难度带优先**：小模型在目标集上的准确率应落在约 15%–85%，避免地板/天花板效应（量化损失只有在“还会做但开始做错”的区间才可测）；
4. **语言可解释**：同一场景内不混入无法用同一模板/判题处理的语言；跨场景保证中英都有；
5. **最小数据面**：优先复用已有本地原始数据，新增下载只补缺口（本方案仅新增 MBPP 一个数据集）；
6. **样本即契约**：每条样本自带 prompt + 判题所需全部输入，判题器不读外部状态。

### 1.3 非目标

- 不与公开榜单（MMLU 5-shot、HumanEval pass@1 等）横向比较：提示词、抽样口径、判题实现均不同；
- 不测多轮工具循环、Web 浏览、代码仓库级修改、图像/视频理解（VLM 的多模态能力不在本精度方案内）；
- 不测安全/对齐/毒性；不测训练数据污染；
- 不评价推理框架性能（TTFT/TPS 属于性能基准，另一套方案）。

---

## 2. 业界开源测试集调研（按场景）

下表是 2026 年前后社区最常用的开源测试集及其适用场景、判题方式和“对 1B 级小模型是否合适”的判断。判题方式的“确定性”是本次选型的硬门槛。

| 场景 | 数据集 | 语言 | 判题方式 | 规模（约） | 小模型适配 |
|---|---|---|---|---:|---|
| 知识 | MMLU | en | 4 选 1 字母 | 14k | 中（1B 约 40–60%） |
| 知识 | MMLU-Pro | en | 10 选 1 + CoT | 12k | 低（1B 接近随机） |
| 知识 | CMMLU | zh | 4 选 1 字母 | 11k | 中 |
| 知识 | C-Eval | zh | 4 选 1 字母 | 13k | 中（部分题格式特殊） |
| 知识/多语 | **Global-MMLU** | 42 语 | 4 选 1 字母 | 590k | 中（本次采用） |
| 多语 | MMMLU | 14 语 | 4 选 1 字母 | 196k | 中 |
| 多语 | Belebele | 122 语 | 4 选 1 字母 | 490k | 中（阅读问答，非知识） |
| 多语 | XNLI | 15 语 | 3 选 1 | 112k | 中（NLI 任务） |
| 指令 | **IFEval** | en | 官方规则 checker | 541 | 中（本次采用） |
| 指令 | IFBench | en | 规则 checker（OOD） | 300 | 低（1B 常 <10%） |
| 指令 | Multi-IF | 8 语 | 规则 checker（多轮） | 4.5k | 低（多轮 + 格式复杂） |
| 指令 | FollowBench | en/zh | 规则 + 人工 | 1.4k | 低 |
| 指令 | ComplexBench | zh | 规则 + LLM | 4k | 低（依赖 LLM judge） |
| 长文 | **LongBench v1** | en+zh | F1 / ROUGE | 4.7k | 中（本次采用） |
| 长文 | LongBench v2 | en | 4 选 1 | 503 | 低（100K+ 字符） |
| 长文 | RULER | en | 规则（合成） | 13k | 低（合成任务） |
| 长文 | ∞Bench | en+zh | F1 / 规则 | 3.9k | 低（100K+） |
| 长文 | L-Eval | en+zh | F1 / ROUGE | 2k | 中 |
| 数学 | **GSM8K** | en | 数值精确匹配 | 8.8k | 中（本次采用） |
| 数学 | MATH / MATH-500 | en | 符号等价 | 12.5k / 500 | 低（1B 低于 15%） |
| 数学 | AIME | en | 数值 | 1k | 极低（<5%） |
| 数学 | **CMATH** | zh | 数值 | 1.7k | 中（本次采用） |
| 数学 | MathBench | zh+en | 规则 + LLM | 5.7k | 中 |
| 逻辑 | BBH | en | 精确匹配 | 6.5k | 低（格式复杂） |
| 编程 | **HumanEval** | en | 单元测试 | 164 | 中（本次采用） |
| 编程 | HumanEval+ | en | 增强单测 | 164 | 低 |
| 编程 | **MBPP（sanitized）** | en | 单元测试 | 427 | 中（本次采用） |
| 编程 | MBPP+ | en | 增强单测 | 378 | 低 |
| 编程 | LiveCodeBench | en | 单元测试（新题） | 1k+ | 极低（1B <5%） |
| 编程 | CRUXEval | en | 精确匹配 | 800 | 中（代码执行推理，非生成） |
| Agent | **BFCL v4（single-turn）** | en | possible-answer 结构匹配 | 2.7k | 中（本次采用） |
| Agent | τ-bench / τ²-bench | en | 环境 + 用户模拟器 | 1.6k | 极低（需 LLM 模拟器） |
| Agent | AgentBench | en | 多环境判定 | 1k+ | 低（部署重） |
| Agent | ToolBench | en | 环境 / LLM | 126k | 低（API 不稳定） |
| 搜索 | **HotpotQA（distractor）** | en | EM / F1 | 113k | 中（本次采用） |
| 搜索 | 2WikiMultiHopQA | en | EM / F1 | 192k | 中（可替代） |
| 搜索 | Musique | en | EM / F1 | 25k | 中低（3 跳） |
| 搜索 | FRAMES | en | 精确 / LLM | 824 | 低（需检索环境） |
| 搜索 | GAIA | en | 精确 / LLM | 466 | 极低（需网络+多工具） |
| 搜索 | BrowseComp | en | 精确 / LLM | 1,266 | 极低 |
| 搜索 | xbench-DeepSearch | zh+en | LLM judge | 100 | 低（judge 依赖大模型） |
| 搜索 | WebArena | en | 环境判定 | 812 | 低（需浏览器环境） |

**从“热门”到“可用”的三条筛选结论：**

1. **榜单热度不等于小模型可测**：MMLU-Pro、MATH/MATH-500、AIME、LiveCodeBench、BrowseComp 这类面向 7B+ 模型的难题，1B 模型分数接近随机（10%–15%）甚至 0，量化损失无法与噪声区分，直接排除；
2. **判题必须闭环**：依赖 LLM judge（ComplexBench、xbench）、用户模拟器（τ-bench）、外部检索/网络（GAIA、BrowseComp、WebArena）、动态 API（ToolBench）的方案不可复现，排除；
3. **优先“短答案 + 客观判分”**：4 选 1 字母、数值、短 span F1、单测、工具调用结构匹配，这五类足以覆盖 8 个场景，正好对应 5 个判题族。

---

## 3. 数据集选型

### 3.1 选型准则

| # | 准则 | 说明 |
|---|---|---|
| S1 | 能力直接命中 | 任务形态与场景定义一致（如“指令遵循”用带约束 checker 的任务，而不是开放式问答） |
| S2 | 判题确定性 | 有官方 checker / 参考答案 / 单测；同一响应重复判分结果完全一致 |
| S3 | 难度带 | 参考模型（1B）在候选集上的准确率约 15%–85%，且题型分布均匀 |
| S4 | 语言与样本量 | 单场景语言可单语，但整体必须中英都有；单数据集可用题量 ≥ 250 或可与同能力集拼满 250 |
| S5 | 数据源精简 | 同能力优先复用一个数据集；能覆盖两个场景的数据集优先 |
| S6 | 许可与可得 | 本地已有或可公开下载；许可允许研究使用；文件可校验 SHA-256 |

### 3.2 场景 × 数据集矩阵（最终）

| 场景 | 数据集 | 语言 / 配比 | 题池 | 抽样方式 | 判题 |
|---|---|---|---|---|---|
| 知识问答 | Global-MMLU | en 125 + zh 125（125 题配对） | 14,042 题/语（跨语言对齐后配对池 13,802） | 56 学科分层，125 题 × 2 语 | `mc_letter` |
| 指令遵循 | IFEval | en 250 | 510 条可用 | 8 个一级约束类型分层 | `ifeval` |
| 长上下文 | LongBench v1（5 en + 2 zh 任务） | en 150 + zh 100 | 700 en / 294 zh（4K–32K，按 prompt 去重后） | 语言 × 长度桶 × 任务三层分层 | `normalized_match`（F1/中文 F1/ROUGE-L） |
| 数学与逻辑 | GSM8K + CMATH | en 125 + zh 125 | 1,319 + 1,098 | 推理步数 / 年级分层 | `normalized_match`（数值） |
| 编程 | HumanEval + MBPP | en 250 | 164 + 427 | 题面长度四分位分层 | `code_exec` |
| 通用 Agent | BFCL v4 | en 250 | 2,741（11 类） | 11 类分层，小类保底 12 | `tool_call` |
| 搜索 Agent | HotpotQA distractor | en 250 | 7,405（全 hard） | 题型 × 跳数（2/3/4+）分层 | `normalized_match`（F1） |
| 多语言 | Global-MMLU | 10 语 × 25（25 题配对） | 14,042 题/语（跨语言过滤后候选 13,552） | 6 学科大类分层选 25 题 | `mc_letter` |

### 3.3 逐场景选型依据（含备选与弃选理由）

> 本节引用的“实测”均指 §8 的参考模型难度试点（每场景 12 条），只用于判断难度带，不是正式精度结论。

#### 3.3.1 知识问答 → Global-MMLU（en + zh 配对）

- **候选**：MMLU、MMLU-Pro、CMMLU、C-Eval、Global-MMLU、AGIEval。
- **选 Global-MMLU**：同一 `sample_id` 有 42 种语言版本（en 即原版 MMLU），可以 125 道题 × 中英两版 = 250 行做**配对设计**，直接分离“知识”与“语言”两个因素；4 选 1、判题只比字母；1B 模型实测约 50%，处于可测区间。
- **弃 MMLU-Pro**：10 选项 + 官方 5-shot CoT，1B 模型接近 10% 随机线，量化差异被噪声淹没；弃 **MATH 类**同理由 §3.3.4；弃 **CMMLU/C-Eval**：只有中文，无法做双语配对，且 C-Eval 部分题目含特殊格式；弃 **AGIEval**：法律/公务员等子集题型混杂、难度两极。
- **配对设计的代价**：知识场景只有 125 个唯一题（250 个评测行）。对本项目的“bf16 vs 量化”同一模型对比，配对反而提高检出力；单场景准确率 95% CI 约 ±6.2pp（250 行）。
- **上游对齐过滤（2026-11 审计修复）**：Global-MMLU 的 zh/ko `high_school_world_history` 在 `sample_id` 上与英文行错位（该学科 237 行中 zh 有 176 行的 answer 字母与 en 不同，其余 56 学科为 0），因此构建期不再只断言 `sample_id` 集合相等，而是：
  1. 只保留同一 `sample_id` 的 en/zh answer 字母相等的配对；
  2. 按学科统计 en/zh 不一致率，任一学科不一致率 > 0 即整科退出配对池（本版剔除 `high_school_world_history` 237 题）；
  3. 剔除已知上游机翻错误行：`abstract_algebra/test/1`（zh 把 `index` 误译为“幂”，gold C=24 会被读成“24 次幂”）；
  4. 配对池还需 en/zh 两侧结构合法（4 个非空选项、答案 A–D；本版剔除 zh 选项为空的 2 条 virology 行）。
  过滤后配对池 **13,802 题**、覆盖 **56 学科**；排除明细（subject/count/reason）记录在 `manifest.scenarios.knowledge_qa.sampling.alignment_excluded`。

#### 3.3.2 指令遵循 → IFEval

- **候选**：IFEval、IFBench、Multi-IF、FollowBench、ComplexBench。
- **选 IFEval**：25 类可程序验证的约束（长度、大小写、标点、关键词、格式、起止、组合），Google 官方 checker 可直接复用；1B 模型 prompt-level strict 实测约 62%，既不是全过也不是全错。541 条中排除 31 条 `language:response_language`（依赖 `langdetect`，非纯规则），可用池 510 条，满足 250。
- **弃 IFBench**：58 类 OOD 新约束，1B 模型通常个位数通过率，地板效应；弃 **Multi-IF**：多轮 + 多语言，checker 口径与单轮 IFEval 不完全一致，且多轮不在本方案运行协议内；弃 **FollowBench/ComplexBench**：含人工/LLM 评分成分，违反“判题不依赖大模型”。
- **语言**：IFEval 官方只有英文；按约束 C2 允许单场景单语言，中文能力由知识、数学、长上下文场景覆盖。

#### 3.3.3 长上下文 → LongBench v1（4K–32K 分层）

- **候选**：LongBench v1、LongBench v2、RULER、∞Bench、L-Eval、NIAH。
- **选 LongBench v1**：真实长文档 + 短答案抽取任务，官方指标判题（英文 5 任务 SQuAD token F1，中文用官方中文指标）；英文取 narrativeqa / qasper / hotpotqa / 2wikimqa / multifieldqa_en，中文取 dureader / multifieldqa_zh，天然中英双语；用目标模型 tokenizer 过滤到 4,096–32,768 tokens 后仍有 en 700 / zh 294 的池（按 prompt 去重后；未去重为 713/295）。
- **弃 LongBench v2**：上下文中位约 10 万 token，250 条预填成本高、且 1B 模型在 100K 长度上近乎全错；弃 **RULER/NIAH**：合成检索任务，只测“找针”，不测真实长文理解；弃 **∞Bench**：同样超长；弃 **L-Eval**：部分任务为摘要/主观评分，F1/ROUGE 对 1B 生成不稳定。
- **判题口径**：英文 5 任务 `metric=f1`（SQuAD token F1）；`dureader` 用官方 ROUGE-L（`metric=rouge_zh`）；`multifieldqa_zh` 用官方中文 token F1（`metric=f1_zh`，jieba 分词 + 中文归一化）。上游 20 组重复 prompt 中 18 组答案不同，构建期把同组全部可接受答案按出现顺序 union 进 `judge.answers`（本版入选行中 7 行 narrativeqa 因此恢复被丢弃的参考答案，如 `A monk.` / `a priest`）。
- **已知取舍**：16–32K 桶只有英文（中文 LongBench v1 文档最长约 1.7 万 token，仅 1 条候选）；中文长文集中在 4–16K。长度分层是**抽样口径**，不是性能档位。

#### 3.3.4 数学与逻辑 → GSM8K（en）+ CMATH（zh）

- **候选**：GSM8K、MATH、MATH-500、AIME、CMATH、MathBench、BBH。
- **选 GSM8K + CMATH**：小学/初中应用题，答案唯一数字，判题只需数值归一化（分数、百分数、千分位、单位）；1B 模型实测 GSM8K 可解，CMATH 约 50%，处于可测区间；两数据集一中一英，各 125 条。
- **弃 MATH/MATH-500/AIME**：1B 模型正确率 <15%（AIME 接近 0），无法测量化损失；弃 **MathBench**：部分子集依赖 LLM/规则混合判题；弃 **BBH**：逻辑题选项/格式复杂，小模型大量输在格式而非推理。**逻辑推理**未单独设集，由数学应用题的多步推理间接覆盖，避免再引入一个判题族。
- **分层**：GSM8K 按解题中计算步骤数（`<<...>>` 个数）分 5 桶；CMATH 按年级 1–6 分层，保证难度梯度均匀。

#### 3.3.5 编程 → HumanEval + MBPP

- **候选**：HumanEval、HumanEval+、MBPP、MBPP+、LiveCodeBench、BigCodeBench、CRUXEval。
- **选 HumanEval + MBPP（sanitized）**：两者都有官方单元测试，pass@1 判题确定；HumanEval 164 题 + MBPP 427 题，各抽 125 补满 250；1B 模型试点 pass@1 约 38%，高于地板且远低于天花板。
- **弃 LiveCodeBench/BigCodeBench**：新题/复杂依赖，1B 模型通过率 <5%；弃 **HumanEval+/MBPP+**：同题增强测试，只会进一步压低通过率，不增加能力覆盖面；弃 **CRUXEval**：测“代码执行结果预测”，不是代码生成能力，可作为编程场景的地板备选（§9）。
- **判题**：HumanEval 按官方协议 `prompt_prefix + completion` 拼接后执行官方 `test`；MBPP 执行 `test_imports + test_list`。子进程 6s 超时 + CPU/句柄/文件大小限制；不是安全沙箱，只用于受控测试机。
- **MBPP 题面（2026-11 审计修复）**：sanitized MBPP 的 `prompt` 只含任务描述、不含被测函数名（审计中 124/125 行不可判分），因此构建期从参考解 `code` 的首个顶层 `def` 提取入口函数名：题面写明 “Your code must be named `<entry_point>`” 并引用官方 `test_list` 前 3 条测试，同时 `judge.entry_point` 写入该名字（`prompt_prefix` 保持 `null`）；题面长度四分位按新渲染 prompt 的 token 数分层，保证与 `input_tokens` 口径一致。

#### 3.3.6 通用 Agent → BFCL v4（11 个单轮类）

- **候选**：BFCL v4、τ-bench、τ²-bench、AgentBench、ToolBench、API-Bank。
- **选 BFCL v4 single-turn**：11 个类别覆盖 simple（单函数）、multiple（多函数选一）、parallel（并行多调用）、parallel_multiple、irrelevance（不该调用）以及 live_* 真实请求；ground truth 是“函数名 + 每个参数可接受值”，结构匹配即可判分，无需用户模拟器。2,741 条池，11 类按比例分层、小类保底 12 条。
- **弃 τ-bench、τ²-bench**：需要 LLM 用户模拟器 + 环境状态，判题链路引入大模型，违反 C3；弃 **AgentBench**：多环境部署重、部分环境需交互式容器；弃 **ToolBench**：依赖不稳定真实 API；弃 **API-Bank**：可作为备选，但类别覆盖不如 BFCL。
- **实现约定**：模型以文本 JSON 形式输出调用（`{"name":..., "arguments":{...}}` 或数组），不依赖模型原生 tool_calls 接口，保证不同推理框架下可比；多调用类别按无序匹配。

#### 3.3.7 搜索 Agent → HotpotQA distractor

- **候选**：HotpotQA、2WikiMultiHopQA、Musique、FRAMES、GAIA、BrowseComp、xbench-DeepSearch、WebArena。
- **选 HotpotQA distractor**：每题的 10 段检索结果固定，模型需要在多段中定位并跨文档合并证据，答案短、EM/F1 判题确定；1B 模型试点平均 F1 约 0.34，可测。7,405 条校验集全部为 hard，天然避免简单题。
- **弃 GAIA/BrowseComp/xbench**：需要实时网络 + 大模型判题，不可复现；弃 **FRAMES**：需要外部 Wikipedia 检索环境；弃 **WebArena**：需要浏览器与站点环境；弃 **2Wiki/Musique**：与 HotpotQA 能力重叠，可作为同场景替换备选。
- **口径说明**：本场景是“**给定检索结果的单轮检索增强问答**”，测“跨文档证据定位与聚合”，不测多轮 search 工具循环（后者需要 runner 支持工具环境，不在样本设计范围内）。

#### 3.3.8 多语言 → Global-MMLU（10 语 × 25 题配对）

- **候选**：MMMLU、Global-MMLU、Belebele、XNLI、Flores-200。
- **选 Global-MMLU**：与知识场景同一数据集，零新增数据源；选 10 个跨语系语言（fr/de/es/pt/ru/ja/ko/ar/hi/vi），25 道题在 10 种语言下各评测一次 = 250 行，配对设计可比较“同一题不同语言”的稳定性；判题与知识场景共用 `mc_letter`。
- **弃 MMMLU**：本地仅有 6 种语言文件，补全需额外下载，且与 Global-MMLU 能力重复；弃 **Belebele/XNLI**：阅读问答/NLI，不是知识能力，且需要另一套判题；弃 **Flores-200**：翻译任务需 BLEU/chrF（连续指标 + 分词器），且翻译能力不在 8 场景定义内。
- **跨语言一致性过滤（2026-11 审计修复）**：候选 `sample_id` 必须在 10 个语种中 answer 字母完全一致，且每个语种的 4 个选项两两去重后仍为 4 个非空选项（上游 ko 等存在翻译坍塌，如 `high_school_macroeconomics/test/49` 的 4 个选项只剩 2 种文本）；过滤后按 6 大类合格池重新分配 25 题配额（本版仍为 Humanities 8 / STEM 5 / Social Sciences 5 / Medical 3 / Business 2 / Other 2）。排除明细（subject/count/reason，本版 365 个 sample_id）记录在 `manifest.scenarios.multilingual.sampling.alignment_excluded`。
- **zh/en 的位置**：多语言场景的 10 种语言**不含** zh/en（这两语在知识场景已作锚点）；因此“多语言”报告是相对 zh/en 的跨语种泛化剖面。每语言仅 25 题，单语言 CI 约 ±19pp，**只用于剖面，不用于单语排名**。

### 3.4 数据集复用与语言分布

- 9 个上游数据集：Global-MMLU、IFEval、LongBench v1、GSM8K、CMATH、HumanEval、MBPP、BFCL v4、HotpotQA；其中 Global-MMLU 复用 2 个场景（知识、多语言）。
- 语言分布（2,000 行）：en 1,400（70.0%）、zh 350（17.5%）、其他 250（12.5%）。中文覆盖知识 125、数学 125、长上下文 100；英文覆盖全部 8 场景。

---

## 4. 抽样方法与原则

### 4.1 总体流程

```
dataset/raw/<asset>/                    上游原始文件与工具资产（只读，gitignored）
      │  0. fetch：按 tool/dataset/accuracy/raw_spec.json 下载并校验
      │  1. 读取 + 结构校验（字段/选项/答案/单测/ground truth）
      │  2. 过滤（非法答案、超长/超短、重复题、需第三方依赖的指令）
      │  3. 分层（学科/类型/长度/难度/类别/语言）
      │  4. 分层配额（最大余数法）
      │  5. 层内确定性抽样（sha256(seed|upstream_id) 排序取前 k）
      ▼
dataset/test/accuracy/testset.jsonl         2,000 行 = 8 场景 × 250，单文件，逐行带判题契约
dataset/test/accuracy/manifest.json         抽样参数 + 分场景统计 + 数据源 URL/许可 + SHA-256（无时间戳）
reports/dataset-build/accuracy/selection_report.md      自动生成的抽样明细（中间产物）
```

### 4.2 可复现性

| 机制 | 做法 |
|---|---|
| 随机源 | 不使用有状态的 `random`；层内按 `sha256("42|" + upstream_id)` 升序取前 k |
| 顺序无关 | 抽样结果只依赖 id 集合，与文件行序、目录顺序无关 |
| 配额确定性 | 最大余数法；余数并列时按 `sha256(stratum)` 决出，不依赖字典序以外的状态 |
| 可校验 | `manifest.json` 记录 `testset.jsonl` 的字节数与 SHA-256；`build.py --check` 复算 |
| 上游可审计 | manifest 记录每个上游文件的相对路径、字节数、SHA-256、来源 URL、许可 |

同一份 `dataset/raw/` 重跑得到逐字节相同的单个 `testset.jsonl`；改 seed、改配额或改上游文件都会改变 manifest 中的 SHA-256。

### 4.3 分层与配额

| 场景 | 分层键 | 池 | 配额规则 | 实际选中 |
|---|---|---|---:|---|---|
| 知识问答 | 56 个学科（跨语言对齐过滤后） | en 13,802 | 每学科保底 2，余量按池比例（最大余数） | 125 题（2–3/学科）× en/zh |
| 指令遵循 | 一级约束类型（8 类） | 510 | 按池比例，每类保底 1 | detectable_format 56、keywords 42、length 37、change_case 32、combination 27、startend 25、punctuation 16、content 15 |
| 长上下文 | 语言 × 长度桶 × 任务 | en 700 / zh 294 | (en 4–8K 50 / 8–16K 55 / 16–32K 45) + (zh 4–8K 50 / 8–16K 50)；桶内按任务池比例 | narrativeqa 46、qasper 23、hotpotqa 37、2wikimqa 27、multifieldqa_en 17、dureader 68、multifieldqa_zh 32 |
| 数学与逻辑 | GSM8K 计算步数 5 桶；CMATH 年级 6 层 | 1,319 / 1,098 | 各 125，按池比例，每层保底 1 | GSM8K 9/34/34/27/21；CMATH 19/28/27/14/15/22 |
| 编程 | 数据集 × 题面长度四分位 | 164 / 427 | 各 125，四分位按池比例（MBPP 按新渲染 prompt 的 token 长度分层） | HumanEval 32/31/31/31；MBPP 31/31/32/31 |
| 通用 Agent | 11 个 BFCL 单轮类 | 2,741 | 每类保底 12，余量按池比例 | live_multiple 59、simple_python 30、live_simple 23、irrelevance 22、multiple 21、parallel 20、parallel_multiple 20、simple_java 16、simple_javascript 14、live_parallel_multiple 13、live_parallel 12 |
| 搜索 Agent | 题型（bridge/comparison）× 跳数（2/3/4+） | 7,405 | 6 层按池比例，每层保底 1 | bridge 198（128/51/19）、comparison 52（39/9/4） |
| 多语言 | 6 个学科大类 | 跨语言过滤后 13,552 题 × 10 语 | 25 题配额按大类合格池比例（8/5/5/3/2/2），每题在 10 语各评测一次 | 25 题 × 10 语 |

### 4.4 过滤、去重与互斥规则

1. **结构校验**：选择题必须 4 个选项齐全、答案为 A–D；代码题必须有可执行测试；BFCL 必须有 ground truth（irrelevance 的空列表视为“不调用即正确”）。
2. **跨语言对齐过滤**：知识与多语言的配对/多语行还要求同一 `sample_id` 的 answer 字母一致（知识按 56 学科统计不一致率，任一学科 > 0 即整科剔除；多语言要求 10 语 answer 全一致且每语 4 选项去重后仍为 4 个非空选项），并附加已知上游错误排除表（`abstract_algebra/test/1`）；排除明细写入 manifest 的 `sampling.alignment_excluded`。
3. **依赖过滤**：IFEval 排除 31 条 `language:response_language`（依赖 `langdetect`，破坏纯规则判题）；编码/长文本不依赖第三方包。
4. **长度窗口**：长上下文只保留 4,096–32,768 个 MiniCPM token（含模板），不截断、不拼接，避免“短文本混入长文桶”或“超窗跳过”。
5. **去重**：LongBench 按 `(task, 渲染后 prompt)` 去重，同组全部可接受答案按上游出现顺序 union 进 `judge.answers`；HotpotQA 按 question 去重；GSM8K/CMATH 按题面去重；同一场景内 `upstream_id` 唯一。
6. **跨场景互斥**：知识问答与多语言共用 Global-MMLU 底池，先抽知识 125 题，多语言再从剩余池中选 25 题，保证两场景不重复（同一题的 en/zh 两版属于配对行，不算重复）。
7. **不做的事**：不做“按难度动态配平”、不做拒绝采样、不做模型在环筛选——抽样必须与模型无关，否则 bf16/量化两次评测不可比。

### 4.5 均匀性

- **层间**：配额与池规模成比例（最大余数），小层保底 1–12 条，避免长尾类别完全缺席；
- **层内**：sha256 排序近似均匀随机，且不依赖原始文件顺序（原始文件常按时间/来源排序，直接取前 k 会引入偏差）；
- **可验证**：`selection_report.md` 给出每层池大小与选中数；`build.py --check` 校验行数、id 唯一性、判题族合法性和 SHA-256，并校验跨语言对齐、LongBench 答案 union、MBPP 题面函数名、`judge.max_new_tokens`/`metric` 契约（§7.3）；
- **配对**：知识与多语言采用“同一题多语言”配对（只使用通过跨语言 answer 对齐过滤的行），保证语言间比较不受题目难度差异干扰。

### 4.6 样本量与统计功效

| 层级 | 行数 | p=0.5 时 95% CI 半宽（正态近似/Wald，未做题级聚类修正） |
|---|---:|---:|
| 单场景 | 250 | ±6.2pp |
| 知识单语 | 125 | ±8.8pp |
| 多语言单语 | 25 | ±19.6pp（仅剖面） |
| 8 场景合计 | 2,000 | 宏平均约 ±2.2pp（独立近似） |

- 单场景 250 条可以分辨约 10pp 以上的量化损失；对“bf16 vs 量化”的**同一批样本**，用配对检验（McNemar / 配对 bootstrap）比独立 CI 更敏感；
- **注意**：knowledge 的 250 行实为 125 个题对、multilingual 为 25 题×10 语，行间不独立；上表是行级独立近似，正式报告应按题级聚类 bootstrap，不应把 ±6.2pp 当作双语配对场景的精确 CI。
- 不建议按单语言 25 条做语言排名，只做“跨语言是否整体退化”的剖面结论。

### 4.7 抽样脚本使用指导

| 步骤 | 命令 | 说明 |
|---|---|---|
| 1. 下载原始数据 | `python tool/dataset/accuracy/fetch.py` | 按 `raw_spec.json` 下载到 `dataset/raw/`；标准库实现，换机可先跑 |
| 2. 校验原始数据 | `python tool/dataset/accuracy/fetch.py --verify` | 只读校验字节数 + SHA-256；exit 0 表示齐全 |
| 3. 生成测试集 | `python tool/dataset/accuracy/build.py` | 按 §4.3 分层配额生成单文件 2,000 条 + manifest/yaml/报告 |
| 4. 校验测试集 | `python tool/dataset/accuracy/build.py --check` | 行数、id 唯一性、字段白名单、判题族、SHA-256 |

- 分层键、配额、过滤规则、prompt 模板都在 `tool/dataset/accuracy/build.py`；抽样算法在
  `tool/dataset/accuracy/sampling.py`；上游文件规范在 `tool/dataset/accuracy/raw_spec.json`。
- 修改 seed/配额/过滤条件等价于发布新版本测试集：manifest 与 `testset.jsonl` 的 SHA-256 会变，
  必须重新做 §8 的难度试点，不能与旧版本结果混比。
- 脚本保证总量恒定（8 × 250），任一场景达不到 250 会直接断言失败，不静默降级。
- Windows 把 `.venv/bin/python` 换成 `.venv\Scripts\python.exe`，参数不变。

---

## 5. 样本规格与判题契约

### 5.1 文件布局

```
dataset/test/accuracy/
├── testset.jsonl     # 2,000 行单文件（8 场景 × 250；gitignored）
└── manifest.json     # 抽样/统计/数据源/SHA-256（提交；status=offline-asset）
reports/dataset-build/accuracy/
├── selection_report.md   # 自动生成的抽样明细（中间产物）
├── pilot/                # 参考模型难度试点（§8，非正式结果）
└── archive/              # 旧版本产物（仅追溯）
```

### 5.2 单行 schema

```json
{
  "id": "knowledge_qa:global_mmlu:abstract_algebra/test/1|en",
  "scenario": "knowledge_qa",
  "dataset": "global_mmlu",
  "language": "en",
  "upstream_id": "abstract_algebra/test/1|en",
  "input_tokens": 101,
  "messages": [{"role": "user", "content": "<完整 prompt>"}],
  "judge": {"family": "mc_letter", "answer": "C", "max_new_tokens": 1536},
  "meta": {"sample_id": "abstract_algebra/test/1", "subject": "abstract_algebra",
           "subject_category": "STEM", "stratum": "subject=abstract_algebra|lang=en"}
}
```

- **只保留有实际作用的字段**：场景、数据集来源、语言、上游 id、输入 token 数、prompt、
  判题契约、分场景辅助信息（学科/任务/类别/年级/难度层）；不写构造时间等冗余字段。
- `messages` 已含完整 prompt（含场景模板与指令），runner 不再二次渲染；
- `judge` 是判题契约，判题器只读这份字段 + 模型响应；**每行必须包含 `judge.max_new_tokens`**（§6.1），不再依赖场景默认回退；
- `input_tokens` 由参考 tokenizer（MiniCPM5-1B）对 prompt 内容计数，不含 chat template 控制 token；
- `dataset` 的名称到上游 URL/许可/SHA-256 的映射在 `manifest.json` 的 `datasets` 字段；
- `upstream_id` 可回溯到上游行且**全局唯一**（math_logic 的 GSM8K/CMATH 分别用 `gsm8k/test/<i>`、`cmath/test/<i>` 前缀）；`id` 全局唯一；manifest 记录 `upstream_id_format`。

### 5.3 Prompt 模板（每场景一句话）

| 场景 | 模板要点 |
|---|---|
| 知识 / 多语言 | 4 选 1，要求“只输出字母；若推理，最后写 The answer is <letter> / 答案是 <字母>” |
| 指令 | IFEval 原始 prompt，不加任何系统提示，避免额外约束干扰 checker |
| 长上下文 | LongBench v1 官方 `dataset2prompt` 模板（保留上游 `asconcisely` 原始拼写）；重复 prompt 的全部可接受答案写入 `judge.answers` |
| 数学 | 要求逐步推理，最后一行 `#### <数值>`（GSM8K 官方口径）/ 中文同构 |
| 编程 | HumanEval：官方函数签名 + docstring，要求补全完整函数；MBPP：任务描述 + **必须使用的函数名**（从参考解提取）+ **前 3 条官方测试** + “只输出 Python 代码” |
| 通用 Agent | 文本化函数目录（JSON Schema）+ 用户请求 + “只输出 JSON 调用” |
| 搜索 | 10 段带编号段落 + 问题，要求“用短语回答；适用时回答 yes/no” |

所有场景均为 **0-shot**（长上下文/指令遵循本身就是 0-shot 任务）；温度 0、top_p=1，避免采样噪声。

### 5.4 五个判题族（恰好 5 种）

| 判题族 | 覆盖场景（行数） | 输入契约 | 判定算法 | 指标 |
|---|---|---|---|---|
| `mc_letter` | 知识 250 + 多语言 250 | `answer`（A–D） | 优先匹配 `answer is X` / `答案是 X`，否则取括号外独立字母；与 gold 比 | 0/1 |
| `ifeval` | 指令 250 | `instruction_id_list` + `kwargs` | 官方 `instruction_following_eval` checker；响应为空视为不遵循 | prompt-level strict 0/1（另报 loose / instruction-level） |
| `normalized_match` | 长上下文 250 + 搜索 250 + 数学 250 | `answers[]`、`metric`、`max_new_tokens`（+ CMATH `tolerance`） | `f1`：SQuAD 归一化后 token F1 取最大；`f1_zh`：jieba 分词 + 中文归一化 F1（multifieldqa_zh）；`rouge_zh`：jieba + ROUGE-L F（dureader）；`numeric_exact`：抽取最后一个数字，支持分数/百分数/千分位/货币符号，1e-6 容差；CMATH 非整数 gold 追加 `tolerance=relative_1pct`（1% 相对容差），GSM8K 保持精确且不带该字段 | F1 均值（0–1）或 0/1 |
| `code_exec` | 编程 250 | `test_code`、`prompt_prefix`、`entry_point`、`timeout_s`、`max_new_tokens` | 抽取代码块 → 拼接官方 prompt 前缀 → 子进程执行单测（6s、CPU/句柄/文件大小限制）；MBPP 的 `entry_point` 必非空且出现在题面中 | pass@1 |
| `tool_call` | 通用 Agent 250 | `ground_truth`（函数名 + 参数可接受值）、`functions`、`max_new_tokens` | 解析 JSON/JSONL 调用；函数名匹配 + 每个参数命中可接受值；多调用无序匹配；irrelevance 要求空调用 | 0/1 |

判题器自检（`tool/system/validate_judges.py`，含 negative controls）：把参考答案/官方代码/ground truth 本身喂给判题器，**1,750 条正例全部通过**，且错字母、错误代码、错误数值、空 IFEval 响应等负例全部被拒绝，输出 `RESULT: PASS`（见 §10 附录 B）。

---

## 6. 精度测试评估方法

### 6.1 运行协议

| 项 | 约定 |
|---|---|
| 输入 | 直接使用 jsonl 的 `messages`，不重渲染、不截断 |
| 解码 | temperature=0、top_p=1、不重复惩罚；每条请求独立，不使用 prompt cache 复用（避免“缓存命中”影响可比性） |
| 生成长度 | 知识/多语言 1,536；指令 1,024；长上下文 32–128（按任务官方 maxlen）；搜索 32；数学 512；编程 768；Agent 256（均已写入每行 `judge.max_new_tokens`，无场景默认回退） |
| 重试 | 网络/服务错误重试 1 次；重试仍失败记 `error`，该样本计 0 并在报告中单列错误率 |
| 汇总 | 每个模型独立 run_id；逐样本记录响应、判分结果、token 用量、时延；聚合只能由逐样本记录复算 |
| 禁止 | 不按结果挑样本、不事后调 prompt、不用模型自评/LLM judge |

### 6.2 指标定义

- **单样本**：0/1（mc、指令、数学、编程、Agent）或 F1∈[0,1]（长上下文、搜索）；
- **场景分**：`mean(sample score)`；同时报 pass 率（F1 场景）与格式失败率（无法解析/空响应）；
- **总分**：8 场景等权宏平均（每个场景先算场景分），避免长上下文/数学行数多寡影响权重；
- **分组**：按语言（en/zh/other）、数据集、难度层（长度桶/学科/类别）出分，用于定位退化来源。

### 6.3 量化损失分析

对同一批 2,000 条样本，分别在 bf16 与量化模型上各跑一次，逐样本配对：

| 统计量 | 定义 | 用途 |
|---|---|---|
| Δ（pp） | `Acc_bf16 − Acc_quant`（场景分同理） | 主结论：绝对精度损失 |
| 相对损失 | `Δ / Acc_bf16` | 跨场景比较（避免高分区小差异被夸大） |
| 配对 bootstrap | 对 2,000 个样本对重采样 10,000 次，取 Δ 的 2.5%/97.5% 分位 | Δ 的 95% CI |
| McNemar | 2×2 表（bf16 对/量化错 vs 反之）精确检验 | 单场景 Δ 是否显著 |
| Holm 校正 | 8 个场景同时检验时控制族错误率 | 多重比较 |
| 噪声基线 | bf16 重复跑 2 次（同 prompt、温度 0） | 提供“非量化因素”的波动下限（服务/内核差异） |

报告模板（示例格式，非本次结果）：

```
场景             n    bf16    quant    Δ(pp)   95%CI        p(Holm)   判定
knowledge_qa    250   0.512   0.487    -2.5    [-5.4, +0.3]  0.21     无显著差异
...
总体（宏平均）  2000   ...     ...      ...     ...           ...      ...
```

判定口径建议：`p<0.05（Holm）` 且 `Δ` 超过噪声基线 2 倍，才标记为“显著退化”；否则只报点估计与 CI，不宣称退化。

### 6.4 失败与边界处理

| 情况 | 处理 |
|---|---|
| 空响应 / 无法解析 | 该样本计 0；格式失败率单独统计，若某场景 >20% 认为“模型不适配该输出格式”，需在报告中标注 |
| 超时 | 单请求超时（如 300s）记错误，重试 1 次；长上下文超窗（>模型窗口）不计入分母，记 `skipped(over_window)` |
| 判题器异常 | 不允许静默吞掉：记 `judge_error`，该样本按 0 计并计入错误率，同时保留响应原文 |
| 服务错误（OOM/507 等） | 不盲目重试；记录错误码，该 cell 标 `incomplete`，不与其他 cell 合并 |

### 6.5 时间预算（参考本机 1B 模型实测）

| 场景 | 250 条预计耗时 | 说明 |
|---|---:|---|
| 长上下文 | ≈2–2.5 h | 2.97M prompt token，按约 330 tok/s 预填估算 |
| 知识 + 多语言 | ≈20 min | 平均 1.4–3.1 s/条（生成 200–400 token） |
| 指令 + 数学 + 编程 | ≈16 min | 平均 0.7–1.8 s/条 |
| Agent + 搜索 | ≈3 min | 输出很短（<30 token） |
| **合计/模型** | **≈3 h** | bf16 + 量化两次约 6 h；可用并发/批处理加速，但不改变样本 |

（估算基于本机 oMLX + MiniCPM-V-4.6 试点；不同硬件/后端差异较大，仅用于排期。）

---

## 7. 数据来源与校验

### 7.1 目录

```
dataset/raw/                     # 上游原始文件与工具资产（只读，gitignored）
├── global_mmlu/{en,zh,fr,de,es,pt,ru,ja,ko,ar,hi,vi}/test.parquet
├── ifeval/ifeval_input_data.jsonl + instruction_following_eval/  # 官方 checker
├── longbench_v1/data.zip + data/*.jsonl + prompts/*.json
├── gsm8k/main/test-*.parquet
├── cmath/cmath_test.jsonl
├── humaneval/openai_humaneval/test-*.parquet
├── mbpp/sanitized-mbpp.json
├── bfcl/BFCL_v4_*.json + possible_answer/BFCL_v4_*.json
├── hotpotqa/distractor/validation-*.parquet
├── tokenizer/MiniCPM5-1B/tokenizer.json   # 参考 tokenizer（input_tokens）
└── nltk_data/{punkt.zip,punkt_tab.zip,tokenizers/}  # 官方 IFEval checker 依赖
dataset/test/accuracy/           # 精度测试集（jsonl 不提交；manifest 提交；无 testset.yaml，保持隐藏）
reports/dataset-build/accuracy/              # 抽样报告、难度试点等中间产物（不提交）
```

### 7.2 数据源与许可

> 上游 URL 与许可以 `dataset/test/accuracy/manifest.json` 的 `datasets` 字段（`source_url`/`license`）
> 为准，表内为人工核对副本；抽样配额字段见 manifest `per_scenario`（250）/`scenarios.<name>`
> （`rows`/`datasets`/`languages`/`judges`）。**内容风险**：上游数据集可能含用户贡献内容
> （HotpotQA/LongBench 网页摘录等），仅限本地研究与基准测试用途，不得再分发 raw 数据。

| 数据集 | 上游 | 许可 | 本方案用途 |
|---|---|---|---|
| Global-MMLU | [CohereLabs/Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU) | Apache-2.0 | 知识（125 题 × en/zh）、多语言（25 题 × 10 语） |
| IFEval | [google-research/instruction_following_eval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) | Apache-2.0 | 指令遵循 250 条 + 官方 checker |
| LongBench v1 | [THUDM/LongBench](https://github.com/THUDM/LongBench) | MIT | 长上下文 250 条 + 官方 prompt 模板 |
| GSM8K | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | MIT | 数学 en 125 条 |
| CMATH | [XiaoMi/cmath](https://github.com/XiaoMi/cmath) | 见上游仓库（研究用途） | 数学 zh 125 条 |
| HumanEval | [openai/human-eval](https://github.com/openai/human-eval) | MIT | 编程 125 条 |
| MBPP | [google-research/mbpp](https://github.com/google-research/google-research/tree/master/mbpp) | CC-BY-4.0 | 编程 125 条（sanitized 427 题池） |
| BFCL v4 | [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) | Apache-2.0 | 通用 Agent 250 条 + possible-answer |
| HotpotQA | [hotpotqa/hotpot_qa](https://huggingface.co/datasets/hotpotqa/hotpot_qa) | CC-BY-SA-4.0 | 搜索 Agent 250 条（distractor validation） |

### 7.3 完整性校验

- `dataset/test/accuracy/manifest.json` 对每个上游文件记录相对路径、字节数、SHA-256、来源 URL 与许可；
- 抽样后行数校验：单个 `testset.jsonl` 共 2,000 行、每场景 250 行、id 唯一、判题族合法、prompt 非空、`input_tokens` 为正整数；
- `build.py --check` 额外校验契约：知识 125 对 en/zh answer 相等、无错位学科/已知错误行；多语言 25 组 × 10 语 answer 全相等且每行 4 个选项互异；长上下文 `judge.answers` 从 raw 复算为完整 union（本版 7 行受益）；MBPP `entry_point` 非空且出现在题面中；每行 `judge.max_new_tokens` 存在且与 §6.1 一致；`metric` 取值合法（`f1`/`f1_zh`/`rouge_zh`/`numeric_exact`）；
- 判题器校验：`tool/system/validate_judges.py` 的 1,750 条 gold 正例 + negative controls 全部通过（`RESULT: PASS`，§10 附录 B）；
- 数据源全部来自公开仓库/模型卡；本地下载仅使用 HF 镜像与 GitHub raw，未做任何内容改写（仅按上述规则排除/联合上游行）。

### 7.4 构造与移植指导（GitHub → 新设备）

**提交与不提交**：仓库提交 `tool/dataset/accuracy/`（获取 + 构造逻辑 + `raw_spec.json`）与
`dataset/test/accuracy/manifest.json`；**不提交** `dataset/raw/`（约 232 MB 上游文件）
和 `dataset/test/accuracy/testset.jsonl`（派生数据）。新设备按以下 5 步重建，结果与发布版本逐字节一致。

| 步骤 | 命令（Linux/macOS） | 说明 |
|---|---|---|
| 1. 环境 | `./modelbench setup` + `.venv/bin/pip install -e ".[dev,datasets,eval]"` | 构造需要 `pyarrow`/`tokenizers`；判题自检另需 `eval` |
| 2. 下数据 | `.venv/bin/python tool/dataset/accuracy/fetch.py` | 默认 `hf-mirror.com`；`--mirror` 可切 HF 官方；`--only` 只下子集；`--dry-run` 预览 |
| 3. 校验数据 | `.venv/bin/python tool/dataset/accuracy/fetch.py --verify` | 只读 size + SHA-256，exit 0 表示齐全 |
| 4. 构造集 | `.venv/bin/python tool/dataset/accuracy/build.py` | 生成 `dataset/test/accuracy/testset.jsonl` + manifest + 报告（不生成 testset.yaml） |
| 5. 校验集 | `.venv/bin/python tool/dataset/accuracy/build.py --check` | 行数/id/字段/判题族/SHA-256 |

Windows 把解释器路径换成 `.venv\Scripts\python.exe`，参数不变；无法装 `pip` 的 uv 环境用
`uv pip install --python .venv/bin/python -e ".[dev,datasets,eval]"`。

**工具位置**：

| 文件 | 作用 |
|---|---|
| `tool/dataset/accuracy/raw_spec.json` | 可提交的 raw 规范：50 个文件的 path/url/bytes/sha256 + 解压/生成规则 |
| `tool/dataset/accuracy/fetch.py` | 仅标准库：下载、`size+SHA-256` 校验、解压 LongBench/NLTK、写包标记 |
| `tool/dataset/accuracy/sources.py` | 数据集元信息（上游链接/许可）与 spec 读取 |
| `tool/dataset/accuracy/sampling.py` | 确定性哈希抽样 + 分层最大余数配额 |
| `tool/dataset/accuracy/prompts.py` | prompt 模板与 gold 答案归一化 |
| `tool/dataset/accuracy/build.py` | 8 场景构造：单 jsonl + manifest + yaml + 抽样报告 |
| `tool/common/fs.py` / `tool/common/tokenizer.py` | 路径/JSONL/SHA-256；参考 tokenizer 与 `input_tokens` |
| `tool/system/judges.py` | 5 个确定性判题族 |
| `tool/system/validate_judges.py` | gold 答案判题自检 |
| `tool/system/pilot.py` | 参考模型难度试点（调用 oMLX） |

约定：

- **确定性**：seed=42 + `sha256(seed|upstream_id)` 排序 + 最大余数配额；同一份 raw 重建得到
  逐字节相同的 `testset.jsonl`（可用 manifest 中的 SHA-256 校验）。
- **无时间戳**：manifest 与样本行只保留来源、场景、语言、`input_tokens`、判题契约等有实际作用的字段。
- **路径可迁移**：本机是工作区 `dataset/`；部署到 Linux 服务器时把仓库 `dataset/` 映射为 `/dataset/`
  （即 `/dataset/raw`、`/dataset/test/accuracy`），仓库内统一使用 `dataset/` 根，脚本无需改路径。
- **失败不静默**：raw 缺失/SHA 不符时 `fetch.py` 非 0 退出；任一场景不足 250 条时
  `build.py` 直接断言失败，不会生成残缺测试集。

---

## 8. 参考模型难度试点（MiniCPM-V-4.6，12 条/场景）

### 8.1 方法

- 模型：本机 oMLX 上的 `MiniCPM-V-4.7-1B`（当前 v1.1 testset，2026-11-10 重跑）；
- 样本：每场景从 250 条中按分层轮转取 12 条（覆盖各层，非随机挑简单题）；
- 协议与正式测试一致（同 prompt、同判题、温度 0）；
- 目的：验证“小模型不会全错/全对”，并暴露判题与格式问题；**样本量小，结果不是正式精度结论**。

### 8.2 结果（v1.1 testset，2026-11-10）

| 场景 | n | pass / mean | 难度带 | 平均生成 token | 说明 |
|---|---:|---|---|---:|---|
| 知识问答 | 12 | 0.583（7/12） | 可用 | 219 | v1.1 抽样后中英分层 |
| 指令遵循 | 12 | 0.667（8/12） | 可用 | 203 | 失败均为确实违反约束（非判题错误） |
| 长上下文 | 12 | 0.566（F1 均值，3/12 pass） | 可用 | 14 | 短答案；4–8K/8–16K/16–32K 均有覆盖 |
| 数学与逻辑 | 12 | 0.917（11/12） | **天花板** | 242 | GSM8K 5/5，CMATH 6/7——偏易，bf16 对比时区分度可能不足 |
| 编程 | 12 | 0.250（3/12） | 可用 | 97 | HumanEval 2/8，MBPP 1/4 |
| 通用 Agent | 12 | 0.667（8/12） | 可用 | 44 | 失败集中在并行多调用（只输出 1 个调用） |
| 搜索 Agent | 12 | 0.308（F1 均值，2/12 pass） | 可用 | 3 | 短答案，部分答案与 gold 部分重叠 |
| 多语言 | 12 | 0.750（9/12） | 可用 | 440 | 分层轮转覆盖多个语种（非语言均衡切片）；单语种生成偏长 |

**结论：7/8 场景落在 15%–85% 可用带；`math_logic` 0.917 触发天花板判定（>0.85）。**
数学侧对 1B 模型偏易，后续 bf16-vs-量化对比若需在数学场景保留区分度，可考虑换更难题池
（如 MATH-500 子集）或把该场景标注“饱和风险”。试点只用于校准，不能作为最终分数；
正式测试需 250 条 × 2 个模型版本。

> **历史记录（v1.0，2026-11-09，`MiniCPM-V-4.6` @ 旧 testset）**：旧表数字已被上方 v1.1
> 重跑取代，仅存于 git 历史与 `reports/dataset-build/accuracy/pilot/`（中间产物）。

### 8.3 试点暴露的问题与设计调整

| 问题 | 调整 |
|---|---|
| 选择题模型会先长篇推理，短 `max_tokens` 会在给出答案前截断 | 选择题 `max_new_tokens=1536`；判题优先匹配“answer is X / 答案是 X”，且要求最终行给字母 |
| HumanEval 模型常重复函数签名，导致判题时缺少 `from typing import List` 等导入 | 判题按官方协议**始终拼接 prompt 前缀**（重复 def 无害，导入保留） |
| 并行工具调用模型按“一行一个 JSON”输出，不是数组 | 判题器增加 JSONL 多对象解析（仍要求每个对象含 name/arguments） |
| 长上下文中文文档偏短，16–32K 桶没有中文 | 抽样时固定“16–32K 仅英文”，中文集中在 4–16K，并在报告中注明 |
| 试点耗时受模型 prompt cache 影响（长文重复请求命中） | 正式 run 使用独立 prompt、关闭缓存复用；时间预算按 token 数估算 |

### 8.4 试点局限

- 每场景 12 条，95% CI 很宽（例如 0.5 时约 ±28pp），只能判断“是否在带内”，不能估计精确分数；
- 仅一个模型（MiniCPM-V-4.6 bf16），未覆盖其他小模型与量化版本；
- 未做重复测量，无法给出模型自身波动。

---

## 9. 已知局限与后续工作

1. **搜索 Agent 是单轮检索模拟**：固定 10 段检索结果，测“证据定位/聚合”，不测多轮 search 工具循环；若后续 runner 支持工具环境，可在同一样本集上增加“允许调用 search/lookup”的协议。
2. **BFCL 判题为结构化近似**：按 possible-answer 的函数名 + 参数可接受值匹配，未启用官方 `ast_checker` 的语言相关 AST 归一（Java/JS 的表达式解析、类型强制转换差异可能造成少量误判）；建议正式跑分时同时输出官方 checker 对照。
3. **中文长文上限**：LongBench v1 中文任务最长约 1.7 万 token，16–32K 只有英文；若需中文超长文，可补充 LongBench v2 中文子集或构造中文 NIAH（会引入新数据源）。
4. **单场景 250 的统计分辨率**：约 ±6.2pp；对 <5pp 的量化损失，单场景可能不显著，应看 8 场景宏平均与配对检验。
5. **多语言每语 25 条**：只做剖面，不做单语排名。
6. **数据污染**：Global-MMLU/HotpotQA/GSM8K 均为公开数据，不能排除预训练见过；本方案用于**同模型 bf16 vs 量化**的相对比较，污染对两版模型近似同向，不影响主结论，但不能与公开榜单横比。
7. **备选切换条件**（若正式测试暴露地板/天花板）：
   - 编程全量 <10% → 改用 MBPP-only（或 CRUXEval 执行预测）并保留 HumanEval 作参考；
   - 数学 >85% → 增加 MATH-500 中难度子集（Level 1–3）替换部分 GSM8K；
   - 知识 <15% → 知识场景改用 CMMLU（中文）+ MMLU（英文）或降低学科难度分层；
   - 长上下文 F1 <0.1 → 缩短到 2–16K 或改用 L-Eval 短子集。

---

## 10. 附录

### 附录 A：抽样统计（由 `selection_report.md` 自动生成）

- 总行数 2,000；每场景 250；语言 en 1,400 / zh 350 / other 250；
- 长上下文 token：min 4,317 / median 9,411 / mean 11,874 / max 32,176，总 prompt token 2.97M；20 组重复 prompt 中 18 组答案不同，入选行中 7 行恢复被丢弃的参考答案；
- 知识 125 题覆盖 56 个学科（每学科 2–3 题），学科大类分布 STEM 80、Humanities 60、Social 54、Medical 30、Business 16、Other 10（行数）；跨语言对齐过滤剔除 240 个配对（world_history 237、virology 2、abstract_algebra 1）；
- 多语言 25 题覆盖 6 个学科大类，10 种语言各 25 行；跨语言过滤在知识场景已占用的 125 题之外剔除 365 个 sample_id（答案不一致 122、选项坍塌 243）；
- `testset.jsonl` 的字节数/SHA-256、各场景 `input_tokens` 统计与 `sampling.alignment_excluded` 明细见 `dataset/test/accuracy/manifest.json`（SHA-256 以 manifest 为准，不在文档中写死）。

### 附录 B：判题器自检结果

| 判题族 | 验证方式 | 结果 |
|---|---|---|
| `mc_letter` | gold 字母必过、错误字母必不过（500 条） | 500/500 |
| `normalized_match`（F1） | gold 答案自评 F1=1.0（500 条） | 500/500 |
| `normalized_match`（数值） | gold 数值必过（250 条） | 250/250 |
| `code_exec` | HumanEval canonical solution + MBPP 官方代码跑单测（250 条） | 250/250 |
| `tool_call` | BFCL ground truth 自身匹配（250 条，含 irrelevance 空调用） | 250/250 |
| `ifeval` | 选中样本的指令 id 全部在官方注册表（0 个未知 id）；空响应必不过（250/250） | 通过 |
| negative controls | 错字母、错误代码（含 `sys.exit(0)`）、错误数值、空 IFEval 响应均被拒绝 | 10/10 通过 |

`tool/system/validate_judges.py` 整体输出 `RESULT: PASS`。

### 附录 C：文件清单

| 文件 | 作用 |
|---|---|
| `tool/dataset/accuracy/fetch.py` | 原始数据下载/校验/解压（仅标准库） |
| `tool/dataset/accuracy/build.py` | 确定性抽样器（读 raw → 单文件 2,000 条 + manifest/报告） |
| `tool/dataset/accuracy/raw_spec.json` | 可提交的上游文件规范（URL/size/SHA-256） |
| `tool/system/judges.py` | 5 个判题族实现（无 LLM、无网络） |
| `tool/system/validate_judges.py` | gold 答案判题自检 |
| `tool/system/pilot.py` | 参考模型难度试点（调用 oMLX） |
| `dataset/test/accuracy/testset.jsonl` | 单文件测试集（2,000 条） |
| `dataset/test/accuracy/manifest.json` | seed/配额/数据源/SHA-256 |
| `reports/dataset-build/accuracy/selection_report.md` | 抽样明细（中间产物） |
| `reports/dataset-build/accuracy/pilot/pilot_summary.json` | 试点汇总（中间产物） |

---

**一句话总结**：用 9 个公开数据源、5 个确定性判题族、8×250 条分层抽样，构造了一套面向 1B 级小模型、中英双语、可逐字节复现、难度经参考模型校准的精度测试样本集；正式评估时对 bf16 与量化模型各跑一遍，按场景做配对比较与显著性检验。
