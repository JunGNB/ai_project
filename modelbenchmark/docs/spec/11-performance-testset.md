# 本地小模型性能测试：样本集设计与抽样方案

> 版本：v1.1（2026-11-10）
> 范围：**只覆盖性能测试的样本选择**（TTFT / TG TPS / TPOT / 峰值内存），不涉及精度判分与能力结论。
> **固定一套样本、全设备/全模型复用**：不为单个模型/机器重新构造，也不做运行时裁剪/补齐；
> 被测对象：本地小模型（MiniCPM 级，1B 量级，bf16 约 2.4 GB，最大上下文 262,144）。
> 产物：**8 场景 × 1K/8K/32K × 10 = 240 条**样本、24 个 cell、5 个上游数据集、可逐字节复现的构造脚本与 manifest。
> 上游数据：`dataset/raw/`；产物：`dataset/test/performance/`；构造器：`tool/dataset/performance/`。

---

## 0. 摘要

| # | 场景 | 上游数据集（本地证据） | 1K/8K/32K 构造方式 | 语言 | 每档 target |
|---|---|---|---|---:|---:|
| 1 | 知识问答 | LongBench v2 `Single-Document QA`（116→106 可用） | 同文档字符前缀 + 原题/选项 | en | 10 |
| 2 | 指令遵循 | IFEval（541 条，510 可用） | 目标指令 + 同集指令元数据参考块（旋转起点，单份不复制） | en | 10 |
| 3 | 长上下文 | LongBench v2 `Long Structured Data`（32） | 同文档字符前缀 + 原题/选项 | en | 10 |
| 4 | 数学与逻辑 | GSM8K（1,319） | 目标题 + 同集已解例题（旋转起点） | en | 10 |
| 5 | 编程 | LongBench v2 `Code Repository`（48） | 同代码仓库文档字符前缀 + 原题/选项 | en | 10 |
| 6 | 通用 Agent | BFCL v4（3 类 800 条 / 642 唯一函数） | 目标请求 + target-first 函数目录 | en | 10 |
| 7 | 搜索 Agent | LongBench v2 `Multi-Document QA`（94） | 同多文档字符前缀 + 固定 search/lookup schema | en | 10 |
| 8 | 多语言 | Global-MMLU（10 语种 × 14,042） | 目标题 + **同语种**题库参考块（旋转起点） | 10 语种 | 10 |

**六个关键决策**

1. **5 个数据集覆盖 8 个场景**：LongBench v2 一库四用（知识/长文/编程/搜索）；Global-MMLU 承担多语言；指令、数学、通用 Agent 各 1 个。新增依赖为 0（17 个上游文件 + 2 个 tokenizer 资产，全部已在 `dataset/raw/`，`raw_spec.json` 为准）。
2. **每格 10 条、共 240 条**：性能指标是连续量，10 条/格给中位数与 IQR 已足够；相比 20 条/格方案把测量请求从 480 降到 240，24 个 cell 的对比结构不变。
3. **长度是唯一自变量**：同一 `target` 贯穿 1K/8K/32K，题面/指令/请求逐字符一致，只有上下文长度变化；文档场景用字符前缀保证 1K⊂8K⊂32K。
4. **长短分离、按形状选数据集**：长文场景用原生长文（不拼接），短任务场景用"同集参考块"扩展并标 `constructed=true`；避免"同一段文本反复出现"的病态 prefill 形状。
5. **确定性抽样**：`seed=42` + `sha256(seed|upstream_id)` 排序 + 分层最大余数配额，与文件行序无关；同一份 raw 重跑逐字节一致，SHA-256 写入 manifest。
6. **不判分、不横比**：`score` 恒 null；样本只用于 TTFT/TG TPS/内存曲线，任何"能力/准确率"结论都不从本集推导。

---

## 1. 目标、约束与非目标

### 1.1 硬约束

| # | 约束 | 本方案落实 |
|---|---|---|
| C1 | 必须覆盖 8 个场景 | 8 个场景各 10 个 target，无缺席 |
| C2 | 每场景必须覆盖 1K/8K/32K | 每场景 × 3 档 = 30 行；同 target 三档 |
| C3 | 长度可在范围附近波动 | 目标 1024/8192/32768 上下文：声明值按 §4.4 校准预算模型定标（构造 ±5%），运行期实际 token ±10% 入带 |
| C4 | 数据集不要太多、能复用则复用 | 5 个数据集；LongBench v2 复用 4 个场景 |
| C5 | 可复现、可审计 | 固定 seed、无时间戳、manifest 记录逐文件 SHA-256 与抽样明细 |
| C6 | 负载形状真实 | 长文用原生文档；短任务用同集单元按序拼接后取字符前缀扩展（末尾单元可能被截断，见 §9）；tools 只放完整函数对象 |
| C7 | 面向本地小模型、单请求串行 | 单请求 `chat/completions`；OSL 50；不测并发/批处理 |

### 1.2 非目标

- 不判分、不报告准确率；**不与任何公开榜或官方 benchmark 数字横比**；
- 不测多轮工具循环、Web 浏览、代码仓库级修改、多模态；
- 不测并发吞吐（单实例串行是项目的性能测量前提）；
- 精度测试是另一套独立方案（见 [12-accuracy-testset.md](12-accuracy-testset.md)），两套数据集与结论互不推导。

---

## 2. 业界开源数据集调研（性能视角）

### 2.1 筛选标准

性能测试与精度测试的选型门槛不同：**判分难度不是门槛**（`score` 恒 null），真正的门槛是下面四条。

1. **长度可控**：能否稳定产出 1024 / 8192 / 32768 token 的 prompt，且三档同源（可裁前缀或可生成）；
2. **负载形状真实**：长文档 prefill、短任务 + 参考块、tools JSON、多语言文本，不能是重复串或无标点长串；
3. **可离线复现**：整包可下载、无 gated、无联网、无滚动版本；
4. **许可清晰**：MIT / Apache-2.0 / CC-BY 优先，研究许可需标注。

> 性能不判分意味着"老数据集/已污染数据集"仍然可用：污染改变的是正确率，不改变 token 数、attention 计算量与显存占用——而后者才是本方案要测的量。

### 2.2 调研总表

下表覆盖 8 个场景的 26 个主流开源数据集（完整 35 个见调研底稿）。"长度能力"指能否原生/低成本得到 1K/8K/32K。

| 场景 | 数据集 | 语言 | 规模（约） | 长度能力 | 许可 | 性能适用性 |
|---|---|---|---|---:|---|---|
| 知识 | MMLU-Pro | en | 12,032 | 原生 <1K；拼接可达 8K/32K | MIT | 一般：只能做 1K 短基线，8K/32K 靠堆叠 |
| 知识 | C-Eval / CMMLU | zh | 13,948 / 11,528 | 原生 <1K；拼接 | CC BY-NC-SA / Apache-2.0 | 一般：中文短基线；C-Eval 非商用 |
| 知识 | **LongBench v2** | en | 503 | **原生 8K–2M 词，可裁前缀** | Apache-2.0 | **适合**：天然三档同源长文（本方案采用） |
| 指令 | **IFEval** | en | 541 | 原生 <1K；参考块扩展到 32K | Apache-2.0 | **适合**：短指令 + 参考块，三档最省（采用） |
| 指令 | IFBench | en | 300 | 同上 | ODC-BY-1.0 | 适合但池小：做参考/交叉 |
| 指令 | Multi-IF | 8 语 | 4,501 会话 | 多轮累积到 8K/32K | Apache-2.0 | 适合多轮形状，非单请求 |
| 长文 | LongBench v1 | en+zh | 4,750 | 原生 1K–20K+；32K 需拼接 | MIT | 适合 1K/8K，32K 吃力 |
| 长文 | RULER | en | 13 任务 × 6 长度档 | **可按 token 生成 4K–128K** | Apache-2.0 | 很适合长度校准；合成任务、形状单一 |
| 长文 | ∞Bench | en+zh | 3,900 | 原生 100K+，截到 32K | MIT | 适合 32K 档；需截断 |
| 长文 | L-Eval | en+zh | 2,043 QA | 原生 3K–200K | 混合 | 适合；许可需逐子任务确认 |
| 长文 | NIAH | en | 任意 | **可配置 1K–1M** | MIT | 适合做长度阶梯；只有检索形状 |
| 数学 | **GSM8K** | en | 1,319 | 原生 <1K；例题块扩展 | MIT | **适合**：短题 + 已解例题（采用） |
| 数学 | MATH-500 | en | 500 | 原生 <1K；解答 2–3K | MIT | 适合长输出；池偏小 |
| 数学 | AIME | en | 90–900 | 原生 <1K | 题目版权 MAA | 一般：池太小 |
| 数学 | CMATH | zh | 1,700 | 原生 <1K | CC BY 4.0 | 适合中文短题 |
| 编程 | HumanEval / MBPP | en | 164 / 427 | 原生 <1K | MIT / CC-BY-4.0 | 适合 1K；需多题堆叠 |
| 编程 | BigCodeBench | en | 1,140 | 原生 1–3K | Apache-2.0 | 适合 1K 代码 prompt |
| 编程 | **LongBench v2** `Code Repository` | en | 48（≥120K 字符） | **原生 100K–16M 字符** | Apache-2.0 | **适合**：代码仓库长上下文 prefill（采用） |
| 编程 | LiveCodeBench | en | 滚动 400–1,055 | 原生 0.5–2K | 混合 | 一般：版本滚动，锁 release 才可复现 |
| Agent | **BFCL v4** | en | 2,694 唯一题 / 642 函数 | 原生 1–4K；目录可扩到 32K | Apache-2.0 | **适合**：tools JSON 最真实（采用） |
| Agent | τ²-bench | en+6 语 | 3 域 × 50 任务 / 900 calls | 原生 2–8K，多轮漂移 | MIT | 适合多轮；需 user simulator |
| Agent | ToolBench | en | 126K 实例 | 1–4K；目录可拼到 8K+ | Apache-2.0 | 适合大目录 prefill；API 不稳定 |
| 搜索 | **LongBench v2** `Multi-Document QA` | en | 94（≥120K 字符） | 原生 100K–7M 字符 | Apache-2.0 | **适合**：多文档 + tools（采用） |
| 搜索 | HotpotQA | en | 7,405 | 原生 2–4K；段落堆叠 | CC BY-SA 4.0 | 适合离线；换源增加一个数据集 |
| 搜索 | 2WikiMultiHopQA / MuSiQue | en | 192K / 24K | 同上 | Apache-2.0 / CC BY 4.0 | 适合；与 HotpotQA 能力重叠 |
| 搜索 | GAIA / BrowseComp | en | 466 / 1,266 | 题面 <1K，正文需联网 | gated / 密文 | **不适合**：离线不可复现 |
| 多语 | **Global-MMLU** | 42 语 | 590K（每语 ~14K） | 原生 <1K；**同语种题库可扩到 32K** | Apache-2.0 | **适合**：`sample_id` 跨语种对齐（采用） |
| 多语 | MMMLU | 14 语 | 196K | 原生 <1K；拼接 | MIT | 适合：语种较少，作交叉 |
| 多语 | Belebele | 122 语 | 490K | 原生 <1K（短文段） | CC BY-SA 4.0 | 适合多语冒烟；8K 需拼接 |
| 多语 | XNLI / Flores-200 | 15 / 200 语 | 112K / 3,001 | 句子级 | CC BY-NC / CC BY-SA | 一般：只适合分词/填充料 |

### 2.3 三条筛选结论

1. **"长度可控性"是第一属性**：热门短题集（MMLU-Pro、GSM8K、HumanEval）题面 200–800 token 且不可配置；塞进 8K/32K 只能同源堆叠，会引入重复文本的病态 attention 形状。真正能精确给出三档的是 **LongBench v2**（字符前缀）与 **RULER/NIAH**（按 token 生成）。→ 长文优先"可裁/可生成"，短题只做 1K 或改用参考块扩展。
2. **负载形状按 prefill 类型分工**：长文档 = 长 prefill + 短输出（决定 TTFT 与峰值内存）；短题 + 参考块 = 中等 prefill；tools JSON = 模板/tokenizer 开销；多轮 = KV 增长。本方案固定 OSL=50，因此**数据集只影响 prefill 形状**，decode 形状由测量协议统一控制。
3. **离线可复现 + 许可清晰优先于"新"**：滚动榜单（LiveCodeBench 月更）、gated/联网数据集（GAIA、BrowseComp）、需模拟器的多轮环境（τ²-bench）不进入本 CI；本方案 5 个数据集全部整包可下、无 gated、许可明确。

---

## 3. 数据集选型

### 3.1 场景 × 数据集矩阵

| 场景 | 主数据集 | 长度构造 | tools | 语言 |
|---|---|---|---|---|
| 知识问答 | LongBench v2：Single-Document QA | 同文档前缀（原生长文） | 无 | en |
| 长上下文 | LongBench v2：Long Structured Data | 同文档前缀（原生长文） | 无 | en |
| 编程 | LongBench v2：Code Repository | 同文档前缀（原生长代码） | 无 | en |
| 搜索 Agent | LongBench v2：Multi-Document QA | 同文档前缀 + 固定 2 工具 schema | search/lookup | en |
| 多语言 | Global-MMLU（10 语种） | 目标题 + 同语种题库参考块 | 无 | fr/de/es/pt/ru/ja/ko/ar/hi/vi |
| 指令遵循 | IFEval | 目标指令 + 同集指令参考块 | 无 | en |
| 数学与逻辑 | GSM8K | 目标题 + 同集已解例题 | 无 | en |
| 通用 Agent | BFCL v4（simple/multiple/parallel） | 目标请求 + 函数目录前缀 | 目标自身函数 | en |

### 3.2 逐场景选型依据

**3.2.1 知识问答 → LongBench v2 `Single-Document QA`**

- 候选：MMLU-Pro、Global-MMLU、C-Eval/CMMLU、LongBench v2。
- 为什么选：本场景要的是"长文档 + 问题"的 prefill 形状，LongBench v2 的 175 条 Single-Document QA 文档中位 31.6 万字符，`≥120K 字符` 过滤后仍有 116 条、按 context 去重 107 条、按 question 去重 106 条——足够 10 个 target，且 32K 档无需拼接。题面为 4 选 1，短输出，真实（开卷问答）。
- 不选：MMLU-Pro/Global-MMLU/C-Eval 都是短选择题（<1K），8K/32K 只能同源堆叠，prefill 形状失真，且会新增数据集；Global-MMLU 在精度侧已承担"知识 + 多语言"，性能侧无需重复。
- 局限：本场景是"开卷长文 MCQ"，不是闭卷短问答；报告不得用它下"知识能力"结论。

**3.2.2 长上下文 → LongBench v2 `Long Structured Data Understanding`**

- 候选：LongBench v1、RULER、∞Bench、L-Eval、NIAH、LongBench v2。
- 为什么选：该子域是结构化长数据（表格/知识图谱/事件序列），原生文档中位 57 万字符（≥120K 过滤后 32 条），三档前缀切片天然单调；结构化文本的 token 密度与纯自然语言不同，正好补一个"结构化长 prefill"形状。
- 不选：LongBench v1 中位 2–15K token，32K 需拼接；RULER/NIAH 是合成任务、只有检索一种注意力形状，且易被针对性优化；∞Bench 100K+ 只能截断到 32K；L-Eval 许可混合。把它们放进主集会让"场景"退化成"长度阶梯"。
- 局限：仅 32 条主池，10 个 target 的抽样空间有限，但性能不要求题型覆盖。

**3.2.3 编程 → LongBench v2 `Code Repository Understanding`**

- 候选：HumanEval、MBPP、LiveCodeBench、BigCodeBench、LongBench v2 代码仓库。
- 为什么选：性能关注 32K 级代码上下文的 prefill。代码仓库文档中位 194 万字符（≥120K 过滤后 48 条），是本地唯一能稳定切 32K 的真实代码语料；单请求只读，不需要执行/生成，测量链路简单。
- 不选：HumanEval/MBPP 题面 100–400 token，32K 需多题堆叠且语义失真；LiveCodeBench 版本滚动、跨 run 不可比；BigCodeBench 单题 1–3K，到 32K 仍需堆叠。代码生成形状由统一 OSL=50 覆盖，无需在本场景单独承担。
- 局限：只测"读代码"的 prefill，不测代码生成质量/长输出；报告需标注。

**3.2.4 搜索 Agent → LongBench v2 `Multi-Document QA` + 固定工具 schema**

- 候选：HotpotQA、2WikiMultiHopQA、MuSiQue、LongBench v2 多文档、GAIA（排除）。
- 为什么选：多文档 QA 的原生多文档上下文（125 条，≥120K 过滤后 94 条）就是 RAG/搜索 Agent 的典型 prefill 形状；`tools` 固定 search/lookup 两个 schema，测"带工具定义的单请求"，不执行工具循环——与性能模式单请求约束一致。
- 不选：HotpotQA/2Wiki/MuSiQue 需要把 10 段检索结果拼进 prompt（可做，但要新增数据集与语料管线）；GAIA/BrowseComp 需联网或 gated，离线不可复现。
- 局限：单请求，不测多步工具循环与 KV 增长；"搜索"能力结论不成立。

**3.2.5 多语言 → Global-MMLU（10 语种）**

- 候选：Global-MMLU、MMMLU、Belebele、XNLI、Flores-200。
- 为什么选：Global-MMLU 的 10 个测试语种各 14,042 行，`sample_id` 集合跨语种完全一致（scout 已全量比对），因此可以"同一题在 10 种语言里各有本地素材"；性能侧按语种取 1 个 target，用**同语种其他题目**拼参考块达到 1K/8K/32K，保持整段 prompt 单语，测的是各语种 tokenizer 效率与 prefill。
- 不选：MMMLU 只有 14 语种且本地仅 6 种；Belebele 是阅读问答（换能力）、8K 仍需拼接；XNLI 句子级、Flores 是翻译语料，长度与形状都不合适。
- 局限：多语言档是"同语种题库扩展"而非原生长文；10 个 target = 10 语种 × 1 题（对角线设计），不用于跨语种能力/难度比较；语言选择与精度侧一致（fr/de/es/pt/ru/ja/ko/ar/hi/vi），覆盖拉丁、西里尔、CJK、阿拉伯、天城文五种书写系统。

**3.2.6 指令遵循 → IFEval**

- 候选：IFEval、IFBench、Multi-IF、FollowBench。
- 为什么选：541 条（排除 31 条依赖 `language:` 的样本后 510 条）、8 类可验证指令，池足够 10 个 target 且能按类型分层；纯 prompt、无答案依赖，最适合"目标指令 + 同集参考块"扩到 32K。
- 参考块格式：每条参考 item 渲染为 `INSTRUCTION {k}: ids=<instruction_id_list JSON> kwargs=<kwargs JSON>` 后接 prompt，目标自身不入参考块；509 条单份合计约 43K token（cl100k，`reference_items=509`），已超过 32K 校准下界 31,094，所以 32K 档是**一次不重复的完整遍历**，1K/8K/32K 只是同一字符串的递增前缀（`reference_pool_cycles=1`）。加入 `ids`/`kwargs` 元数据是为了让单份素材超过 32K 而不复制同一段文本（审计 P-01 修复）。
- 不选：IFBench 仅 300 条且精度侧已用；Multi-IF 是多轮、长度随轮次漂移，单请求测量不匹配；FollowBench 需额外判据。
- 局限：参考块按字符前缀切分，末尾可能落在某条指令的 prompt 中间（这是截断、不是复制，见 §9）；只测"读参考材料 + 执行指令"的形状，不判是否遵守约束。

**3.2.7 数学与逻辑 → GSM8K**

- 候选：GSM8K、MATH-500、AIME、CMATH。
- 为什么选：1,319 条、MIT、题面短、自带解题过程；"目标题 + 已解例题"能扩展出任意长度且保持"题-解"结构真实（不是重复同一段）。10 个 target 与例题池不重叠（构建断言）。
- 不选：MATH-500 仅 500 条且 LaTeX 多、扩展 32K 需要更多条；AIME 池太小；CMATH 中文小学题、与多语言场景重叠。精度侧用 MATH-500 是因为要判难题，性能侧不需要。
- 局限：数学只作为"短题 + 已解例题"的形状探针，不测推理能力。

**3.2.8 通用 Agent → BFCL v4（3 类）**

- 候选：BFCL v4、τ²-bench、ToolBench、API-Bank。
- 为什么选：BFCL 的 `function` 字段是真实 JSON Schema，可直接作为 `tools`；3 个本地请求文件（simple_python 400、multiple 200、parallel 200）共 800 条、642 个唯一函数，足够 10 个 target 与 32K 级函数目录。目标是"函数目录 + user request"的真实 function-calling prefill。
- 不选：τ²-bench 需要 user simulator 与环境、长度随轮次漂移；ToolBench 依赖不稳定真实 API；API-Bank 类别覆盖弱。multi-turn BFCL 需要环境，不纳入。
- 局限：单请求、不执行工具调用；镜像里函数目录是 **target-first 前缀**（先放目标自身函数），32K 档也放不下全量 642 函数（约 88.5K token），所以是"大目录"而非"完整目录"。

### 3.3 数据集复用与取舍

- **LongBench v2 承担 4/8 场景**：一个 `data.json`（465 MB）同时提供单文档、多文档、代码仓库、结构化长数据四种原生长文，是"数据集最少"的关键；代价是 4 个场景共享同一语料家族、场景独立性有限，报告必须按 `domain` 标注，不得据此下能力结论。
- **Global-MMLU 同时服务精度与性能**：精度侧 42 语种 × 100 配对题；性能侧 10 语种 × 1 target。数据同一份，抽样与用途独立。
- **与精度侧数据集不完全相同**：性能 `coding` 用代码仓库长文，精度 `coding` 用 HumanEval + MBPP（代码生成）；性能 `knowledge_qa` 用 LongBench v2 长文，精度用 Global-MMLU 短题。两套测试目标不同，**不要求数据集一致**，但报告必须写清"场景—数据源—测量口径"。
- **未选 RULER/NIAH 做长度阶梯**：合成任务长度可编程是优点，但只有"检索"一种注意力形状，容易让 1K/8K/32K 曲线过于平滑。若有需求（如 64K/128K 扩展），建议作为**独立 testset** 增加，不混入本 240 行。

---

## 4. 长度构造与 token 口径

### 4.1 三档定义与接受边界

| 档位 | 目标内容 token | 接受下界 `floor(0.95t)` | 接受上界 `floor(1.05t)` |
|---|---:|---:|---:|
| 1k | 1,024 | 972 | 1,075 |
| 8k | 8,192 | 7,782 | 8,601 |
| 32k | 32,768 | 31,129 | 34,406 |

- 每行按**整数 token**校验；不满足即构建失败（不放宽配额）。
- `tokens = messages 内容 token + tools JSON token`；同 target 三档的 `tools` 逐字节一致。
- 边界是"内容 token"，不含 chat template 控制符（见 §4.4）。

### 4.2 五种构造模式

| 模式 | 适用场景 | 做法 | `constructed` |
|---|---|---|---|
| `document_prefix` | 知识问答、长上下文、编程 | 取同一文档的字符前缀，后面接原题面；三档是 1K⊂8K⊂32K 前缀链 | false |
| `doc_prefix_with_tools` | 搜索 Agent | 同上 + 固定 search/lookup 两个 schema | false |
| `reference_rotation` | 指令遵循 | 目标指令原样 + 同集其他指令（每条带 `ids`/`kwargs` 元数据）按固定起点拼接；目标自身不入参考块；单份 509 条约 43K token（cl100k），不复制不循环 | true |
| `example_rotation` | 数学与逻辑 | 目标题 + 同集其他题的"题 + 解题过程"；起点 step=41 旋转；排除 10 个目标 | true |
| `function_catalog` | 通用 Agent | 目标请求 + target-first 的 whole-function 目录（自身函数→同类→其余），只放完整 JSON 函数 | true |
| `same_language_bank` | 多语言 | 目标题 + **同语种**其他题目/选项按固定起点拼接；排除目标 `sample_id` | true |

短任务场景的 8K/32K 用同集其他样本的单元（指令 / 已解例题 / 题目）按固定顺序拼接，再取**字符前缀**：单元内容完整，但**切点可能落在最后一个单元内部**（半条指令/半道例题/半道题干，见 §9），且**不复制**任何整段文本——IFEval 32K 已改为单份约 43K token（cl100k）的「元数据 + prompt」参考块，`reference_pool_cycles` 恒为 1；general_agent 只放 whole-function JSON，不存在半 JSON。

### 4.3 三档同源一致性

- 同一 `target_id` 的 1K/8K/32K 三条：题面/指令/用户请求**逐字符一致**，`tools` 逐字节一致，只有上下文长度不同；
- 文档场景：`1K 内容 ⊂ 8K 内容 ⊂ 32K 内容`（字符前缀链，构建时断言）；
- 参考块场景：8K 是 1K 的超集、32K 是 8K 的超集（同一起点向后追加）；
- 这样"长度效应"不会被"换题"混淆，TTFT/内存随长度的曲线可归因。

### 4.4 token 口径与 runner 对齐

- 参考 tokenizer：**cl100k_base**（OpenAI 的业界通用口径，`dataset/raw/tokenizer/cl100k_base/`：官方 `cl100k_base.tiktoken` blob + 由 `derive_cl100k.py` 确定性派生的 `tokenizer.json`，SHA-256 双双钉死在 `raw_spec.json`）。**不绑定任何模型**：声明值是模型无关的通用计数，不同目标模型的可信度一致；
- **`tokens` 是内容 token**，不含 chat template 控制符、不含任何特殊 token（cl100k 无 BOS 计入问题）；
- **校准预算模型**：不同 tokenizer 对同一文本计数不同（代码/数学更密、非拉丁文更松），所以每条样本的声明值不是裸 bucket 值，而是按场景/语种校准的预算
  `budget = (bucket − RESERVE) / INFL`，
  其中 RESERVE 是**固定渲染预留**（带 tools 行 270、纯文本行 90，实测于 240-cell 校准 run `20261110T112641Z-8349`），tools JSON 本身作为内容随 INFL 一起膨胀、不再单独扣除（旧公式的双重扣除曾把 1K tool 档饿到带外：实际 796 < 带下沿 921）；
  INFL 是**内容类型膨胀系数**（代码 1.07、数学 1.15、长文 1.24、知识 1.05、指令 1.04、Agent 1.00–1.03、多语言按语种 0.50–0.94），为实测中位数校准值（通用内容类型属性，不与特定模型绑定；来源与公式写进 manifest `budget_model` 与 `deviations.tokenizer-generic-cl100k`）。效果：声明值收敛到让**目标模型实际渲染 token** 居中落进运行期 ±10% ISL 带（离线逐行模拟 95% 入带）；
- 运行期追加的 decode-guard 指令已在固定渲染预留中一并覆盖（`deviations.decode-guard-suffix`），不再把渲染 prompt 推出带外；
- tools 的 JSON 文本按其序列化 token 计入（`tools_tokens`），模板 wrap 由固定 215 系数预留；
- manifest 的 `tokens_semantics`/`budget_model` 字段写死该口径，报告引用时必须同时引用。

### 4.5 实际输入大小记录、聚合带与窗口判定

- **固定样本**：本集是**唯一一套**性能样本，所有设备/模型/后端复用；构造时用 cl100k 把内容
  token 控制在**校准预算** ±5%（如 32K 档：长文场景约 27,065±5%，知识/指令场景约 32,731±5%），
  校准方向使目标模型实际渲染 token 落入 ±10% ISL 带。**不为单个模型重新
  构造/裁剪/补齐**；32K target 固定 **32,768**，与 1K/8K 一样是普通 bucket，**无特殊处理**。
- **运行期 actual 记录（不驱动跳过）**：请求前用**目标模型** `POST /v1/tokenize` 对最终
  messages+tools（含 chat template/decode guard）计数，写入
  `request.tokenized_prompt_tokens` 与 `context_meta.isl_tolerance = {target, low, high, actual, ok}`。
  该计数只用于两件事：**(a)** 记录 actual 并标记入带/出带；**(b)** 模型窗口判定。
  **不会因 tokenizer/actual 偏差提前跳过**：所有能测的样本都产生 measured 记录。
- **聚合带（±10%，所有档位对称，区间权威表见 [20-methodology.md](20-methodology.md) §5）**：**只对入带样本求
  平均**；出带样本仍保留在 `records.jsonl`/`samples.csv`（`isl_band=false`），但不进
  `aggregates.overall/cells` 的均值。计数：`n_isl_in_band`/`n_isl_out_of_band`
  （run.json outcome 与 aggregates 行都有）；in-band 样本过少时报告须标记
  `insufficient_samples`，不得静默当作有效结论。
- **为什么不是"构造 ±5% 就够"**：运行时由目标模型 tokenizer 计数，且不同模型 tokenizer 对
  同一文本可差很多（实测同一 32K 内容跨模型 **12,810–54,964** token）。这正是引入 §4.4
  预算模型的原因——INFL 系数把最常见的跨 tokenizer 差异（内容类型/语种密度）吸收进构造
  侧，使多数模型直接入带；残余差异仍靠运行时记录 actual + ±10% 入带聚合兜底。固定一套
  样本 + 校准构造 + 运行时 actual 记录，既保持"同一套样本"的可移植性，又不把不同 token
  工作量混进同一均值；跨模型只能作"同一套样本"对比，actual 分布必须一起报告（见
  [13-testset-portability.md](13-testset-portability.md)）。
- **窗口判定（唯一跳过理由之一）**：用**实际值**校验 `tokenized_prompt_tokens + max_tokens 50 +
  context_safety_margin 128 > model_context_window` → 该样本跳过（not-testable，写入
  `run.json.outcome.skipped`），**不截断**；tokenize 失败时回退声明值+242 判定并 warning。
  请求失败（网络/服务错误）同样按 skipped/error 记录，不影响其余样本。
- **32K 档**：target 仍是 32,768，声明值为校准预算（按场景 25,712–34,367 不等）；是否跳过只
  取决于**目标模型实际窗口**（actual+50+128 是否放得下），不做 32K 专属构造/特殊阈值。校准
  后膨胀场景（长文/数学等）的声明值低于裸 bucket，在 32,768 窗口上的可测性更好；知识/
  指令等 INFL≈1 场景上界仍可能超窗（构建期 ±5% 上界只是**数据集构建验收边界**，不是运行时
  门槛）。
- 结论：报告需单列 not-testable 行数、`n_isl_in_band`/`n_isl_out_of_band` 与 actual 分布，
  不得把未测行当 0，也不得把出带样本算进平均。

---

## 5. 抽样方法与均匀性

### 5.1 总体流程

```
dataset/raw/<dataset>/                    上游原始文件（只读，gitignored）
   │  0. fetch：按 tool/dataset/performance/raw_spec.json 校验 size+SHA-256
   │  1. 读取 + 结构校验（字段/选项/答案/函数 schema）
   │  2. 过滤（长度下限、依赖排除、题面预算、非法行）
   │  3. 去重（context sha1 / question / 目标素材互斥）
   │  4. 分层（子域/类别/题型的池大小）
   │  5. 分层配额（最大余数，小层保底 1）
   │  6. 层内确定性抽样（sha256("42|"+upstream_id) 升序取前 k）
   │  7. 三档构造 + token 边界校验（÷ 失败即中止）
   ▼
dataset/test/performance/testset.jsonl     240 行 = 8 场景 × 3 档 × 10
dataset/test/performance/manifest.json     抽样明细 + 统计 + 数据源 SHA-256（无时间戳）
reports/dataset-build/performance/selection_report.md  逐层池/选中/跳过明细（中间产物）
```

> 生成路径以 `tool/dataset/performance/build.py`（`OUT_REPORT`）为准：`reports/dataset-build/
> performance/selection_report.md`。仓库里现存的 `reports/dataset-build/performance-selection_report.md`
> 是旧版平铺遗留，下次重建后由 `performance/` 目录内的报告取代（gitignored 中间产物，可删）。

### 5.2 可复现性机制

| 机制 | 做法 |
|---|---|
| 随机源 | 不使用有状态的 `random`；层内按 `sha256("42|"+upstream_id)` 升序取前 k |
| 顺序无关 | 结果只依赖 id 集合，与文件行序、目录顺序无关 |
| 配额确定性 | 最大余数法；余数并列按 `sha256(stratum)` 决出 |
| 可校验 | manifest 记录 `testset.jsonl` 的字节数与 SHA-256；`build.py --check` 复算 |
| 上游可审计 | manifest 记录每个上游文件的相对路径、字节数、SHA-256、来源 URL、许可 |
| 无时间戳 | 产物只含来源、场景、长度、抽样参数等有效字段，重跑逐字节一致 |

### 5.3 分层与配额

| 场景 | 分层键 | 本地池 | 配额规则 | 实际选中 |
|---|---|---|---:|---|---|
| 知识问答 | domain（唯一题面） | 106 | 层内哈希取 10 | 10 |
| 长上下文 | domain | 31 | 取 10 | 10 |
| 编程 | domain | 47 | 取 10 | 10 |
| 搜索 Agent | domain | 94 | 取 10 | 10 |
| 指令遵循 | 8 类首要指令类型 | 510 | 最大余数 + 每类保底 1 | 10 |
| 数学与逻辑 | 无（同质池） | 1,319 | 层内哈希取 10 | 10 |
| 通用 Agent | 3 类请求（simple/multiple/parallel） | 400/200/200 | 4 / 3 / 3 | 10 |
| 多语言 | 语种（10）× `subject_category`（6） | 10 × 14,042 | 6 类配额 2/2/2/2/1/1；层内 10 个不同 subject；每语种 1 题 | 10 |

> 文档 4 场景先按 `sha1(context)` 全局去重，再分别抽 10 个；选中集 40 个 context 互不重复（构建断言）。

### 5.4 过滤、去重与互斥规则

1. **长度过滤**：LongBench v2 只保留 `context ≥ 120,000 字符`（≈ ≥29K token），保证 32K 档可达且无需拼接；题面 token 超过该档 0.6 倍的 target 直接跳过并记 skip 原因。
2. **去重**：文档场景按 `sha1(context)` 去重，knowledge_qa 再按 `question` 去重；同一场景内 `upstream_id` 唯一；跨场景 `upstream_id` 全局唯一。
3. **目标素材互斥**：IFEval 目标指令不入参考块；GSM8K 10 个目标题不入例题块；多语言目标 `sample_id` 不入参考块。
4. **依赖过滤**：IFEval 排除 31 条含 `language:` 指令的样本（依赖第三方语言检测，且与性能无关）。
5. **不做的事**：不做难度配平、不做模型在环筛选、不按"跑得快/慢"挑样本——抽样必须与模型无关，否则跨模型/跨版本不可比。

### 5.5 均匀性论证

- **层间**：配额与池规模成比例（最大余数），小层（IFEval 8 类）保底 1，避免长尾类型完全缺席；
- **层内**：哈希排序近似均匀随机，且不依赖原始文件顺序（原始文件常按来源/时间排序，直接取前 k 会引入偏差）；
- **长度维**：每档固定 10 个 target，且三档同 target，长度是唯一变量；
- **语言维**：多语言 10 语种各 1 题，书写系统覆盖拉丁/西里尔/CJK/阿拉伯/天城文；题目按 6 个 `subject_category` 分层（最大余数 + 每类保底 1：Humanities 2 / STEM 2 / Social 2 / Medical 2 / Other 1 / Business 1）、取 10 个不同 subject，避免全落在单一学科；
- **可验证**：`selection_report.md` 给出每层池大小与选中数；`build.py --check` 校验行数、id 唯一、三档齐全、token 边界、重复与 manifest SHA。

### 5.6 为什么每格 10 条

- 性能指标是连续量，不是比例：TTFT/TG TPS/峰值内存用**中位数 + IQR** 描述，10 条/格足以看出长度趋势与量级差异；这与精度测"正确率"需要的 250 条不同（比例指标才有 ±6pp 的 CI 概念）。
- 24 个 cell × 10 = 240 条 measured 请求，单模型单轮约 3.36M prefill token（80×32K + 80×8K + 80×1K），比 480 条方案省一半机时，便于做多模型/多配置对比。
- 若用于**结论性对比**（bf16 vs 量化、两个模型 A/B），建议 `repeats=3` 并做配对比较（同一 target 三次取中位数）；本集只负责"样本固定"，重复次数由 run 配置控制。
- 10 条的局限：单格中位数仍受热漂移/启动抖动影响，报告应同时给 IQR 与重复次数，不应只看均值。

---

## 6. 样本规格与产物

### 6.1 文件布局

```
tool/dataset/performance/          # 构造器（提交）
├── fetch.py                       # 仅标准库：下载/校验 → dataset/raw/
├── raw_spec.json                  # 可提交规范：17 个上游文件 + 2 个 tokenizer 资产的 path/url/bytes/sha256（tokenizer.json 走 generated 规则）
├── sources.py                     # 数据集元信息（URL/许可/used_by）
├── sampling.py                    # 确定性哈希抽样 + 分层最大余数
└── build.py                       # 8 场景构造 + manifest + yaml + 报告
dataset/raw/                          # 上游原始文件（只读，gitignored；部署机为 /dataset/raw）
dataset/test/performance/
├── testset.jsonl                  # 240 行（gitignored，可重建）
├── manifest.json                  # 抽样/统计/数据源/SHA-256（提交）
└── testset.yaml                   # 测试集定义（提交）
reports/dataset-build/performance/
├── selection_report.md            # 抽样明细（中间产物）
```

### 6.2 单行 schema

```json
{
  "id": "perf-knowledge_qa-1k-000",
  "scenario": "knowledge_qa",
  "bucket": "1k",
  "target_id": "tgt-knowledge_qa-000",
  "dataset": "longbench_v2:Single-Document QA",
  "tokens": 1013,
  "messages_tokens": 1013,
  "tools_tokens": 0,
  "messages": [{"role": "user", "content": "<完整 prompt>"}],
  "tools": null,
  "constructed": false,
  "meta": {
    "workload": "longbench_v2:Single-Document QA (open-book long-doc MCQ)",
    "lang": "en",
    "input_tokens": 1013,
    "upstream_id": "longbench_v2:66fcffd9...",
    "context_sha1": "…"
  }
}
```

- `tools` 只在 `general_agent`（目标请求自身函数 1–3 个）与 `search_agent`（固定 search/lookup）出现；
- `meta` 随场景追加：文档场景 `task/domain`、IFEval `instruction_ids`、GSM8K `grade`、BFCL `category/bfcl_id/catalog_functions/own_functions`、多语言 `subject/subject_category/sample_id`；
- `meta.upstream_id` 全局唯一，用于去重与溯源。

### 6.3 manifest 关键字段

- `schema_version:"performance/v1"`、`rows:240`、`per_scenario:30`、`targets_per_cell:10`、`seed:42`；
- `tokenizer` + `tokens_semantics`（cl100k 内容 token，不含模板/特殊 token）、`budget_model`（公式 + INFL 表 + 37/215 常数 + 验收语义）、`buckets`、`accept_rule`/`accept_bounds`；
- `run_defaults`（`max_tokens:50`、`decode_guard:true`、`warmup_per_model:1`、`warmup_requests:0`、`repeats:1`）；
- `window_requirements`（32K 每请求预留 242 token 与 not-testable 事实）；
- `provenance`：5 个数据集、17 个上游文件 + cl100k blob/派生 tokenizer.json 的 `path/bytes/sha256` + `source_url` + `license`；
- `scenarios.<name>.sampling`：池、去重、跳过原因、配额、tokens 统计；
- `deviations`：与官方口径的偏差（内容 token、10 条/格、前缀切片、Agent 单请求、多语言题库扩展、不判分等）；另显式记录**参考块末尾截断、文档前缀半结构、cl100k 通用口径与预算校准、decode-guard 预留、IFEval 元数据参考块（单份不重复）** 等偏差。

### 6.4 testset.yaml

`dataset/test/performance/testset.yaml` 用可读形式声明 8 个场景 → `perf_jsonl`、`dataset_params.scenario`、`path: dataset/test/performance/testset.jsonl`，并记录 `sampling`/`buckets`/`osl`。它是"配置契约"，不是产物。

### 6.5 校验命令

```bash
.venv/bin/python tool/dataset/performance/build.py --check
.venv/bin/python -m pytest test/test_perf_dataset.py
```

---

## 7. 数据来源、许可与下载

### 7.1 数据源与许可

| 数据集 | 上游 | 许可 | 用途 | 本地文件数 / 体量 |
|---|---|---|---|---:|
| LongBench v2 | [zai-org/LongBench-v2](https://huggingface.co/datasets/zai-org/LongBench-v2) | Apache-2.0 | 知识/长文/编程/搜索（4 场景） | 1 / 465.5 MB |
| IFEval | [google-research/instruction_following_eval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) | Apache-2.0 | 指令遵循参考池 | 1 / 0.2 MB |
| GSM8K | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | MIT | 数学例题池 | 1 / 0.4 MB |
| BFCL v4 | [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) | Apache-2.0 | 通用 Agent（3 个请求文件） | 3 / 0.8 MB |
| Global-MMLU | [CohereLabs/Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU) | Apache-2.0 | 多语言（10 语种；en 作题目选择参考池） | 11 / 51.5 MB |
| cl100k_base 编码数据 | [openai/tiktoken](https://github.com/openai/tiktoken)（OpenAI 官方 blob） | MIT | 通用参考 tokenizer（声明 token 计数） | 2 / 4.4 MB |

合计 **5 个数据集 / 17 个上游文件 + 2 个 tokenizer 资产 / 约 518 MB**（十进制 MB，不含 tokenizer 4.4 MB = 1.7 MB 官方 blob + 2.7 MB 派生 tokenizer.json）；体量**以
`dataset/raw/download_manifest.json` 的逐文件 `size`/`sha256` 为准**（表内为人工核对值，重建后如有
出入以 manifest 为准）；全部整包可下、无 gated、无联网依赖。`global_mmlu` 11 文件 = build 消费的
10 语种 `test.parquet` + en 参考池；`dataset/raw/global_mmlu/` 实存 42 语种目录（fetch 全量下载），
超出部分为冗余资产，未被 build 引用。tokenizer.json 由 `tool/dataset/performance/derive_cl100k.py`
从官方 blob 确定性派生（字节级可复现，SHA-256 钉死在 `raw_spec.json` 的 `generated` 规则中）。

### 7.2 目录

- 仓库根下只有一套数据根：`dataset/raw/`（上游输入）与 `dataset/test/`（产物）；
- 部署到别的机器时直接同步整个 `dataset/`，`raw_spec.json`、`manifest` 与构造器路径都不变；
- 旧 v3 的 480/240 双集布局已删除，不再维护。

### 7.3 下载与校验

```bash
python tool/dataset/performance/fetch.py            # 缺什么下什么（默认 hf-mirror.com）
python tool/dataset/performance/fetch.py --verify   # 只读 size+SHA-256，exit 0 = 齐全
python tool/dataset/performance/fetch.py --dry-run  # 只打印计划
```

- 下载先写 `<目标>.part`，校验 size+SHA-256 通过后原子改名；失败不留下损坏文件；
- 本机现状：19 个文件（17 上游 + 2 个 tokenizer 资产）已全部就绪，`--verify` 通过（exit 0），**无需新增下载**；
- 上游版本冻结在 `raw_spec.json` 记录的 SHA-256；换版本必须重新生成测试集并更新 manifest，不能与旧结果混比。

---

## 8. 复现、验证与接线

### 8.1 五步复现

| 步骤 | 命令（macOS/Linux） | 说明 |
|---|---|---|
| 1. 环境 | `./modelbench setup` + `.venv/bin/pip install -e ".[dev,datasets]"` | 构造需要 `tokenizers`、`pyarrow` |
| 2. 下数据 | `.venv/bin/python tool/dataset/performance/fetch.py` | 已有 raw 可跳过 |
| 3. 校验数据 | `.venv/bin/python tool/dataset/performance/fetch.py --verify` | exit 0 = 19 个文件（17 上游 + 2 个 tokenizer 资产）齐全 |
| 4. 构造集 | `.venv/bin/python tool/dataset/performance/build.py` | 生成 testset/manifest/yaml/报告 |
| 5. 校验集 | `.venv/bin/python tool/dataset/performance/build.py --check` | 行数/id/三档/边界/SHA 复算 |

### 8.2 实测构建结果

> 以 `dataset/test/performance/manifest.json` 为准（数字由构建器复算）。

| 项 | 结果 |
|---|---|
| 行数 / target / cell | 240 / 80 / 24（每格 10） |
| `testset.jsonl` | 12,720,373 B，SHA-256 `f56f3af0ecf84a6b8e845341da053adcf5d8360fd6298bd1a5aee4feaae34442` |
| `manifest.json` / `testset.yaml` | 88,802 B / 1,591 B |
| tokens（按场景/档校准预算，示例） | coding 1k 911 / 8k 7,548 / 32k 30,303；long_context 1k 809 / 8k 6,733 / 32k 27,043；instruction 1k 987 / 32k 32,731；multilingual 8k 按语种 8,677（fr）–16,347（ar） |
| constructed 行 | 120（IFEval 30、GSM8K 30、BFCL 30、多语言 30）；另 120 行为原生文档/请求 |
| IFEval 32K 参考块 | 单份 509 条「元数据 + prompt」≈43K token（cl100k），32K 只切单份；`reference_pool_cycles=1`、重复块 token 0（每行 378–387 个不重复指令单元） |
| 32K 窗口需求 | 运行期按目标模型 tokenize 实际值 + max_tokens 50 + 安全边距 128 判定；声明值为校准预算（25,712–34,367/行不等）；32K target 不特殊处理 |
| 聚合 band | 所有档位 target ±10%（区间见 [20-methodology.md](../spec/20-methodology.md) §5 权威表）。**只对入带样本求平均**；出带样本保留但不进均值（`n_isl_in_band`/`n_isl_out_of_band`） |
| 构建耗时 / 确定性 | 单次约 108 s；连续两次构建 `testset.jsonl` 逐字节一致 |
| 验证 | `build.py --check` PASS；`pytest test/test_perf_dataset.py` 25 passed（含 cl100k/预算模型/校准边界用例）；dry-run 24 cells × 10 |

### 8.3 与 runner 的接线

- 默认配置已切换到本集：`datasets.testset_dir: dataset/test`、`datasets.raw_dir: dataset/raw`；
- 默认 `scenarios.yaml` 的 8 个场景直接读 `dataset/test/performance/testset.jsonl`，场景 8 为 `multilingual`；
- 旧 `config/perf_v1*.yaml` 过渡 profile 已删除，验证与运行使用默认配置：

```bash
./modelbench validate --offline
./modelbench run --dry-run
```

### 8.4 质量门禁

- [ ] 240 行；8 场景 × 3 档 × 10；`id` 唯一；80 个 `target_id` 各 3 条；
- [ ] 每行 `tokens == messages_tokens + tools_tokens`，整数在 ±5% 边界内；
- [ ] 同 target 三档题面/指令/请求逐字符一致、`tools` 逐字节一致；
- [ ] 文档 4 场景 1K⊂8K⊂32K 前缀链；选中 context 无重复；
- [ ] `tools` 仅两个 Agent 场景、JSON 可解析、`parameters` 是合法 JSON Schema；
- [ ] BFCL 类别 4/3/3、目录 whole-function、10/10 自身函数三档可见；
- [ ] 目标素材互斥（IFEval / GSM8K / 多语言）成立；
- [ ] manifest 与逐行复算一致；连续两次构建 `testset.jsonl` 逐字节相同。

### 8.5 运行预算

| 项 | 数值 |
|---|---|
| measured 请求 | 240（每档 80；每场景 30） |
| 24 cells | 8 场景 × 3 档；每格 10 条 |
| prefill 内容 token | 3,358,836（实测合计；理论下界 80×32,768 + 80×8,192 + 80×1,024 = 3,358,720） |
| decode token（OSL=50） | 上限 12,000，实际按 `finish_reason` |
| 缓存 | 每条 measured 前清 hot+SSD，`cached_tokens==0` |
| 预热/重复 | 模型级 warmup 1 次；`repeats=1`（结论性对比建议 3） |

---

## 9. 已知局限与后续工作

1. **场景独立性有限**：4/8 场景共用 LongBench v2，且都是"长 prefill + 短输出"；报告必须按 `domain` 标注，不得把 4 个场景当 4 份独立语料证据。
2. **短任务场景的 8K/32K 是构造**：IFEval/GSM8K/BFCL/多语言的参考块扩展只保证长度与结构真实，不承诺语义连贯；只测形状，不测能力。
3. **Agent 单请求**：不测多步工具循环、KV cache 随轮次增长；τ²-bench 与 Multi-IF 这类多轮形状可作为后续独立 testset。
4. **多语言是"同语种题库"**：不是原生多语长文，10 个 target 为对角线设计，不用于跨语种排名。
5. **编程只测读代码**：代码生成/长输出形状由统一 OSL=50 覆盖，本场景不单独承担。
6. **32K 档的可测性受模型窗口限制**：32,768 窗口模型上膨胀场景（长文/数学等，声明预算低于裸 bucket）可测性更好；INFL≈1 场景上界仍可能超窗；不截断，运行期以目标模型 tokenize 实际值判定，构建期 ±5% 上界只是验收边界。
7. **10 条/格的方差**：单次测量受热漂移影响，结论性对比应 `repeats≥3` 并做配对统计。
8. **不可与公开榜横比**：内容 token 口径、10 条/格、前缀切片、Agent 单请求、OSL=50 都与官方 benchmark 不同。
9. **参考块末尾可能被截断**：IFEval/math_logic/multilingual 的参考块是同集单元按序拼接后取**字符前缀**，最后一条单元可能只保留半条指令/半道例题/半道题干（多语言则可能切在题干/选项内部）；这是“截断”而非“重复”，不改变长度变量的结论，但 prompt 不以完整单元结尾（manifest `deviations.reference-block-prefix-slice`）。
10. **文档前缀可能切在结构内部**：long_context/coding 的字符前缀可能落在 JSON 对象/数组/字符串内部或打开的代码围栏内（long_context 实测 19/30 行括号不平衡），不保证文档结构在切点闭合（manifest `deviations.document-prefix-mid-structure`）。
11. **构造 tokenizer 为通用 cl100k + 预算校准**：声明值用 cl100k_base（业界通用口径）计数，不绑定模型；按场景/语种的经验膨胀系数 INFL 把声明值定标到校准预算，使多数目标模型的实际渲染 token 入带。系数是通用内容类型属性（代码/数学密、非拉丁文松），校准来源与公式记录在 manifest `budget_model` 与 `deviations.tokenizer-generic-cl100k`。
12. **运行期 decode guard 改写生成任务**：runner 会追加“忽略上文问题、改写 ≥250 词技术总结”指令（37 cl100k token，已在构造预算中预留，guard 不再把渲染 prompt 推出带外）；生成内容不是数据集任务，只用于测 decode 阶段吞吐（manifest `deviations.decode-guard-suffix`）。
13. **actual token 与聚合带（不是跳过理由）**：目标模型 tokenizer 对同一 32K 内容可计出 12,810–54,964 token；运行时**不因 tokenizer/actual 偏差跳过**，所有能测样本都 measured，只把 actual 记录进 `records.jsonl` 并按 ±10% 带筛选聚合（出带保留、不进均值）。因此实际 measured 数一般仍为 240，只有窗口不足/请求失败才 skip；报告须同时给出 actual 分布、`n_isl_in_band`/`n_isl_out_of_band`、not-testable 数，不能把"入带样本数"当成"measured 数"。跨系统比较准入见 [13-testset-portability.md](13-testset-portability.md)。

**备选切换条件**：若后续需要"长度可编程的纯净阶梯"，新增 RULER/NIAH 独立 testset（64K/128K）；若需要多轮 KV 增长，新增 τ²-bench 单场景 testset；两者都不混入本 240 行。

---

## 10. 附录

### 附录 A：抽样统计

**规模与 token**：240 行 = 8 场景 × 3 档 × 10；24 个 cell 各 10 行；80 个 target 各 3 行。
逐档 tokens（n=80）：1k min 997 / median 1,024 / mean 1,026.0 / max 1,074；
8k 8,106 / 8,192 / 8,190.7 / 8,252；32k 32,704 / 32,768 / 32,768.8 / 32,841。
`constructed=true` 120 行（IFEval / GSM8K / BFCL / 多语言各 30）。产物 `testset.jsonl` 13,273,802 B、
SHA-256 `5bba5d7266642ab7b66e6c0af34b5c8fa094c6a13fa1cc436b1c4f087f858890`。
IFEval 32K 参考块单份 509 条（`reference_pool_cycles=1`），每行 378–387 个不重复指令单元；
对 10 行 32K 检查参考区最后 20,000 字符不再出现在前文、1,024/2,048 字符重复窗口均为 0，
最长自然重复子串 256–318 字符（上游共享样板句，非块复制）。

**逐场景抽样明细**

| 场景 | 池 → 候选（去重/过滤） | 配置 | 选中 | 跳过 |
|---|---|---|---|---:|---|
| knowledge_qa | 175 → 106（context 去重 −9、question 去重 −1） | 层内哈希 | 10 | 32k 不可达 1 |
| instruction_following | 510（排除含 `language:` 的 31 行） | 8 类：detectable_format 2 / keywords 2 / 其余 6 类各 1 | 10 | 0 |
| long_context | 33 → 31（context 去重 −1） | 层内哈希 | 10 | 0 |
| math_logic | 1,319 | 层内哈希；例题旋转 step=41 | 10 | 0 |
| coding | 50 → 47（context 去重 −1） | 层内哈希 | 10 | 0 |
| general_agent | 800（400/200/200） | 4 simple / 3 multiple / 3 parallel | 10 | 1k 不可达 2 |
| search_agent | 125 → 94 | 层内哈希 | 10 | 题面过长 3 |
| multilingual | 14,042（en 参考池） | 6 类配额 Business 1 / Humanities 2 / Medical 2 / Other 1 / STEM 2 / Social 2；10 个不同 subject | 10 | 0 |

多语言 10 个 target 的（语种, subject, category）：fr marketing/Business、de formal_logic/Humanities、
es moral_scenarios/Humanities、pt medical_genetics/Medical、ru nutrition/Medical、
ja miscellaneous/Other、ko college_chemistry/STEM、ar electrical_engineering/STEM、
hi econometrics/Social Sciences、vi us_foreign_policy/Social Sciences。

**函数目录规模**（general_agent，whole-function 前缀）：1k 档 3–7 个函数、8k 档 54–59、32k 档 229–234；
10/10 target 的自身函数在三档均可见（构建硬断言）。

**验证**：`build.py --check` → `PASS rows=240 targets=80 cells=24 sha=5bba5d726664`；
`pytest test/test_perf_dataset.py` → 16 passed；`./modelbench validate --offline`
→ 240 samples / 8 scenarios；`run --dry-run` → 24 cells × 10、estimated measured requests ~240。

### 附录 B：文件清单

| 文件 | 作用 |
|---|---|
| `tool/dataset/performance/fetch.py` | 下载/校验 19 个文件 = 17 上游 + 2 个 tokenizer 资产（仅标准库；派生文件由 generator 规则按需重建） |
| `tool/dataset/performance/raw_spec.json` | 可提交上游规范（path/url/bytes/sha256） |
| `tool/dataset/performance/build.py` | 8 场景构造 + manifest + yaml + 报告 |
| `tool/dataset/performance/sampling.py` | 确定性抽样 + 分层配额 |
| `dataset/test/performance/testset.jsonl` | 240 行测试集（可重建） |
| `dataset/test/performance/manifest.json` | 抽样/统计/SHA-256（提交） |
| `dataset/test/performance/testset.yaml` | 测试集定义（提交） |
| `reports/dataset-build/performance/selection_report.md` | 抽样明细（中间产物） |

---

**一句话总结**：用 5 个公开数据源、5 种确定性构造模式、8×3×10 分层抽样，构造了一套面向本地小模型、长度可控、三档同源、可逐字节复现的性能测试样本集；它只回答"在 1K/8K/32K 负载下 TTFT、TG TPS、峰值内存是多少"，不回答"模型答得对不对"。
