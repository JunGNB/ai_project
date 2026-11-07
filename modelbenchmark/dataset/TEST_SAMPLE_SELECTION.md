# 本地模型测评：测试集设计（v3）

> 本文定义两套彼此独立的测试：**性能基准**（快不快）与**精度测试**（行不行）。
> 样本由 `tool/dataset/` 下的构造器从 `dataset/raw/` 的上游原始文件确定性生成，
> 产物落在 `dataset/test/{performance,accuracy}/`，统计与溯源写在各自的 `manifest.json`。
>
> 设计约束：8 个场景（知识问答、长上下文、编程、搜索 Agent、中文长文、指令遵循、数学与逻辑、通用 Agent）；
> 性能分 1K/8K/32K 三档，每场景每档 **10 条**与 **20 条**两套；精度不分性能档位
> （long_context 的 4K–32K 长度分层是**抽样口径**，不是性能档位）。
>
> **2026-11 现场口径**（本文数字与状态以两份 manifest 为准）：
> - 性能侧 8 场景中的 `chinese_long_context` 是 LongBench v1 `dureader` 中文单语长文；
>   原 `multilingual` 场景名已更名，精度侧真正的 42 语种场景仍叫 `multilingual`；
> - 精度 6,589 行是**离线数据资产（`manifest.status=offline-asset`）**：不接入 runner、
>   没有判题实现，`judges.*.implementation_status` 全部为 `contract-only`；
> - 性能 480 行 / 24 cells / 160 targets；快速集 240 行是严格子集；
>   精度 **6,589 = 250×6 + 889（BFCL 11 类）+ 4,200（Global-MMLU 42×100）**。
>   逐场景明细见 `manifest` 的 `sampling`、`stats`、`totals`。

---

## 1. 设计目标

用固定、可复现、可审计的样本回答两个独立问题：

1. **性能**：在给定输入长度与负载形状下，TTFT、TG TPS、TPOT、端到端时延、峰值内存是多少；
2. **精度**：在固定判题契约下，8 项能力各自正确率是多少（**当前只到数据/契约层**，
   运行器未接入，见 §4.1）。

两套样本**分开构建、分开运行、分别汇总**，结论不互相推导。性能样本不判分，精度样本不测速。

### 1.1 不变量

| # | 不变量 |
|---|---|
| 1 | 性能：8 场景 × 3 档 × 20 = **480**；快速集 × 10 = **240**；两套来自同一批 target，不可当独立证据 |
| 2 | 精度：**6,589 行** = 6 场景 × 250 + general_agent **889** + multilingual **4,200**；不截断 prompt，long_context 只保留 4K–32K 并按 100/99/51 分层（抽样口径，非性能档位）；但它是**离线资产**，不接入 run、不可当作“已就绪的精度测试” |
| 3 | 精度状态契约：`manifest.status=offline-asset`、`manifest.runner=not wired`、每个 `judges.*.implementation_status=contract-only`；实现判题前不得声称可跑精度 |
| 4 | 性能同一 `target_id` 的三条行：题面/指令/用户请求**逐字符一致**，只有上下文长度不同 |
| 5 | 性能 `tokens = messages_tokens + tools_tokens`，按**整数** token 落在目标长度 ±5% 内：`floor(0.95×target) ≤ tokens ≤ floor(1.05×target)`（1K 下界 972，不是 972.8） |
| 6 | 精度每行都携带其判题契约所需的全部输入（选项/参考答案/指令 id 与 kwargs/隐藏测试/候选答案/检索语料），且有全局唯一的 `meta.upstream_id` |
| 7 | `tools` 必须是合法 JSON Schema；`messages` 中的 JSON（如函数目录）必须是完整可解析 JSON，按 whole-function 填充、绝不截断半个函数 |
| 8 | 溯源按测试集过滤：每个上游文件的 `source_url`、许可以及**逐文件 SHA-256**写入对应 manifest（performance 5 数据集 / 7 文件；accuracy 8 数据集 / 70 文件） |
| 9 | 固定 seed、确定性抽样：同一份 raw 重建得到逐字节相同的样本；选中的 target 之间全量 context sha1 / 题面无重复（long_context 1K body 的上游 boilerplate 碰撞单独记录，不算抽样重复） |

---

## 2. 场景 × 数据集矩阵

| 场景 | 性能（形状探针） | 精度（能力 + 判题契约） |
|---|---|---|
| 知识问答 | LongBench v2 `Single-Document QA`：长文档 + 选择题 | MMLU-Pro：14 学科、3–10 选项选择题（0-shot CoT，`Answer: X`）→ `mc_letter` |
| 长上下文 | LongBench v2 `Long Structured Data Understanding`：结构化长文档 | LongBench v1 `narrativeqa / qasper / hotpotqa / 2wikimqa` → `qa_f1` |
| 编程 | LongBench v2 `Code Repository Understanding`：代码仓库长上下文（只读，不生成） | LiveCodeBench 最新 release 窗口（v5+v6，2024-09~2025-04）：**150 条 stdin + 100 条 functional** 隐藏测试 → `code_pass1` |
| 搜索 Agent | LongBench v2 `Multi-Document QA` + 固定 search/lookup 工具 schema（单请求） | HotpotQA distractor 规范子集 + search/lookup 工具 + 语料 → `qa_f1` |
| 中文长文 | LongBench v1 `dureader`（中文单语长文） | Global-MMLU（本地 **42 语种 × 100 配对题** = 4,200，en 100 行标 `is_en_reference`）→ `mc_letter` |
| 指令遵循 | IFEval：目标指令 + 同集指令参考块 | IFBench：官方 checker（strict/loose × prompt/instruction）→ `ifbench` |
| 数学与逻辑 | GSM8K：目标题 + 同集已解例题 | MATH-500：`\boxed{}` + 数学等价判定 → `math` |
| 通用 Agent | BFCL v4：目标请求 + **target-first 的 whole-function 函数目录前缀**（非全量目录） | BFCL v4 **11 个 single-turn 类别**（含 irrelevance、无 multi-turn）：官方 possible-answer AST 匹配 → `bfcl_ast` |

> 精度两列是**数据 + 判题契约**，不是可运行结果：截止 2026-11，runner 未接入 accuracy，
> 所有 judge 只有契约、没有实现（见 §4.1/§4.2）。

### 2.1 选型原则

1. **能力对齐**：精度集直接测目标能力；性能集只要求负载形状（长度、上下文结构、工具目录大小）真实。
2. **可程序判定**：精度每题都有确定性判题契约，不引入 LLM judge / rubric。
3. **难度非饱和**：知识用 MMLU-Pro（多选项降低猜测）、数学用 MATH-500、编程用 LiveCodeBench；GSM8K/HumanEval 只用于性能形状。
4. **时间窗与抗污染**：精度优先选择**最新可得的 held-out / 时间切片**数据；性能不判分，可使用较旧语料，但必须记录时间窗。任何数据集都记录本地文件窗口与 SHA-256，报告注明“不可与公开榜横比”。
5. **本地可得 + 许可清晰**：全部来自本地 `dataset/raw/`，许可与 SHA-256 进入 manifest。
6. **一场景一主数据集**：性能与精度可以使用不同数据集（如 coding 性能=代码仓库读取，精度=代码生成），但报告按“场景-数据源-能力”显式标注。
7. **同源长度**：性能同一 target 贯穿三档，避免“换题”混淆长度效应。

### 2.2 已知取舍

| 取舍 | 说明 |
|---|---|
| LiveCodeBench 只取最新 release 窗口 | 使用官方 `code_generation_lite` 的 v5+v6 增量窗口（2024-09-22~2025-04-06），选中 250 题 = 全部 171 条可判 v6 + 最新 79 条可判 v5（日期 2024-11-22~2025-04-06）；不是全量 LCB，也不是官方榜单的固定 release 口径。隐藏测试已从上游解码入库（保留 9,184 条；单条 JSON >256KB 的 331 条压测用例省略并逐行记录）。 |
| LCB 两类题面混排 | 有 starter code 的 LeetCode 行用函数级 prompt（`testtype=functional`，100 条，附 `meta.fn_name`）；其余 AtCoder 行用 stdin/stdout prompt（`testtype=stdin`，150 条）。判题器必须两类都支持，不能假设“全部是隐藏 stdin/stdout”。 |
| Global-MMLU 42 语种 × 100 配对题 | 本地 42 个语种的 `test.parquet`（各 14,042 行，`sample_id` 集合完全一致）；以 en 行按 57 个 subject round-robin 选出**同一组 100 个 sample_id**，42 语种共用（语言间比较的是同一题的不同语言翻译，属**配对设计**）；每语种 100 行，en 100 行标 `is_en_reference=true`。逐语种报告 + 宏平均，en 单列；**不要当成 42 个独立样本**。 |
| MMLU-Pro 用 0-shot CoT | 官方口径是 5-shot；本地 250 行按 14 学科分层抽 250 题，prompt 为 0-shot CoT 并要求末行 `Answer: X`、选项重排为 A–J，**不再有字面量 `$LETTER`**。不可与 5-shot 公开榜横比。 |
| LongBench v1 用官方模板 + 去重 + 长度分层 | 长上下文 4 任务使用官方 `dataset2prompt` 模板（保留 upstream 的 `asconcisely` 拼写），并先按 `(task, input, context)` 去重、answers 保序 union：narrativeqa 200→182 行（合并 18 行）；再**只保留 4,096 ≤ tokens ≤ 32,768**，按长度分层抽 250（4k-8k 100 / 8k-16k 99 / 16k-32k 51，16–32K 池恰 51 全取），任务配额随之变为 narrativeqa 77 / qasper 47 / hotpotqa 62 / 2wikimqa 64（旧的 63/63/62/62 已退役）。**4K 下界与长度分层是抽样设计，不可与 LongBench 官方全量分数横比。** |
| BFCL 只测单轮 11 类 | 精度覆盖 BFCL v4 **全部 11 个 single-turn 类别**（simple_python / simple_java / simple_javascript / multiple / parallel / parallel_multiple / irrelevance / live_simple / live_multiple / live_parallel / live_parallel_multiple），每类目标 100 行、池小取全部 → **889**；跨类按 user question 全局去重（共跳过 47 行重复题干），判题用官方 possible-answer。**`irrelevance` 已含在内**（上游无 possible_answer，gold 为空列表，“不调用工具”即正确）；multi-turn 类别仍排除（需环境与 user simulator），因此测不到多步 agent 能力。 |
| HotpotQA 只有 hard 题 | distractor validation 的 7,405 行**全部 `level=hard`**（上游属性）；search_agent 的 250 行因此天然都是 hard，不得当作 easy/medium 混合能力解读，也不可与按全难度抽样的公开榜横比。 |
| 性能长文为前缀切片 | 1K/8K/32K 取同一文档前缀；只测 prefill/decode 形状，**不承诺答案可定位**，不得用于能力结论。 |
| 性能 Agent 单请求 | 性能模式只发一次请求、不执行工具循环；测的是“大工具目录/长上下文预填充”，不是多步 agent 能力。`tools` 只带目标请求自身函数（1–3 个），函数目录在 messages 内按 whole-function 前缀填充。 |
| 性能短任务为同集素材扩展 | IFEval/GSM8K/BFCL 的 8K/32K 由同数据集素材扩展到目标长度（IFEval 参考池按 target 旋转起点、池不足 32K 时循环一次；GSM8K 例题块按 target 旋转并排除 20 道目标题），标 `constructed=true`（共 220 行），只测形状。 |
| 精度规模不齐 | 6 个场景各 250 行；general_agent 889 行；multilingual 4,200 行但**每语种仅 100 行**。p≈0.5 时 Wilson 95% CI：250 行约 ±6.2pp、100 行约 ±9.8pp；单场景/单语种只能分辨约 10–20pp 的差异，宏平均/全量汇总更稳。 |
| 精度长上下文 4K–32K token | 为保证 32K 窗口模型可测、又避免大量短文混入，长上下文样本在**抽取时**只保留 `4,096 ≤ tokens ≤ 32,768`（不截断；实际 4,102–32,187），并按长度分 3 档抽 100/99/51；运行期超窗样本记 `skipped(not-testable)`。 |
| 精度暂为离线资产 | 6,589 行数据与 6 个判题契约已就绪，但 runner/evaluator 未接入（`status=offline-asset`）。当前任何“精度分数”都不可引用，除非未来单独任务接入并跑独立 run_id。 |

### 2.3 逐数据集选型依据

> 每个数据集按“服务场景 / 本地证据 / 为什么选它 / 不选谁 / 合理性”说明。数字均为本地实测。

#### 2.3.1 LongBench v2（性能：知识问答、长上下文、编程、搜索 Agent）

- **本地证据**：503 条；上下文 ≥120K 字符的四个子域池为 Single-Document QA 116、Multi-Document QA 94、Code Repository 48、Long Structured Data 32（合计 290 行）；单条上下文中位约 41.7 万字符，最大 1,600 万字符。抽样前按 `sha1(raw context)` 去重：290→279（6 组/11 行），knowledge_qa 再按 question 去重 1 行，得到候选 278（106/94/47/31）；最终 80 个 target（每场景 20），跳样 6 条（2/3/0/1，原因：32K 不可达 1、低 token 密度 2、题面过长 3），选中集重复 context/题面 0。
- **为什么选**：本地唯一能“单文档切 1K/8K/32K 三档”的长文语料，四个子域分别对应长文事实问答、结构化长文、代码仓库、多文档四种负载；Apache-2.0、已缓存，零新增下载。
- **不选**：LongBench v1（单文档中位 2–15K token，无法稳定切 32K，只能拼接）；NIAH/RULER（合成任务，负载形状单一且易被针对性优化）；真实网页/文档（许可与可复现性不可控）。
- **合理性**：作为**性能形状探针**合理（不判分、只测长度与结构）；但一个数据集承担 4/8 性能场景（240/480），场景独立性有限，报告必须按子域标注、不得据此下能力结论。long_context 的 1K 档存在上游 boilerplate 碰撞（20 条里只有 15 个唯一 1K body，已写入 `manifest.sampling.long_context.known_1k_prefix_collisions`），属上游属性、不是抽样错误。

#### 2.3.2 LongBench v1（性能：中文长文 `chinese_long_context`；精度：长上下文）

- **本地证据**：`data.zip` 含 34 个任务；`dureader` 200 条（全中文，上下文中位 1.5 万字符）；精度任务 `narrativeqa` 200（中位 10.7 万字符）、`qasper` 200（2.3 万）、`hotpotqa` 200（6.3 万）、`2wikimqa` 200（2.6 万）。精度侧 narrativeqa 去重后 182 行；4 任务在 4K–32K 窗口内的池为 4k-8k 263 / 8k-16k 279 / 16k-32k 51，最终抽 100/99/51 = 250（任务分布 narrativeqa 77 / qasper 47 / hotpotqa 62 / 2wikimqa 64）。
- **为什么选**：长上下文精度需要“长文档 + 短答案 + 可程序判 F1”的任务，这 4 个任务均有官方模板与参考答案；性能侧 `dureader` 是本地唯一中文长文 QA，能切 1K、再拼接 8K/32K。该性能场景**只有中文**，故场景名为 `chinese_long_context` 而非 `multilingual`（后者保留给精度侧真正的 42 语种 Global-MMLU）。
- **不选**：LongBench v2 做精度（中位约 10 万 token，250 条预填成本过高、超窗跳过率高，模型间不可比）；`multifieldqa_zh` 做性能（中位 6K 字符，到不了 32K）；CMMLU/C-Eval（短选择题，不是长文）。
- **合理性**：合理。限制：精度长上下文只保留 4K–32K token 并按长度分层（保证 32K 窗口可测、避免短文混入；4K 下界与分层是抽样设计，不是 LongBench 官方口径，故不可与官方全量分数横比）；dureader 只用于性能形状，语言维度不据此下结论。

#### 2.3.3 IFEval（性能：指令遵循）

- **本地证据**：541 条、约束类型多样、纯 prompt、无答案依赖；同集指令池约 2.5 万 token，循环一次可覆盖 32K 档。20 个 target 的参考块按不同起点旋转（`0,37,…,518`，排除目标题），cell 内公共前缀为 0。
- **为什么选**：社区最常用的指令基线，纯 prompt 天然适合“目标指令 + 同集素材”做长度扩展；与精度侧 IFBench 共用“指令遵循”能力但数据不重叠。
- **不选**：IFBench 做性能（仅 300 条、且精度侧已用，避免两侧完全重叠）；Multi-IF（多轮/多语言，形状复杂、本地未缓存）。
- **合理性**：性能不判分，只测“读参考材料 + 执行指令”的 prefill/decode 形状；合理。

#### 2.3.4 IFBench（精度：指令遵循）

- **本地证据**：300 条、58 类 OOD 新约束、每条含 `instruction_id_list` / `kwargs`；官方 `evaluation_lib` 在 `dataset/raw/ifbench/`。
- **为什么选**：IFEval 在强模型上接近天花板；IFBench 是更新的 held-out 约束集，官方 checker 可程序判定，能覆盖“约束类型”而不是只测通用问答。
- **不选**：IFEval 做精度（饱和、且性能侧已用）；Multi-IF（多轮/多语言，判题成本与依赖更高、本地未缓存）；FollowBench 等（本地不可得，社区口径不统一）。
- **合理性**：合理但有两个前提：① 判题必须接官方 checker（58 类，不能只实现 IFEval 子集）；② 250/300 = 83% 饱和，且按“首要 instruction id”分层，稀有约束权重会被放大。

#### 2.3.5 GSM8K（性能：数学与逻辑）

- **本地证据**：1,319 条，自带解题过程与短答案，MIT，已缓存。20 个 target 的已解例题块使用不同起点（`0,41,…,779`），并排除 20 道目标题；cell 内公共前缀仅 10 字符。
- **为什么选**：目标题 + 已解例题天然适合“短任务扩长”，三档可共享同一目标题；题目与答案不进入参考材料（排除目标 20 题）。
- **不选**：MATH-500 做性能（解答长、LaTeX 多，扩到 32K 需更多条，且精度侧已用）；AIME（题量小，长度扩展素材不足）。
- **合理性**：性能只测形状；合理。

#### 2.3.6 MATH-500（精度：数学与逻辑）

- **本地证据**：500 题，难度 1–5（43/90/105/128/134），答案短（整数/分数/LaTeX），MIT。
- **为什么选**：GSM8K 已饱和，AIME 仅 90 题不足 250；MATH-500 题量与难度跨度合适，是数学能力的事实标准之一。
- **不选**：GSM8K（饱和；性能侧已用）；AIME（题量不足）；CMATH（中文小学难度，饱和且窄）；Omni-MATH 等（本地不可得）。
- **合理性**：合理；但判题必须用 `\boxed{}` + math-verify 等价判定，不能“取最后一个数字”（否则分式/根式/区间会系统性误判）。

#### 2.3.7 LiveCodeBench（精度：编程）

- **本地证据（2026-11 更新）**：官方 `code_generation_lite` 最新增量 `test5.jsonl`（167 条，2024-09-22~2025-01-04）+ `test6.jsonl`（175 条，2025-01-04~2025-04-06）；解码隐藏测试后可判 163 + 171 条。选中 250 题 = 全部 171 条可判 v6 + 最新 79 条可判 v5，日期 **2024-11-22~2025-04-06**。其中 AtCoder **150 条 stdin**、LeetCode **100 条 functional**；functional 行 100/100 带 `meta.fn_name` + starter code，`meta.testtype` 与隐藏测试一致；隐藏测试保留 9,184 条（单项 JSON >256KB 的 331 条省略并记录）。
- **为什么选**：执行判定 + 时间切片是编程能力评测的正确形态；HumanEval/MBPP 已饱和且泄漏严重；SWE-bench 需要仓库环境与镜像，成本高；CodeContests 等本地不可得。选择官方最新的增量 release，而不是旧的全量快照，是为了尽量靠近当前模型训练截止。
- **不选**：`test.jsonl` 旧全量（1.25GB，窗口只到 2024-03，反而更旧，且是 v1 口径）；`test4.jsonl` 等更早累计文件（>1GB、窗口更旧）；HumanEval/MBPP（饱和+泄漏）。
- **合理性**：合理但必须标注窗口与题面类型：最新窗口仍不等于“训练截止之后”，报告需注明“LCB v5+v6 窗口、150 stdin + 100 functional、隐藏测试裁剪（>256KB 省略）、与官方榜不可比”；不得外推为“全量 LCB 能力”。

#### 2.3.8 BFCL v4（性能/精度：通用 Agent）

- **本地证据**：本地 BFCL v4 single-turn 共 **11 类**：三类旧集 simple_python 400 / multiple 200 / parallel 200，加 simple_java 100 / simple_javascript 50 / parallel_multiple 200 / irrelevance 240 / live_simple 258 / live_multiple 1,053 / live_parallel 16 / live_parallel_multiple 24；合计 2,741 行、跨类去重后 2,694 个唯一题干。性能侧只使用三类共 800 条，全库唯一函数 642 个（simple 370 / multiple 443 / parallel 186 的并集）；`possible_answer` 提供每参数可接受值集合；Apache-2.0。
- **性能样本**：20 targets 按 **simple 10 / multiple 5 / parallel 5** 分层配额抽取（类别内固定 seed 随机）；1K 不可容纳的候选跳过 28 条（simple 23 / multiple 3 / parallel 2），全部记录。函数目录是**target-first 的 whole-function 前缀**：先放该 target 自身函数（与请求 `tools` 定义逐字节一致），再放同类别、再放其余，按函数名去重、绝不截断半个 JSON；实测每档可容纳 1K 3–7、8K 57–61、32K 228–248 个函数，20/20 target 的自身函数在三档都可见。全量 642 函数约 88.5K token，超出 32K 档，因此**不是“完整函数目录”**。
- **为什么选**：工具调用的结构化判定标准；`possible_answer` 支持多候选、可选参数与并行调用；single-turn 不需要环境与 user simulator，成本与方差可控。
- **不选**：τ²-bench / ToolBench（多轮 + 环境/真实 API，成本高、方差大）；API-Bank（数据较旧、口径不一）；BFCL multi-turn（需要环境与用户模拟器，长度不可控）。
- **合理性**：合理；局限：只测**单轮**工具调用（不含 multi-turn 与多步 agent）。精度侧已补齐 `irrelevance`（“不该调用”，100 行空 gold），但“过度调用”只在 AST 匹配下间接体现。精度侧覆盖全部 11 类，按 user question 全局去重（共跳过 47 行），每类 100（池小取全部：simple_javascript 50 / live_parallel 15 / live_parallel_multiple 24），合计 **889** 个唯一题干。

#### 2.3.9 MMLU-Pro（精度：知识问答）

- **本地证据**：12,032 题、14 学科、选项数 3–10（10 选项 9,981；9 选项 801；4 选项 606 等）；MIT。精度 250 题按 14 学科 round-robin（每类 17–18 题）。
- **prompt 口径**：0-shot CoT，明确要求“think step by step, then end with a single line of the form `Answer: X`”，选项重排为 A–J；构建断言禁止出现字面量 `$LETTER`（早期版本会把占位符原样喂给模型）。
- **为什么选**：MMLU 已饱和（强模型 >90%），MMLU-Pro 选项更多、难度更高、学科覆盖稳定；250 题按 14 学科分层，样本量充裕。
- **不选**：MMLU（饱和）；MMMLU（机器翻译，语言能力混淆）；GPQA（仅 198 题，领域窄）；C-Eval/CMMLU（中文单语，作为多语言补充而非知识主集）。
- **合理性**：合理；注意两点：① 0-shot CoT 与官方 5-shot 不同口径，不能横比；② 选项数混合（16% 非 10 选项），猜测基线不恒定，报告需说明。

#### 2.3.10 HotpotQA（精度：搜索 Agent）

- **本地证据**：distractor validation 7,405 条（全部 `level=hard`）；规范子集（10 段 + 2 supporting facts）4,940 条；答案短、EM/F1 可判。与长上下文场景做跨场景去重：long_context 已选的 62 条 hotpotqa 题中有 48 条在同池，排除后池 4,892，最终选 194 bridge / 56 comparison（共 250，跨场景重复 0）。
- **为什么选**：多跳、可离线给定语料、答案可程序判定；适合“给定语料 + search/lookup 工具的检索式 Agent”测试（无需真实网页）。
- **不选**：FRAMES/BrowseComp（需要实时网页/冻结快照，不可复现）；2WikiMultiHopQA/MuSiQue（本地未缓存，且换源需要新的判题/模板管线）。
- **合理性**：作为“闭环检索 Agent”合理；但 HotpotQA 是 2018 年数据，污染风险高，结论只用于**相对比较**，不得与公开榜横比。检索语料 `meta.paragraphs` 是输入而非 prompt，单独记录 `meta.corpus_chars` / `meta.corpus_tokens`，**不计入** `tokens`；性能侧 `search_agent` 不发语料、只发固定工具 schema。

#### 2.3.11 Global-MMLU（精度：多语言）

- **本地证据**：**42 个语种**（am ar bn cs de el en es fa fil fr ha he hi id ig it ja ko ky lt mg ms ne nl ny pl pt ro ru si sn so sr sv sw te tr uk vi yo zh），每语种 14,042 条、`sample_id` 集合完全一致；人工翻译、含文化敏感标注。精度用**配对设计**：以 en 行按 57 个 subject round-robin 选同一组 **100 个 sample_id**，42 语种共用 → 42 × 100 = **4,200** 行；覆盖 57/57 subjects、6 个 subject_category；en 的 100 行标 `meta.is_en_reference=true`（其余 false），宏平均时 en 单列。
- **为什么选**：原生多语种知识题、每语种题量充足、可逐语种报告；MMMLU 是机器翻译版本，会把翻译质量混入语言能力；CMMLU/C-Eval 是单语中文，无法测跨语言。
- **不选**：MMMLU（翻译版本）；CMMLU/C-Eval（单语）；MGSM（数学任务，不是知识问答）。
- **合理性**：合理但需限定口径——覆盖本地全部 42 语种，但**每语种仅 100 题，且 42 语种共用同一组 100 个 sample_id（配对设计）**：跨语言差异比较的是“同一题的不同语言翻译”，不是 42 组独立样本，不能按独立样本做显著性检验。结论限定为“42 语种、逐语种 + 宏平均、en 单列”。注意这个精度场景名仍是 `multilingual`，与性能侧已更名的 `chinese_long_context` 不是同一件事。

### 2.4 时间窗与抗污染策略

“用更新的数据”是降低污染风险的必要条件，但不是充分条件：公开数据集的发布时间不等于模型训练截止时间，我们无法证明某题“未被见过”。因此采取**记录窗口 + 优先最新 held-out + 报告标注**三件事：

| 数据集 | 本地窗口 / 版本 | 类型 | 污染风险 | 本轮处理 |
|---|---|---|---|---|
| LiveCodeBench | v5+v6（2024-09~2025-04；选中 2024-11~2025-04） | 时间切片、执行判定 | 中低 | **2026-11 从 2023-05~2024-03 旧截断头更新到最新增量 release**，隐藏测试入库，窗口写进 manifest |
| IFBench | 2025 发布，300 条新约束 | held-out OOD | 低 | 全量后分层抽 250 |
| MMLU-Pro | 2024 发布 | 静态知识题 | 中 | 0-shot + 混合选项，报告注明不可与 5-shot 榜横比 |
| Global-MMLU | 2024 发布 | 人工翻译知识题 | 中 | 42 语种 × 100 配对题，逐语种报告 |
| LongBench v2 | 2024/2025 发布 | 长文 | 中 | 仅性能形状，不判分 |
| LongBench v1 | 2023 发布 | 长文 QA | 中 | 仅精度长上下文；4K–32K 长度分层；报告标注 |
| BFCL v4 | 2024/2025 发布 | 工具调用 | 中低 | 11 个单轮类别（含 irrelevance）+ 多候选判题 |
| MATH-500 | MATH（2021）衍生 | 数学题 | 中高 | 数学题相对稳定；判题用等价判定 |
| GSM8K | 2021 发布 | 小学数学 | 高 | 仅性能形状，不判分 |
| HotpotQA | 2018 发布 | 多跳 QA | 高 | 仅相对比较；报告明确标注 |
| IFEval | 2023 发布 | 指令遵循 | 中高 | 仅性能形状，不判分 |
| dureader | 中文长文 QA | 中文长文 | 中高 | 仅性能形状（`chinese_long_context`），不判分 |

**LiveCodeBench 更新说明**：

1. 旧本地文件是 `test.jsonl` 的 820 MB 截断头（315 条，2023-05-07~2024-03-02），恰好是训练覆盖概率最高的一段；
2. 2026-11 从 hf-mirror 下载官方**最新增量** `test6.jsonl`（134 MB，2025-01~2025-04）与 `test5.jsonl`（558 MB，2024-09~2025-01），两者合计可判 334 题；
3. 采样规则“最新优先”：全部 171 条可判 v6 + 最新 79 条 v5 = 250 题，窗口 2024-11-22~2025-04-06；
4. 不下载 `test.jsonl` 全量（1.25 GB）或 `test4.jsonl`（1.2 GB）：前者窗口反而更旧（到 2024-03），后者也早于 v5/v6，且超过 1GB 下载预算；
5. 每个上游文件记录 URL + SHA-256；窗口与“隐藏测试裁剪”写入 manifest，报告必须注明“不可与官方榜横比”。

**2026-11 精度集扩展说明**（build v5，2,000 → 6,589）：

1. LongBench v1 只保留 **4K–32K** 并按长度分层（100/99/51），解决旧版 89% 行 <16K、`tokens_min=1,259`（短文混入）的问题；16–32K 池恰 51 条全部取用，任务配额因此不再是 63/63/62/62；
2. BFCL 补齐到 **11 个 single-turn 类别**（含 `irrelevance`，其 possible_answer 缺省为空列表），按跨类题干全局去重后 889 行；multi-turn 仍不覆盖；
3. Global-MMLU 使用本地全部 **42 语种**（`dataset/raw/global_mmlu/` 下 42 个 `test.parquet`，每语种 14,042 行、`sample_id` 集合完全一致；raw 目录 gitignored），以 en 行按 subject round-robin 选出同一组 100 个配对 id，42 语种共用 → 4,200 行；
4. 溯源文件数从 8 数据集/21 文件变为 **8 数据集/70 文件**（bfcl 21 = 11 个请求 + 10 个 possible_answer；global_mmlu 42 个 `test.parquet`）。

**刷新流程**（后续新 release）：下载 `testN+1.jsonl` → 校验可判题数与日期 → 重建精度集 → manifest 更新窗口/SHA → `verify.py` + `gate`。

---

## 3. 性能测试设计

### 3.1 测量口径

| 项 | 规则 |
|---|---|
| 请求 | 单请求流式 `chat/completions`，`temperature=0`；`max_tokens`（OSL）默认 **50**（CLI/Web 可覆盖，记录在结果中） |
| 档位 | 1K / 8K / 32K（目标 token 数 1024 / 8192 / 32768）；实测声明值 1K 978–1072、8K 8023–8201、32K 32455–32779（均值 13,980） |
| token 口径 | `tokens = messages_tokens + tools_tokens`：messages 走参考 tokenizer 的 chat template，tools 按其 JSON 文本计 token；archive 的旧“双计”已修正 |
| 参考 tokenizer | MiniCPM5-1B `tokenizer.json` + 官方 `chat_template.jinja`（`add_generation_prompt=True`），与 oMLX `/v1/tokenize` 逐条对齐 |
| 容差 | 整数 token：`floor(0.95×target) ≤ tokens ≤ floor(1.05×target)`；1K = [972, 1075]、8K = [7782, 8601]、32K = [31129, 34406]（1K 下界是 972，不是 972.8） |
| 冷缓存 | 每条 measured 请求前清 hot+SSD，`cached_tokens==0`；缺失/命中即污染，不进聚合 |
| 每次请求预留 | `max_tokens 50 + context_safety_margin 128 + decode_guard_reserve 64 = 242` token（声明 tokens 不含 decode guard） |
| 窗口不足 | `声明 tokens + 242 > 模型窗口` → 该行记 `not-testable`（写入 `summary.skipped`），**不截断**。32K 档需模型窗口 32,697–33,021：32,768 窗口模型只有 1/160 行可测，其余 159 行 not-testable；≥33,021 窗口可全部测 |
| 运行默认 | `manifest.run_defaults = {max_tokens: 50, decode_guard: true, warmup_per_model: 1, warmup_requests: 0, repeats: 1}`，与 runner 默认一致 |
| 判分 | 不判分；`score` 恒为 null |

> 注意：`tokens` 是**参考 tokenizer 的声明值**，运行时以目标模型实际 tokenizer 计数；逐样本记录 `input_context_tokens`，二者差异在结果中可复算。

### 3.2 两套规模

| 测试集 | 文件 | 规模 | 用途 |
|---|---|---|---|
| `performance` | `performance/testset.jsonl` | 480（8×3×20） | 正式对比/归档 |
| `performance_10` | `performance/testset_10.jsonl` | 240（8×3×10） | 快速回归/冒烟 |

`performance_10` 是 `performance` 的确定性派生：每场景按 `target_id` 排序取偶数位 10 个 target（`sorted(target_id)[::2]`），每个 target 保留 1K/8K/32K 三条 = 240 行 / 24 cells；逐行与全量 dict 相等，是严格子集。两套同源，不构成独立证据。

### 3.3 构造规则（逐场景）

| 场景 | 上下文来源 | 三档构造 | constructed |
|---|---|---|---|
| 知识问答 | 同一篇 LongBench v2 单文档 | 同一文档前缀切片 + 原题/选项 | 否 |
| 长上下文 | 同一篇 LongBench v2 结构化长文档 | 同上 | 否 |
| 编程 | 同一份 LongBench v2 代码仓库上下文 | 同上（只读，不要求生成代码） | 否 |
| 搜索 Agent | 同一份 LongBench v2 多文档上下文 | 同上 + 固定 search/lookup 工具 schema（单请求，不执行工具） | 否 |
| 中文长文 | 目标题自己的 dureader 文档 | 1K 切目标文档；8K/32K 以目标文档开头、拼接其他同任务文档后切片 | 8K/32K 是 |
| 指令遵循 | 同数据集 IFEval 指令池 | 目标指令 + 同集指令参考块（按 target 旋转起点；池不足 32K 时循环一次） | 是 |
| 数学与逻辑 | 同数据集 GSM8K 已解例题 | 目标题 + 已解例题（按 target 旋转起点，排除 20 道目标题） | 是 |
| 通用 Agent | BFCL 共享 642 函数池 | 目标请求 + **target-first 的 whole-function 前缀**（自身函数→同类别→其余；不切 JSON 中间，也不是全量目录） | 是 |

规则：

- 上下文只来自真实数据，不生成新题/新答案/新单测；
- 同集素材不得包含目标题的答案或解法（构建断言：目标指令不进入 IFEval 参考块、目标题不进入例题块）；
- 函数目录必须始终是合法 JSON，且 schema 类型合法；目标 `tools` 中的函数定义与目录内同名定义逐字节一致；
- 长文切片为前缀（`head`），只用于形状测量；5 类文档场景的 1K⊂8K⊂32K 字符前缀链全部成立；
- `constructed=true` 共 220 行（中文长文 8K/32K 40 + 指令遵循 60 + 数学 60 + 通用 Agent 60）。

### 3.4 预算与耗时

| 项 | 数值 |
|---|---|
| measured 请求（`performance`） | 480（1K/8K/32K 各 160；场景各 60） |
| measured 请求（`performance_10`） | 240（每档 80；场景各 30） |
| OSL | 默认 50 token（`perf.max_tokens`，CLI/Web 可覆盖；触顶不丢弃，按实际生成窗口计算 TG TPS） |
| 预热 / 轮数 | 模型级 `warmup_per_model=1`（加载后 1 条短请求，丢弃）；per-cell `warmup_requests=0`；`repeats=1`（`run_defaults` 记录在 manifest） |
| 数据体量 | 全量 `testset.jsonl` 约 27.5 MB；快速子集约 13.7 MB |
| 32K 可测性 | 32K 行声明 32,455–32,779 token；32,768 窗口模型仅 1/160 可测，其余 159 行 `not-testable`（不截断） |

### 3.5 可比性边界（重要）

本性能集是**本地口径的形状基准**，不是官方榜：默认做一次模型级短预热（`warmup_per_model=1`，仿 oMLX 消除 JIT/编译），但 per-cell `warmup_requests=0`、
`max_tokens=50`（OSL 解码窗口很短，TG TPS/TPOT 对短窗口更敏感）、`repeats=1`（单次测量，
靠样本量而非重复平均），且冷缓存逐请求清理。这与 vLLM / oMLX 官方 benchmark 的预热、生成长度、
并发/批处理口径都不同，**不得与任何公开榜或官方 benchmark 数字直接横比**；跨 run 比较也要求
`max_tokens`、缓存模式、预热/轮数一致（记录在 `manifest.config_snapshot`）。

---

## 4. 精度测试设计

### 4.1 原则与当前状态

- **状态（最重要）**：6,589 行是**离线数据资产**（`status=offline-asset`、`runner=not wired`），
  6 个 judge 全部 `implementation_status=contract-only`——只有契约描述，没有对应 evaluator/runner
  （仓库的 legacy `domain/evaluators/` 未接入，也不覆盖 IFBench/LCB/BFCL/math-verify 等官方判据）。
  在接入判题并跑出独立 run_id 之前，**不得引用任何“精度分数”**。
- **固定模板、无示例**：原始题目 + 固定官方模板；无 few-shot、无示例、无 k 次采样。
  MMLU-Pro 用 0-shot CoT（`Answer: X`）；LongBench v1 用官方 `dataset2prompt` 模板；
  LiveCodeBench 按题面类型用函数级或 stdin/stdout prompt；BFCL 用原始请求。
- **不截断、不分性能档位**：直接使用原始 prompt，不截断；长上下文在**抽取时**只保留
  `4,096 ≤ tokens ≤ 32,768`（实际 4,102–32,187）并按长度分层（4k-8k 100 / 8k-16k 99 / 16k-32k 51，
  每行 `meta.length_band`）——这是抽样口径，不是性能档位；超窗样本运行时 `skipped(not-testable)`。
- **判题契约前置**：每行的 `judge` 决定其 `answer/choices/meta` 必须包含哪些字段；缺字段即构建失败。
- **可追溯**：每行 `meta.upstream_id` 全局唯一（6,589/6,589），另保留上游语义主键（question_id / sample_id / bfcl_id / ifbench_key / task 等）。

### 4.2 判题契约

| judge | 适用场景 | 输入 | 判据 | 指标 |
|---|---|---|---|---|
| `mc_letter` | 知识问答、多语言 | `choices`、`answer`（字母） | 抽取显式答案短语 → 括号字母 → 单字母行 → 唯一独立字母；超出 `len(choices)` 的字母判错（MMLU-Pro A–J、Global-MMLU A–D） | acc@1 |
| `ifbench` | 指令遵循 | `meta.instruction_id_list`、`meta.kwargs`、`meta.ifbench_key` | 官方 IFBench `evaluation_lib` + instructions registry（58 类） | prompt/instruction × strict/loose 四组 acc |
| `qa_f1` | 长上下文、搜索 Agent | `meta.answers`（多参考） | SQuAD 归一化；F1 取参考最大值 | F1（EM 次指标） |
| `math` | 数学与逻辑 | `answer`（LaTeX） | 抽取最后一个 `\boxed{}`，math-verify 符号/数值等价判定 | acc |
| `code_pass1` | 编程 | `meta.tests`（**functional + stdin 两类隐藏测试**，单条 JSON ≤256KB；functional 行附 `meta.fn_name`/starter code）、`meta.public_tests` | 子进程按 testtype 执行生成程序，全部隐藏测试通过为 pass | pass@1 |
| `bfcl_ast` | 通用 Agent | `meta.possible_answers` | 官方 AST 匹配：候选集合、参数级多值、可选参数、parallel 顺序无关 | acc |

判题失败（无答案/格式错/执行失败/超时）单独统计，不并入正确率分母。**实现状态**：以上 6 个 judge
在 manifest 中均为 `implementation_status="contract-only"`；运行器未接入，没有可执行的判题路径。

### 4.3 抽样与分层

| 场景 | 数据集 | n | 分层 |
|---|---|---:|---|
| 知识问答 | MMLU-Pro（12,032 题） | 250 | 14 学科 round-robin（17–18/学科）；0-shot CoT |
| 指令遵循 | IFBench（300 题） | 250 | 首要 instruction id（58 类）round-robin |
| 长上下文 | LongBench v1（4 任务） | 250 | 长度分层 4k-8k 100 / 8k-16k 99 / 16k-32k 51（band 内按任务 round-robin）；只保留 4,096–32,768 token；任务实际分布 77/47/62/64 |
| 数学与逻辑 | MATH-500（500 题） | 250 | (subject, level) round-robin |
| 编程 | LiveCodeBench（v5+v6 最新窗口，可判 334 条） | 250 | 最新优先：全部 171 条可判 v6 + 最新 79 条 v5；150 stdin + 100 functional |
| 通用 Agent | BFCL v4 single-turn（11 类） | 889 | 每类目标 100、池小取全部（simple_javascript 50 / live_parallel 15 / live_parallel_multiple 24）；类别内随机；跨类 user question 全局去重（跳过 47） |
| 搜索 Agent | HotpotQA canonical 子集（4,940 → 排除 48 → 4,892） | 250 | type（bridge/comparison）按池比例 round-robin；194 bridge / 56 comparison |
| 多语言 | Global-MMLU（42 语种） | 4,200 | 42 语种 × 同一组 100 个配对 `sample_id`（subject round-robin 选 id）；en 100 行标 `is_en_reference` |

分层计数与去重明细逐项记录在 `dataset/test/accuracy/manifest.json` 的 `strata_counts` 与 `sampling`。
LongBench v1 先按 `(task, input, context)` 去重并 union answers：narrativeqa 200→182 行（合并 18 行），
再只保留 4,096–32,768 token；band 池为 4k-8k 263 / 8k-16k 279 / 16k-32k 51（合计 593），按 100/99/51 抽样；
其余任务 200 行均唯一。

搜索 Agent 的样本包含：问题、search/lookup 工具、`meta.paragraphs`（10 段语料）、`meta.supporting_facts`、参考答案。
语料单独计量（`meta.corpus_chars`/`meta.corpus_tokens`），不计入 prompt `tokens`；运行器在给定语料上执行确定性检索后回答，
判题只看最终答案。long_context 先构建并占用 62 条 hotpotqa 题，search_agent 池排除同题（48 条），
构建后跨场景重复题 = 0。

### 4.4 超窗与失败

- 精度不分长度档：prompt 超过模型窗口 → 运行时 `skipped(not-testable)`，不计入分母；
  长上下文已在抽取时限定到 4,096–32,768 参考 token（实际 4,102–32,187），32K 窗口模型可直接跑；
- 判题器异常 → 记 `error`，单列；判题失败（无答案/格式错/执行失败/超时）不计入正确率分母；
- 同一模型的多轮/多版本结果必须来自独立 run_id；
- **当前 runner 未接入 accuracy**：以上是未来接入时的约定，不是已实现行为。

---

## 5. 数据规格

### 5.1 性能行

```json
{
  "id": "perf-chinese_long_context-1k-000",
  "scenario": "...", "bucket": "1k|8k|32k", "target_id": "tgt-<scenario>-<idx>",
  "dataset": "longbench_v2:Single-Document QA | longbench_v1:dureader | ...",
  "tokens": 1011, "messages_tokens": 1011, "tools_tokens": 0,
  "messages": [{"role": "user", "content": "..."}],
  "tools": null,
  "constructed": false,
  "meta": {"workload": "...", "lang": "zh|en", "domain": "...", "context_sha1": "..."}
}
```

`tools` 只在 `general_agent`（目标请求自身函数 1–3 个）与 `search_agent`（固定 search/lookup 2 个）
出现；BFCL 的 large catalog 在 messages 内。`meta` 随场景增加
`workload`、`lang`、`domain`/`task`、`context_sha1`；BFCL 另有
`category/category_detail/bfcl_id/catalog_functions(642)/own_functions/own_functions_in_catalog`；
IFEval/GSM8K 另有 `reference_rotation_start`/`example_rotation_start`。

### 5.2 精度行

```json
{
  "id": "acc-<scenario>-<idx>",
  "scenario": "...", "dataset": "...", "judge": "mc_letter|ifbench|qa_f1|math|code_pass1|bfcl_ast",
  "tokens": 190,
  "messages": [{"role": "user", "content": "..."}],
  "choices": ["A. ...", "B. ..."] | null,
  "answer": <string | list | null>,
  "tools": null,
  "meta": {"lang": "...", "upstream_id": "...", "...": "judge-specific inputs"}
}
```

`answer` 的类型由 judge 决定：`mc_letter`/`math`/`qa_f1` 为字符串（`qa_f1` 的完整参考列表在
`meta.answers`）；`ifbench`/`code_pass1` 为 `null`（判据在 `meta`）；`bfcl_ast` 为官方
possible-answer 列表（当前与 `meta.possible_answers` 相同；`irrelevance` 行为空列表）。
`tools` 只在两个 Agent 场景出现（与性能侧同构）；
`search_agent` 的 `meta.paragraphs` 是检索语料，单独记录 `corpus_chars`/`corpus_tokens`、不计入 `tokens`。
分层元数据：`long_context` 每行有 `meta.length_band`（4k-8k/8k-16k/16k-32k）；`general_agent` 有
`meta.category`（11 个完整类别名）与 `meta.category_type="single_turn"`；`multilingual` 有 `meta.lang`、
`meta.is_en_reference`、`meta.sample_id`、`meta.subject`、`meta.subject_category`。

### 5.3 manifest

两份 manifest 均包含：

- `schema_version`、`generator_version`、`seed`、`tokenizer`（模型/模式/语义）、`tokenizer_cache`、`osl`、`tokens_semantics`；
- `provenance`：**只含该测试集实际用到的上游数据集**（performance 5 个/7 文件；accuracy 8 个/70 文件），
  每个数据集 `source_url`、`license`、逐文件 `bytes` + `sha256`；
- `stats`：每个 cell/场景的 n、tokens min/median/max/mean、constructed、langs、dataset；
- `totals`。

性能额外包含：`buckets`、`accept_range`、`accept_bounds`/`accept_rule`（整数 floor 口径）、
`run_defaults`、`workloads`/`scenarios.*.workload`、`window_requirements`（32K 每请求预留与
not-testable 事实）、`sampling`（逐场景 source/candidates/dedup/selected/skipped/selected_targets；
BFCL 另有类别池/配额/目录规模；long_context 有 `known_1k_prefix_collisions`）、`files`、两套规模。

精度额外包含：`status`、`runner`、`length_processing`/`construction`（当前均 `none`）、
`judges`（含逐 judge `implementation_status`）、`strata_counts`（long_context 含 `length_band`、
general_agent 11 类、multilingual 42 语种）、`totals.per_scenario`（逐场景映射）、
`sampling`（含跨场景排除与 `upstream_id_format`；long_context 记 band targets/ranges/pools/selected，
general_agent 记逐类 upstream/unique/skipped/selected，multilingual 记 languages/per_language/paired_ids/
subjects_covered）、`deviations`（11 条与官方口径的偏差）、每场景分层与 metric。

---

## 6. 质量门禁

构建时强制断言（失败即中止，不写产物）：

**性能**
- [ ] 行数 480；24 cells × 20；`id` 唯一；`tokens = messages_tokens + tools_tokens`，每行在整数
      `[floor(0.95×target), floor(1.05×target)]` 内；
- [ ] 160 个 `target_id`，每个恰好 3 条；三档题面/指令/请求逐字符一致，tools 三档一致；
- [ ] 选中集无重复 context（sha1）；knowledge_qa 无重复题面；`tools` 仅出现在两个 Agent 场景且 schema 合法；
- [ ] 文档场景 1K⊂8K⊂32K 字符前缀链；`constructed` 与 §3.3 规则一致（共 220 行）；
- [ ] BFCL：类别 10/5/5；catalog 是 target-first whole-function、JSON 可解析；20/20 target 自身函数三档可见；
- [ ] ifeval 目标指令不进入参考块、GSM8K 目标题不进入例题块；
- [ ] manifest：`osl=50`、`run_defaults` 5 键、`sampling.selected=20`、`accept_bounds/accept_rule`、
      `window_requirements`（32K not-testable 159/160）、provenance 5 数据集/7 文件；
- [ ] `performance_10`：`sorted(target_id)[::2]` 10 个 target、三档齐全，240 行逐行等于全集。

**精度**
- [ ] 行数 **6,589**；逐场景 n = {knowledge_qa 250、instruction_following 250、long_context 250、
      math_logic 250、coding 250、general_agent 889、search_agent 250、multilingual 4,200}；
      `id` 唯一；`tokens > 0`；
- [ ] `judge` 与场景契约一致；`answer/choices/meta` 满足 judge 要求；每个 judge 的
      `implementation_status=contract-only`（未实现判题时不得静默变成“已就绪”）；
- [ ] 无字面量 `$LETTER`、无未渲染占位符；场景内 messages+tools 组合唯一；
- [ ] `meta.upstream_id` 6,589/6,589 且全局唯一；BFCL 题干唯一（889 个）；跨场景重复题 = 0；
- [ ] long_context：每行 `4096 ≤ tokens ≤ 32768`，`meta.length_band` 与按 tokens 重算的 band 一致，
      band 计数恰为 100/99/51，4 个任务全部出现且各非空；
- [ ] general_agent：11 个 single-turn 类别计数与 manifest 一致
      （100/100/50/100/100/100/100/100/100/15/24 = 889），`category_type=single_turn`，
      irrelevance `answer=[]` 且 `meta.possible_answers=[]`，其余 10 类非空；
- [ ] multilingual：42 语种、每语种恰 100、42 个语种的 `sample_id` 集合完全相同（各 100 个），
      en 100 行 `is_en_reference=true`，同一 sample_id 的 subject 跨语种一致；
- [ ] coding：`testtype` 与隐藏测试一致（150 stdin / 100 functional），functional 行有 `fn_name`+starter code；
- [ ] `tools` JSON Schema 合法；`meta` 中的测试/语料/候选答案可解析；
- [ ] manifest：`status=offline-asset`、`runner=not wired`、provenance 8 数据集/70 文件、
      `strata_counts` 各行求和 = 对应场景 n、`totals.per_scenario == sampling.per_scenario`、
      `deviations` 11 条、统计与逐行复算一致。

**通用**
- [ ] `./modelbench validate --offline` 通过（`performance`=480、`performance_10`=240）；
- [ ] 只读数据集校验 `tool/dataset/verify.py` 通过（当前 73,301 checks）；
- [ ] `harness.gate` 通过；
- [ ] manifest 的 SHA-256 与 `dataset/raw/` 现场文件一致。

---

## 7. 复现

```bash
cd "/Users/huawei/Desktop/模型管家相关资料/modelbenchmark"
PY=.venv/bin/python

$PY tool/dataset/build_perf.py          # 480 行 + manifest
$PY tool/dataset/build_perf_subset.py   # 240 行 + manifest_10
$PY tool/dataset/build_acc.py           # 6,589 行 + manifest

./modelbench validate --offline
./modelbench validate --offline --testset performance_10
$PY -m harness.gate
```

token 计数优先使用本地离线模式（oMLX.app 自带 Python + 模型 tokenizer 文件），不可用时回退到 oMLX `/v1/tokenize`；两种模式共用 `dataset/.cache/tokenize.json`，按 payload 的 SHA-256 缓存。

---

## 8. 数据集清单与许可

| 数据集 | 用途 | 许可 | 来源 |
|---|---|---|---|
| LongBench v2 | 性能：知识/长文/编程/搜索（4 场景） | Apache-2.0 | https://huggingface.co/datasets/zai-org/LongBench-v2 |
| LongBench v1 | 性能：中文长文 `chinese_long_context`（dureader）；精度：长上下文 4 任务 | research use | https://huggingface.co/datasets/zai-org/LongBench |
| IFEval | 性能：指令遵循参考池 | Apache-2.0 | https://huggingface.co/datasets/google/IFEval |
| IFBench | 精度：指令遵循 | ODC-BY-1.0 | https://huggingface.co/datasets/allenai/IFBench_test |
| GSM8K | 性能：数学形状 | MIT | https://huggingface.co/datasets/openai/gsm8k |
| MATH-500 | 精度：数学 | MIT | https://huggingface.co/datasets/HuggingFaceH4/MATH-500 |
| LiveCodeBench | 精度：编程（v5+v6 最新窗口） | see repo | https://huggingface.co/datasets/livecodebench/code_generation_lite |
| BFCL v4 | 性能/精度：通用 Agent（请求文件 + possible answers；性能侧不含 possible answers） | Apache-2.0 | https://github.com/ShishirPatil/gorilla |
| MMLU-Pro | 精度：知识问答 | MIT | https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro |
| HotpotQA | 精度：搜索 Agent | CC BY-SA 4.0 | https://huggingface.co/datasets/hotpotqa/hotpot_qa |
| Global-MMLU | 精度：多语言（42 语种） | see card | https://huggingface.co/datasets/CohereLabs/Global-MMLU |

溯源按测试集过滤：performance 只记录实际使用的 5 个数据集 / 7 个文件（longbench_v1、longbench_v2、
ifeval、gsm8k、bfcl 请求 3 文件；无 livecodebench）；accuracy 记录 8 个数据集 / 70 个文件
（bfcl 21 个 = 11 个请求 + 10 个 possible_answer、global_mmlu 42 个 `test.parquet`、
LCB `test5.jsonl`+`test6.jsonl` 等）。
逐文件 SHA-256 与字节数见两份 `manifest.json` 的 `provenance` 字段。
