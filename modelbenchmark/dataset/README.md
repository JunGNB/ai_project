# dataset/ — 本地模型测评数据集

本目录是 [TEST_SAMPLE_SELECTION.md](TEST_SAMPLE_SELECTION.md)（v3）的落地产物：
**性能**（480 + 240）与**精度**（6,589）两套 testset 已按设计从 `dataset/raw/` 确定性构建。
来源 URL、许可与逐文件 SHA-256 记录在各自的 `manifest.json`。

## 1. 结论

| 项 | 实际 | 说明 |
|---|---|---|
| 性能（完整） | **480** = 8 场景 × 1K/8K/32K × 20 | 正式对比/归档 |
| 性能（快速） | **240** = 8 场景 × 1K/8K/32K × 10 | `performance` 的确定性子集，快速回归 |
| 精度 | **6,589** = 250×6 + 889 + 4,200 | 不截断 prompt；long_context 按 4K–32K 长度分层抽样 |
| 判题契约 | 6 种：`mc_letter` / `ifbench` / `qa_f1` / `math` / `code_pass1` / `bfcl_ast` | 见设计文档 §4.2 |
| 参考 tokenizer | MiniCPM5-1B `tokenizer.json` + 官方 chat template | 与 oMLX `/v1/tokenize` 逐条对齐；离线可复算 |
| 性能 token 实测 | 1K 978–1072；8K 8023–8201；32K 32455–32779（含 tools；均值约 13,980） | 目标 ±5%（整数 floor 边界） |
| 精度 token 实测 | 27–32,187（中位 190） | 长上下文样本筛选为 4K–32K token |

## 2. 目录

```
dataset/
├── TEST_SAMPLE_SELECTION.md       # 设计（场景/数据源/构造/判题契约/规格）
├── README.md                      # 本文件
├── raw/                           # 上游原始文件（只读）
├── .cache/tokenize.json           # token 计数缓存（sha256(payload) -> count）
└── test/
    ├── performance/
    │   ├── testset.yaml           # 默认测试集（480）
    │   ├── testset.jsonl          # 480 行
    │   ├── manifest.json
    │   ├── testset_10.jsonl       # 240 行快速子集
    │   └── manifest_10.json
    ├── performance_10/
    │   └── testset.yaml           # 快速测试集（240）
    └── accuracy/
        ├── testset.jsonl          # 6,589 行
        └── manifest.json
```

## 3. 性能样本（480）

同一 `target_id` 的三条为“同题不同长”：题面/指令/用户请求一致，只有上下文长度不同。

| 场景 | 数据源 | 1K/8K/32K 构造 | `constructed` |
|---|---|---|---|
| knowledge_qa | LongBench v2 `Single-Document QA` | 同一文档前缀切片 + 原题/选项 | 否 |
| long_context | LongBench v2 `Long Structured Data` | 同上 | 否 |
| coding | LongBench v2 `Code Repository` | 同上 | 否 |
| search_agent | LongBench v2 `Multi-Document QA` | 同上 + 固定 search/lookup 工具 schema（单请求） | 否 |
| chinese_long_context | LongBench v1 `dureader`（**中文单语**） | 1K 切目标文档；8K/32K 目标文档开头 + 其他同任务文档拼接 | 8K/32K 是 |
| instruction_following | IFEval | 目标指令 + 同集指令参考块 | 是 |
| math_logic | GSM8K | 目标题 + 同集已解例题（排除 20 道目标题） | 是 |
| general_agent | BFCL v4 | 目标请求 + 函数目录前缀（目标函数优先、同类别次之；whole-function 填充，非全量目录） | 是 |

- 字段：`id / scenario / bucket / target_id / dataset / tokens / messages_tokens / tools_tokens / messages / tools / constructed / meta`。
- `tokens = messages_tokens + tools_tokens`（tools 按其 JSON 文本计 token）。
- 每场景每档 20 条，共 24 cells；`manifest.json` 有每 cell 的 token min/median/max/mean，以及 `sampling`（去重/跳样逐条记录：共跳 34 条，选中集重复 context/题面 0）。
- 32K 档声明 32,455–32,779 token；运行时另需 242 token（OSL 50 + 安全边距 128 + decode guard 64），故需模型窗口 32,697–33,021；32,768 窗口模型仅 **1/160** 32K 行可测，其余 159 行 `not-testable`（不截断）。
- 性能样本只测 prefill/decode 形状：长文为前缀切片，**不承诺答案可定位**，不用于能力结论。
- 场景名是"负载形状"而非能力名：`knowledge_qa`=开卷长文 MCQ、`coding`=读代码仓库、`search_agent`=单请求固定工具 schema 预填、`general_agent`=target-first 函数目录前缀预填、`chinese_long_context`=中文长文；测量协议（OSL 50、per-cell warmup 0、模型级 `warmup_per_model=1`、repeats 1、decode_guard）见 `docs/TEST_PLAN.md`，不可与 vLLM/oMLX 官方榜直接横比。

## 4. 精度样本（6,589）

| 场景 | 数据集 | 条数 | judge | 指标 |
|---|---|---:|---|---|
| knowledge_qa | MMLU-Pro | 250 | `mc_letter` | acc@1 |
| instruction_following | IFBench | 250 | `ifbench` | prompt/instruction × strict/loose acc |
| long_context | LongBench v1（4 任务，4K–32K 长度分层 100/99/51） | 250 | `qa_f1` | F1（多参考取 max）/ EM |
| math_logic | MATH-500 | 250 | `math` | acc（数学等价） |
| coding | LiveCodeBench（最新 release 窗口 v5+v6，2024-11~2025-04） | 250 | `code_pass1` | pass@1（隐藏测试；150 stdin + 100 LeetCode functional） |
| general_agent | BFCL v4 single-turn（11 类） | 889 | `bfcl_ast` | AST 匹配 acc |
| search_agent | HotpotQA canonical 子集 | 250 | `qa_f1` | EM / F1 |
| multilingual | Global-MMLU（42 语种 × 100 配对题） | 4,200 | `mc_letter` | acc@1（逐语种 + 宏平均） |

- 字段：`id / scenario / dataset / judge / tokens / messages / choices / answer / tools / meta`；每行都有 `meta.upstream_id`（可回指上游主键）。
- 判题所需输入全部在行内：选项、参考答案列表、IFBench `instruction_id_list/kwargs/key`、LiveCodeBench 隐藏测试（单条 JSON ≤256KB，超大压测用例省略并记录；functional 行附 `meta.fn_name`）、BFCL possible answers、HotpotQA 语料与 supporting facts。
- **状态：离线数据资产（offline-asset）**。`docs/TEST_PLAN.md` 当前只运行性能基准；accuracy 不接入 runner、`judge` 只是契约描述（`implementation_status=contract-only`）。仓库里有 legacy `domain/evaluators/` 代码，但它们不覆盖这 6 个官方判题契约（IFBench 官方 checker/LCB 隐藏测试/BFCL AST/math-verify 等），因此**没有可执行的精度判题路径**。不要把它当作"已就绪的精度测试"。
- 无 few-shot、无 `bucket`；MMLU-Pro 用 0-shot CoT（`Answer: X`，无 `$LETTER`）；长上下文使用官方 LongBench 模板，只保留 4K–32K token 并按长度分层抽样（4k-8k 100 / 8k-16k 99 / 16k-32k 51，每行 `meta.length_band`；**不与 LongBench 官方全量分数横比**）；narrativeqa 上游重复题已按 `(task,input,context)` 合并 answers（200→182）；LCB 混合 150 stdin + 100 functional 两类题面。
- BFCL 覆盖 11 个 single-turn 类别（含 `irrelevance`，其 gold 为空；不含 multi-turn），889 行题干全局唯一；Global-MMLU 为 42 语种 × 100 个**配对** `sample_id`（同一组题的各语言翻译），语言间差异是配对比较、不是 42 个独立样本。
- manifest 另含 `status`/`runner`/`strata_counts`/`sampling`/`deviations`/`judges.implementation_status`，可审计每场景分层、去重、跨场景排除与 11 条口径偏差。

## 5. 溯源与复现

- 每份 manifest 的 `provenance` **只记录该测试集实际使用的上游数据集**（performance 5 个/7 文件、accuracy 8 个/70 文件），含 `source_url`、`license`、逐文件 `bytes` 与 `sha256`。
- token 计数优先使用本地离线模式（oMLX.app 自带 Python + 模型 tokenizer 文件），不可用时回退 oMLX `/v1/tokenize`；两种模式共用 `.cache/tokenize.json`。

```bash
cd "/Users/huawei/Desktop/模型管家相关资料/modelbenchmark"
PY=.venv/bin/python

$PY tool/dataset/build_perf.py          # 480 + manifest
$PY tool/dataset/build_perf_subset.py   # 240 + manifest_10
$PY tool/dataset/build_acc.py           # 6,589 + manifest
$PY tool/dataset/verify.py              # 只读校验（73,301 checks，不写产物）

./modelbench validate --offline
./modelbench validate --offline --testset performance_10
$PY -m harness.gate
```

## 6. 使用约定

- **性能 run 不判分**：`score` 恒为 null；冷缓存；1K/8K/32K 分档报告，不跨档合并。
- **精度不做 run**：6,589 行是离线数据资产（`manifest.status=offline-asset`）；若未来接入，`judge` 决定判题路径，判题失败单列，不并入正确率。
- **不混算**：`constructed=true` 的性能样本只用于长度压力；模型 thinking on/off 分开报。
- **同源不可独立**：`performance_10` 是 `performance` 的子集，两者不是两份独立证据。
- **边界**：LiveCodeBench 使用本地最新 release 增量窗口（v5+v6，选中题目 2024-11-22~2025-04-06），隐藏测试中单条 JSON >256KB 的压测用例省略，pass@1 在保留测试上计算，不可与官方榜横比；Global-MMLU 为本地 42 语种 × 100 配对题（en 100 行标 `is_en_reference`，宏平均时单列），不可外推为 42 个独立样本；long_context 的 4K 下界与长度分层是**抽样设计**（只保留 4K–32K），不可与 LongBench 官方全量分数横比；性能长文切片不承诺答案可定位。
