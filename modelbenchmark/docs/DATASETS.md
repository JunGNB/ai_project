# 数据集来源与本地缓存（DATASETS.md）

本文件记录 benchmark 的数据源、构建方式、样本预算、溯源与校验口径。
**唯一权威来源**是 `dataset/TEST_SAMPLE_SELECTION.md`（设计）与两份 manifest（统计/溯源）：

- `dataset/test/performance/manifest.json`、`dataset/test/performance/manifest_10.json`
- `dataset/test/accuracy/manifest.json`

上游原始文件位于 `dataset/raw/`，只读；构造器位于 `tool/dataset/`。

## 1. 数据流

```
dataset/raw/<dataset>/...                    上游原始文件（只读）
        │  tool/dataset/build_perf.py        （token 计数 + 构造 + 质量断言）
        │  tool/dataset/build_perf_subset.py （派生 10 条/档子集）
        │  tool/dataset/build_acc.py         （精度 6,589 行）
        ▼
dataset/test/performance/{testset.jsonl, testset_10.jsonl, manifest*.json}
dataset/test/accuracy/{testset.jsonl, manifest.json}
        │
        ▼
运行时：src/modelbench/adapters/datasets/perf_jsonl.py（性能）
        src/modelbench/adapters/datasets/…（legacy 精度适配器，不在默认 run 中）
```

性能测试集是**运行时唯一默认数据源**；精度 testset 是离线数据资产（当前版本
`docs/TEST_PLAN.md` 说明精度判分不接入 run）。当前构造器版本见 manifest 的
`generator_version`：`tool/dataset/build_perf.py v4`、`build_perf_subset.py v4`、
`build_acc.py v5`。

## 2. 默认样本预算（1 模型）

| 测试集 | 文件 | 规模 | cells |
|---|---|---|---|
| `performance` | `dataset/test/performance/testset.jsonl` | **480** = 8 场景 × 1K/8K/32K × 20 | 24 |
| `performance_10` | `dataset/test/performance/testset_10.jsonl` | **240** = 8 场景 × 1K/8K/32K × 10 | 24 |
| 精度 | `dataset/test/accuracy/testset.jsonl` | **6,589** = 250×6 + 889 + 4,200 | 8 场景 |

`performance_10` 是 `performance` 的确定性子集（每场景 `target_id` 排序取偶数位 10 个），
两者同源、不可作为两份独立证据。

## 3. 性能数据源与构造

| 场景 | 数据源 | 构造 | `constructed` |
|---|---|---|---|
| knowledge_qa | LongBench v2 `Single-Document QA` | 同一文档前缀切片 + 原题/选项 | 否 |
| long_context | LongBench v2 `Long Structured Data Understanding` | 同上 | 否 |
| coding | LongBench v2 `Code Repository Understanding` | 同上 | 否 |
| search_agent | LongBench v2 `Multi-Document QA` | 同上 + 固定 search/lookup 工具 schema（单请求） | 否 |
| chinese_long_context | LongBench v1 `dureader`（**中文单语**） | 1K 切目标文档；8K/32K 目标文档开头 + 其他同任务文档拼接 | 8K/32K 是 |
| instruction_following | IFEval | 目标指令 + 同集指令参考块（池不足 32K 时循环一次） | 是 |
| math_logic | GSM8K | 目标题 + 同集已解例题（排除目标 20 题） | 是 |
| general_agent | BFCL v4 | 目标请求 + 函数目录前缀（目标函数优先、同类别次之；whole-function 填充，非全量目录） | 是 |

- 同一 `target_id` 三档的题面/指令/用户请求逐字符一致；`tokens = messages_tokens + tools_tokens`，每档按**整数** token 落在目标 ±5%：`floor(0.95N) ≤ tokens ≤ floor(1.05N)`（1K = 972–1075）。
- 场景名是负载形状而非能力名；`multilingual` 已更名为 `chinese_long_context`（原 dureader 只有中文）。
- 性能样本只测 prefill/decode 形状：长文为前缀切片，不承诺答案可定位，不用于能力结论。
- 32K 档声明 32,455–32,779 token；运行时另需 242 token（OSL 50 + 安全边距 128 + decode guard 64），需模型窗口 32,697–33,021；32,768 窗口模型仅 1/160 可测，其余 159 行 `not-testable`（不截断）。
- 性能 `manifest` 另有 `run_defaults`（5 键：max_tokens 50 / decode_guard true / warmup_per_model 1 / warmup_requests 0 / repeats 1）、`accept_bounds/accept_rule`、`window_requirements`、`sampling`（逐条跳样与去重）、`workloads`。
- `tools` 在构造期递归规范化为 JSON Schema（`dict→object`、`float→number`、`tuple→array`）。

## 4. 精度数据源与判题契约

| 场景 | 数据集 | n | judge | 指标 |
|---|---|---:|---|---|
| knowledge_qa | MMLU-Pro | 250 | `mc_letter` | acc@1 |
| instruction_following | IFBench | 250 | `ifbench` | prompt/instruction × strict/loose acc |
| long_context | LongBench v1（narrativeqa/qasper/hotpotqa/2wikimqa，4K–32K 长度分层 100/99/51） | 250 | `qa_f1` | F1（多参考取 max）/ EM |
| math_logic | MATH-500 | 250 | `math` | acc（数学等价） |
| coding | LiveCodeBench（最新 release 窗口 v5+v6，2024-11~2025-04） | 250 | `code_pass1` | pass@1（隐藏测试；150 stdin + 100 LeetCode functional） |
| general_agent | BFCL v4 single-turn（11 类，含 irrelevance） | 889 | `bfcl_ast` | AST 匹配 acc |
| search_agent | HotpotQA distractor 规范子集 | 250 | `qa_f1` | EM / F1 |
| multilingual | Global-MMLU（本地 42 语种 × 100 配对题） | 4,200 | `mc_letter` | acc@1（逐语种 + 宏平均） |

判题输入全部在行内：`choices`/`answer`、`meta.answers`、`meta.instruction_id_list`+`kwargs`+`ifbench_key`、
`meta.tests`（隐藏测试）、`meta.possible_answers`、`meta.paragraphs`+`supporting_facts`；每行 `meta.upstream_id` 可回指上游。

**状态：离线数据资产（`manifest.status=offline-asset`）**。当前 `docs/TEST_PLAN.md` 只运行性能基准，accuracy 不接入 runner；`judge` 是契约描述（`implementation_status=contract-only`），判题实现尚未接入。长上下文使用官方 LongBench 模板，只保留 **4K–32K** prompt token（不截断）并按长度分层抽样（4k-8k 100 / 8k-16k 99 / 16k-32k 51，每行 `meta.length_band`；该 4K 下界与分层是抽样设计，不可与 LongBench 官方全量分数横比）；narrativeqa 上游重复题已合并 answers（200→182）；MMLU-Pro 为 0-shot CoT（`Answer: X`，无 `$LETTER`）；LiveCodeBench 为 150 stdin + 100 functional 两类题面，隐藏测试保留 9,184 条（>256KB 的 331 条省略并记录）；BFCL 覆盖 11 个 single-turn 类别（含 `irrelevance`，gold 为空；不含 multi-turn），889 行题干唯一；Global-MMLU 为 42 语种 × 100 个配对 `sample_id`（同一组题的各语言翻译），语言间差异是配对比较。`manifest` 另有 `runner`、`strata_counts`、`sampling`、`deviations`（11 条）。

## 5. 溯源与校验

每份 manifest 的 `provenance` **只记录该测试集实际使用的上游数据集**（performance 5 个/7 文件；accuracy 8 个/70 文件），由 `tool/dataset/config.py` 的 `provenance_for(set_name)` 按测试集过滤，每个数据集：

```json
{
  "source_url": "https://huggingface.co/datasets/...",
  "license": "MIT",
  "files": [{"logical_path": "test.parquet", "bytes": 4144185, "sha256": "..."}]
}
```

校验：

```bash
cd "/Users/huawei/Desktop/模型管家相关资料/modelbenchmark"
PY=.venv/bin/python

# 重建（幂等；token 计数走本地离线模式，失败回退 oMLX /v1/tokenize）
$PY tool/dataset/build_perf.py
$PY tool/dataset/build_perf_subset.py
$PY tool/dataset/build_acc.py

# 结构/测试集解析
./modelbench validate --offline
./modelbench validate --offline --testset performance_10

# 只读数据集校验（73,301 checks；不重建、不写产物）
$PY tool/dataset/verify.py

# 文档/路径/测试门禁
$PY -m harness.gate
```

## 6. legacy 说明

仓库保留了早期按数据集划分的 adapter（`src/modelbench/adapters/datasets/` 下的
`mmlu`、`gsm8k`、`ifeval`、`humaneval`、`mmmlu`、`hotpotqa`、`bfcl` 等）。
它们只用于历史对照，**不是默认 run 的数据源**；默认运行由 `perf_jsonl` 读取预构造测试集。
`dataset/raw/download_manifest.json` 是历史下载记录，**不再作为在线校验依据**——
当前以两份 testset manifest 中的逐文件 SHA-256 为准。
