# dataset/ — 本地模型测评数据集

本目录是当前唯一的数据根：`dataset/raw/` 是 gitignored 的上游输入，
`dataset/test/` 是可重建的测试集产物。数据由 `tool/dataset/` 的确定性构造器生成：

| 测试集 | 产物 | 规模 | 状态 |
|---|---|---|---|
| 性能 `performance` | `dataset/test/performance/{testset.jsonl,manifest.json,testset.yaml}` | 240 = 8 场景 × 1K/8K/32K × 10，24 cells | 默认参与 run |
| 精度 `accuracy` | `dataset/test/accuracy/{testset.jsonl,manifest.json}` | 2,000 = 8 场景 × 250 | **隐藏离线资产**（`status=offline-asset`、runner 未接入、无 `testset.yaml`） |

## 1. 目录

```
dataset/
├── README.md                    # 本文件
├── raw/                         # 上游原始文件（只读、gitignored、不提交）
└── test/
    ├── performance/
    │   ├── testset.jsonl        # 240 行（派生、gitignored、可重建）
    │   ├── manifest.json        # 抽样/统计/数据源 SHA-256
    │   └── testset.yaml         # 测试集定义（提交）
    └── accuracy/
        ├── testset.jsonl        # 2,000 行（派生、gitignored、可重建）
        └── manifest.json        # 抽样/统计/数据源 SHA-256（精度集保持隐藏）
```

`dataset/raw/` 与 `dataset/test/**/testset.jsonl` 不提交 git；新 clone 先按 §2 下载 raw，
再用 §3 重建。`dataset/raw/` 里保留历史/备用数据集（CEval、MMLU 等），
但当前两套测试集只读取各自的 `raw_spec.json`；这些额外目录不进入 manifest。

## 2. 下载并校验 raw

```bash
PY=.venv/bin/python                       # Windows: .venv\Scripts\python.exe
$PY tool/dataset/performance/fetch.py             # 18 个文件（默认 hf-mirror.com）
$PY tool/dataset/performance/fetch.py --verify    # 只读 size + SHA-256
$PY tool/dataset/accuracy/fetch.py                # 50 个文件（含 tokenizer / NLTK）
$PY tool/dataset/accuracy/fetch.py --verify
```

下载先写 `<目标>.part`，校验 size + SHA-256 后原子改名；`--dry-run` 只打印计划。
`--raw-dir` 可覆盖目标目录，`--mirror` 可切回 HF 官方。

## 3. 重建测试集

```bash
$PY tool/dataset/performance/build.py            # 240 行 + manifest + testset.yaml
$PY tool/dataset/performance/build.py --check    # 行数/target/三档/边界/SHA 复算
$PY tool/dataset/accuracy/build.py               # 2,000 行 + manifest（不写 testset.yaml）
$PY tool/dataset/accuracy/build.py --check       # 行数/字段/upstream_id 唯一/SHA
```

同一份 `dataset/raw/` 重跑得到逐字节相同的结果（固定 seed=42、无时间戳）；manifest
记录每个上游文件的 `source_url`、许可、字节数与 SHA-256。

## 4. 使用约定

- **性能 run 不判分**：`score` 恒为 null；逐请求冷缓存；按 `(model, context_length)` 分档报告，
  1K/8K/32K 不跨档合并。
- **精度集保持隐藏**：没有 `testset.yaml`，不会出现在 Web/CLI 测试集列表；manifest 的
  `status=offline-asset`、`runner=not wired`，不得引用任何“精度分数”。
- **不可横比**：性能默认 OSL=50、模型级 warmup 1 次、per-cell warmup 0、`repeats=1`、
  逐请求冷缓存，与 vLLM / oMLX 官方 benchmark 口径不同。
- **设计文档**：[docs/spec/11-performance-testset.md](../docs/spec/11-performance-testset.md)、
  [docs/spec/12-accuracy-testset.md](../docs/spec/12-accuracy-testset.md)。

## 5. 数据流与数据源许可

```
dataset/raw/<dataset>/...                    上游原始文件（只读、gitignored）
        │  tool/dataset/performance/build.py  （token 计数 + 构造 + 质量断言）
        │  tool/dataset/accuracy/build.py     （2,000 行精度资产，隐藏）
        ▼
dataset/test/performance/{testset.jsonl, manifest.json, testset.yaml}
dataset/test/accuracy/{testset.jsonl, manifest.json}
        ▼
运行时：src/modelbench/adapters/datasets/perf_jsonl.py（性能）
```

两侧合并许可表（**以两份 manifest 的 `provenance`/`datasets` 字段为准**，表内为人工核对副本）：

| 数据集 | 上游 | 许可 | 用途 |
|---|---|---|---|
| LongBench v2 | https://huggingface.co/datasets/zai-org/LongBench-v2 | Apache-2.0 | 性能 4 个场景 |
| LongBench v1 | https://huggingface.co/datasets/zai-org/LongBench | MIT | 精度长上下文 |
| IFEval | https://github.com/google-research/google-research/tree/master/instruction_following_eval | Apache-2.0 | 性能参考池 + 精度指令 |
| GSM8K | https://huggingface.co/datasets/openai/gsm8k | MIT | 性能数学 + 精度数学（en） |
| CMATH | https://github.com/XiaoMi/cmath | 见上游仓库（研究用途） | 精度数学（zh） |
| HumanEval | https://github.com/openai/human-eval | MIT | 精度编程 |
| MBPP | https://github.com/google-research/google-research/tree/master/mbpp | CC-BY-4.0 | 精度编程 |
| BFCL v4 | https://github.com/ShishirPatil/gorilla | Apache-2.0 | 性能/精度 Agent |
| HotpotQA | https://huggingface.co/datasets/hotpotqa/hotpot_qa | CC-BY-SA-4.0 | 精度搜索 |
| Global-MMLU | https://huggingface.co/datasets/CohereLabs/Global-MMLU | Apache-2.0 | 性能多语言 + 精度知识/多语言 |
| MiniCPM5-1B tokenizer | 本地模型资产 | 随上游模型许可 | 参考 tokenizer |

逐文件 `path/bytes/sha256` 见两份 manifest。**内容风险**：上游含用户贡献内容
（HotpotQA/LongBench 网页摘录等），仅限本地研究与基准测试用途，不得再分发 raw 数据。

## 6. 校验

```bash
./modelbench validate --offline
$PY -m pytest test/test_perf_dataset.py test/test_perf.py
$PY -m harness.gate
```
