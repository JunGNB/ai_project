# tool/ — 数据构造与离线工具

`tool/` 只放工程脚本，不被运行时导入（除测试外）。当前数据根为 `dataset/`：

```
tool/
├── bootstrap.py                 # `./modelbench setup` 的安装实现
├── common/
│   ├── fs.py                    # 路径（dataset/raw、dataset/test、output）+ JSONL/SHA-256
│   └── tokenizer.py             # MiniCPM5-1B 参考 tokenizer（内容 token）
├── dataset/
│   ├── accuracy/                # 隐藏精度集（8 × 250）
│   │   ├── fetch.py             # 下载/校验 → dataset/raw/
│   │   ├── raw_spec.json        # 精度集 50 个文件的 path/url/bytes/sha256
│   │   ├── build.py             # 2,000 行 + manifest（不写 testset.yaml）
│   │   └── {sampling,sources,prompts}.py
│   └── performance/             # 性能集（8 场景 × 1K/4K/32K × 10 = 240）
│       ├── fetch.py             # 下载/校验/派生 cl100k → dataset/raw/
│       ├── derive_cl100k.py     # 由官方 tiktoken blob 确定性生成参考 tokenizer
│       ├── build.py             # 240 行 + manifest + testset.yaml（cl100k 超量基线）
│       ├── align.py             # 可选预检：按目标模型 tokenizer 验证 240 行能否精确对齐
│       └── {sampling,sources}.py
├── migrate/
│   └── legacy_v2_to_v3.py       # 一次性：把 archive/legacy-v2 的 schema-2.x bundle 重序列化为 3.0
├── system/                      # 判题工具（精度 run 用 domain/judges；这里保留自检/试点/兼容转发）
│   ├── judges.py                # 兼容转发 → src/modelbench/domain/judges/
│   ├── validate_judges.py
│   └── pilot.py
└── download/                    # 旧数据层兼容下载入口（不是 dataset/raw 初始化路径）
```

## 1. 性能集（默认）

```bash
PY=.venv/bin/python                       # Windows: .venv\Scripts\python.exe
$PY tool/dataset/performance/fetch.py
$PY tool/dataset/performance/fetch.py --verify
$PY tool/dataset/performance/build.py
$PY tool/dataset/performance/build.py --check
$PY tool/dataset/performance/align.py \\
    --tokenizer /path/to/model/tokenizer.json --base-url http://127.0.0.1:8000 --model <model-id>
```

`build.py` 用通用参考 tokenizer（cl100k_base）把正文**超量**构建到 2× 档位
（1K→2,048、4K→8,192、32K→65,536 token），任务块单独成第二条 user 消息；**runner 在发送前按目标
模型 tokenizer 把 body 裁剪/补齐**到 1,024/8,192/32,768 精确等值（0 容差，含 warmup 学到的
`usage.prompt_tokens - /v1/tokenize` 常量偏移），不跳过、不做 ±10% 误差带；结果逐样本记录
`context_meta.prompt_alignment`。`align.py` 仅作可选预检（离线/服务器两种模式），不再是跑分前置条件；
只有 tokenizer 不可用/无法对齐才记 `runtime_alignment` skip。oMLX 有鉴权时，`align.py` 取
`--api-key`、`OMLX_API_KEY` 或 `modelbench` 配置中的 `server` key（与 runner 同源）。

产物：`dataset/test/performance/{testset.jsonl,manifest.json,testset.yaml}`；
`testset.jsonl` 与 `dataset/raw/` 不提交 git。场景 8 为 Global-MMLU `multilingual`（10 语种）。

## 2. 精度集（`method=accuracy`）

```bash
$PY tool/dataset/accuracy/fetch.py
$PY tool/dataset/accuracy/fetch.py --verify
$PY tool/dataset/accuracy/build.py
$PY tool/dataset/accuracy/build.py --check
```

产物：`dataset/test/accuracy/{testset.jsonl,manifest.json,testset.yaml}`；
`manifest.status=wired`、`runner=accuracy`、`hidden=false`，测试集出现在 Web/CLI 列表并接入
`./modelbench run --method accuracy`（独立 run_id，不与性能 run 混算）。`testset.yaml` 随仓库提交，
`testset.jsonl` 与 `dataset/raw/` 不提交。`build.py --check` 校验行数/id/判题族/SHA-256 与
`upstream_id` 唯一性。

## 3. 判题工具（精度 run 的单一事实源）

`src/modelbench/domain/judges/` 实现 5 个确定性判题族（`mc_letter` / `ifeval` /
`normalized_match` / `code_exec` / `tool_call`），由 `AccuracyOrchestrator` 在精度 run 中逐样本调用；
`tool/system/judges.py` 仅保留兼容转发（构造自检脚本使用）。`validate_judges.py` 用 gold 答案做
判题自检（1,750 正例 + negative controls）；`pilot.py` 做小样本模型试点。判题**无 LLM judge、
无网络**；判题异常/依赖缺失落 `judge_error`（计 0 分且留在分母），不静默剔除。

## 4. 路径约定

- 所有构造器通过 `tool/common/fs.py` 的 `DATA_DIR` / `RAW_DIR` / `TEST_DIR` 定位数据，
  rename 数据根时只需改这一处；
- 性能集参考 tokenizer 为通用 `cl100k_base`（`dataset/raw/tokenizer/cl100k_base/tokenizer.json`，由官方 tiktoken blob 确定性派生）；精度集参考 tokenizer 见其 `sources.py`；
- `dataset/raw/` 里保留的历史/备用数据集文件不属于当前两套测试集，manifest 只记录
  `sources.py`（性能）与 `raw_spec.json`（精度）中实际使用的文件。
