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
│   │   ├── raw_spec.json        # 50 个文件的 path/url/bytes/sha256
│   │   ├── build.py             # 2,000 行 + manifest（不写 testset.yaml）
│   │   └── {sampling,sources,prompts}.py
│   └── performance/             # 性能集（8 × 1K/8K/32K × 10 = 240）
│       ├── fetch.py             # 下载/校验 → dataset/raw/
│       ├── raw_spec.json        # 18 个文件的 path/url/bytes/sha256
│       ├── build.py             # 240 行 + manifest + testset.yaml
│       └── {sampling,sources}.py
├── migrate/
│   └── legacy_v2_to_v3.py       # 一次性：把 archive/legacy-v2 的 schema-2.x bundle 重序列化为 3.0
├── system/                      # 离线判题工具（当前不接入 runner）
│   ├── judges.py
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
```

产物：`dataset/test/performance/{testset.jsonl,manifest.json,testset.yaml}`；
`testset.jsonl` 与 `dataset/raw/` 不提交 git。场景 8 为 Global-MMLU `multilingual`（10 语种）。

## 2. 精度集（隐藏离线资产）

```bash
$PY tool/dataset/accuracy/fetch.py
$PY tool/dataset/accuracy/fetch.py --verify
$PY tool/dataset/accuracy/build.py
$PY tool/dataset/accuracy/build.py --check
```

产物：`dataset/test/accuracy/{testset.jsonl,manifest.json}`；**没有 testset.yaml**，
`manifest.status=offline-asset`、`runner=not wired`、`hidden=true`，不会出现在 Web/CLI
测试集列表，也不接入 run。`build.py --check` 会校验 `upstream_id` 全局唯一。

## 3. 判题工具（当前不接 runner）

`judges.py` 实现 5 个确定性判题族（`mc_letter` / `ifeval` / `normalized_match` /
`code_exec` / `tool_call`）；`validate_judges.py` 用 gold 答案做自检；`pilot.py` 做小样本
模型试点。它们是离线工具，**没有接入性能 runner**，也不构成可引用的精度结果。

## 4. 路径约定

- 所有构造器通过 `tool/common/fs.py` 的 `DATA_DIR` / `RAW_DIR` / `TEST_DIR` 定位数据，
  rename 数据根时只需改这一处；
- 参考 tokenizer 由两个 `fetch.py` 下载到 `dataset/raw/tokenizer/MiniCPM5-1B/tokenizer.json`；
- `dataset/raw/` 里保留的历史/备用数据集文件不属于当前两套测试集，manifest 只记录
  `raw_spec.json` 中实际使用的文件。
