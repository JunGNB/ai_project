# 结果数据管理（RESULTS）

> 本文定义 run bundle 的目录结构、`output/index.json` 目录索引、`analysis/` 二次分析投影
> 与使用方式。Web 控制台与 `./modelbench runs` 都以本页为准。

## 1. 目录约定

```text
output/
  index.json                      # 目录索引（runs + batches，工具/分析入口）
  batches/<batch_id>.json         # 一次测试动作的分组元数据（含各模型 run_id）
  result_analysis/<batch_id>_<method>_<models>/  # Excel 结果总结（见 §6）
  <batch_id>_<method>_<model>/    # 每个模型一个 bundle：时间 + 测试方法 + 模型
  <run_id>/                       # 旧版（未分批）run，仍可读取
    manifest.json                 # 配置快照、硬件、oMLX 快照、模型、git commit
    hardware.json                 # 芯片/内存/GPU/OS/oMLX 版本 + upload_fields
    omlx.json                     # oMLX 版本、全局设置、每模型 settings/flags、upload_projection
    models.json                   # /v1/models/status 全量
    records.jsonl                 # 逐请求权威原始证据（最大文件，含请求/响应/指标）
    summary.json                  # 按 (model, 档位) 的 overall + cells + 逐样本 samples
    summary.csv                   # 整体平均值（每模型×上下文档位一行）
    cells.csv                     # 数据集粒度
    samples.csv                   # 逐样本结果（含输入上下文大小）——二次分析主表
    report.md                     # 人读摘要
    analysis/                     # 小体积分析投影（新 run 才有）
      run.json                    # 运行元数据（模型/跳过项/perf/run/计数）
      samples_long.csv            # 长表：每行 (sample, metric, value)
      environment.json            # 硬件/oMLX/feature flags/参数
      README.md                   # 字段与 pandas 示例
    raw/samples/                  # 未截断的原始测试样本
    raw/<model>/<scenario>/<bucket>/  # 逐记录快照（bucket=1k/8k/32k；legacy 为 native）
    logs/run.log                  # 运行日志（含模型加载/跳过/复位原因）
```

命名规则（新布局）：

- `batch_id`：一次测试动作的批次标识，默认是 UTC 时间戳（`YYYYMMDDTHHMMSSZ`），
  也可用 `--run-id` 指定；
- `method`：测试方法，目前是 `performance`（未来精度测试用 `accuracy`）；
- `model`：模型 alias/id 的安全化形式；
- 例：`output/<batch_id>_performance_<model>/`；对应 Excel 目录为
  `output/result_analysis/<batch_id>_performance_<model>/`（多模型用 `_` 连接）。
- 同一批次选择多个模型时，**每个模型写自己的 bundle**（records/summary/analysis/raw 互不混合），
  通过 `batches/<batch_id>.json` 与 `index.json.batches` 关联。

规则：

1. **run_id 不可变**：目录名就是标识；不要在 `output/` 里手改或移动正在运行的目录。
2. **原始证据优先**：`records.jsonl` 永不被改写；`summary.*` 可由它复算。
3. **索引是派生物**：`output/index.json` 可随时重建（Web「重建索引」或 `./modelbench index`），
   即使删除/新增历史目录也能恢复。
4. **旧 run 兼容**：schema 1.x 的旧 run 仍会进入索引（标记 `legacy=true`），
   Web 详情可查看 cells，但没有 `overall`/`analysis/`。
   旧 schema 2.0 run（`overall` 无 `context_length`）在 Web 中显示为 `全部*`，表示
   “跨 1K/8K/32K 合并的历史值”，**不要与新口径按档位结果混比**；旧 bundle 里已落盘的
   `summary.csv`/`report.md` 仍是合并值，需要按档位结论时请从 `records.jsonl`
   用当前 `aggregate_overall` 复算，或直接用新版本重跑。
5. **索引写入有锁**：并发 job 更新索引时会在 `output/` 下使用 `.index.lock`（0 字节临时文件）
   串行化 read-modify-write，避免并发 upsert 丢失条目；该文件不参与 `n_runs`，可安全删除。

## 2. `output/index.json`

```json
{
  "schema_version": "1.0",
  "generated_at": "...",
  "output_dir": ".../output",
  "n_runs": 28,
  "n_batches": 9,
  "batches": [{"batch_id": "20261105T155245Z", "method": "performance",
               "n_models": 2, "n_samples": 2, "run_ids": ["..."]}],
  "runs": [{
    "run_id": "web-20261104T085646Z-ca93",
    "schema_version": "2.1",
    "mode": "performance",
    "created_at": "...",
    "duration_s": 5484.4,
    "models_selected": ["minicpm5-1b", "minicpm5-2b"],
    "models_measured": ["MiniCPM5-1B"],
    "n_models_selected": 2,
    "n_models_measured": 1,
    "n_skipped": 1,
    "skipped": ["model MiniCPM5-2B: load ... HTTP 507 ..."],
    "datasets": ["perf_jsonl", "..."],
    "n_samples": 480,
    "memory_errors": 0,
    "files": {"summary.json": {"size": 429140}, "...": "..."},
    "analysis": [{"name": "samples_long.csv", "size": 123}],
    "bytes_total": 142716928,
    "bytes_total_human": "136.1 MB",
    "legacy": false
  }]
}
```

用途：

- 同一批次的多模型结果通过 `batch_id` 关联，Web 历史按「批次 → 模型」分组显示；
- Web 历史列表直接读索引，不再逐个解析大 `summary.json`；
- 脚本/笔记本用 `output/index.json` 枚举 run，再按需读取 `samples.csv`；
- `models_selected` 与 `models_measured` 的差值解释"为什么勾选两个模型但只有一份结果"
  （另一个模型加载失败被跳过，原因在 `skipped`）；新布局下每个模型仍各自有目录。

## 3. 二次分析用法

```python
import pandas as pd

run = "output/verify-final-01"

# 逐样本主表（推荐）
samples = pd.read_csv(f"{run}/samples.csv")
print(samples.groupby("model")[["ttft_ms", "gen_tps", "peak_memory_gb"]].mean())

# 按数据集/输入上下文分层
print(samples.groupby(["dataset", "model"])["input_context_tokens"].describe())

# 长表 + pivot（适合画图/BI）
long = pd.read_csv(f"{run}/analysis/samples_long.csv")
pivot = long.pivot_table(
    index=["model", "dataset", "sample_id"], columns="metric", values="value"
)

# 环境信息
import json
env = json.load(open(f"{run}/analysis/environment.json"))
print(env["hardware"]["upload_fields"], env["omlx"]["feature_flags"])
```

跨 run 对比：

```bash
./modelbench runs                     # 按批次分组列出：batch -> 每个模型一行
./modelbench runs --json > /tmp/runs.json
./modelbench analyze output/<batch_id>_performance_<model_a> \
                    output/<batch_id>_performance_<model_b>
```

## 4. Web 入口

- 历史运行列表：点击 run 打开详情（整体/数据集/逐样本/环境/报告）。
- 「原始文件」：直接打开 `summary.json`、`samples.csv`、`report.md` 等；
  `records.jsonl` 体积大，默认下载。
- 「Finder」：在 macOS Finder 中打开对应 run 目录。
- 「重建索引」：重建 `output/index.json`；「打开输出目录」：打开 `output/`。

## 5. 保留与归档建议

- 不要删除 `output/index.json` 之外的正在运行目录；历史 run 可整体移动到外部归档盘，
  需要时放回 `output/<run_id>` 再重建索引即可。
- 需要长期保留的分析结果，优先复制 `analysis/`、`samples.csv`、`summary.json`
  （体积小），`records.jsonl` 与 `raw/` 可作为冷备。
- 跨机器/跨版本对比前，先看 `manifest.json` 的 `git_commit`、`omlx.json` 的
  `feature_flags`/settings 与 `hardware.json` 的 `upload_fields` 是否一致。

## 6. Excel 结果总结（result_analysis）

每个批次完成后会自动生成：

```text
output/result_analysis/<batch_id>_<method>_<models>/
  summary.xlsx     # 3 个页签
  batch.json       # 批次元数据副本
  README.md        # 页签/列说明
```

`summary.xlsx` 页签：

| 页签 | 内容 |
|---|---|
| 1 `硬件配置` | 批次/测试配置、芯片/内存/GPU/OS/oMLX 版本、oMLX 全局设置、每个模型的生效配置（量化/上下文/MoE/feature flags） |
| 2 `模型性能平均` | 每个模型一行：样本/有效/错误数、TTFT 均值/中位数/P95、TG TPS、TPOT、E2E、Prefill、峰值内存均值/最大、峰值 CPU、输入/输出 token 均值、耗时、跳过/内存错误数 |
| 3 `详细数据` | 每个 measured 样本一行：测试模型、模型ID、**样本来源（dataset）**、**样本所属场景（scenario）**、样本ID、重复、**API 输入内容**、**输入 prompt token 大小**、输入字符数、**输出结果**、**输出 token 大小**、输出字符数、**峰值内存占用**，以及 TTFT/TG TPS/TPOT/E2E/Prefill/Total TPS/CPU/cached/状态/截断/错误/时间戳 |

说明：

- 生成时机：`BatchRunner` 在批次结束后自动生成；失败只记 warning，不影响 benchmark 结果；
- 重新生成：

```bash
./modelbench export-analysis <batch_id>
./modelbench export-analysis --all
./modelbench export-analysis <batch_id> --output-dir DIR
```

- 超长文本（long_context 的 API 输入/输出）超过 Excel 单元格上限时会保留头尾并标注原长度。
- Excel 依赖 `openpyxl`（已加入项目依赖）。写入前会清洗 XML 非法控制字符（如 `\x07`/`\x0b`），
  否则 openpyxl 会抛 `IllegalCharacterError`，批次只留下空目录并把原因写进
  `output/batches/<batch_id>.json` 的 `result_analysis_error`；此时可修复后用
  `./modelbench export-analysis <batch_id>`（或 `--all`）重新生成。
