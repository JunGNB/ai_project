# 结果数据管理（RESULTS）

> 本文是 `output/` 的**现状契约**（bundle schema **3.0** / 索引 schema **2.0**）：
> 目录层次、每个文件的职责与权威性、索引字段、二次分析方式、派生工具与保留策略。
> **`runs/` 是一层平铺**：一个 run 一个目录，目录名就是 `run_id`；批次元数据单独放在
> `output/batches/<batch_id>.json`。Web 控制台、`./modelbench` 的
> `runs/analyze/export-analysis/gc/verify/rebuild/pack` 都以本页为准；布局的唯一代码实现在
> `src/modelbench/adapters/storage/layout.py`。设计依据见
> [OUTPUT_LAYOUT.md](../archive/OUTPUT_LAYOUT.md)，测量口径见 [20-methodology.md](../spec/20-methodology.md)。

---

## 1. 总览

```text
output/
├── index.json                              # D  全局索引（唯一入口，可重建）
├── blobs/                                  # M  全局内容寻址池（跨 run/批次去重）
│   └── prompts/
│       └── <sha[:2]>/<sha:64>.json         # M  实际发送的 messages（正文只存一份）
├── runs/                                   # M  全部运行结果：一层平铺
│   └── <run_id>/                           # M  一个模型一次运行；目录名 == run_id
│       ├── run.json                        # M  运行元数据 + 唯一环境快照
│       ├── MANIFEST.json                   # M  产物清单 + sha256（最后写）
│       ├── records.jsonl                   # M  权威逐样本证据（prompt 只存摘要）
│       ├── samples.csv                     # D  逐样本主表（Web 分页/pandas）
│       ├── aggregates.json                 # D  overall + cells
│       ├── report.md                       # D  人读摘要
│       └── logs/run.log                    # O  运行日志
├── batches/                                # M  批次元数据（一次测试动作 = 一个文件）
│   └── <batch_id>.json                     # M  权威批次元数据（相对路径、无绝对路径）
├── exports/                                # D  人类交付物（batch 结束自动生成，可整目录删除）
│   └── <batch_id>.xlsx
├── archive/                                # A  退役 run/旧格式 bundle（仍是证据，GC 引用根）
└── .state/                                 # J  运行时状态（非证据，可清空）
    ├── index.lock
    ├── benchmark.lock
    └── tmp/
```

`output/` 根固定 **7 个条目**：`index.json`、`blobs/`、`runs/`、`batches/`、`exports/`、
`archive/`、`.state/`（`make build` 产生的 `output/build/` 是构建临时目录，不属证据树，
可随时删除，不参与索引与体积统计）。

图例：**M** 权威（不可再生，只存一份）｜**D** 派生（可由权威数据重建）｜
**O** 运维｜**A** 归档（证据，保留，且是 GC 引用根）｜**J** 运行时/垃圾。

**为什么是平铺**：`batch_id` 本身已含日期，`method` 单独成层没有信息量。压平后
`ls output/runs/` 一次看到全部结果；同一次测试动作产生的多个模型 run 靠 `run_id` 前缀
（`<batch_id>__…`）、`batches/<batch_id>.json` 的 `model_runs[].run_id` 与
`run.json.batch_id` 三处关联，不再需要“批次目录”这一层。

`ls output/runs/` 的真实示例（本仓库历史 8 个 run，由旧 2.x bundle 归档名换算）：

```text
20261107T032029Z-4184__performance__gemma-4-e4b-oq4/
20261107T032029Z-4184__performance__minicpm-v-4.7-1b/
20261107T032029Z-4184__performance__qwen3.6-35b-a3b-oq4/
20261107T115732Z-215b__performance__gemma-4-e4b-oq4/
20261107T115732Z-215b__performance__minicpm-v-4.6/
20261107T115732Z-215b__performance__minicpm-v-4.7-1b/
20261107T115732Z-215b__performance__minicpm5-2b/
20261107T115732Z-215b__performance__qwen3.6-35b-a3b-oq4/
```

**四条硬规则**

1. **一份事实只落一个文件**：prompt 正文只在 `blobs/` 一份；环境快照只在 `run.json` 一份；
   批次元数据只在 `batches/<batch_id>.json` 一份；逐样本只有 `records.jsonl`（权威）与
   `samples.csv`（投影）。
2. **权威与派生物分层**：`records.jsonl` 永不被改写；`samples.csv`/`aggregates.json`/
   `report.md`/`MANIFEST.json` 都可以用 `./modelbench rebuild` 从它重建。
3. **索引是派生物**：`index.json` 可随时重建（`./modelbench index`），删除后不丢信息。
4. **`output/` 是证据目录**：只读、不提交 git；删除/搬移归档按 [AGENTS.md](../../AGENTS.md) §8
   需人类确认。

---

## 2. 命名规则

| 标识 | 规则 | 例 |
|---|---|---|
| `batch_id` | `<UTC 时间戳>-<4 hex>`；不带 `web-` 前缀，来源写进 `batches/<batch_id>.json` 的 `source` | `20261107T032029Z-4184` |
| `method` | 固定枚举：`performance`（精度 `accuracy` 未来同构） | `performance` |
| `model_slug` | 模型 alias/id 的 `sanitize_name()`（非字母数字转 `_`） | `gemma-4-e4b-oq4` |
| `run_id` | **单段**稳定键 `<batch_id>__<method>__<model_slug>`，不含 `/`；**同时是 `runs/` 下的目录名** | `20261107T032029Z-4184__performance__gemma-4-e4b-oq4` |
| `run_path` | 相对 `output/` 的路径，索引与 Web/CLI 导航用，恒为 `runs/<run_id>` | `runs/20261107T032029Z-4184__performance__gemma-4-e4b-oq4` |
| 磁盘目录 | `runs/<run_id>/`（一层，无日期/方法中间层） | 同上 |

- `run_id` 刻意做成单段：目录索引以它为主键，Web API 也按 URL 单段解析，且现在**目录名就是
  它**；因此**导航用 `run_path`、索引键用 `run_id`**，两者可互相派生
  （`src/modelbench/adapters/storage/layout.py`）。
- 同批次关联：`run_id` 以 `<batch_id>__` 开头；`batches/<batch_id>.json` 的
  `model_runs[].run_id` 列出成员；每个 `runs/<run_id>/run.json` 的 `batch_id` 再确认归属。
- `run_id` 不含独立的日期段（`batch_id` 已有时间戳），因此把 run 目录移入 `archive/` 后身份不变。
- 由 Web 发起的批次，`BatchRunner` 会把历史 `web-` 前缀剥掉并写 `source="web"`；
  CLI 发起为 `source="cli"`。

---

## 3. 全局目录

### 3.1 `output/index.json`（派生，索引 schema `2.0`）

唯一入口、纯派生：只存**相对 `output/` 的路径**（根从索引文件自身位置推导，不含
`output_dir` 绝对路径），因此整个 `output/` 搬家后仍可解析。由
`src/modelbench/adapters/storage/catalog.py` 写入，字段如下：

```jsonc
{
  "schema_version": "2.0",
  "generated_at": "2026-11-09T13:20:00+00:00",
  "n_batches": 2,
  "n_runs": 8,
  "n_blobs": {"prompts": 240, "bytes": 13520000},
  "totals": {"bytes": 21600000, "files": 299},
  "batches": [{
    "batch_id": "20261107T032029Z-4184",
    "created_at": "2026-11-07T03:20:29+00:00",
    "path": "batches/20261107T032029Z-4184.json",       // 批次元数据文件（相对 output/）
    "methods": ["performance"],
    "models": [{
      "model": "gemma-4-E4B-it-oQ4", "model_id": "gemma-4-E4B-it-oQ4",
      "model_slug": "gemma-4-e4b-oq4",
      "run_id": "20261107T032029Z-4184__performance__gemma-4-e4b-oq4",
      "run_path": "runs/20261107T032029Z-4184__performance__gemma-4-e4b-oq4",
      "status": "ok", "n_samples": 240, "duration_s": 3012.4
    }],
    "run_ids": ["…"], "n_models": 1, "n_samples": 240, "duration_s": 3012.4
  }],
  // 派生扁平行：Web 列表页 / `./modelbench runs` 一次读完即渲染
  "runs": [{
    "run_id": "20261107T032029Z-4184__performance__gemma-4-e4b-oq4",
    "run_path": "runs/20261107T032029Z-4184__performance__gemma-4-e4b-oq4",
    "batch_id": "…", "batch_index": 0, "batch_models_total": 3,
    "method": "performance", "mode": "performance", "legacy": false,
    "model": "gemma-4-E4B-it-oQ4", "model_id": "…", "model_slug": "gemma-4-e4b-oq4",
    "model_size_gb": 5.2, "created_at": "…", "duration_s": 3012.4,
    "status": "ok", "cancelled": false,
    "models_selected": ["…"], "models_measured": ["…"],
    "n_models_selected": 1, "n_models_measured": 1,
    "n_skipped": 0, "skipped": [], "memory_errors": 0,
    "n_samples": 240, "n_records": 240, "n_cells": 24, "datasets": ["…"],
    "bytes_total": 1027604, "bytes_total_human": "1.0 MB",
    "files": {"records.jsonl": {"bytes": 758123, "sha256": "…"}}
  }]
}
```

要点：

- `runs[].run_path` 恒为 `runs/<run_id>`；`batches[].path` 恒为
  `batches/<batch_id>.json`；`batches[].models[].run_path` 同样指向 run 目录。索引内没有
  `run_dir`/`batch_dir` 这类绝对或嵌套路径字段。
- `runs[].files` 直接抄该 run 的 `MANIFEST.json`，重建索引时不需要 `rglob` 全盘 stat；
  `bytes_total` 取 `MANIFEST.totals.bytes`（缺失时回退为 `files` 求和）。
- `batches[]` 是 `batches/<batch_id>.json` 的派生分组视图：即使某批次在写出任何模型 run
  之前就失败，只要批次元数据文件存在，它仍会出现在 `batches[]` 里（`models` 为空），
  不会被隐藏。
- `n_blobs.prompts/bytes` 来自 `blobs/prompts` 池本身；`totals` 为 run 产物字节/文件数 +
  blob 池 + 各批次元数据文件的合计（不是递归 stat）。
- 只索引 schema 3.0 的 `runs/`：`archive/` 里的退役 run 与旧格式 bundle 刻意不可见
  （但仍是 GC 引用根，见 §7）。

### 3.2 `output/blobs/prompts/`（权威，全局唯一）

存放**实际发送给模型的完整输入**，内容寻址，跨 run / 跨批次 / 跨模型 / 跨 repeat 只存一份。
路径 `blobs/prompts/<sha[:2]>/<sha>.json`；文件内容：

```jsonc
{
  "sha256": "<64 hex>",
  "chars": 12034,                    // 各 message.content 长度之和
  "messages": [                      // 精确等于发送时的 request.messages
    {"role": "user", "content": "…", "name": null, "tool_call_id": null, "tool_calls": null}
  ],
  "tools": null,                     // 场景带 tools 时一并纳入哈希
  "chat_template_kwargs": {"enable_thinking": false},
  "decode_guard_version": 1
}
```

- **自校验**：`sha256` 是「去掉 `sha256` 字段后的规范化 JSON」的 SHA-256
  （`sort_keys=True`、`separators=(",", ":")`、`ensure_ascii=False`、UTF-8、无尾换行），
  即文件内容可独立验证（`src/modelbench/adapters/storage/blobstore.py`）。
- **幂等**：同一份 payload 重复写入直接跳过；原子写（临时文件 + `os.replace`）。
- **与 testset 的区别**：`dataset/test/<name>/testset.jsonl` 是**输入**（可重建），blob 是
  **实际发送内容**（含 decode guard 追加消息与 `chat_template_kwargs`，不可再生）；
  `batches/<batch_id>.json` 的 `inputs.testset.sha256` 记录输入文件指纹，二者合起来才能自证
  「这一次到底发了什么」。

### 3.3 `output/exports/`（派生）

人类交付物：每个批次一个 `exports/<batch_id>.xlsx`（三个页签：硬件配置 / 模型性能平均 /
详细数据）。**batch 结束时自动生成**（T-068；失败只告警不失败 run，见 §8），`./modelbench
export-analysis` 可随时重建；整目录可任意删除，重新导出即可恢复。Excel「API 输入内容」列由
blob 池解析填入真实 prompt，并保留单元格截断（约 3 万字符，超出保留头尾并标注原长度）。

### 3.4 `output/archive/`（归档证据，GC 引用根）

退役 run、旧 schema bundle 存放处，可自定子目录组织。`archive/` **不是删除区**：里面的
`records.jsonl` 仍引用 `blobs/prompts`，所以 GC 必须把 `archive/` 与 `runs/` 一起当作引用根
（见 §7.2）。归档动作（示例）：把 `runs/<run_id>/` 与对应的 `batches/<batch_id>.json`
一起移动到 `archive/<batch_id>/` 下，再 `./modelbench index` 重建索引；run 目录身份与 blob
引用都不变。需要时 `./modelbench analyze` 仍可读归档 run 目录。

### 3.5 `output/.state/`（运行时，非证据）

`index.lock`（索引 read-modify-write 串行化）、`benchmark.lock`（**output 根**跨进程单实例，
先取 endpoint 锁再取本锁）、`tmp/`（浏览器 profile 等临时物）。不参与索引与体积统计，可随时清空。

> **另一把锁在系统临时目录**（不属于 `output/`，也不进本目录清单）：oMLX endpoint 锁，
> 公式与获取顺序见 [30-architecture.md](30-architecture.md) §4.3（单一事实源）。
> 两把锁共同保证同一 oMLX 实例（即使 output 根不同）只跑一个 benchmark；
> 不同 `base_url` 可并发。

---

## 4. 单 run bundle：7 个条目逐个说明

一个模型 run 目录固定 7 个条目（1 个目录 + 6 个文件）。同一次测试动作的多个模型**各写自己的
run 目录**，互不混合；批次关系由 `batches/<batch_id>.json` 与 `index.json.batches` 表达。

| 条目 | 性质 | 职责 | 权威性 |
|---|---|---|---|
| `records.jsonl` | 逐行 JSON | 每条请求一条记录：身份键、输入指纹/计量、响应全文、归一化指标、冷缓存证据、模型快照引用 | **权威**；聚合只能由它复算 |
| `run.json` | JSON | run 元数据 + 配置 + **唯一**环境快照 + `outcome` 对账计数 | 权威（环境证据只此一份） |
| `MANIFEST.json` | JSON | 同目录文件的 path/bytes/sha256 清单 + blob 引用统计 | 权威（完整性） |
| `samples.csv` | CSV | `records.jsonl` 的逐样本扁平投影，Web「逐样本」分页与 pandas 主表 | 派生（可重建） |
| `aggregates.json` | JSON | `overall` + `cells` 聚合分布（唯一机器可读聚合形态） | 派生（可重建） |
| `report.md` | Markdown | 人读摘要（配置/环境/整体/数据集/逐样本），Web 报告页直接渲染 | 派生（可重建） |
| `logs/run.log` | 文本 | 运行日志（模型加载、跳过/复位原因、警告） | 运维；排障用 |

写盘时序由 `RunWriter`（`src/modelbench/adapters/storage/writer.py`）固定：
`run.json` 初始化 → 逐条 append `records.jsonl`（每条 flush）→ `finalize()` 写
`aggregates.json`/`samples.csv`/`report.md` → 更新 `run.json.outcome` → **最后**写
`MANIFEST.json` → 刷新 `index.json`。因此中途崩溃时 `MANIFEST.json` 或 `run.json.outcome`
可能缺失，可用 `./modelbench verify` 与 `./modelbench rebuild` 判定/恢复。

### 4.1 `records.jsonl`（权威逐样本证据）

一行一个请求（warmup 也落盘、但 `phase="warmup"` 不进聚合；污染/错误记录同样保留）。
关键结构（实测由 `RunWriter` 写出的字段）：

```jsonc
{
  "schema_version": "3.0",   // 新 run / 新迁移件为 3.0；旧迁移件可能仍是 2.1（见 §10）
  "run_id": "20261107T032029Z-4184__performance__gemma-4-e4b-oq4",
  "timestamp": "2026-11-07T…+00:00",
  "model": "gemma-4-E4B-it-oQ4",
  "model_env_ref": "3f9c…(16)",              // → run.json.env.model_env
  "scenario": "coding",
  "dataset": "longbench_v2:Code Repository Understanding",
  "context_length": 1024,                    // 档位（1024/8192/32768）
  "sample_id": "perf-coding-1k-000",
  "repeat": 0, "phase": "measured", "status": "ok", "error": null,

  "request": {                               // 发送侧：指纹 + 参数，不含 prompt 正文
    "model": "…", "max_tokens": 50, "temperature": 0.0, "top_p": null, "seed": null,
    "stream": true, "extra": {}, "chat_template_kwargs": {"enable_thinking": false},
    "target_context_tokens": null, "actual_prompt_tokens": 1013,   // 数据集声明值（参考 tokenizer）
    "tokenized_prompt_tokens": 1174,        // 目标模型 tokenize 实际值（含 template+decode guard）
    "prompt_sha256": "ab12ef…(64)",          // → blobs/prompts/ab/ab12ef….json
    "prompt_chars": 12034,
    "truncated": false, "padded": false,
    "context_meta": {"bucket": "1k", "decode_guard": true,
                     "window_check_basis": "tokenized",
                     "tokenize_latency_ms": 3.7,   // 测量窗口外（tokenize 仅用于口径判定）
                     "isl_tolerance": {"target": 1024, "low": 921, "high": 1126,
                                       "actual": 1174, "ok": false}, "…": "…"}
  },

  "response": {                              // provider 原文（审计归一化是否正确）
    "text": "…", "reasoning_text": "", "tool_calls": [], "finish_reason": "length",
    "usage": {"prompt_tokens": 1174, "completion_tokens": 50, "cached_tokens": 0, "…": "…"},
    "num_chunks": 12, "first_token_ts": 1.2587, "last_token_ts": 2.0068
  },

  "metrics": {                               // 归一化指标：唯一指标来源
    "ttft_ms": 1258.8, "ttft_visible_ms": 1258.8, "tpot_ms": 15.27, "e2e_ms": 2007.3,
    "gen_tps": 65.5, "prefill_tps": 932.7, "total_tps": 609.77,
    "input_context_tokens": 1174, "input_context_chars": 3873,
    "completion_tokens": 50, "prompt_tokens": 1174, "cached_tokens": 0,
    "peak_cpu_pct": 120.7, "mean_cpu_pct": 43.9, "cpu_seconds": 0.877,
    "peak_rss_mb": 6376.7, "peak_memory_gb": 6.7,
    "cpu_samples": 29, "cpu_sample_window_ms": 2015.7,
    "decode_window_ms": 748.1,               // 解码窗口（即使 TG TPS 被抑制也保留）
    "gen_tps_invalid": null                   // 被抑制时的原因枚举，见 §4.1 注意项
  },

  "cache": {                                 // 冷缓存合规的唯一证据
    "mode": "cold", "cleared_before_request": true, "clear_latency_ms": 64.9,
    "cached_tokens": 0, "polluted": false, "note": "hot+ssd cleared (settle 800ms)"
  },

  "sample_meta": {"dataset_meta": {"bucket": "1k", "lang": "en"}},  // 迁移件另有 migrated_from_schema
  "score": null                              // 性能模式恒 null（判分器不接入 run）
}
```

注意：

- **prompt 正文不在记录里**：只留 `request.prompt_sha256`（→ blob）与 `prompt_chars`。
  因此「某条记录实际发了什么」= 记录里的 `prompt_sha256` + `output/blobs/prompts/` 中的
  blob；blob 本身可自校验（§3.2）。这是 AGENTS.md §2 第 13 条「逐样本可追溯」在
  schema 3.0 的成立方式：`records.jsonl` 保留全部逐样本字段，正文外置为不可变内容寻址文件。
- **模型配置快照不在记录里**：`model_env_ref` 指向 `run.json.env.model_env[<16hex>]`；
  同一次 run 内快照恒定只存一份，若中途重载模型会追加新条目，后续记录指向新键
  （不静默丢弃环境证据）。
- `metrics` 与 `response.usage` 在 token 数上重合是**刻意对照**（归一化值 vs provider 原始值），
  不属重复。`gen_tps/tpot_ms` 仅在解码窗口可信时输出（`completion_tokens >= 16` **且**
  `decode_window_ms >= 50`、`finish_reason != tool_calls`、值 ≤ 10,000）；否则置 null 并在
  `metrics.gen_tps_invalid` 记录原因（完整枚举见 [20-methodology.md](../spec/20-methodology.md) §2）。
- **实际输入大小（actual ISL）**：`request.tokenized_prompt_tokens` 是目标模型对最终
  messages+tools（含 chat template 与 decode guard）的 tokenize 结果；`context_meta.isl_tolerance`
  记录 `{target, low, high, actual, ok}`，`context_meta.window_check_basis` 说明窗口判定用的是
  `tokenized` 还是回退的 `declared`。**actual 不驱动跳过**：所有能测样本都 measured；
  聚合只使用 actual 在 **target ±10%**（所有档位对称，区间见 [20-methodology.md](../spec/20-methodology.md) §5）的入带样本，出带样本保留在 records/samples.csv（`isl_band=false`）但
  不进平均值。计数在 `run.json.outcome.n_isl_in_band/n_isl_out_of_band` 与
  `aggregates.*` 同名计数；`samples.csv` 在 `actual_prompt_tokens` 后有 `isl_band` 列。
  原因：数据集构造用参考 tokenizer 控制内容 token，运行时按目标模型 tokenizer 计数
  且叠加 template/guard（构造 ±5% 与运行 ±10% 的完整推导见
  [11-performance-testset.md](../spec/11-performance-testset.md) §4.5 单一事实源），故用
  ±10% 入带聚合而非跳过；跨系统比较准入见
  [../spec/13-testset-portability.md](../spec/13-testset-portability.md)。
- **tokenize 在测量窗口之外**：`context_meta.tokenize_latency_ms` 记录"tokenize → 清 hot+SSD →
  measured 请求"流程中**第一步**的耗时，**不进入 TTFT/TPS/内存**（测量窗口从清缓存后的请求开始）。
  cold 模式的 hot+SSD 清理发生在 tokenize **之后**，因此 tokenize 自身产生的缓存/预热影响不会
  进入测量；32K 首次 tokenize 约 80ms，后续客户端按 prompt 缓存复用，240 条/模型总开销约数秒。
- warmup / 污染 / 错误记录不参与聚合，但**保留在文件里**可审计。

### 4.2 `run.json`（运行元数据 + 唯一环境快照）

```jsonc
{
  "schema_version": "3.0",
  "run_id": "…", "batch_id": "…", "method": "performance",
  "model": {"id": "…", "alias": "…", "model_slug": "…"},
  "project_version": "0.3.0",
  "created_at": "…", "started_at": "…", "finished_at": "…", "duration_s": 3012.4,
  "git": {"commit": "…", "dirty": false},
  "config": {                              // 当时生效的完整配置（密钥脱敏）
    "server": {}, "run": {}, "perf": {}, "datasets": {},
    "models": [], "scenarios": {}, "selection": {}
  },
  "env": {
    "hardware": {"chip": "…", "chip_name": "…", "memory_gb": 24, "arch": "arm64",
                 "cpu": {}, "memory": {}, "os": {}, "gpu": [], "python": {},
                 "upload_fields": {}},
    "omlx": {"base_url": "…", "omlx_version": "…", "global_settings": {},
             "device": {}, "models": {}, "feature_flags": [], "default_config": true},
    "server_models": {                       // 原 models.json：服务端全量 /v1/models/status
      "server": {"final_ceiling": 16826541616, "current_model_memory": 0,
                 "model_count": 12, "loaded_count": 0,
                 "load_seconds_per_gb_estimate": 0.675, "load_time_observations": 13,
                 "models": ["… 服务端全部模型的状态条目 …"]},
      "selected": ["… 本次选中的模型 …"]
    },
    "model_env": {
      "3f9c…(16)": {"id": "…", "engine_type": "vlm", "quantization": "oQ4",
                    "estimated_size": 5612040826, "context_length": 131072,
                    "settings": {}, "feature_flags": [], "loaded": true}
    }
  },
  "outcome": {
    "status": "ok", "cancelled": false, "memory_errors": 0,
    "n_selected": 240, "n_measured": 240, "n_ok": 240, "n_records": 240, "n_samples": 240, "n_cells": 24,
    "n_gen_tps_invalid": 3,        // TG TPS/TPOT 被抑制的 measured 记录数（原因见记录）
    "n_isl_in_band": 232,          // actual 在 target ±10% 带内、进入平均的 measured 记录数
    "n_isl_out_of_band": 8,        // actual 出带：保留测量但不进平均的 measured 记录数
    "skipped": ["…"], "warnings": [], "duration_s": 3012.4
  },
  "notes": null
}
```

- **环境只此一份**：硬件 / oMLX / 服务端模型清单 / 每模型生效配置都内嵌在 `env`；
  不再有独立的 `hardware.json` / `omlx.json` / `models.json` 文件（内容逐字段保留，无丢失）。
  `model_env` 的键 = 该条目规范化 JSON 的 SHA-256 前 16 位，`records[].model_env_ref` 指向它。
- `env.server_models` 对应 schema 2.1 的 `models.json`：它保留了**服务端全量**模型状态
  （含 `final_ceiling` 内存上限、`load_seconds_per_gb_estimate`、`load_time_observations`
  与全部模型清单），这些量没有别的产物能复现，因此不能省。`env.omlx.models` 只含**被测模型**
  的生效设置，`env.hardware.omlx_health` 则是采集时刻的健康快照。
- `config` 是当时生效的完整配置（`server` 密钥已脱敏）；`config.perf`（含 `context_windows`、
  `max_tokens`、`decode_guard`）与 `config.run`（缓存模式/预热/轮数等）是跨 run 对比前必须对齐
  的项。
- `outcome` 是对账口径：`n_measured` = `records.jsonl` 中 `phase=measured` 的行数
  （**error/timeout 失败尝试也入账**），`n_ok` = 其中 `status=ok` 的行数，
  `n_selected` = `n_measured + len(skipped)`，`n_records` = 文件行数；`./modelbench verify`
  逐项校验（§7.1；无 `n_ok` 的历史 run 按旧口径 ok-only 对账）。`skipped` 为字符串列表，每条带 `model/scenario/sample/repeat` 与原因
  （便于定位，不静默丢弃）。`n_gen_tps_invalid` = TG TPS/TPOT 被抑制的 measured 记录数；
  `n_isl_in_band`/`n_isl_out_of_band` = actual 在 target ±10% 带内/带外的 measured 记录数；
  **只有入带记录进入 averages**，出带记录保留但不进均值。旧口径的 `n_isl_skipped` /
  `n_isl_out_of_tolerance` 已删除（历史 run 字段可忽略）。

### 4.3 `MANIFEST.json`（完整性清单，最后写）

```jsonc
{
  "schema_version": "3.0", "run_id": "…", "generated_at": "…", "algorithm": "sha256",
  "files": [
    {"path": "aggregates.json", "bytes": 6889, "sha256": "…"},
    {"path": "logs/run.log", "bytes": 34695, "sha256": "…"},
    {"path": "records.jsonl", "bytes": 754974, "sha256": "…"},
    {"path": "report.md", "bytes": 38118, "sha256": "…"},
    {"path": "run.json", "bytes": 41234, "sha256": "…"},
    {"path": "samples.csv", "bytes": 45123, "sha256": "…"}
  ],
  "totals": {"files": 6, "bytes": 981000},
  "blob_refs": {"prompts": 240, "bytes": 13520000, "missing": 0}
}
```

- `files` 覆盖 run 的规定产物（`run.json`、`records.jsonl`、`samples.csv`、`aggregates.json`、
  `report.md`、`logs/run.log`；实际不存在的条目跳过），按 `path` 排序；`records.jsonl` 的
  sha256 必填。
- `blob_refs.prompts` = 本 run 引用的**去重** prompt 数；`bytes` = 这些被引用 blob 的字节数
  （不是全库总量，多个 run 共享的 blob 会被各自计入，所以不能相加当磁盘占用）；
  `pool_bytes` = 整个 `blobs/prompts` 池的当前字节数（这才是磁盘口径）；
  `missing` 在生成时与 `./modelbench verify` 时都必须为 0。
- 索引重建直接读它，`archive/` 与手工拷贝的 bundle 也一样。

### 4.4 `samples.csv`（逐样本主表，派生）

一行一个 measured 样本，列顺序**冻结**（`SAMPLES_CSV_FIELDS`）：

```text
model,model_id,scenario,dataset,subtask,context_length,sample_id,repeat,phase,status,
finish_reason,length_capped,input_context_tokens,input_context_chars,actual_prompt_tokens,isl_band,
ttft_ms,ttft_visible_ms,tpot_ms,gen_tps,e2e_ms,prefill_tps,total_tps,
completion_tokens,cached_tokens,cache_mode,cache_cleared,cache_polluted,
peak_memory_gb,peak_rss_mb,peak_cpu_pct,mean_cpu_pct,
prompt_sha256,truncated,error,timestamp
```

- `model` 与 `model_id` 目前同值（都是模型 id，`SampleResult.model_id` 已产出）；`model_id`
  保留为"服务端 id"的独立列，以便未来 alias 与 id 分离时不必改列序。
- `actual_prompt_tokens`（数据集声明 token）、`total_tps`、`mean_cpu_pct` 均已产出并写入。
- 冷缓存审计字段（`cached_tokens`/`cache_mode`/`cache_cleared`/`cache_polluted`）直接在此表，
  不必回读 `records.jsonl`。
- `prompt_sha256` 可回查 blob 取 prompt 正文（§6.3）。
- `gen_tps`/`tpot_ms` 对 `metrics.gen_tps_invalid` 非空的记录写 **null**（列数不变、不加新列）；
  统计口径以 `run.json.outcome.n_gen_tps_invalid` 为准。
- `actual_prompt_tokens` 是数据集声明值；`isl_band`（紧随其后）= 目标模型 actual 是否在
  该 bucket target ±10% 带内（入带样本才进入 averages）。目标模型实际值在 `records.jsonl`
  的 `request.tokenized_prompt_tokens`；`samples.csv` 只加 `isl_band` 一列（列序冻结）。
- `length_capped=true` 表示输出触顶（`finish_reason=length`），**不是丢弃**：样本照常参与
  TTFT/内存/吞吐聚合，`gen_tps/tpot_ms` 按实际 `completion_tokens` 计算。

### 4.5 `aggregates.json`（聚合，派生）

```jsonc
{
  "schema_version": "3.0", "run_id": "…", "generated_at": "…",
  "n_records": 240, "n_samples": 240, "n_cells": 24,
  "overall": [{
    "model": "…", "context_length": 1024,
    "n_datasets": 3, "n_samples": 80, "n_valid": 80, "n_warmup": 0,
    "n_contaminated": 0, "n_error": 0, "n_length_capped": 12,
    "n_gen_tps_invalid": 2, "n_isl_in_band": 78, "n_isl_out_of_band": 2,
    "stability_cv": 0.03, "unstable": false,
    "perf": {"ttft_ms": {"n": 78, "mean": 1258.8, "median": 1200.1,
                          "std": 80.2, "min": 900.0, "max": 2100.0, "p95": 1800.0,
                          "ci95_low": 1241.2, "ci95_high": 1276.4}},
    "perf_all": {"ttft_ms": {"n": 80, "mean": 1261.0, "…": "同 Distribution"}},
    "peak_rss_mb_max": 6376.7, "peak_memory_gb_max": 6.7
  }],
  "cells": [{
    "model": "…", "scenario": "…", "dataset": "…", "context_length": 1024,
    "n_samples": 10, "n_repeats": 1, "n_valid": 10, "n_length_capped": 2,
    "n_gen_tps_invalid": 0, "n_isl_in_band": 10, "n_isl_out_of_band": 0,
    "stability_cv": 0.02, "unstable": false,
    "perf": {"…": "同 Distribution"}, "perf_all": {"…": "同 Distribution"}
  }]
}
```

- `overall`：每 `(model, context_length)` 一行——1K/8K/32K 的 TTFT/预填充相差一个数量级，
  **绝不跨档位合并**；`peak_memory_gb_max` 是该档位全局最大值。
- `cells`：每 `(model, scenario, dataset, context_length)` 一行。
- `perf` / `perf_all` 是「指标名 → `Distribution`」映射，`Distribution` 含
  `n/mean/median/std/min/max/p95/ci95_low/ci95_high`；
  `n` 是该指标真实参与统计的样本数（`gen_tps` 只统计可信解码窗口的样本，见 §4.1）。
- **双口径（T-067）**：`perf` 只统计 actual 在 target ±10% 入带的样本（跨 run/跨系统
  比较口径，列名带「（±10%过滤）」）；`perf_all` 统计该行全部 valid measured 样本
  （默认展示口径，含出带）。两者除 ISL 带过滤外完全同规则（warmup/污染/错误、
  `gen_tps`/`tpot` 可信窗口判定一致）；schema 3.0 增量字段，旧 run 无 `perf_all`
  时消费端从 `records.jsonl` 现算兜底。
- **`ci95_low` / `ci95_high`**：**run 内** mean 的正态近似 95% 置信区间
  `mean ± 1.96·std/√n`（`n<2` 时为 null）。它只描述"本次 run 内样本均值的不确定性"，
  **不是 run-to-run 噪声/重复性区间**；跨 run 波动要用 `repeats≥2` 的重复测量或配对比较，
  不能把 ci95 当成可复现性证据。
- `n_gen_tps_invalid` / `n_isl_in_band` / `n_isl_out_of_band`：本行（`overall` 行或 `cells` 行）
  内被抑制 TG TPS 的样本数、actual 入带（进入 `perf`/（±10%过滤）列）与出带（保留但不进入
  `perf`）的样本数；与 `run.json.outcome` 的全局计数同源、可复算。`perf` 各 Distribution 的
  `n` 等于本行入带样本中该指标有效的数量（如 `gen_tps_n ≤ n_isl_in_band`；
  `perf_all` 的 `n ≤ n_valid`）。in-band 样本不足以支撑均值（默认阈值 8 条入带样本）时，
  `report.md`/Excel/Web 标记 `insufficient_samples`（该标记针对（±10%过滤）口径）。
- 这是唯一的机器可读聚合形态；需要 CSV 时用 pandas 从 `samples.csv`/本文件导出，
  不再常驻 CSV 汇总。

### 4.6 `report.md` 与 `logs/run.log`

- `report.md`：人读摘要（运行配置、环境、整体结果、数据集结果、逐样本表 + 口径注脚），
  Web 报告页直接渲染；内容由 `records.jsonl` 复算，`rebuild` 可重生成。报告显示
  **ISL入带/ISL出带** 计数，并把 `n_isl_out_of_band` 注明"仍测量但不进（±10%过滤）列，
  但计入默认全样本列"；整体/数据集表均为双口径：默认列（TTFT/TG TPS/TPOT/E2E 等）
  为全样本口径（`perf_all`），带「（±10%过滤）」的列为入带口径（`perf`）；
  in-band 样本不足时标记 `insufficient_samples`。
- `logs/run.log`：运行日志，运行中由 Web 任务面板增量读取，结束后仅排障用；随归档清理。

---

## 5. 批次与索引

### 5.1 `batches/<batch_id>.json`（批次权威元数据，唯一副本）

位于 `output/batches/<batch_id>.json`。**一次测试动作 = 一个批次元数据文件**；它不拥有目录，
成员是 `runs/` 下的一批 run 目录。字段：

```jsonc
{
  "schema_version": "3.0",
  "batch_id": "20261107T032029Z-4184",
  "source": "web",                              // web | cli
  "method": "performance",
  "created_at": "2026-11-07T03:20:29+00:00", "finished_at": "…",
  "project_version": "0.3.0", "duration_s": 3600.0, "cancelled": false,
  "models_selected": ["gemma-4-E4B-it-oQ4", "…"],
  "models_measured": ["gemma-4-E4B-it-oQ4", "…"],
  "n_models_selected": 3, "n_models_measured": 3,
  "model_runs": [{                              // 成员 run（无 "run_dir" 字段）
    "model": "…", "model_id": "…", "model_slug": "…", "method": "performance",
    "run_id": "20261107T032029Z-4184__performance__gemma-4-e4b-oq4",
    "run_path": "runs/20261107T032029Z-4184__performance__gemma-4-e4b-oq4",  // 相对 output/
    "model_size_gb": 5.2, "status": "ok",
    "n_records": 240, "n_samples": 240, "duration_s": 3012.4,
    "memory_errors": 0, "skipped": [], "cancelled": false
  }],
  "inputs": {                                   // 输入自证（schema 3.0 新增）
    "testset": {
      "path": "dataset/test/performance/testset.jsonl",
      "sha256": "…", "rows": 240,
      "context_windows": [1024, 8192, 32768],
      "decode_guard_version": 1
    }
  },
  "config": {"perf": {}, "run": {}, "scenarios": []},
  "retention": {"pinned": false, "keep_until": null},
  "provenance": {"git_commit": null, "host": {"python": "3.12.x"}},
  "skipped": [], "warnings": [], "errors": []
}
```

- `model_runs[]` 用 **`run_id` + `run_path`** 指明成员，`run_path` 相对 `output/`
  （恒为 `runs/<run_id>`）；不再有 `run_dir`（旧布局中相对批次目录的字段）。`batch_id` 不含来源前缀，
  来源在 `source`。
- 批次归属的三重校验：`run_id` 前缀 = `<batch_id>__`、本文件 `model_runs[].run_id`、
  各 run 的 `run.json.batch_id`。`./modelbench verify` 会检查它们一致。
- `inputs.testset.sha256` 是输入文件指纹（`dataset/test/<name>/manifest.json` 已记录同一值），
  与 `blobs/prompts/` 一起构成「这一次跑了哪 240 行 + 实际发了什么」的自证链。
- `retention.pinned/keep_until` 是保留标记（预留给未来配额/清理策略；当前 `gc` 只按引用
  关系标记清扫，不读该字段，见 §7.2）。
- **文件内不写任何绝对路径**，因此整个 `output/` 可整体搬移；`BatchRunner.run()` 的返回值里
  才有 `batch_dir`（现指 `output/batches`）与 `batch_file` 绝对路径，那是进程内便利字段，
  不落盘。

### 5.2 索引重建

`index.json` 完全由 `runs/*/run.json` + `runs/*/MANIFEST.json` 与 `batches/*.json` 派生：

```bash
./modelbench index                 # 全量重建（原子写 + .state/index.lock）
./modelbench runs --rebuild        # 先重建再按批次列出
```

删除 `index.json` 后再重建，内容一致（除 `generated_at`）。`archive/` 不参与索引。

---

## 6. 二次分析用法

### 6.1 读逐样本（`samples.csv`）

```python
import pandas as pd

run = "output/runs/20261107T032029Z-4184__performance__gemma-4-e4b-oq4"

samples = pd.read_csv(f"{run}/samples.csv")          # 逐样本主表（36 列，列序冻结）
print(samples.groupby(["dataset", "context_length"])[
    ["ttft_ms", "gen_tps", "tpot_ms", "peak_memory_gb"]
].mean())

# 冷缓存合规审计：无需回读 records.jsonl
print(samples[["cache_mode", "cache_cleared", "cached_tokens"]].value_counts())

# 输出触顶只看不剔除
print(samples["length_capped"].mean(), samples["finish_reason"].value_counts())
```

### 6.2 读聚合（`aggregates.json`）

```python
import json

run = "output/runs/20261107T032029Z-4184__performance__gemma-4-e4b-oq4"
agg = json.load(open(f"{run}/aggregates.json"))

overall = agg["overall"]        # 每 (model, context_length) 一行，不跨档位合并
cells = agg["cells"]            # 每 (model, scenario, dataset, context_length) 一行
print([(row["context_length"], row["perf"]["ttft_ms"]["mean"]) for row in overall])
```

`./modelbench analyze` 也是读 `aggregates.json`，参数可以是 run 目录、`run_id`
或显式 `aggregates.json` 路径：

```bash
./modelbench analyze <run_id_a> <run_id_b>
./modelbench analyze output/runs/<run_id_a> output/runs/<run_id_b> \
  --metrics ttft_ms gen_tps peak_memory_gb input_context_tokens
./modelbench analyze <run_id_a> <run_id_b> --json
./modelbench analyze <run_id_a> <run_id_b> --paired      # 配对比较（读两份 samples.csv）
./modelbench analyze <run_id_a> <run_id_b> --paired --json
```

> `--paired` 只比较两份 run 中 `(scenario, dataset, context_length, sample_id, repeat)`
> 相同的 `status=ok` 样本，给出逐样本 Δ(B-A)、配对 bootstrap 95% CI 与“CI 是否不含 0”。
> 这是 **run 内抽样不确定性**，不是 run-to-run 噪声；要跨 run 噪声基线仍需
> `repeats≥2` 或整轮重复并比较。两份 `samples.csv` 都必须存在。

### 6.3 回查 prompt 正文（`records.jsonl` → blob）

prompt 正文不在记录里，按摘要回 blob 取：

```python
import hashlib, json
from pathlib import Path

root = Path("output")
run = root / "runs/20261107T032029Z-4184__performance__gemma-4-e4b-oq4"

with open(run / "records.jsonl", encoding="utf-8") as fh:
    record = json.loads(fh.readline())

sha = record["request"]["prompt_sha256"]
blob = json.loads((root / "blobs/prompts" / sha[:2] / f"{sha}.json").read_text())
assert hashlib.sha256(
    json.dumps({k: v for k, v in blob.items() if k != "sha256"},
               sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
).hexdigest() == sha          # 内容自校验

messages = blob["messages"]   # 精确等于发送时的完整输入（含 decode guard 追加消息）
```

- 解析口径：**`records.jsonl` 是证据，`blob` 是正文**。Excel 的「API 输入内容」列、
  `report.md` 的 decode guard 说明都按同一路径取。
- 同一 prompt 被多个模型/批次/重复使用时只存一份，所以 `records.jsonl` 里的
  `prompt_sha256` 会在同一批次内大量重复，这是预期的。

### 6.4 校验一个 bundle（`MANIFEST.json` → 文件字节）

```python
import hashlib, json
from pathlib import Path

run = Path("output/runs/<run_id>")
manifest = json.loads((run / "MANIFEST.json").read_text())
for item in manifest["files"]:
    digest = hashlib.sha256((run / item["path"]).read_bytes()).hexdigest()
    assert digest == item["sha256"], item["path"]
assert manifest["blob_refs"]["missing"] == 0
```

等价命令：`./modelbench verify`（§7.1）。

---

## 7. 完整性、GC 与保留

### 7.1 `./modelbench verify`

对账五类事实（`src/modelbench/adapters/storage/tools.py`）：

1. 批次 ↔ run 归属：`batches/<batch_id>.json` 的 `batch_id` 必须等于文件名、
   `models_selected` ⊆ `model_runs`，每个 `model_runs[].run_id` 必须是单段且其
   `run_path` == `runs/<run_id>`、磁盘上 `runs/<run_id>/run.json` 存在且 `run.json` 的
   `run_id`/`batch_id` 与声明一致；反过来，`runs/` 下任何未被 `batches/*.json` 引用的 run
   也会被报出；
2. 每个 run：`outcome.n_measured` == `records.jsonl` 中 `phase=measured` 行数（含
   error/timeout），`n_ok` == 其中 `status=ok` 行数（无 `n_ok` 的历史 run 按 ok-only
   口径对账），`n_selected == n_measured + len(skipped)`，`n_records` == 文件行数；
3. 每个 run：所有 `request.prompt_sha256` 都能在 `blobs/prompts` 解析（`missing == 0`），
   且 `MANIFEST.json` 中每个文件的大小/sha256 与实际一致；
4. **聚合可复算**：`aggregates.json` 的 `overall[]`/`cells[]` 必须能由 `records.jsonl`
   重算——**双口径都查**：`perf`（±10% 入带）与 `perf_all`（全样本）的计数、各指标
   Distribution 的 n/mean/median/std/min/max/p95/ci95；旧 run 缺 `perf_all` 时按
   复算结果同样缺失（该桶无 valid 记录）才通过，否则报 issue；
   派生文件过期或被手改时 `verify` 会报 issue，用 `./modelbench rebuild` 恢复；
5. **测量有效性/ISL 带**：统计 `gen_tps_invalid`（有 `metrics.gen_tps_invalid` 的记录数）、
   `isl_in_band`/`isl_out_of_tolerance`（按 actual 与 bucket target ±10% 复算的入带/出带
   measured 记录数；与 `outcome.n_isl_in_band`/`n_isl_out_of_band` 对账，不一致报 issue）、
   `legacy_gen_tps_invalid`（历史 run 的非法 `gen_tps`，>10,000 或 <0）；旧 run 的非法值只计入
   `legacy_gen_tps_invalid`，**不导致 verify 失败**；若同一条记录同时有 `gen_tps` 与
   `gen_tps_invalid`（新逻辑自相矛盾）则报 issue。

```bash
./modelbench verify            # 人读：统计 + 问题明细；有问题退出码 1
./modelbench verify --json     # 机器读：{"ok":…, "issues":[…], "stats":{…}}
```

### 7.2 `./modelbench gc`（prompt 池标记清扫）

**引用根 = `runs/` ∪ `archive/`**（两者都递归找 `records.jsonl`）。归档 run 仍是证据，
若只扫 `runs/` 会把它们引用的 prompt 正文删掉（records 还在、正文永久丢失）。

```bash
./modelbench gc --dry-run     # 只报告将删除多少个、释放多少字节
./modelbench gc               # 先 verify，再删除无引用的 blob 并清理空分片目录
```

保护行为：`runs/` 与 `archive/` 都不存在时**拒绝执行**（否则像「全是垃圾」）；
`verify` 报 `blob_missing > 0` 时**拒绝执行**（带着缺口删除会让证据不可恢复）。

### 7.3 `./modelbench rebuild`

从权威的 `records.jsonl` 重建派生文件：

```bash
./modelbench rebuild                    # 全部 run
./modelbench rebuild <run_id>           # 指定单段 run_id
./modelbench rebuild <run_id> --json
```

重建 `samples.csv`、`aggregates.json`、`report.md`、`MANIFEST.json`（`run.json` 与
`records.jsonl` 只读），随后重建 `index.json`。重建后**语义等价**（字段与取值相同；
浮点格式化与时间戳会变，不保证字节级相同）。删掉派生物后用它可以恢复。

### 7.4 `./modelbench pack`（可移植打包）

```bash
./modelbench pack <batch_id> --out /tmp/<batch_id>.zip
```

按 `batches/<batch_id>.json` 的成员清单，把**批次元数据文件 + 每个成员 `runs/<run_id>/`
+ 它们引用的全部 prompt blob** 打进 zip；压缩包内路径相对 `output/` 根：

```text
batches/<batch_id>.json
runs/<run_id_a>/…          # 该批次的每个 run
runs/<run_id_b>/…
blobs/prompts/<sha[:2]>/<sha>.json
```

解压到某个 `output/` 根后仍是合法批次，`./modelbench verify` 可通过；有成员 run 或 blob
缺失时拒绝打包并报缺失明细。

### 7.5 保留与归档策略

- **长期保留**：优先 `pack` 整个批次（自包含 + prompt 池），或把
  `runs/<run_id>/` + `batches/<batch_id>.json` 一起移入 `archive/`（同盘、blob 引用不变）。
- **`archive/` 是 GC 引用根**：移入 `archive/` 的 run 不会被 `gc` 误删 blob；不要用
  「删除 `runs/` 下目录」代替归档。
- **整体搬盘**：搬整个 `output/`（索引与 `batches/*.json` 里的证据路径都是相对的），或先
  `pack` 指定批次。
- **可删的只有派生层**：`exports/` 可整目录删；`samples.csv`/`aggregates.json`/`report.md`/
  `MANIFEST.json` 可删后用 `rebuild` 恢复；`index.json` 可删后用 `index` 重建。
  `records.jsonl`、`run.json`、`batches/*.json`、`blobs/` **不可再生**，删除=丢证据。
- **`reports/` 不是证据**：根目录 `reports/`（评审/审计/一次性分析记录）可随时归档或删除，
  索引与保留策略见 `reports/README.md`（gitignored 本地文件）；长期结论应进入 `docs/`
  （规范/设计）或 `docs/records/`（时间性记录），**不要**把 `reports/` 当作 run 结果。

---

## 8. Excel 交付物（`export-analysis`）

Excel 在**每个 batch 结束时自动生成**到 `output/exports/<batch_id>.xlsx`（T-068：无需再手动执行；
生成是尽力而为——失败只向 `batch.json.warnings` 和 stderr 告警，绝不让基准 run 本身失败，
`exports` 字段记为 `null`，之后可用下面的命令补出）。手动命令用于重建/补出历史批次：

```bash
./modelbench export-analysis <batch_id>       # 生成一个批次
./modelbench export-analysis --all            # 重建全部批次
./modelbench export-analysis <batch_id> --output-dir DIR
```

三个页签（数据源：`batches/<batch_id>.json` + 各 run 的
`run.json`/`aggregates.json`/`records.jsonl` + blob 池）：

| 页签 | 内容 |
|---|---|
| `硬件配置` | 批次/测试配置、芯片/内存/GPU/OS/oMLX 版本、oMLX 全局设置、每个模型的生效配置（量化/上下文/MoE/feature flags） |
| `模型性能平均` | 每个模型 × 上下文档位一行：样本/有效/错误/触顶数、TTFT 均值/中位数/P95、TG TPS（均值与有效数）、TPOT、E2E、Prefill、峰值内存均值/最大、峰值 CPU、输入/输出 token 均值、耗时、跳过/内存错误数，以及 **`TG无效数`**、**`ISL入带数`**（进入（±10%过滤）列的样本数）、**`ISL出带数`**（仍测量但不进入（±10%过滤）列的样本数）。**默认列为全样本口径**（该桶全部 measured+ok 记录，含出带；读 `aggregates.json` 的 `perf_all`，旧 run 无该字段时从 records 现算）；随后是**（±10%过滤）列**：**`TTFT均值(ms)（±10%过滤）`**、**`TG TPS均值（±10%过滤）`**、**`TG TPS有效数（±10%过滤）`**、**`E2E均值(ms)（±10%过滤）`**（只统计入带样本，跨 run 比较口径，读 `perf`）与 **`出带TTFT波动范围`**（全样本 vs 入带 TTFT 均值的相对差，如 `+100.0%`；任一均值缺失时留空）。口径由列名携带，仅一行表尾短注释说明，不设说明页签 |
| `详细数据` | 每个 measured 样本一行：模型/数据集/场景/样本/重复、**API 输入内容**（从 blob 解析）、输入 token/字符数、**输出结果**、输出 token/字符数、峰值内存，以及 TTFT/TG TPS/TPOT/E2E/Prefill/Total TPS/CPU/cached/状态/截断/错误/时间戳，末尾 **`ISL入带`**（`context_meta.isl_tolerance.ok`，缺失时按 actual vs target ±10% 判定，无依据按入带保留）与 **`ISL偏差%`**（`(actual-target)/target*100`，1 位小数；无 actual/target 时空） |

- 生成失败不影响任何证据：批次 bundle 不变；修复后重跑 `export-analysis` 即可。
- 超长文本超过 Excel 单元格上限（32,767 字符）时保留头尾并标注原长度（不改 blob 全文）。
- 写入前清洗 XML 非法控制字符并对 `= + - @` 前缀加 `'`，避免 openpyxl 报错与公式注入。
- 双口径来自同一 `aggregates.json`（`perf_all` = 全样本、`perf` = ±10% 入带，两者都由
  `records.jsonl` 复算；`export-analysis` 是派生层可随时重建）。旧 run（无 `perf_all`）的
  全样本列由 records 现算兜底。

---

## 9. Web 控制台入口

- **历史运行**：按批次分组，列表直接读 `output/index.json`（不逐个解析大文件）；
  行内显示模型/样本数/耗时/体积/已测已选/跳过数。
- **详情页**：整体（按档位）、数据集（cells）、逐样本、环境、报告五个页签。
  - 顶部汇总行显示 **`ISL入带 N`**（`outcome.n_isl_in_band`）、**`ISL出带 N`**
    （`n_isl_out_of_band`）与 **`TG无效 N`**（`n_gen_tps_invalid`）；整体/数据集表为
    双口径：默认列（TTFT/TG TPS 等）为全样本口径（`perf_all`），随后是带
    「（±10%过滤）」的入带口径列（`perf`）；`n_isl_in_band`/`n_isl_out_of_band`
    同表展示，出带注明"仍测量但不进（±10%过滤）列"。
  - 逐样本页**分页**读取 `samples.csv`：`GET /api/runs/<run_id>/samples?offset=&limit=`
    （`limit` 默认 200、上限 1000；可选过滤 `scenario=`、`dataset=`、`status=`、`model=`、
    `context_length=`、`q=`），返回 `{"total":N,"offset":…,"limit":…,"rows":[…]}`；
    `limit` 之外不要假设一次拿全量。
  - 环境页读 `run.json`（唯一环境快照，含 `env.server_models`）；报告页渲染 `report.md`。
- **原始文件**：白名单 `run.json`、`MANIFEST.json`、`records.jsonl`（默认下载）、
  `samples.csv`、`aggregates.json`、`report.md`、`logs/run.log`；
  下载走 `/api/runs/<run_id>/file?path=`，服务端做路径穿越校验。
- **重建索引**：等价 `./modelbench index`；**打开输出目录**：打开 `output/`；
  **Finder**：打开当前 run 目录。

---

## 10. 迁移与兼容

- **schema 3.0 与 2.x 不兼容，一次切换**：旧 bundle 已整体移入 `output/archive/legacy-v2/`，
  新索引/新工具不读取它们（运行时不再保留兼容读取分支）。
- **一次性重序列化（可选，已执行过一次）**：`tool/migrate/legacy_v2_to_v3.py` 把
  `archive/legacy-v2/` 的 2.x bundle *逐值* 重写成 3.0 布局（prompt → blob、记录瘦身、
  环境快照上提、派生文件由 writer 重算），并断言「记录数不变 + 每个 overall 分布的 mean
  与旧 `summary.json` 完全一致」。原件留在 `archive/` 不动，可随时对照或重跑：

  ```bash
  .venv/bin/python tool/migrate/legacy_v2_to_v3.py   # archive → output/runs, output/blobs, output/batches
  ./modelbench index && ./modelbench verify
  ```

  迁移件的 `batches/<batch_id>.json` 中 `inputs.testset.sha256` 为 `null` 并带 `note`：2.x
  从未记录测试集指纹，且该文件之后被重建过，写今天的指纹会构成**假 provenance**；但该次
  测量的**实际 prompt** 已按内容寻址完整保留在 `blobs/prompts/`，可复现性不受影响。
- **迁移记录的 schema 版本（2026-11-10 起）**：`legacy_v2_to_v3.py` 新写出的
  `records.jsonl` 每条 `schema_version="3.0"`，并把原版本写入
  `sample_meta.migrated_from_schema`（如 `"2.1"`）。**现有 8 个已迁移 run 的 records 仍是
  2.1**（当时迁移器未写 3.0，历史件保留不改写）；读取/校验时按"新迁移件=3.0 /
  旧迁移件=2.1 legacy"区分，不要以单条 record 的 `schema_version` 判定 bundle 是否为 3.0
  （bundle 布局以 `runs/<run_id>/` + `MANIFEST.json` 为准）。
- **第二轮布局调整（2026-11-09，用户要求）**：`runs/` 由
  `runs/<YYYY-MM>/<batch_id>/<method>/<model_slug>/` 4 层嵌套压平为 `runs/<run_id>/` 一层；
  批次元数据由批次目录内的 `batch.json` 移到 `output/batches/<batch_id>.json`。
  `run_id`/`index.json` schema 均不变，旧嵌套目录不再被索引。
- 同一份 `index.json` 的版本（`2.0`）与 bundle 版本（`3.0`）独立演进：索引纯派生，可单独重建。
- `archive/` 中的旧 bundle 仍是证据：不要删；GC 也不会回收它们引用的 prompt。
