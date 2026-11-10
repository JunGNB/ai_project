# 输出目录与产物规范（OUTPUT_LAYOUT，历史）

> **状态**：archived ｜ **superseded_by**：[../design/31-output-contract.md](../design/31-output-contract.md)
> **归档说明**：本文是 `output/` 重构的设计论证/实施记录；现状契约与逐字段规范以
> [31-output-contract.md](../design/31-output-contract.md)（RESULTS）为准。保留本文仅用于
> 追溯"为什么这样设计/如何测算收益"；其中的目录树、迁移计划、待确认项均为历史。
>
> **历史状态（当时）**：已实施（bundle schema 3.0 / `index.json` schema 2.0），2026-11-09 落地。
> 本文保留设计论证、去重测算与实测审计过程作为依据；§4/§5 的目录树是第一次落地的结构
> （已被第 9 条压平修订），§5.2/§5.5/§6–§8 的字段与去重规范仍适用；
> **逐字段现状契约以 [31-output-contract.md](../design/31-output-contract.md) 为准**，路径/字段以
> `src/modelbench/adapters/storage/layout.py`、`blobstore.py`、`writer.py` 等实现为准。
> §9/§10 的实施计划与 §12 的待确认表是实施前的历史记录（其中「S1 需人类确认」等已完成），
> 不再代表现状。实施与设计的差异（以代码/磁盘产物为准）：
>
> 1. `run.json.env.hardware` 仍保留顶层扁平字段与 `upload_fields` 两份（设计 §6.3 曾计划
>    只留 `upload_fields`）；`run.json.env.omlx` 仍写 `upload_projection`（设计 R11 曾计划删除，
>    当前零读取方，保留无副作用）。
> 2. `run.json.outcome.skipped` / 批次元数据（`batches/<batch_id>.json`）的 `skipped` 实为
>    **字符串列表**（每条含位置与原因），不是设计中的结构化对象；`verify` 只按条数对账。
> 3. ~~`batch.json` 写绝对路径~~ → **已修**：`batch_dir`/`batch_file` 仅存在于
>    `BatchRunner.run()` 返回值，不落盘；§11.9「batch.json 无绝对路径」已满足。
> 4. ~~`samples.csv.model_id` 留空~~ → **已修**：`SampleResult.model_id` 与
>    `actual_prompt_tokens`/`total_tps`/`mean_cpu_pct` 均已产出，契约 §6 的列不再有空列。
> 5. `MANIFEST.json.blob_refs.bytes` = 本 run **引用**的 blob 字节数；另有
>    `blob_refs.pool_bytes` 给出全库池字节（磁盘口径）。
> 6. 设计 R4 把旧的 `models.json` 并入 `run.json`，但**不是丢弃**：服务端全量
>    `/v1/models/status` 原样存于 `run.json.env.server_models`（含 `final_ceiling`、
>    加载耗时观测与全部模型清单）；`env.omlx.models` 只含被测模型的生效设置。
> 7. `gc` 只按引用关系（`runs/` ∪ `archive/`）标记清扫，**尚未**读取 `retention.pinned/keep_until`
>    做豁免；配额同样未强制。
> 8. 精度测试（`accuracy/` 方法层）未随本次落地，当前只有 `performance`。
> 9. **布局已按第二轮压平**（2026-11-09，用户要求）：`runs/` 由
>    `runs/<YYYY-MM>/<batch_id>/<method>/<model_slug>/` 4 层嵌套改为 **`runs/<run_id>/` 一层**
>    （目录名即 `run_id`），批次元数据由批次目录内的 `batch.json` 移到
>    `output/batches/<batch_id>.json`，`output/` 根变为 7 个条目（新增 `batches/`）。
>    本文 §4/§5/§9/§10 保留第一次落地的历史论证，相关段落已就地加注；**现状以
>    [31-output-contract.md](../design/31-output-contract.md) 为准**。
> 10. **离线构建中间产物已迁出 `output/`**（2026-11-10）：`tool/dataset/*` 现写到
>    `reports/dataset-build/performance/selection_report.md`、
>    `reports/dataset-build/accuracy/selection_report.md` 与
>    `reports/dataset-build/accuracy/pilot/pilot_summary.json`；`output/` 不再产生
>    `performance/`、`accuracy/` 目录。本文 §2.4/附录 A 的历史行保留原路径，仅作审计记录。
>
> 审计依据：`output/` 全量实测（591.1 MB / 4,311 条目 / 8 个 run）、
> 逐文件读写链路代码追溯（含 `文件:行号`）、功能侧数据契约反推。三份审计材料可复现。

---

## 0. 一页结论

**三个根因**

1. **层次缺失**：`output/` 根下 19 个平铺条目（8 个 run 目录 + 11 个其它），每个 run
   内部又是 10 个文件 + `analysis/` + `raw/` + `logs/` 平铺，没有"时间 → 批次 → 方法 → 模型"的
   组织维度；目录名把 `batch/method/model` 三段用下划线拼成 55 字符的长名字，既难读也无法导航。
2. **重复压倒性**：一个 run 57.8 MB 里 **42.3 MB（73.0%）是重复副本**——
   `raw/<model>/**` 是 `records.jsonl` 的逐条等价副本（1920/1920 条对象全等），
   `raw/samples/**` 在 8 个 run 之间 240/240 个路径字节完全相同。
   而真正独占的内容只有 **13.5 MB**：全部 8 个 run 用的是**同一批 240 条 prompt**。
3. **产物没有分层**：权威证据（`records.jsonl`）、聚合派生（`summary.json`/3 个 CSV）、
   展示派生（`report.md`）、环境快照（4 个文件互相内嵌复制）、分析投影（`analysis/` 4 件）
   混在同一层目录里，无法判断哪个能删、哪个能重建。

**目标态（本设计与实测无关的两条硬承诺）**

- 目录层级：`output/runs/<YYYY-MM>/<batch_id>/<method>/<model>/`，一层一个概念。
- 单 run 产物体积：**57.8 MB → ≈0.98 MB**；8 个 run 合计 **463.0 MB → ≈7.8 MB**，
  再加全局 prompt 池 **13.5 MB（全库只存一份）**、批次元数据与索引，
  `output/` 总量 **591.1 MB → ≈21.6 MB（−96%）**，文件数 **4,311 → ≈299**，功能面零损失。

**去重的根本手段**：把"实际发送的 prompt 全文"提升为**全局内容寻址池**
（`blobs/prompts/<sha256>`），逐样本记录只留 64 位引用。同一批 prompt 无论被多少
模型、多少批次、多少 repeat 使用，磁盘上永远只有一份。

---

## 1. 审计范围与方法

| 维度 | 方法 |
|---|---|
| 文件/体积 | 对 `output/` 递归 `stat`，逐目录汇总；8 个 run 逐文件字节对照 |
| 重复 | ① 按 `(sample_id, repeat)` 把 `records.jsonl` 与 `raw/<model>/**` 做 JSON 对象相等比对；② 按相对路径对 8 个 run 做 sha256 比对；③ 全 `output/` 内容寻址（sha256）去重组统计 |
| 生成链路 | 逐文件反查写入函数（`writer.py` / `benchmark.py` / `batch.py` / `excel_report.py` / `catalog.py`），标注行号与触发时机 |
| 消费链路 | 逐文件反查读取方（Web 路由 + 前端、analysis CLI、Excel、门禁、测试），判定"删掉会破坏什么" |
| 需求 | 从 9 类使用场景（Web 列表/详情、批次分组、跨 run 对比、Excel、复现审计、二次分析、精度）反推最小字段集 |

判定口径：

- **权威（M）**：不可再生，删了无法复算 → 必须持久化且只存一份。
- **派生物（D）**：可由权威数据确定性重算 → 可以不在每次 run 落盘。
- **冗余（X）**：与别处内容重复 → 删除。
- **垃圾（J）**：非本系统产物或运行时临时物 → 移出 `output/`。

---

## 2. 现状问题

### 2.1 问题一：层次（平铺）

现状 `output/` 根（19 个条目，按体积降序）：

```text
output/
├── web-20261107T032029Z-4184_performance_gemma-4-e4b-oq4/      57.8 MB   ← 8 个这样的平铺 run
├── web-20261107T032029Z-4184_performance_minicpm-v-4.7-1b/     57.8 MB
├── web-20261107T032029Z-4184_performance_qwen3.6-35b-a3b-oq4/  57.8 MB
├── web-20261107T115732Z-215b_performance_gemma-4-e4b-oq4/      57.8 MB
├── web-20261107T115732Z-215b_performance_minicpm-v-4.6/        57.8 MB
├── web-20261107T115732Z-215b_performance_minicpm-v-4.7-1b/     57.8 MB
├── web-20261107T115732Z-215b_performance_minicpm5-2b/          57.8 MB
├── web-20261107T115732Z-215b_performance_qwen3.6-35b-a3b-oq4/  57.8 MB
├── M4PRO_testdata2.zip                                        100.7 MB   ← 垃圾
├── result_analysis/                                            17.2 MB   ← 派生物混在证据区
├── index.json                                                  24.9 KB   ← 派生物
├── accuracy/                                                  243.9 KB   ← 数据集构建中间产物（位置错）
├── performance/                                                30.9 KB   ← 同上
├── batches/                                                    13.1 KB   ← 批次元数据（第 2 份）
├── .tmp-chrome-profile/                                         3.9 MB   ← 垃圾
├── build/                                                         空      ← 垃圾
├── .benchmark.lock / .index.lock / .DS_Store                     ~12 KB   ← 运行时/OS 垃圾
```

单个 run 内部（495 个文件 / 57.8 MB）同样是平的（**13 个条目**）：

| 条目 | 体积 | 占比 | 性质 |
|---|---|---|---|
| `raw/` | 40.3 MB | 73.0% | **全部冗余**（见 §2.2） |
| `records.jsonl` | 13.9 MB | 25.2% | 权威 |
| `analysis/` | 0.52 MB | 0.9% | 派生物，零机器消费者 |
| `summary.json` | 0.26 MB | 0.5% | 派生（含 0.13 MB 冗余 `samples[]`） |
| `report.md` | 37 KB | 0.1% | 派生 |
| `samples.csv` | 43 KB | 0.1% | 派生（与 `summary.samples` 同构） |
| `manifest.json` | 41 KB | 0.1% | 元数据（内嵌 3 份副本） |
| `cells.csv` / `summary.csv` | 12 KB | 0.02% | 派生 |
| `models.json` / `omlx.json` / `hardware.json` | 30 KB | 0.05% | 元数据（`manifest` 内已有） |
| `logs/run.log` | 34 KB | 0.06% | 运维 |

问题：**没有"批次"这一层**。同一次测试动作（同一批 240 样本、同一时间窗、同一台机器）
产生的 3～5 个模型 run 在磁盘上彼此无关，只能靠读 `batches/<id>.json` 才能知道它们属于
同一个 batch。而"一次测试动作"恰恰是用户心智里的单位。

### 2.2 问题二：重复（一张表）

| # | 重复内容 | 副本 1 | 副本 2 | 实测 | 判定 |
|---|---|---|---|---|---|
| R1 | **整条逐样本记录** | `records.jsonl`（13.9 MB） | `raw/<model>/<scenario>/<bucket>/*.json`（14.8 MB） | 1920 条记录 → 1920 个文件，**JSON 对象 1920/1920 全等**；差异仅 `indent=2`（多 1.36% 字节） | **X：删** |
| R2 | **原始样本 + prompt 全文** | `raw/samples/**`（27.46 MB/run） | 8 个 run 各一份 + 文件内 `question` 与 `metadata.messages` 又互抄一次 | 240 个相对路径在 **8/8 个 run 中 sha256 完全相同**；物理 219.7 MB → 唯一 27.46 MB（**87.5% 可去重**）；单文件内 `question` == `messages` 拼接（240/240） | **X：删** |
| R3 | **prompt 全文** | `records.jsonl.request`（占 records **95.03%**，其中 `messages` 本身占 93.55%） | R1 副本 + R2 副本 + Excel 明细页 | 全 `output/` 只有 **240 条唯一 prompt / 12.06 M 字符**，却存在约 32 份拷贝 → 约 440 MB | **改造为内容寻址，只存一份** |
| R4 | **环境快照** | `manifest.json` 内嵌 `hardware`/`omlx`/`models` | 独立 `hardware.json` / `omlx.json` / `models.json` | 三份独立文件与 manifest 内嵌值**逐字节相等**；独立文件 30.6 KB 占 manifest 41.6 KB 的 73% | **X：合并为 1 份** |
| R5 | **逐样本表** | `summary.json.samples[]`（240×22） | `samples.csv`（240×22） | 行数、列数、取值完全同构 | **X：合并为 1 份** |
| R6 | **逐样本长表** | `analysis/samples_long.csv`（535 KB） | 即 R5 的 12 个 metric 字段 melt 展开 | 反比 `samples.csv` 大 11× | **X：删**（pandas 一行可 melt） |
| R7 | **聚合结果** | `summary.json.overall/cells` | `summary.csv` + `cells.csv` | 同一 `aggregate_*` 输出的两种序列化 | **合并**：留 `aggregates.json`，CSV 可选导出 |
| R8 | **批次元数据** | `output/batches/<id>.json` | `result_analysis/<batch>/batch.json` + `index.json.batches` | 3 份内容重叠的批次元数据 | **X：只留批次目录内 1 份** |
| R9 | **prompt 全文（第 3 处）** | Excel「详细数据」页 | R3 的同一批文本 | `summary.xlsx` 单文件 3.6–9.0 MB，17.2 MB/3 批，主要是 prompt 文本 | **D：改为按需生成** |

> 量化结论分两个口径，都基于实测：
>
> 1. **按文件 sha256 去重**：591.1 MB → 唯一 397.8 MB，即 **193.3 MB（32.71%）
>    是逐字节重复的整份文件**（主要是 `raw/samples/**` 被 8 个 run 各存一份，
>    以及 `raw/<model>/**` 与 records 同对象）。
> 2. **按内容语义去重**（本设计的目标口径）：同一段 prompt 文本被装进
>    `records.jsonl`、`raw/<model>/*.json`、`raw/samples/*.json`、Excel 单元格 4 种
>    容器，字节不同所以口径 1 抓不到；把这些容器统一后，`output/` 只需 **≈21.6 MB**。
>    换言之，591.1 MB 里有 **约 569 MB（96%）是可消除的重复或派生物**。
>
> 这也是为什么不能只靠"sha256 去重"：它能发现 193 MB，却漏掉更大的一块
> ——"同样的内容、不同的包装"。§6 的分层 + 内容寻址才是对症手段。

### 2.3 问题三：只写不读

以下产物**没有任何生产代码消费方**（`src/`、`tool/`、`harness/` 均无解析点）：

| 产物 | 写入点 | 消费情况 | 体积 |
|---|---|---|---|
| `raw/<model>/**` | `writer.py:130-140` | 零读取（连测试都不读） | 14.8 MB/run |
| `raw/samples/**` | `benchmark.py:1784-1793`（调用 `1296`） | 仅 `test_integration.py:621-622` 断言存在；指向它的 `sample_meta.raw_sample_path`（`benchmark.py:1754`）**零读取** | 27.5 MB/run |
| `analysis/run.json`、`analysis/environment.json`、`analysis/samples_long.csv`、`analysis/README.md` | `writer.py:255/283/284/285` | 无 JSON/CSV 解析 | ① 被 `index.json.runs[].analysis` 列出；② Web 文件条下载按钮（`app.js:889`）；③ `test/test_results.py:93-111` 确实 `json.loads` 它们做断言（测试契约，非产品消费者） | 0.52 MB/run |
| `summary.csv`、`cells.csv`、`samples.csv` | `writer.py:375-430/432-482/484-487` | 仅测试断言表头 | 0.06 MB/run |
| `result_analysis/**/batch.json`、`README.md` | `excel_report.py:454-459` | 零读取（权威副本在 `output/batches/`） | 4 KB/批 |

**这是设计的关键许可**：重复副本没有消费者，因此去重不需要保留兼容行为，
只需要把**唯一权威源**与**真正的消费方**对齐。

不能删的反例（有真实消费方）：`report.md`（Web 报告页 `server.py:302-303` + `app.js:877`）、
`omlx.json`（门禁 `gate.py:504` 的 `env_ok` 存在性判定 + Excel `excel_report.py:183`）、
`records.jsonl`（Excel 明细 + 门禁 + 复算）、`summary.json`（Web 三个页签 + `analyze` + Excel 平均值页）。

### 2.4 问题四：垃圾与越界

| 条目 | 体积 | 说明 |
|---|---|---|
| `M4PRO_testdata2.zip` | 100.7 MB | 人工放入的测试数据包，非本系统产物 |
| `result_analysis/**/summary.xlsx` | 17.2 MB | 派生交付物，却与证据同级混放，且自动生成常驻 |
| `.tmp-chrome-profile/` | 3.9 MB | 浏览器 profile 临时目录（311 个文件），应进系统 temp |
| `accuracy/`、`performance/selection_report.md` | 0.27 MB | 由 `tool/dataset/{performance,accuracy}/build.py` 写入，属**数据集构建中间产物**，不是测试结果（**现状已迁出**：见头部第 10 条，写 `reports/dataset-build/…`） |
| `build/` | 空 | 残留空目录 |
| `.benchmark.lock`、`.index.lock`、`.DS_Store` | ~12 KB | 运行时锁与 OS 元数据 |

### 2.5 问题五：命名与索引缺陷

1. **目录名承载语义**：`web-20261107T032029Z-4184_performance_gemma-4-e4b-oq4` 把
   batch 前缀、method、model 三段拼在一起，55 字符；模型名一改格式就不可读。
2. **索引存绝对路径**：`index.json.runs[].path` 是
   `/Users/.../output/web-...`，整个 `output/` 一搬家，索引里所有路径失效。
3. **批次元数据同样存绝对路径**（`batch.py:204-220`），跨机器/移动即失效。
4. **无完整性清单**：`index.json.runs[].files` 只记 `{size}`，没有 sha256，
   无法回答"这个 bundle 有没有被手改过"；且 `files` 只列 10 个顶层文件、**不含 `raw/` 与 `logs/`**，
   未覆盖的体积占 73.11%，导致 `bytes_total` 无法由 `files` 复算。
5. **无法自证输入**：`manifest.config_snapshot` 记录了
   `dataset/test/performance/testset_10.jsonl`——这个文件**不存在**（悬空引用），
   且没有 testset 的 sha256，所以今天只能靠"存一份 prompt 全文"来保证可复现。

---

### 2.6 附带发现：派生 CSV 的均值不可作为结论

审计过程中发现 2 条异常记录（`perf-general_agent-8k-018`、`perf-general_agent-32k-018`）
`tpot_ms = 0.0`，据此算出的 `gen_tps` 分别为 **583,332** 与 **345,477**（中位数区间本应 ~59/45），
把 `summary.csv` 的 `gen_tps_mean` 抬到 **8,049 / 4,713**。

这条不属于目录设计范畴（属指标侧缺陷，需独立任务确认根因），但它直接支撑本文的一个论点：

- **聚合派生值（CSV 里的 mean）容易被单条脏记录污染，而逐样本表不会。**
  因此 `samples.csv` 必须保有完整逐样本字段（含 `tpot_ms`/`gen_tps`/`cache_*`），
  让"异常值 → 回查原始记录"成为一步操作；
- 也说明 `summary.csv`/`cells.csv` 这类**丢失分母与分位数**的投影
  （实测：`summary.csv` 的 mean 只有 **22/30** 能由 `cells.csv` 精确复算，逐指标分母齐全时才 30/30）不适合作为权威，
  应降级为按需导出。

---

## 3. 设计原则

| # | 原则 | 直接后果 |
|---|---|---|
| P1 | **一份事实只落一个文件** | 每个数据项在全 `output/` 中只存一份；其余为显式引用 |
| P2 | **按"可再生性"分三层**：证据 / 快照 / 投影 | 权威证据不可再生；快照是 run 时点；投影随时可重建，允许不落盘 |
| P3 | **批次是可移植与归档单元** | 一次测试动作 = 一个目录；跨模型对比、搬移、归档都以批次为单位 |
| P4 | **大文本内容寻址** | 唯一超 10 MB 的可变内容（prompt）进全局 blob 池，记录只存 sha256 |
| P5 | **目录层级表达导航，文件名表达内容** | 时间/批次/方法/模型进路径；语义不再拼进目录名 |
| P6 | **派生物离开证据树** | 派生物可以删、可以重建，不占据权威位置 |
| P7 | **可校验** | 每个 run 有 `MANIFEST.json`（path/bytes/sha256）；索引只存相对路径；提供对账命令 |

---

## 4. 目标结构

### 4.1 总览

> **已按第二轮压平（2026-11-09，用户要求）**：现状是 `runs/<run_id>/` 一层平铺 +
> `output/batches/<batch_id>.json`（`output/` 根 7 个条目）。下面这棵树是设计初稿的
> 4 层结构，仅作历史论证；现状目录树见 [31-output-contract.md](../design/31-output-contract.md) §1。

```text
output/
├── index.json                              # D  全局索引（唯一入口，可重建）
├── blobs/                                  # M  全局内容寻址池（跨 run / 跨批次去重）
│   └── prompts/
│       ├── ab/
│       │   └── ab12ef…(64).json            # M  实际发送的 messages（正文只存一份）
│       └── …
├── runs/                                   # M  全部测试产物，按 时间 → 批次 → 方法 → 模型 分层
│   └── 2026-11/
│       └── 20261107T032029Z-4184/          # M  批次目录（= 一次测试动作，可移植单元）
│           ├── batch.json                  # M  批次权威元数据（相对路径）
│           ├── labels.jsonl                # M* 仅精度批次：sample_id → answer/choices
│           ├── performance/                # M  测试方法层
│           │   ├── gemma-4-e4b-oq4/        # M  模型 run（一个模型一个目录）
│           │   │   ├── run.json            # M  运行元数据 + 环境快照（唯一）
│           │   │   ├── MANIFEST.json       # M  产物清单 + sha256
│           │   │   ├── records.jsonl       # M  权威逐样本证据（prompt 以 sha256 引用）
│           │   │   ├── samples.csv         # D  逐样本主表（Web 分页 / pandas）
│           │   │   ├── aggregates.json     # D  overall + cells
│           │   │   ├── report.md           # D  人读摘要
│           │   │   └── logs/run.log        # O  运行日志
│           │   ├── minicpm-v-4.7-1b/
│           │   └── qwen3.6-35b-a3b-oq4/
│           └── accuracy/                   # M  精度测试与性能同构（未来）
│               └── <model>/…
├── exports/                                # D  人类交付物（按需生成，可整目录删除）
│   └── <batch_id>.xlsx
├── archive/                                # A  退役批次/旧格式 bundle（仍是证据，不删）
└── .state/                                 # J  运行时状态（非证据）
    ├── index.lock
    ├── benchmark.lock
    └── tmp/                                # 浏览器 profile 等临时物
```

图例：**M** 权威（不可再生）｜**D** 派生（可重建）｜**O** 运维｜**A** 归档（证据，保留，且是 GC 引用根）｜**J** 垃圾。

### 4.2 为什么是这四层

> **已按第二轮压平**：`runs/<YYYY-MM>/`、`<batch_id>/`、`<method>/` 三层已合并为
> `runs/<run_id>/` 一层；下表保留当时“为什么需要层次”的论证，现状规则见
> [31-output-contract.md](../design/31-output-contract.md) §2。

| 层级 | 概念 | 回答的问题 | 现状对应 |
|---|---|---|---|
| `runs/<YYYY-MM>/` | 时间 | "上个月测了什么" | 无（混在目录名前缀里） |
| `<batch_id>/` | **一次测试动作** | "那次 4 个模型一起跑的结果在哪" | 靠读 `batches/<id>.json` 才隐式存在 |
| `<method>/` | 测试方法 | "这次跑的是性能还是精度" | 编码在目录名里 |
| `<model>/` | 被测对象 | "这个模型的表现" | 目录名尾部 |

现状把后三层压进一个 55 字符目录名，导致"批次"这个概念在文件系统上**不可见**；
新结构让每一层都能被 `ls` 直接看到，也让跨模型对比（同批次内 `ls` 一下）变得自然。

同时，`batch.json` 成为批次元数据的**唯一**权威副本（现在有 3 份），
`index.json.batches[]` 退化为它的派生视图。

### 4.3 可移植 / 归档单元 = 批次目录

> **已按第二轮压平**：没有“批次目录”这一层。可移植/归档单位是
> `batches/<batch_id>.json` + 其 `model_runs[].run_id` 指向的 `runs/<run_id>/` 集合；
> `pack` 就是按这个单位打包。

一个批次目录自包含：批次元数据 + 所有模型 run + （精度）标签。
prompt 正文在全局 `blobs/` 中，按 P2 属于"输入证据"层，与批次解耦——因为
**同一份 testset 的 prompt 会被后续批次反复使用**（实测：2 个批次 8 个 run 共用
同一批 240 条 prompt）。

因此归档语义调整为：

- **在 `output/` 内部移动**：直接 `mv runs/2026-11/<batch> archive/`，索引重建即可，blob 引用不变。
- **整体搬到外部盘**：搬整个 `output/`，或先打包（见 §8.3，打包会把被引用的 blob 一并收进去）。
- **过期清理**：`blobs/` 的垃圾回收按"是否被现存 run 引用"做标记清扫（见 §8.2）。

---

## 5. 文件规范

### 5.0 命名规范（消除"名字承载语义"）

现状目录名把 3 段语义拼成 55 字符，且前缀 `web-` 还编码了"从哪发起"。新规则：

| 标识 | 规则 | 例 | 说明 |
|---|---|---|---|
| `batch_id` | `<UTC秒级时间戳>-<4 hex>`；**不再带 `web-` 前缀** | `20261107T032029Z-4184` | 发起来源（web/cli）写进 `batch.json.source`，不进名字 |
| `method` | 固定枚举 | `performance` / `accuracy` | 目录段 |
| `model_slug` | `sanitize_name(alias or id)`（沿用现有规则） | `gemma-4-e4b-oq4` | 目录段 |
| `run_id` | **单段稳定键** `<batch_id>__<method>__<model_slug>` | `20261107T032029Z-4184__performance__gemma-4-e4b-oq4` | 不含 `/`，可直接做索引键 / URL 参数 |
| `run_path` | 相对 `output/` 的多段路径 | `runs/2026-11/20261107T032029Z-4184/performance/gemma-4-e4b-oq4` | 导航用；**必须**校验拼接后仍在 `output/` 内 |
| 磁盘路径 | `runs/<YYYY-MM>/<batch_id>/<method>/<model_slug>/` | 同上 | `YYYY-MM` 由 `batch.json.created_at` 派生，**不是身份的一部分** |

> **已按第二轮压平**：`run_path` = `runs/<run_id>`，磁盘路径 = `runs/<run_id>/`（一层），
> 批次元数据在 `batches/<batch_id>.json`；下表第 5–6 行的多段路径/`YYYY-MM` 是初稿，
> 现状见 [31-output-contract.md](../design/31-output-contract.md) §2。

设计要点：

1. `run_id` **刻意做成单段**。独立评审确认：若让 `run_id` 含 `/`，会同时击穿
   `catalog.py:188`（`run_id = run_dir.name`）、`catalog.py:375`（按 run_id 去重）、
   `server.py:286-292`（把 run_id 当单段拼 `OUTPUT_DIR`）三处既有假设。
2. 因此**导航用 `run_path`，索引键用 `run_id`**，两者一一对应、可互相派生。
   Web/CLI 接收 `run_path` 并做"解析后必须仍在 `output/` 内"的越界校验（沿用
   `server.py:286-292` 的安全语义，只是校验对象从 `run_id` 换成 `run_path`）。
3. "run_id 不可变"语义：`run_id` 不含日期段，批次在 `runs/` 内按时间重新分桶或移入
   `archive/` 时身份不变，索引重建即可。

### 5.1 `output/index.json`（D，索引 schema `2.0`）

> **已按第二轮压平**：现状 `runs[].run_path` = `runs/<run_id>`、`batches[].path` =
> `batches/<batch_id>.json`；下面示例中的多段路径保留为初稿。

唯一入口，纯派生：删掉后执行索引重建即可完全恢复。**只存相对 `output/` 的路径**（修复 §2.5-2），
**不写 `output_dir` 绝对路径**（现状 `catalog.py:281-288`），根从索引文件自身位置推导。

```jsonc
{
  "schema_version": "2.0",
  "generated_at": "2026-11-09T13:20:00+00:00",
  "n_batches": 2,
  "n_runs": 8,
  "n_blobs": { "prompts": 240, "bytes": 13520000 },
  "totals": { "bytes": 21600000, "files": 299 },
  "batches": [{
    "batch_id": "20261107T032029Z-4184",
    "created_at": "2026-11-07T03:20:29Z",
    "path": "runs/2026-11/20261107T032029Z-4184",
    "methods": ["performance"],
    "models": [{
      "model": "gemma-4-E4B-it-oQ4",
      "model_slug": "gemma-4-e4b-oq4",
      "run_path": "runs/2026-11/20261107T032029Z-4184/performance/gemma-4-e4b-oq4",
      "status": "ok",
      "n_samples": 240,
      "duration_s": 3012.4
    }]
  }],
  // 派生扁平行：Web 列表页 / CLI runs 一次读完即渲染，不必进目录
  "runs": [{
    "run_id": "20261107T032029Z-4184__performance__gemma-4-e4b-oq4",
    "run_path": "runs/2026-11/20261107T032029Z-4184/performance/gemma-4-e4b-oq4",
    "batch_id": "20261107T032029Z-4184", "batch_index": 0, "batch_models_total": 3,
    "method": "performance", "mode": "performance", "legacy": false,
    "model": "gemma-4-E4B-it-oQ4", "model_id": "gemma-4-E4B-it-oQ4", "model_size_gb": 5.2,
    "created_at": "…", "duration_s": 3012.4, "status": "ok", "cancelled": false,
    "models_selected": ["…"], "models_measured": ["…"],
    "n_models_selected": 1, "n_models_measured": 1,
    "n_skipped": 0, "skipped": [], "memory_errors": 0,
    "n_samples": 240, "n_records": 240, "n_cells": 24, "datasets": ["…"],
    "bytes_total": 1027604, "bytes_total_human": "1.0 MB",
    "files": { "records.jsonl": {"bytes": 758123, "sha256": "…"} }
  }]
}
```

`runs[]` 字段是**从现状 Web 列表页的真实需求反推**的（`app.js:98,641,722-790` 用到
`legacy`/`models_selected`/`models_measured`/`n_skipped`/`skipped`/`mode`/`model_size_gb`/
`bytes_total_human`），缺一个列表页就会退化。`files` 直接取自 `MANIFEST.json`
（避免重建索引时 `rglob` 全盘 stat）。

两个版本号独立演进：

| 版本号 | 作用域 | 现值 → 目标 | 说明 |
|---|---|---|---|
| `SCHEMA_VERSION`（`models.py:20`） | **run bundle 内容格式**（`records.jsonl`/`run.json`/`aggregates.json`/`batch.json`） | `2.1` → **`3.0`** | 决定字段结构，不兼容 2.x |
| `INDEX_SCHEMA_VERSION`（`catalog.py:31`） | **仅 `output/index.json` 的格式** | `1.0` → **`2.0`** | 索引是独立派生物，可独立演进 |

索引 2.0 只认 bundle 3.0（本设计不读 2.x，见 §9）。每个落盘 JSON 文件都带
`schema_version` 字段（现状 4 个环境文件缺失）。

### 5.2 `blobs/prompts/<sha256>.json`（M，全局唯一）

**这是整个设计的支点。** 存放"实际发送给模型的完整输入"，内容寻址，跨 run / 跨批次 /
跨模型 / 跨 repeat 只存一份。

```jsonc
{
  "sha256": "ab12ef…(64 hex)",     // 内容自校验：sha256(规范化 payload)
  "chars": 12034,                  // 拼接后字符数（= 输入规模归因）
  "messages": [                    // 精确等于发送时的 request.messages
    {"role": "user",  "content": "…", "name": null, "tool_call_id": null, "tool_calls": null},
    {"role": "user",  "content": "This request is a performance measurement. …"}
  ],
  "tools": null,                   // 若场景带 tools，一并纳入哈希
  "chat_template_kwargs": {"enable_thinking": false}
}
```

**规范化与哈希（必须固定，否则去重失效）**

1. 哈希输入 = payload 去掉 `sha256` 字段后的**规范化 JSON**：
   `sort_keys=True, separators=(",", ":"), ensure_ascii=False`，UTF-8 编码，末尾无换行。
2. `sha256 = hashlib.sha256(canonical_bytes).hexdigest()`，即文件内容自校验。
3. **原子写**：写 `.tmp` 后 `os.replace`；目标已存在则跳过（幂等 = 去重）。
4. 分片目录 `blobs/prompts/<sha[:2]>/`，控制单目录条目数。
5. 仅在 `truncated=true` 时额外存"截断前原文"`prompt_orig_sha256`（正常路径不产生）。
6. 记录侧引用字段：`request.prompt_sha256`。
7. `decode_guard_version` 随 guard 文案变动递增，保证历史 run 的 prompt 可自证。

**与 `dataset/test/performance/testset.jsonl` 的关系（必须交代）**

testset 每行已含 `messages`（实测 13.47 MB / 240 行），而 blob 存的是
`testset.messages` **+ 一条固定的 decode-guard user 消息**。两者体积接近但**语义不同**：

| | `dataset/test/**/testset.jsonl` | `output/blobs/prompts/` |
|---|---|---|
| 层 | 输入（可重建） | 输出侧证据（不可再生） |
| 内容 | build 期产物，`messages` 无 guard | **实际发送**的完整 messages，含 guard |
| 变更 | 重新 build 即变 | 一旦写入永不改 |
| 作用 | 提供样本 | 证明"这一次到底发了什么" |

保留 blob 的理由：① 不依赖 dataset 是否还在/是否被重建；② guard 文案若改动，历史 run
仍可自证。这是"证据自包含"与"零仓库级重复"的取舍：若决定极致省空间，可去掉 blob，
改由 `batch.json.inputs.testset.sha256 + sample_id + decode_guard_version` 解析，
代价是 Excel「API 输入内容」列与任何 prompt 回放都依赖 dataset 在场。

**收益**：实测 8 个 run 共用 240 条唯一 prompt（12.06 M 字符）。今天它们被写了约 32 份
（约 440 MB）；新结构下**全 `output/` 只有 240 个文件、约 13.5 MB**。

### 5.3 `runs/<YYYY-MM>/<batch_id>/batch.json`（M）

> **已按第二轮压平（2026-11-09）**：批次元数据实际位于 `output/batches/<batch_id>.json`；
> `model_runs[]` 改用 `run_id` + `run_path`（相对 `output/`，**无 `run_dir`**）。下文字段
> 结构保留设计讨论，现状示例见 [31-output-contract.md](../design/31-output-contract.md) §5.1。

批次权威元数据的**唯一**副本（取代 `output/batches/<id>.json`、`result_analysis/*/batch.json`
与 `index.json.batches` 三处重复）。

```jsonc
{
  "schema_version": "3.0",
  "batch_id": "20261107T032029Z-4184",
  "source": "web",                                 // web | cli（取代目录名里的 `web-` 前缀）
  "method": "performance",
  "created_at": "2026-11-07T03:20:29Z",
  "finished_at": "2026-11-07T13:00:11Z",
  "models_selected": ["gemma-4-E4B-it-oQ4", "minicpm-v-4.7-1b", "qwen3.6-35B-A3B-oQ4"],
  "model_runs": [                                  // 相对本目录，可迁移
    {"model": "gemma-4-E4B-it-oQ4", "model_id": "…", "model_slug": "gemma-4-e4b-oq4",
     "method": "performance", "run_dir": "performance/gemma-4-e4b-oq4",
     "status": "ok", "n_samples": 240, "duration_s": 3012.4}
  ],
  "inputs": {                                      // 修复 §2.5-5：输入可自证
    "testset": {
      "path": "dataset/test/performance/testset.jsonl",
      "sha256": "…",                               // 文件级指纹
      "rows": 240,
      "context_windows": [1024, 8192, 32768],      // 必须非空（现状 8/8 为空 → Web 显示"全部"）
      "decode_guard_version": 1
    }
  },
  "config": { "perf": {}, "run": {}, "scenarios": ["…"] },
  "retention": { "pinned": false, "keep_until": null },
  "provenance": { "git_commit": "…", "host": "…" },
  "warnings": [], "errors": []
}
```

**输入自证（这是能删掉 prompt 副本的前提）**。今天 `manifest.config_snapshot` 记录的是
`dataset/test/performance/testset_10.jsonl`——**该文件不存在**（悬空引用），且不含任何
testset 指纹，所以 bundle 无法自证"我跑的是哪 240 行"，只能靠存一份 prompt 全文兜底。

实际上指纹**已经有了**，只是没被引用：`dataset/test/performance/manifest.json` 的
`files["testset.jsonl"]` 已含 `rows/bytes/sha256`（实测 `sha256=1099e8b8…`，与文件一致）。
因此 `batch.json.inputs.testset.sha256` 直接抄这个值即可，**不需要新增机制**，
只需在批次开始时把 `path + sha256 + rows` 写进 `batch.json`，并修掉悬空的 `testset_10.jsonl`。

### 5.4 `runs/…/<method>/<model>/` 下的 7 个条目

> **已按第二轮压平**：目录为 `runs/<run_id>/`（一层），7 个条目与职责不变。

#### `run.json`（M）— 合并 6 个旧文件

取代 `manifest.json` + `hardware.json` + `omlx.json` + `models.json`
+ `analysis/run.json` + `analysis/environment.json`。去掉了 §2.2-R4 的内嵌复制。

```jsonc
{
  "schema_version": "3.0",
  "run_id": "20261107T032029Z-4184__performance__gemma-4-e4b-oq4",
  "batch_id": "20261107T032029Z-4184",
  "method": "performance",
  "model": {"id": "gemma-4-E4B-it-oQ4", "alias": "gemma-4-e4b-oq4", "model_slug": "gemma-4-e4b-oq4"},
  "created_at": "…", "started_at": "…", "finished_at": "…", "duration_s": 3012.4,
  "git": {"commit": "…", "dirty": false},
  "config": { "perf": {}, "run": {}, "scenarios": [], "context_windows": [1024, 8192, 32768] },
  // 环境证据：唯一一份。model_env 按规范化哈希索引，正常只有 1 个条目；
  // 若 run 中途重载模型导致快照变化，追加新条目，记录用 model_env_ref 指向当时那份。
  "env": {
    "hardware": { "chip": "…", "hostname": "…", "arch": "arm64",
                  "cpu": {}, "memory": {}, "os": {}, "gpu": [], "python": {},
                  "upload_fields": { /* 上报扁平投影；唯一一份，不再嵌套重复 */ } },
    "omlx": { "version": "…", "feature_flags": {}, "global_settings": {} },
    "model_env": {
      "3f9c…(16)": {"id": "gemma-4-E4B-it-oQ4", "engine_type": "vlm", "model_type": "vlm",
                    "quantization": "oQ4", "estimated_size": 5612040826,
                    "resident_estimated_size": 5612040826, "context_length": 131072,
                    "settings": {}, "feature_flags": {}, "loaded": true}
    }
  },
  "outcome": {
    "status": "ok", "cancelled": false, "memory_errors": 0,
    "n_selected": 240, "n_measured": 240, "n_records": 240, "n_samples": 240, "n_cells": 24,
    "skipped": [                                // 结构化（现在是自由文本，只能 regex）
      {"scenario": "…", "sample_id": "…", "repeat": 0, "reason": "…", "kind": "load_failed"}
    ],
    "warnings": []
  }
}
```

**不变式**：`model_env` 中每个条目的键必须等于其内容的规范化哈希。若一次 run 内模型快照
发生变化（中途 reload），必须追加新条目并让后续记录用新的 `model_env_ref`，
**不允许静默丢弃环境证据**（现状 `model_config_snapshot` 逐条记录，实测 8/8 恒定，
但"恒定"只是观察而非不变式；§11 增加该断言）。

#### `MANIFEST.json`（M）— 新增，可校验

```jsonc
{
  "schema_version": "3.0", "run_id": "…", "generated_at": "…", "algorithm": "sha256",
  "files": [
    {"path": "records.jsonl", "bytes": 754974, "sha256": "…"},
    {"path": "samples.csv",   "bytes": 45123,  "sha256": "…"},
    {"path": "aggregates.json","bytes": 8123,  "sha256": "…"},
    {"path": "report.md",     "bytes": 38118,  "sha256": "…"},
    {"path": "run.json",      "bytes": 41234,  "sha256": "…"},
    {"path": "logs/run.log",  "bytes": 34695,  "sha256": "…"}
  ],
  "totals": {"files": 6, "bytes": 981000},
  "blob_refs": {"prompts": 240, "bytes": 13520000, "missing": 0}
}
```

`records.jsonl` 的 sha256 必填；`blob_refs.missing` 在生成时与 `verify` 时都必须为 0。

#### `records.jsonl`（**M，权威**）— 瘦身版

一行一条记录。**相对现状的三处结构性变化**：① `request.messages` → 64 位引用；
② `model_config_snapshot` → `run.json.env.model_env` + `model_env_ref`；
③ 删 `sample_meta.raw_sample_path`。其余字段一个不少。

```jsonc
{
  "schema_version": "3.0",
  "run_id": "…", "timestamp": "…",
  "model": "gemma-4-E4B-it-oQ4", "model_env_ref": "3f9c…(16)",
  "scenario": "coding", "dataset": "longbench_v2:Code Repository Understanding",
  "context_length": 1024, "sample_id": "perf-coding-1k-000", "subtask": null,
  "repeat": 0, "phase": "measured", "status": "ok", "error": null,

  "request": {                       // 输入：指纹 + 发送参数，不含正文
    "prompt_sha256": "ab12ef…(64)",  // → blobs/prompts/ab/ab12ef….json
    "prompt_chars": 12034,           // 拼接后字符数
    "actual_prompt_tokens": 1013,    // 样本声明/build 期 token（= context_meta.declared_tokens）
    "max_tokens": 50, "temperature": 0.0, "top_p": null, "seed": null, "stream": true,
    "chat_template_kwargs": {"enable_thinking": false},
    "truncated": false, "padded": false,
    "context_meta": { "bucket": "1k", "declared_tokens": 1013,
                      "model_context_window": 131072, "decode_guard": true,
                      "truncation_ratio": null }
  },

  "response": {                      // 输出全文：保留（仅占 1.4%，见 §6.2）
    "text": "…", "reasoning_text": "", "tool_calls": [], "finish_reason": "length",
    "usage": {"prompt_tokens": 1174, "completion_tokens": 50, "total_tokens": 1224,
              "cached_tokens": 0, "time_to_first_token": 1.22,
              "generation_duration": 0.75, "prompt_eval_duration": 1.22,
              "generation_tokens_per_second": 66.66, "total_time": 1.97},
    "num_chunks": 12,
    "first_token_ts": 1.2587, "last_token_ts": 2.0068
  },

  "metrics": {                       // 权威指标：唯一归一化来源
    "ttft_ms": 1258.8, "ttft_visible_ms": 1258.8, "tpot_ms": 15.27, "e2e_ms": 2007.3,
    "gen_tps": 65.5, "prefill_tps": 932.7, "total_tps": 609.77,
    "input_context_tokens": 1174, "input_context_chars": 3873,   // AGENTS.md §2 第13条 硬不变量，只在此处
    "completion_tokens": 50, "prompt_tokens": 1174, "cached_tokens": 0,
    "peak_cpu_pct": 120.7, "mean_cpu_pct": 43.9, "cpu_seconds": 0.877,
    "peak_rss_mb": 6376.7, "peak_memory_gb": 6.7,
    "cpu_samples": 29, "cpu_sample_window_ms": 2015.7
  },

  "cache": {                         // 冷缓存合规的唯一证据（§1/§2 硬不变量）
    "mode": "cold", "cleared_before_request": true, "clear_latency_ms": 64.9,
    "cached_tokens": 0, "polluted": false, "note": "hot+ssd cleared (settle 800ms)"
  },

  "sample_meta": {"dataset_meta": {"bucket": "1k", "lang": "en"},
                  "input_context": {"prompt_tokens": 1013, "chars": 3873, "truncated": false}},

  "score": null                      // 性能模式恒 null（判分器不接入）
}
```

**关于"刻意冗余"的边界（不属于要消除的重复）**

`metrics` 是归一化后的**唯一指标来源**；`response.usage` 保留 provider 原样返回，
用于审计"归一化是否正确"。两者在 `prompt_tokens`/`completion_tokens` 上重合是
**原始值 vs 归一化值**的刻意对照，不列入 §6.5 的重复处置。
除此之外，同一事实不允许在一条记录内出现两次——因此 `input_context_tokens/chars`
只放 `metrics`，`request` 只保留发送侧的 `prompt_sha256`/`prompt_chars`/`actual_prompt_tokens`。

**删掉的字段及去处**

| 旧字段 | 处置 | 理由 |
|---|---|---|
| `request.messages[]`（占 95.03%） | → `blobs/prompts/<sha>.json`，记录留 `prompt_sha256` | 内容寻址去重 |
| `model_config_snapshot`（1.3%） | → `run.json.env.model_env{}`，记录留 `model_env_ref` | 每 run 恒定，逐条重复 240 次 |
| `sample_meta.raw_sample_path` | 删除 | 指向 `raw/samples/`，零读取方，且该目录不再生成 |

体积：单 run **13.9 MB → ≈0.74 MB**。

#### `samples.csv`（D）— 逐样本主表

一行一个 measured 样本，`records.jsonl` 的扁平投影，供 Web 逐样本页分页与 pandas 使用。
**取代** `summary.json.samples[]`（R5）。列在现状 22 列基础上**补齐冷缓存审计字段**：

```text
model, model_id, scenario, dataset, subtask, context_length, sample_id, repeat, phase, status,
finish_reason, length_capped, input_context_tokens, input_context_chars, actual_prompt_tokens,
ttft_ms, ttft_visible_ms, tpot_ms, gen_tps, e2e_ms, prefill_tps, total_tps,
completion_tokens, cached_tokens, cache_mode, cache_cleared, cache_polluted,
peak_memory_gb, peak_rss_mb, peak_cpu_pct, mean_cpu_pct,
prompt_sha256, truncated, error, timestamp
```

补 `cache_*` 的原因：现状 `samples.csv` 没有 `cached_tokens`，做冷缓存审计必须回读
14 MB 的 `records.jsonl`——这是"逐样本主表"名不副实的直接后果。

#### `aggregates.json`（D）

`overall` + `cells` 的唯一机器可读形态，取代 `summary.json` 的聚合段
与 `summary.csv`/`cells.csv`。CSV 版如需，用一条命令导出，不常驻。

```jsonc
{ "schema_version": "3.0", "run_id": "…", "generated_at": "…",
  "n_records": 240, "n_samples": 240, "n_cells": 24,
  "overall": [ /* 每 (model, context_length) 一行：n_*、ttft 均值/中位/P95、gen_tps、tpot、e2e、prefill、峰值内存/CPU、输入/输出 token、duration */ ],
  "cells":   [ /* 每 (model, scenario, dataset, context_length) 一行 */ ] }
```

#### `report.md`（D）

人读摘要，Web 报告页直接渲染（保留现状内容与渲染逻辑）。

#### `logs/run.log`（O）

运行日志。运行中由 Web 任务面板增量读取；run 结束后仅排障用，可随归档清理。

### 5.5 `exports/<batch_id>.xlsx`（D）

批次级 Excel 交付物（现有三页签：硬件配置 / 模型性能平均 / 详细数据）。**改为按需生成**：

- 不再在批次结束时无条件写入（现状 3.6–9.0 MB/批，是 prompt 文本的又一份拷贝）；
- 生成时从 `blobs/` 解析 prompt 填入「API 输入内容」列；
- **必须保留单元格截断**：Excel 单元格上限 32,767 字符，实测 **107/240 条 prompt 超过**
  32,767（最长 **164,737** 字符）。现有 `excel_report.py:25-48` 的
  `_MAX_CELL_CHARS = 30_000` 与 `_clip()`（保留头尾并标注原长）**必须原样保留**，
  不能因为"blob 里有全文"就整段写入；
- 整个 `exports/` 可任意删除，重新导出即可恢复。

### 5.6 `archive/` 与 `.state/`

- `archive/`：退役批次、旧 schema bundle（本设计不兼容 v2，见 §9）。
- `.state/`：`index.lock`、`benchmark.lock`、`tmp/`（浏览器 profile 等）。
  非证据，可随时清空；不参与索引与体积统计。

---

## 6. 去重设计

### 6.1 核心：prompt 内容寻址

三层收益叠加：

| 去重维度 | 现状 | 目标 |
|---|---|---|
| 同一 prompt 在 `records` + `raw/<model>` + `raw/samples` 内 | 3 份/run | 1 份（全局） |
| 同一 prompt 跨同批次多模型 | 每模型 1 份 | 1 份 |
| 同一 prompt 跨批次（同 testset 复用） | 每批 1 份 | 1 份 |
| Excel 明细页内嵌 prompt | 每批 1 份 | 0（按需生成） |

实测：8 个 run / 2 个批次共用 **240 条唯一 prompt**；今天约 32 份拷贝，
目标 **1 份 = 240 个文件 / 13.5 MB**。

### 6.2 逐字段分层（`records.jsonl` 每个字段的去留）

| 层 | 字段 | 处置 | 依据 |
|---|---|---|---|
| L1 身份键 | `run_id`、`model`、`scenario`、`dataset`、`context_length`、`sample_id`、`repeat`、`phase`、`status`、`error`、`timestamp` | **留** | 唯一键 + 聚合分组 + AGENTS.md §2 第13条 溯源 |
| L1 输入计量 | `input_context_tokens`、`input_context_chars`、`actual_prompt_tokens`、`tokenized_prompt_tokens`、`prompt_chars`、`truncated`、`padded`、`context_meta`（含 `isl_tolerance`/`window_check_basis`） | **留** | AGENTS.md §2 第13条 明文要求 + 截断公平性审计 + actual ISL 容差 |
| L1 延迟/吞吐 | `ttft_ms`、`ttft_visible_ms`、`tpot_ms`、`e2e_ms`、`gen_tps`、`prefill_tps`、`total_tps`、`decode_window_ms`、`gen_tps_invalid` | **留** | 主指标；`gen_tps` 需可信解码窗口（C≥16、窗口≥50ms、非 tool_calls、≤10,000），被抑制记原因 |
| L1 资源 | `peak_memory_gb`、`peak_rss_mb`、`peak_cpu_pct`、`mean_cpu_pct`、`cpu_seconds`、`cpu_samples`、`cpu_sample_window_ms` | **留** | AGENTS.md §2 第13条 峰值内存 |
| L1 冷缓存证据 | `cache.*` 全部 | **留** | 冷缓存不变量唯一证据 |
| L1 输出 | `response.text`、`reasoning_text`、`tool_calls`、`finish_reason`、`usage` | **留** | 仅占 1.4%；Excel 明细 + 质量抽查 + 未来判分 |
| L1 收尾 | `score`（恒 null） | **留** | 契约字段，性能模式不判分 |
| L2 输入正文 | `request.messages[]` | **外置 → blob** | 占 95%，内容寻址 |
| L2 环境快照 | `model_config_snapshot` | **上移 → `run.json.env.model_env`**（记录留 `model_env_ref`） | 每 run 恒定，逐条重复 |
| L2 反向指针 | `sample_meta.raw_sample_path` | **删** | 目标目录 + 零读取方 |

### 6.3 环境快照：6 处 → 1 处

现状 `manifest.json` 内嵌三份 + 独立三文件 + `analysis/` 两份投影 = 6 处承载同一事实。
目标：`run.json` 一份，`MANIFEST.json` 只记该文件的 sha256。

同批一并处理的字段级自重复：

| 字段 | 现象 | 处置 |
|---|---|---|
| `hardware.json.upload_fields` | 与 `hardware.json` 顶层 6 个字段完全相同（同文件内自重复；生产者 `hardware.py:153-171`——158-163 写顶层、164-171 写 `upload_fields`） | **保留 `upload_fields`**（读者用它），删掉与之重复的顶层扁平字段；结构化数据留在 `cpu`/`memory`/`os` 块。已核：这 6 个顶层字段无生产消费者，删除安全 | `excel_report.py:185,189-198`；`app.js:1000` |
| `omlx.json.upload_projection` | 全仓**零读取方**（只有写入点 `environment.py:269`） | **删字段** + 同步改断言 | `test/test_integration.py:606-607`、`test/test_perf.py:274-275` |

### 6.4 派生物归位

| 派生物 | 现状 | 目标 |
|---|---|---|
| `summary.csv`、`cells.csv` | 每次 run 落盘 | 不落盘，`aggregates.json` 已有；需要时一条命令导出 |
| `summary.json.samples[]` | 内嵌 | `samples.csv` + 分页 API |
| `analysis/samples_long.csv` | 落盘 535 KB | 删除（pandas `melt` 一行） |
| `analysis/run.json`、`analysis/environment.json` | 落盘 | 合并进 `run.json` |
| `analysis/README.md` | 落盘 | 删除（字段说明进本文档） |
| `summary.xlsx` | 每批自动落盘 | 按需生成到 `exports/` |

### 6.5 重复处置总表

| 编号 | 内容 | 现状体积 | 处置 | 目标 |
|---|---|---|---|---|
| R1 | `raw/<model>/**` ≡ `records.jsonl` | 14.8 MB/run | **删** | 0 |
| R2 | `raw/samples/**`（8 run 全同 + 文件内自重复） | 27.5 MB/run | **删** | 0 |
| R3 | prompt 全文 32 份拷贝 | ≈440 MB 全库 | **内容寻址** | 13.5 MB 全库 1 份 |
| R4 | `manifest` 内嵌 vs 独立三文件 | 30.6 KB/run | **合并** | 1 份（`run.json`） |
| R5 | `summary.samples` vs `samples.csv` | 0.13 MB/run | **合并** | 1 份（`samples.csv`） |
| R6 | `analysis/samples_long.csv` | 0.52 MB/run | **删** | 0 |
| R7 | `summary.json` 聚合 vs 2 CSV | 12 KB/run | **合并** | 1 份（`aggregates.json`） |
| R8 | 批次元数据 3 份 | 3×4 KB/批 | **合并** | 1 份（`batch.json`） |
| R9 | Excel 内嵌 prompt | 3.6–9.0 MB/批 | **按需生成** | 0 常驻 |
| R10 | `hardware.upload_fields`（同文件内自重复） | ~0.3 KB/run | **合并** | 1 份 |
| R11 | `omlx.upload_projection`（零读取方） | ~1.2 KB/run | **删** | 0 |

---

## 7. 体积测算（基于实测，8 个 run / 2 个批次）

### 7.1 单 run

| 层 | 现状 | 目标 | 说明 |
|---|---|---|---|
| L0 元数据 | 0.089 MB（4 文件 + `analysis/` 投影） | 0.024 MB（`run.json`） | 去内嵌复制 |
| L1 权威逐样本 | 14.61 MB（`records.jsonl`，95% 是 request） | 0.74 MB（prompt 引用） | 共享 blob |
| L1 输入正文 | 包含在 L1/raw 中 | 0（全局池 13.5 MB，一次性） | 内容寻址 |
| L2 可复算聚合 | 0.31 MB（`summary.json` + 3 CSV） | 0.14 MB（`aggregates.json` + `samples.csv`） | 去 `samples[]` 与两张 CSV |
| L3 展示派生 | 0.03 MB（`report.md`） | 0.03 MB | 不变 |
| L4 冗余副本 | **42.27 MB**（`raw/` 两棵子树） | **0** | 删除 |
| 运维 | 0.03 MB（`logs/run.log`） | 0.03 MB | 不变 |
| 校验 | — | 0.01 MB（`MANIFEST.json`） | 新增 |
| **合计** | **57.8 MB** | **≈0.98 MB** | **−98.3%** |

### 7.2 全 `output/`

| 项 | 现状 | 目标 | 备注 |
|---|---|---|---|
| 8 个 run | 463.0 MB | ≈7.8 MB | 8 × 0.98 MB |
| prompt 池 | 含在上面（≈440 MB 重复） | 13.5 MB | 240 条唯一，全库一份 |
| 批次元数据 | 13.1 KB（2 份 + 3 份副本） | 8 KB（2 份 `batch.json`） | |
| Excel | 17.2 MB | 0（按需） | 需要时导出 |
| `index.json` | 0.025 MB | ≈0.05 MB | 增加 blob 与 sha256 统计 |
| 垃圾（zip/chrome/DS_Store/空目录） | 104.6 MB | 0（移出） | |
| 数据集构建中间产物（`accuracy/`、`performance/`） | 0.27 MB | 0（移出到数据集侧） | |
| **合计** | **591.1 MB / 4,311 文件** | **≈21.6 MB / ≈299 文件** | **约 −96%** |

### 7.3 设计选择说明：为什么用内容寻址而不是压缩

- 单纯 gzip 整份 `records.jsonl`：14.61 MB → **4.26 MB（29%）**，仍有 3 个模型各存一份，
  且失去可读性/可 grep 性。
- 内容寻址 + 瘦记录：单 run 0.74 MB，`records.jsonl` 保持明文可直接 grep，
  prompt 正文也保持明文可直接查看。
- 两者可叠加（对 `blobs/` 再 gzip 至约 3.5 MB），**本设计不默认压缩**——
  因为内容寻址已消除重复，剩余体积（13.5 MB）不值得用可读性交换。若未来 testset 规模上量，
  在 blob 层加压缩是局部的、无侵入的改动。

---

## 8. 索引与生命周期

### 8.1 索引重建

`index.json` 完全由 `runs/**/batch.json` + `runs/**/MANIFEST.json` 派生：

1. 扫描 `runs/*/*/batch.json` 得到批次清单；
2. 扫描批次下 `*/` 方法与 `*/*/` 模型目录得到 run 清单；
3. 读各 run 的 `MANIFEST.json` 拿 `files`/`totals`（避免 `rglob` 全量 stat）；
4. 合并为 `runs[]` 扁平视图 + `batches[]` 分组视图；
5. 原子写（沿用现有锁语义，锁移到 `.state/index.lock`）。

### 8.2 回收（GC）与保留

- **引用根 = `runs/` ∪ `archive/`**。**必须包含 `archive/`**：按 §4.3 把批次移入
  `archive/` 后，若 GC 只扫 `runs/`，会把它引用的全部 prompt blob 删掉——
  records 还在、正文永久丢失。
- 流程：① `referenced = ∪(所有根下 records.jsonl 的 request.prompt_sha256)`；
  ② `gc` 前先跑 `verify`，若 `blob_refs.missing > 0` 则**拒绝执行**并报缺失明细；
  ③ 删除 `blobs/prompts/**` 中未被引用者；④ 输出"删除 N 个 / 释放 X MB"。
- **保留策略标记**（现状缺失）：`batch.json` 增加 `retention: {"pinned": false, "keep_until": null}`；
  `pinned=true` 的批次及其 blob 引用不被 GC 回收。
- **体积配额**：`index.json.totals` 使配额检查无需全盘扫描。

### 8.3 完整性

- `MANIFEST.json` 覆盖 run 内全部文件（含 `records.jsonl` 的 sha256）。
- `verify` 对账三项（对应 AGENTS.md §2 第13条 的机器化校验）：
  1. `batch.json.models_selected` ⊆ `model_runs` ⊆ 磁盘目录；
  2. 每个 run：`n_measured` == `records.jsonl` 中 `phase=measured && status=ok` 行数，
     且 `n_selected == n_measured + len(skipped)`；
  3. 每个 run：所有 `prompt_sha256` 在 `blobs/` 中可解析（`missing == 0`），
     且 `MANIFEST.json` 中部件的 sha256 与实际一致。
- **打包**（可移植）：把指定批次目录 + 其引用的 blob 打成 zip（含 `MANIFEST`），
  用于外部归档；解开后仍是合法 batch。
- **旧数据**：本设计不兼容 schema 2.x（见 §9），旧 bundle 集中移入 `archive/`。

---

## 9. 切换策略：不允许"半新半旧"

> **第二轮增补**：第一次切换完成后，又按用户要求把 `runs/` 由 4 层压平为 1 层、批次元数据
> 移到 `batches/`（2026-11-09）。下面的 S1–S4 是第一次切换的历史施工记录。

**决策：新布局与 schema 2.x 不兼容，一次切换。** 理由：现状 8 个 bundle 全部为 2.1 且都在
本机；为"保持旧 run 可读"引入的双读分支会把 `legacy` / `2.0 无档位` / `schema 1.x 推导`
三套兼容逻辑（`catalog.py:130-151`、`compare.py:44-51`、`excel_report.py:266-268,311-317`、
`app.js:902`）永久固化在代码里。

**为什么不"双写"过渡**：写入侧双写会让每个 run 同时存在两种布局，索引要同时认两套，
成本高于收益（已确认不保留历史兼容）。因此采用**"结构 + 读取侧同一步切换"**，分 4 刀
（§10 里的 S5 是随 S4 一起落的契约同步清单，不构成第 5 刀）：

| 步骤 | 内容 | 结束状态 |
|---|---|---|
| **S1** | 清理与归档（纯文件操作，0 代码） | 旧 bundle 在 `archive/legacy-v2/`；`output/` 根干净 |
| **S2** | 结构重排：写入新路径 + `run.json` + `MANIFEST` + 索引递归 + **读取侧全改** + 测试改 | 可运行、功能完整，schema 3.0，但仍有 raw/ 冗余 |
| **S3** | 去重：blob + 瘦记录 + 删 raw/ + 合并 env + `samples.csv` 补列 + 分页 API + Excel 从 blob | 可运行，体积达标 |
| **S4** | 派生与新工具：停写 CSV、`exports/` 按需、`rebuild`/`gc`/`verify`/`pack` | 全量验收 |

**硬规则**

1. **S2 内必须同时改写入与读取**。只做写入侧会导致：`server.py:297`（无兜底读
   `summary.json`）500、`gate.py:508-509`（`summary.samples` 是否为 list）FAIL、
   `app.js:943` 逐样本页空、`catalog.py:123-125` run 从索引消失、`excel_report.py:258`
   平均页空。
2. **S3 内删除 `summary.samples[]` 必须与分页 API 同一切换**（牵连 `gate.py:508-509`、
   `app.js:943`、`test/test_core.py:403-404`）。
3. 每一刀结束都是**可运行 + 测试全绿**的状态，可以停在任一刀上（S2 结束时功能完整，
   只是体积未优化）。
4. 迁移动作（S1，一次性）：

```text
1. output/ 下 8 个 web-*_performance_* 目录  →  output/archive/legacy-v2/
2. output/batches/*.json、output/result_analysis/**  →  output/archive/legacy-v2/
3. output/M4PRO_testdata2.zip  →  移出 output/（纯备份；按 AGENTS.md §8 需人工确认）
4. output/.tmp-chrome-profile/ →  删除（无代码生产者，运行时不需要）
5. output/accuracy/、output/performance/selection_report.md  →  reports/dataset-build/{accuracy,performance}/（并改 tool 侧输出根；现已执行）
6. output/build/、output/.DS_Store（×2）→ 删除
7. output/index.json → 删除后按新实现重建
```

> AGENTS.md §8 规定"删除/重写 `output/` 需人类确认"，因此 **S1 的每条删除动作都要先取得
> 确认**（待确认项见 §12 Q0）。

---

## 10. 实施计划

> **第二轮落地记录（2026-11-09）**：`runs/` 压平为 `runs/<run_id>/` 一层；批次元数据移至
> `batches/<batch_id>.json`；`layout.py` 新 API 为 `runs_dir()`/`run_dir(root, run_id)`/
> `run_dir_for()`/`batches_dir()`/`batch_path()`，旧的 `month_bucket()`/`batch_dir()`/
> 5 参 `run_dir()` 已删除。下面的 S1–S4 是第一次切换的历史施工清单。

> 本次交付为设计；以下是实现时的施工清单，按 §9 的 4 刀组织。
> 代码触点来自实测链路追溯（含独立评审修正的行号），行号以当前 HEAD 为准。

### S1 · 清理与归档（0 代码）

- 按 §9 迁移动作执行（需人类确认）。
- 把 `tool/dataset/performance/build.py:238`、`tool/dataset/accuracy/build.py:69`、
  `tool/system/pilot.py:31` 的输出根从 `output/` 改到数据集侧或 `reports/`。

### S2 · 结构重排（写入 + 索引 + 读取 + 测试，一个提交）

| 改动 | 触点 |
|---|---|
| 目录创建改为 `runs/<YYYY-MM>/<batch_id>/<method>/<slug>/` | `writer.py:74-86`；`batch.py:79-88,108` |
| `batch_id` 去 `web-` 前缀、`source` 入 `batch.json` | `jobs.py:320`、`batch.py:67-70`、`cli.py:173-177` |
| 新 `batch.json`（唯一权威、相对路径、testset 指纹、retention） | `batch.py:157-164,204-220,270-271` |
| `run.json` 合并 6 文件 | `writer.py:95-119`（initialize）、`models.py:334-344`（manifest 结构） |
| 新增 `MANIFEST.json` | 新增 storage 模块 |
| 索引递归扫描 + 相对路径 + sha256 + 列表页字段 | `catalog.py:107-116,119-160,188,229-232,281-330,348-354,361-393` |
| 门禁目录定位去硬编码 | `gate.py:426-455`（435/444/446/451） |
| 锁文件移入 `.state/` | `catalog.py:306-308`；`locking.py:30-34` |
| Web 读取改 `run.json`/`aggregates.json`/`samples.csv`；导航改用 `run_path` | `server.py:48,66-83,286-292,295-306,581-616,621,634,666`；`app.js:98,641,722-790,828-1060` |
| **`summary.json` → `aggregates.json`（先保持内容等价搬运）** | `writer.py:167-189`；读取侧同步改名 |
| `analyze` 改读 `aggregates.json` | `compare.py:24-31,44-58`；`analysis/cli.py:25` |
| Excel 改读 `run.json`/`aggregates.json`，**并保留 `output/batches/` 读取回退** | `excel_report.py:96-108,181-184,258-301,437-459,497,516-536` |
| 测试按新布局重写 | `test_integration.py:505-531,600-622,633,1296-1312`；`test_batch.py:60-77,104-110,156-198`；`test_core.py:383-408`；`test_results.py:93-111,141-146` |

> **`output/batches/<id>.json` 不是纯冗余**：`export-analysis` 的
> `_batch_for_regeneration`（`excel_report.py:516-536`）**优先**读它，`gate.py:435-447`
> 也用它定位 run。S2 里改为"先读 `batch.json`，回退读旧 `batches/`"（一个函数两行 if），
> S4 再删回退分支与目录。

### S3 · 去重（收益主体，一个提交）

| 改动 | 触点 |
|---|---|
| **停写** `raw/<model>/**`（R1） | `writer.py:130-140` |
| **停写** `raw/samples/**`（R2） | `benchmark.py:1296,1784-1793`；删 `raw_sample_path` `benchmark.py:1754,1760-1770` |
| 新增 blob 写入器（规范化哈希 + 分片 + 幂等 + 原子写） | 新增 storage 模块；在 `benchmark.py` 构造请求后写入 |
| `records.jsonl` 瘦身（prompt 引用 + `model_env_ref`） | `writer.py:122-140`；`models.py:207`（`SampleRecord` 字段） |
| 删除 `analysis/` 全部 4 件（R6） | `writer.py:221-364` |
| `aggregates.json` **去 `samples[]`/`environment`**（S2 已改名，这里只裁剪字段） | `writer.py:149-218` |
| `samples.csv` 补 `cache_*`/`prompt_sha256`/`subtask` 列 | `writer.py:484-487`；`models.py` `SampleResult` |
| 删除 `hardware.json`/`omlx.json`/`models.json` 独立文件 + 去 `upload_projection` + 去 `upload_fields` 的顶层重复 | `writer.py:105-108`；`environment.py:269`；`hardware.py:153-171`；同步改 `excel_report.py:182-183`、`gate.py:504`、`test_integration.py:600-607`、`test_perf.py:270-277` |
| 逐样本分页 API（`/api/runs/<id>/samples?offset=&limit=`） | 新增路由；`server.py:295-306`；`app.js:943,960,970-996` |
| Excel 从 blob 取 prompt（**保留 `_MAX_CELL_CHARS` 截断**） | `excel_report.py:25-48,96-108` |

### S4 · 派生与新工具（一个提交）

| 改动 | 触点 |
|---|---|
| 停写 `summary.csv`/`cells.csv` | `writer.py:190-191,375-482` |
| Excel 改按需生成到 `exports/<batch_id>.xlsx` | `batch.py:258`；`excel_report.py:437-459` |
| 删除 `result_analysis/**/batch.json` + `README.md` | `excel_report.py:454-459` |
| 删除 `output/batches/` 读取回退 + 目录本身 | `excel_report.py:516-536`；`gate.py:435-447` |
| 新命令：`rebuild`（records → samples.csv/aggregates.json/report.md）、`gc`、`verify`、`pack`、`export` | `cli.py`（新子命令）+ 新工具模块 |

### S5 · 契约同步

- `models.py:20` `SCHEMA_VERSION = "3.0"`；`catalog.py:31` → `"2.0"`。
- 本页转正；删除/重写 `31-output-contract.md` §1/§2/§6；`30-architecture.md:1673-1694` 文件清单；
  `docs/README.md` 版本口径；`20-methodology.md:282-283`。
- 新增 storage 模块后必须跑 `make docs-write`（`docs/design/33-project-structure.md` 自动生成，
  AGENTS.md §12）。
- 新 CLI 子命令必须出现在 `--help`（`docs_contract` 会校验文档里写的 flag）。

---

## 11. 验收标准

1. `output/` 根条目 ≤ 6 个：`index.json`、`blobs/`、`runs/`、`exports/`、`archive/`、`.state/`。
2. 单个 run 目录条目 ≤ 7 个（含 `logs/`）。
3. 同一批 240 样本 × N 模型：`blobs/prompts/` 恰好 **240** 个文件，且 N 增大时不增长。
4. 单 run 产物体积 ≤ 1.2 MB（240 样本、`max_tokens=50` 基准）。
5. 全 `output/` 在 8 run / 2 批规模下 ≤ 25 MB。
6. `verify` 三项对账全过；`blob_refs.missing == 0`。
7. 删除 `samples.csv`/`aggregates.json`/`report.md` 后，`rebuild` 能恢复**语义等价**内容
   （同字段同取值；不要求字节级相同，因为浮点格式化与时间戳会变）。
8. 删除 `index.json` 后重建，内容与原文件一致（除 `generated_at`）。
9. 整个 `output/` 移到他处后，索引与 `batch.json` 中**无绝对路径**，全部可解析。
10. `gc` 在存在 `archive/` 批次时**不删除**被其引用的 blob（回归用例必须覆盖）。
11. `pack` 出的 zip 解开后 `verify` 通过（blob 齐全）。
12. 每个 `run.json.env.model_env` 条目的键 == 其内容规范化哈希。
13. Web 五个页签 + 列表页、`./modelbench runs`、`./modelbench analyze`、
    `./modelbench export-analysis`、`harness.gate` 全部在新结构上通过。
14. `pytest test/` 全绿，`python -m harness.gate` PASS。

---

## 12. 待人类确认

| # | 议题 | 建议 | 影响 |
|---|---|---|---|
| **Q0** | **AGENTS.md §8 要求"删除/重写 `output/`"必须人类确认**——S1 的删除/搬移清单是否获批 | **批准**（含 `M4PRO_testdata2.zip` 移出、chrome profile 删除） | 未获批则 S1 只能搬移不能删 |
| Q1 | `runs/` 是否加时间层 `YYYY-MM` | **加**（批次多时必需） | 目录深度 +1；去掉则退化为 3 层 |
| Q2 | `blobs/` 全局 vs 每批次 | **全局**（跨批次复用，实测已复用） | 全局：单批不可单独搬走（需 `pack`）；每批：多占 13.5 MB/批 |
| Q3 | `samples.csv`/`aggregates.json` 是否常驻 | **常驻**（人读便利；合计 ≈0.14 MB/run，即 §7.1 的 L2 层） | 全去掉则每次查看都要 `rebuild` |
| Q4 | Excel 是否默认生成 | **按需**（省 17.2 MB 且避免第 3 份 prompt 拷贝） | 需用户点一次导出 |
| Q5 | 旧 bundle 是否保留 | **移入 `archive/` 不转换** | 若需可读，须写转换器（工作量 +） |
| Q6 | prompt 池是否压缩 | **不压缩**（去重后仅 13.5 MB，保可读） | testset 上量后再评估 |
| Q7 | 精度测试是否并入 `runs/…/accuracy/` | **并入**（与性能同构） | 需补齐精度侧写盘代码（当前无） |
| Q8 | 上述 `tpot_ms=0.0` → `gen_tps` 爆表的指标缺陷是否本次一并修 | **另开任务**（属指标语义，不属目录设计） | 修则会改变历史可比性，需单独评审 |

---

## 附录 A · 现状产物处置对照表（逐条目）

覆盖当前 `output/` 全部 4311 个文件所属的 19 个顶层条目与 run 内 13 个条目。
**每一条都有明确去向，没有"暂留"**。

### A.1 `output/` 根

| 现状条目 | 体积 | 性质 | 处置 | 目标位置 |
|---|---|---|---|---|
| `web-*_performance_<model>/` ×8 | 463.0 MB | 平铺 run bundle | **重组** | `runs/2026-11/<batch>/performance/<model>/` |
| `index.json` | 24.9 KB | 派生索引 | **重建**（去绝对路径、加 sha256） | `output/index.json` |
| `batches/<id>.json` ×2 | 13.1 KB | 批次元数据（第 2 份）**但有真实读者** | **合并到 `batch.json`**；`export-analysis` 改为读它并保留旧路径回退（S4 去回退） | `<batch>/batch.json` |
| `result_analysis/*/summary.xlsx` ×3 | 17.2 MB | 派生交付物 | **按需生成** | `exports/<batch_id>.xlsx` |
| `result_analysis/*/batch.json` ×3 | 10.7 KB | 批次元数据（第 3 份） | **删** | — |
| `result_analysis/*/README.md` ×3 | 1.6 KB | 说明文本，零读取 | **删**（说明进本文档） | — |
| `performance/selection_report.md` | 30.9 KB | 数据集构建中间产物 | **移出**（`tool/dataset/performance/build.py`） | `reports/dataset-build/performance/` |
| `accuracy/`（pilot + archive） | 243.9 KB | 数据集构建/pilot 中间产物 | **移出**（`tool/dataset/accuracy/build.py`、`tool/system/pilot.py`） | `reports/dataset-build/accuracy/` |
| `M4PRO_testdata2.zip` | 100.7 MB | 人工打的 `output/` 快照备份：2,487 个文件与现存文件逐字节相同、**0 个孤儿文件**、仅 1 个旧版 `index.json` | **移出 `output/`**（纯备份，无独有数据；按 AGENTS.md §8 删除需人工确认） | 待定 |
| `.tmp-chrome-profile/` | 3.9 MB | 浏览器 profile（311 文件，含 3 个 dangling symlink）；**全仓 0 处代码引用**，无本仓代码生产者 | **删除** | — |
| `.benchmark.lock`、`.index.lock` | 51 B | 运行时锁 | **移动** | `.state/` |
| `.DS_Store` ×2（`output/` 12.3 KB + `batches/` 6.1 KB） | 18.0 KB | OS 元数据 | **删 + 加 .gitignore** | — |
| `build/` | 空 | 残留空目录 | **删** | — |

### A.2 run bundle 内部（现状 495 文件 → 目标 7 条目）

| 现状条目 | 体积/run | 性质 | 处置 | 目标 |
|---|---|---|---|---|
| `records.jsonl` | 13.9 MB | **权威**（95% 是 prompt） | **瘦身**：prompt → sha256 引用，`model_config_snapshot` → `run.json` | `records.jsonl`（≈0.76 MB） |
| `raw/<model>/<scenario>/<bucket>/*.json` | 14.8 MB | 与 records **逐条相等**的副本 | **删除** | — |
| `raw/samples/<dataset>/<scenario>/*.json` | 27.5 MB | 原始样本；8 run 全同；文件内 question/messages 互抄 | **删除**（输入改为 `blobs/` + testset 指纹自证） | `blobs/prompts/` |
| `manifest.json` | 41.6 KB | 元数据（内嵌 3 份环境副本） | **拆分合并** | `run.json` + `MANIFEST.json` |
| `hardware.json` | 6.7 KB | 环境证据（与 manifest 内嵌相等） | **合并** | `run.json.env.hardware` |
| `omlx.json` | 6.0–10.9 KB | 环境证据（与 manifest 内嵌相等） | **合并** | `run.json.env.omlx` |
| `models.json` | 13.0 KB | 模型清单（与 manifest 内嵌相等） | **合并** | `run.json.env.model_env` |
| `summary.json` | 0.26 MB | 聚合 + `samples[]` + 环境引用 | **拆分**：聚合→`aggregates.json`，样本→`samples.csv`，环境→`run.json` | `aggregates.json` + `samples.csv` |
| `summary.csv` | 1.9 KB | 聚合的 CSV 序列化（零生产消费） | **删**（按需导出） | — |
| `cells.csv` | 9.8 KB | 同上的单元格粒度（零生产消费） | **删**（按需导出） | — |
| `samples.csv` | 44 KB | 逐样本扁平表（与 summary.samples 同构） | **保留并扩列**（补 `cache.*`、`prompt_sha256`、`subtask`） | `samples.csv`（唯一逐样本表） |
| `report.md` | 38 KB | 人读摘要 | **保留** | `report.md` |
| `analysis/run.json` | 1.5 KB | 运行元数据投影，零解析 | **合并** | `run.json` |
| `analysis/environment.json` | 4.2 KB | 环境投影，零解析 | **合并** | `run.json.env` |
| `analysis/samples_long.csv` | 535 KB | `samples.csv` 的 melt 长表，零解析 | **删**（pandas 一行） | — |
| `analysis/README.md` | 1.1 KB | 分析说明，零解析 | **删**（说明进本文档） | — |
| `logs/run.log` | 34 KB | 运维日志 | **保留** | `logs/run.log` |
| —（新增） | — | 完整性 | **新增** `MANIFEST.json`（path/bytes/sha256） | `MANIFEST.json` |
| —（新增） | — | 运行元数据 + 环境快照 | **新增** `run.json`（合并上述 6 处） | `run.json` |

### A.3 数量对照

| 维度 | 现状 | 目标 | 变化 |
|---|---|---|---|
| `output/` 顶层条目 | 19 | 6 | −13 |
| 单 run 条目 | **13**（含 3 个目录） | 7（含 1 个目录） | −6 |
| 单 run 文件数 | 495 | 7 | −488 |
| 单 run 体积 | 57.8 MB | ≈0.98 MB | −98.3% |
| 全 `output/` 文件数 | 4,311 | ≈299 | −93% |
| 全 `output/` 体积 | 591.1 MB | ≈21.6 MB | −96% |
| prompt 全库拷贝份数 | ≈32 | 1 | −31 |

### A.4 新增产物清单（现状缺失）

| # | 产物 | 解决的问题 |
|---|---|---|
| N1 | `MANIFEST.json`（每 run） | 完整性可校验；索引重建不必全盘 stat |
| N2 | `batch.json.inputs.testset.{path,sha256,rows,context_windows}` | bundle 能自证"跑的是哪 240 行"；这是能删 prompt 副本的前提。注意 `context_windows` 现状 8/8 为空导致 Web 显示"全部" |
| N3 | `run.json.outcome.skipped[]` 结构化 | 现在是自由文本，消费者只能 regex |
| N4 | `run.json.outcome.n_selected/n_measured` 对账口径 | AGENTS.md §2 第13条 硬不变量机器可校验（`verify` 命令） |
| N5 | `samples.csv` 的 `cache_*` / `prompt_sha256` 列 | 冷缓存审计不必回读 14 MB 的 records |
| N6 | `batch.json.retention`（pinned/keep_until） | `output/` 现在无配额/无 TTL/无 pin |
| N7 | `output/blobs/` | prompt 全库只存一份 |
| N8 | `output/exports/` | 人类交付物与证据分离 |
| N9 | `output/.state/` | 运行时锁/临时物与证据分离 |
| N10 | `run.json.env.model_env` 哈希索引 + `model_env_ref` | 环境快照可证明"这条记录用的是哪份模型配置"，且能容忍 run 中途重载 |
