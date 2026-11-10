# 测试方案设计（性能基准测试）

> **状态**：current ｜ **版本**：v2.3（2026-11-10；固定一套样本 + actual 记录 + ±10% 入带聚合，
> 不再因 tokenizer/actual 偏差跳过；32K target 32,768 无特殊处理）｜
> **适用**：`modelbench` 0.3.x 性能基准（bundle schema `3.0` / index schema `2.0`）｜
> **读者**：维护者、验证者、AI Agent ｜ **范围**："测什么、怎么测、如何判定"；实现见
> [30-architecture.md](../design/30-architecture.md)，操作见 [41-user-guide.md](../guides/41-user-guide.md)，指标公式见
> [20-methodology.md](20-methodology.md)，产物契约见 [31-output-contract.md](../design/31-output-contract.md)。

## 1. 目标与原则

在本地 oMLX 上，对选定的**测试集 × 模型**做端到端黑盒性能测试，逐样本采集：

- **TTFT**：首 token 延迟（客户端 SSE 时间戳）；
- **TG TPS**：解码窗口内的输出 token 速度（`gen_tps`）；同时记录 `tpot_ms`；
- **峰值内存**：oMLX 服务进程树在请求窗口内的 `phys_footprint` 峰值（GB/GiB，含 Metal 分配；
  与 oMLX benchmark 的 Metal 峰值 `PeakMem` 口径不同，见 `METHODOLOGY`）；
- **输入上下文大小**：每个样本的**目标模型实际 token 数**（`/v1/tokenize`）与字符数；
- 辅助指标：`e2e_ms`、`prefill_tps`、`total_tps`、峰值/均值 CPU。

对外展示**按上下文档位分别平均的整体结果**（1K / 8K / 32K 各一行，不跨档位合并），
同时**逐样本保留**结果（`records.jsonl` 权威 + `samples.csv` 投影 + `aggregates.json`）。

原则：

1. **不填充对齐、但档位是 cell 维度**：预构造性能样本自带 1K / 8K / 32K 三档
   （参考 tokenizer 计数的声明值，见 `dataset/test/performance/testset.jsonl`），
   每个 `(scenario, dataset, context_length)` 形成独立 cell；不做按目标长度的填充。
   运行前用**目标模型 tokenizer** 对最终 `messages`+`tools`（含 decode guard）计数，
   作为实际输入大小与窗口判定依据（见 §3 与 §4）。
2. **样本数可配置**：`perf.samples_per_dataset` 按**场景**（scenario）控制取多少个样本
   （字段名为历史遗留，实际每个场景各取 N；默认 `performance` 有 8 个场景，所以
   `--samples 5` = 40 个 measured 请求）；**留空（null）= 每个场景全部可用样本**。
3. **冷缓存**：每个测量请求前清理 hot+SSD，校验 `cached_tokens==0`；污染记录剔除聚合。
4. **默认模型配置**：模型以 oMLX 下载/默认配置加载；**只允许**为 MoE 模型显式开启
   `moe_expert_offload_enabled`，不启用 TurboQuant / DFlash / MTP 等任何加速或压缩配置。
5. **可复现**：固定抽样、记录配置快照/硬件/oMLX 设置/模型设置/git commit/原始请求响应。
6. **先加载后测试**：每个模型加载完成后才测量；`restore_state` 默认在结束时卸载本次加载的模型。
7. **不静默降级**：跳过（`run.json.outcome.skipped`）、警告（`outcome.warnings`）、
   测量有效性标记（`gen_tps_invalid` / `isl_tolerance` / `n_gen_tps_invalid` /
   `n_isl_in_band` / `n_isl_out_of_band`）必须落盘可见，不静默剔除、不静默改口径。
   **actual ISL 不触发跳过**：所有能测样本都 measured，只按 ±10% 带筛选聚合
   （入带进均值、出带保留不进均值；见 §3/§4）。

## 2. 测试对象与模型加载

- 模型清单：`src/modelbench/resources/config/models.yaml`（`id` 与 `/v1/models` 一致）。
- 默认模型：`minicpm5-1b`；推荐对比模型：`qwen3.5-9b`。
- **加载策略**：默认不传任何 oMLX 模型设置（即下载后的默认配置）。
  仅 MoE 模型（如 `qwen3.6-35B-A3B`，35B-A3B，约 75 GB，需要流式加载 experts 才能在
  24GB 机器上运行）需要显式设置：

```yaml
  - id: qwen3.6-35B-A3B
    load:
      optional: true                 # 失败降级为 warning，按默认配置继续
      settings:
        moe_expert_offload_enabled: true
```

- 加载设置白名单：只有 `moe_expert_offload_enabled` 允许写入；其它字段（`turboquant_kv_enabled`、
  `dflash_enabled`、`mtp_enabled` 等）会被配置校验直接拒绝，防止"默认配置"被悄悄改掉。
- **历史遗留复位**：oMLX 的模型设置是持久化的，之前的 run 可能留下 `turboquant_kv_enabled` /
  `dflash_enabled` 等非默认值。runner 在加载每个模型前读取其实际生效 settings，若发现已知加速
  开关（DFlash / SpecPrefill / TurboQuant KV / Lightning MTP / VLM MTP / Qwen ANE Prefill /
  Qwen OQ-a8）为 true，会先写回 false 再重新加载，保证本轮测量处于默认配置；仅
  `moe_expert_offload_enabled` 允许为 true。复位动作写入 `logs/run.log`，最终生效值写入
  `run.json.env.omlx`；白名单之外的实验性设置不复位、也不保证出现在 `upload_projection` 中，
  需要时按 oMLX 版本扩展 `environment.py` 的 allowlist。
- 应用流程：卸载已加载实例 → `PUT /admin/api/models/<id>/settings` → 重新加载 →
  `wait_until_loaded`；失败按 `optional` 显式记录（`run.json.outcome.warnings`）。
- 每次 run 采集模型的实际生效设置（oMLX admin 返回的 settings 投影）与加速特性开关，
  写入 `run.json.env.omlx`，结果可追溯"是否使用了非默认配置"。

## 3. 测试计划（唯一模式：性能基准）

| 项 | 规则 |
|---|---|
| 测试集 | `dataset/test/<name>/testset.yaml`（默认 `performance`）；场景 = 数据集 + 参数 |
| 模型 | CLI `--models` / Web 勾选；串行测试 |
| 样本数 | `perf.samples_per_dataset`；`null` = 每个场景全部可用样本 |
| 逐场景样本数 | `perf.samples_by_scenario.<scenario>`（**API 参数 `scenario_samples`**；Web 界面已移除逐场景输入入口），优先于全局样本数 |
| 手选样本 | `perf.sample_ids.<scenario>`（**API 参数 `sample_ids`**；Web 界面已移除手选入口），优先级最高；清空回退到样本数/全部 |
| 上下文窗口筛选 | `perf.context_windows`（Web 多选 / CLI `--context-windows`，**legacy/诊断用途**）；空 = 全部档位；**1K 接受 0～1.5K**，其他 N 接受 **N±0.5K**。新数据集的档位请用场景选择或 `perf_jsonl params.buckets`（1k/8k/32k），两套阈值不要混用。预构造样本按**声明的参考 tokenizer tokens** 筛选，legacy 样本才按目标模型 tokenizer 计数；cell 仍按 `(scenario, dataset, context_length)` 聚合 |
| 实际输入大小（actual ISL，记录/聚合带） | 每个 selected 样本在请求前用目标模型 `POST /v1/tokenize` 对**最终 messages + tools（含 chat template 与 decode guard）**计数，写入 `request.tokenized_prompt_tokens` 与 `request.context_meta.isl_tolerance = {target, low, high, actual, ok}`（以及 `context_meta.window_check_basis = tokenized|declared`）。**所有档位同一 ±10% 带**（区间权威表见 [20-methodology.md](20-methodology.md) §5；target=1,024/8,192/32,768）。该值**不用于跳过**：所有能测样本都 measured，只用于 (a) 记录 actual/`samples.csv.isl_band` 与 (b) 窗口判定。聚合只对入带 measured 样本求平均；出带样本保留在 `records.jsonl`/`samples.csv`（`isl_band=false`）但**不进均值**，计数 `n_isl_in_band`/`n_isl_out_of_band`；in-band 样本少于 **8**（默认每 cell 10 条，阈值 `domain.metrics.MIN_IN_BAND_SAMPLES`）时报告须标记 `insufficient_samples`。tokenize 失败时回退 `声明 tokens + max_tokens + context_safety_margin(128) + decode_guard_reserve(64，默认)=242` 做窗口判定（`window_check_basis=declared`）并记 warning |
| 预热 | **模型级**默认 **1 次/模型**（`run.warmup_per_model=1`，加载后发一条短请求 `max_tokens=8`，仿 oMLX QUICK，丢弃且不记录/不聚合）；per-cell `run.warmup_requests` 默认 **0**，`--warmup N` 仅诊断用（每个 cell 预热 N 条，不计入 measured 预算） |
| 抽样 | 未手选时：**预构造样本先按档位（1k/8k/32k）做最大余数分配，再在每个档位内按 lang 做最大余数分配**，组内按 `sample_id` 取前 k（确定性、跨 run 可复现；与 `perf_jsonl.plan` 同一实现，小 `--samples` 不再整段塌到 1K）；legacy 数据集仍按 `sample_id` 排序取前 N |
| 上下文 | **不设目标长度、不填充、不截断**；预构造样本原样发送（仅追加固定 decode guard 指令）。**窗口判定用目标模型实测量**：`tokenized_prompt_tokens + max_tokens + context_safety_margin(128) > model_context_window` 时该样本**跳过并计入 `run.json.outcome.skipped`**（not-testable）。32K 桶内容 token 32,704–32,841（声明值），按声明+242 估算需窗口 32,946–33,083；对 32,768 窗口模型 32K 桶 **0/80** 可测。本机当前模型窗口 40,960–262,144 均可容纳 |
| 生成长度 | `perf.max_tokens`（默认 **50**，CLI `--max-tokens` / Web「生成长度」可配置）；同时默认开启 `perf.decode_guard=true`，在每条请求后追加固定长输出指令，避免以 `Answer:` 结尾的样本 1 token 结束。**默认 50 + decode guard 下绝大多数输出会 `finish_reason=length`，这是预期解码窗口而非丢弃**；达到 `max_tokens` 的样本照常参与平均，TG TPS 按实际生成 token 计算；`n_length_capped` 显式展示。指令原文记录在 `records.jsonl` 的 `request.context_meta`；`decode_guard=false` 仅用于诊断/历史对比 |
| 轮数 | `run.repeats`（默认 1）；>1 时同一 `sample_id` 产生多条记录，逐样本结果按 `sample_id` 保留 |
| 缓存 | 默认 `cold`（逐请求清理 + 禁写缓存）；`warm` 仅清理 cell 首请求并允许写缓存以测复用；`warm`/`none` 仅诊断，不得与 cold 混算 |
| 工具 | 性能 run **不执行工具循环、不评工具正确性**；agentic 预构造样本只发一次请求且 `tools` 原样保留 |

> **为什么固定一套样本、运行时记录 actual 并按 ±10% 带聚合**：构造 ±5% vs 运行 ±10% 的
> 完整推导（跨模型计数差实测、逐数字）见 [11-performance-testset.md](11-performance-testset.md)
> §4.5（单一事实源）。要点：固定样本可逐字节复现并跨设备/跨模型复用，但运行时输入由
> **目标模型 tokenizer** 计数（chat template + decode guard 额外增加 token）。因此**不因为
> actual 偏差跳过**（否则同一套
> 样本在不同模型上就不可比）：记录 actual、只把入带（target ±10%，所有档位对称）样本纳入均值、
> 出带样本保留但不参与平均，并报告 actual 分布与入带/出带计数。跨系统/跨设备比较准入见
> [13-testset-portability.md](13-testset-portability.md)。

示例：

```bash
# 默认测试集、全部样本、默认模型
./modelbench run --models minicpm5-1b

# 每个场景只取 5 个样本（8 场景共 40 个 measured 请求）
./modelbench run --models minicpm5-1b qwen3.5-9b --samples 5

# 指定测试集（默认 performance；自定义测试集放 dataset/test/<name>/）
./modelbench run --testset performance --models minicpm5-1b --samples 5
```

### 3.1 样本预算（`dataset/test/performance/testset.jsonl`）

| 项 | 数值 |
|---|---|
| 唯一性能样本 | **240 条**（8 个 scenario × 3 档 × 10） |
| 默认 8 场景 measured 请求 | **240**（1K 80 / 8K 80 / 32K 80；场景各 30）；**窗口不足的行进入 skipped，不强行凑满** |
| 默认 cell 数（1 模型） | **24**（8 场景 × 3 档；`(scenario, context_length)` 分组） |
| 投机解码 | 不纳入默认基准（需非默认加速开关，单列专项） |
| 精度样本 | `dataset/test/accuracy/testset.jsonl` **2,000 条**（8 × 250，隐藏离线资产：`manifest.status=offline-asset`、`runner=not wired`、无 `testset.yaml`）；保留在磁盘但**默认不参与**性能 run，也不出现在测试集列表 |

方案与逐场景/语言/来源明细见 [dataset/README.md](../../dataset/README.md)、
[docs/spec/11-performance-testset.md](11-performance-testset.md) 与
[docs/spec/12-accuracy-testset.md](12-accuracy-testset.md)；
投机解码不纳入默认基准：需要非默认加速开关，应作为独立专项实验（不启 MTP/DFlash/SpecPrefill，不算 α/speedup）。

> **场景命名**：性能 8 场景按负载形状命名；场景 8 是 `multilingual`（Global-MMLU 10 语种）；
> 精度侧也保留一个隐藏的 `multilingual` 数据资产（10 语种 × 25 配对题），两者用途不同，不要混淆。
> **精度集**没有 `testset.yaml`，不会出现在 Web/CLI 测试集列表；在接入独立 runner 前不得引用精度分数。
>
> **可比性**：默认模型级预热 1 次（`warmup_per_model=1`）、per-cell 预热 0、`max_tokens=50`、`repeats=1`，且逐请求冷缓存；
> 这与 vLLM / oMLX 官方 benchmark 的预热/生成长度/批处理口径不同，**不可与任何公开榜横比**。
> 提高 `--max-tokens`/`--warmup`（per-cell）/`--repeats` 可做诊断，但结果不得与默认口径混算。

## 4. 指标与统计口径

- **逐样本结果**：每个 `(model, dataset, sample_id, repeat)` 一条 `SampleRecord`，
  含 `metrics.ttft_ms / gen_tps / tpot_ms / e2e_ms / peak_cpu_pct / peak_memory_gb / peak_rss_mb`、
  `metrics.input_context_tokens / input_context_chars`、`metrics.decode_window_ms` 与
  `metrics.gen_tps_invalid`。
- **数据单元汇总（cells）**：`model × scenario × context_length × dataset` 对有效样本求
  mean/median/std/min/max/p95，并给出 **`ci95_low`/`ci95_high`**（run 内 mean 的正态近似
  95% CI：`mean ± 1.96·std/√n`，`n<2` 为 null；**不是 run-to-run 噪声**）。
  （每个档位独立 cell；默认 8 场景共 24 cells / 240 measured）。
- **整体汇总（overall）**：**按 `(model, context_length)` 分别求平均**（1K/8K/32K
  各一行，绝不跨档位合并——TTFT/预填充相差一个数量级）；峰值内存同时给出该档位
  逐样本峰值的平均值与全局最大值。`overall` 行含 `context_length` 字段。
- **计数口径**：`run.json.outcome.n_samples` = measured 记录数（含 repeats）；
  `aggregates.overall[].n_samples` = 唯一样本数；`n_valid` = 参与聚合的有效记录数；
  `n_selected = n_measured + len(outcome.skipped)`。
- **TG TPS 有效性**：`gen_tps/tpot_ms` 仅在**解码窗口可信**时输出，否则置 `null` 并写
  `metrics.gen_tps_invalid` 的完整枚举、判定优先级与 `C<16` 约定见
  [20-methodology.md](20-methodology.md) §2/§8（单一事实源）。
  报告/CSV 用 `mean (n/有效数)` 与 `gen_tps_n` 显式展示覆盖；`aggregates.*.n_gen_tps_invalid`
  给出被抑制样本数，不做插补、不回收异常值。
- **actual ISL 记录与聚合带（不跳过）**：每条 **measured** 记录带 `request.tokenized_prompt_tokens`
  与 `context_meta.isl_tolerance{target,low,high,actual,ok}`；`samples.csv` 在
  `actual_prompt_tokens` 之后新增 **`isl_band`** 列。**所有能测样本都 measured**，actual 不触发
  跳过；聚合只使用 actual 在 **target ±10%**（所有档位对称，区间见 [20-methodology.md](20-methodology.md) §5）的入带样本，出带样本保留但不进平均值。计数：`run.json.outcome` 与
  `aggregates.overall/cells` 的 `n_isl_in_band`/`n_isl_out_of_band`；报告/Excel/Web 显示
  "ISL入带/ISL出带"，出带注明"仍测量但不参与平均"。in-band 样本不足的 cell 必须标记
  `insufficient_samples`。容差定义与理由见 §3。
- **输出触顶不丢弃**：达到 `perf.max_tokens`（`finish_reason=length`）的样本照常参与
  TTFT/内存/TG TPS 聚合（TG TPS 用实际 `completion_tokens` 计算到停止前）；
  `n_length_capped` 在 `aggregates.json` / `report.md` / Excel / Web 中显示触顶数。
- **有效样本**：`phase=measured` 且 `status=ok`；warmup/污染/错误不参与聚合。
- **冷缓存**：`cleared_before_request=true` 且 `cached_tokens==0`，否则污染并重试/剔除。
- 不再计算 accuracy / Wilson CI；`score` 字段恒为空。

## 5. 环境与测量有效性采集（每次 run）

参考 oMLX 内置 benchmark 上传到社区榜时携带的本地参数，run 开始时采集；全部事实**只写入
`run.json` 的 `env`**（不再有独立环境文件），`MANIFEST.json` 记录其大小/sha256：

| 分类 | 字段（`run.json.env.*`） |
|---|---|
| 硬件/系统 | `hardware.chip_name/chip_variant/memory_gb/gpu_cores/os_version`、CPU 核心数、磁盘、可用内存/swap 快照、Python/依赖版本 |
| oMLX | `omlx.omlx_version`、`omlx.base_url`、`omlx.global_settings`（cache/scheduler/memory/sampling 等，密钥脱敏）、`omlx.device` |
| 模型 | `env.server_models`（服务端全量 `/v1/models/status`）、`env.model_env[<16hex>]`（被测模型 id/engine_type/model_type/quantization/max_context_window/max_tokens/estimated_size/生效 settings/feature_flags） |
| 模型设置投影 | `omlx.upload_projection`（oMLX 社区榜上传载荷形状：chip/variant/memory/gpu/os/omlx_version + 每模型 quantization/model_settings/feature_flags），只落盘、不上传 |
| 加速特性 | `omlx.feature_flags`（默认配置下应为空；非空说明存在非默认加速） |
| 逐请求资源 | CPU/RSS/`phys_footprint` 采样窗口写入每条 `records.jsonl` 的 `metrics` |
| 测量有效性 | `records[].metrics.gen_tps_invalid`、`records[].request.tokenized_prompt_tokens`、`records[].request.context_meta.isl_tolerance`（仅 measured 记录）、`context_meta.tokenize_latency_ms`（测量窗口外开销）、`samples.csv.isl_band`；汇总计数在 `run.json.outcome.n_gen_tps_invalid` / `n_isl_in_band` / `n_isl_out_of_band` 与 `aggregates.*` 同名计数 |

## 6. 执行流程

```
选择模型/测试集/样本数
  → 获取双锁（顺序：先 endpoint 后 output；公式见
      [30-architecture.md](../design/30-architecture.md) §4.3 单一事实源）
  → 逐个模型：卸载其它模型 → 应用 MoE 设置（若有）→ 加载模型 → 采集模型设置
  → 逐场景：
       加载并按 scenario 过滤样本 → 上下文窗口筛选 → 取前 N 个（或全部）
       → 按 (dataset, context_length) 分组为 cells
       每个 cell：
         逐样本：tokenize 最终 messages+tools（含 guard；测量窗口外，记 tokenize_latency_ms）
              → 记录 actual（isl_tolerance）与窗口判定
              → 清 hot+SSD（在 tokenize 之后）→ 请求（禁写缓存；窗口不足/请求失败才 skip）
              → 采集 TTFT/usage/CPU/RSS/输入上下文
              → cached_tokens 校验 → gen_tps 有效性 → isl_band 标记 → 记录
  → 汇总 overall/cells（只对 in-band 样本求均值）→ aggregates.json / samples.csv / report.md / MANIFEST.json
  → 写 batches/<batch_id>.json、刷新 index.json
  → 释放双锁；卸载本次加载的模型、清缓存
```

## 7. 输出与验收

输出布局（schema 3.0，`runs/` 一层平铺）：**完整目录树与逐文件字段以
[31-output-contract.md](../design/31-output-contract.md) §1 为单一事实源**（本节不复写），
关键事实：`output/runs/<run_id>/`（`run_id = <batch_id>__<method>__<model_slug>`，目录名即
run_id）内含 `run.json/records.jsonl/aggregates.json/samples.csv/report.md/MANIFEST.json/logs/`；
`batches/<batch_id>.json` 是唯一权威批次元数据；`blobs/prompts/` 外置 prompt 正文；
`archive/` 是 GC 引用根。

**验收标准（可量化，逐条可复核；V&V 方法见
[process/50-validation-plan.md](../process/50-validation-plan.md)）**

| # | 验收项 | 判据 |
|---|---|---|
| AC-1 | selected 样本可追溯 100% | 每个 selected 样本恰有一条 `phase=measured` 记录，或一条 `run.json.outcome.skipped`（含 `scenario/label/sample/repeat/reason`）；`n_selected == n_measured + len(skipped)` |
| AC-2 | 有效测量比例 | `n_ok / n_measured ≥ 95%`（`n_measured`=全部 measured 记录含 error/timeout，`n_ok`=status=ok 的子集；error/timeout 失败尝试入账不静默消失）；`contaminated`/`error`/`timeout` 数在 outcome 与报告中可见，且不进聚合 |
| AC-3 | 非法 TG TPS 为 0 | 聚合与导出中不出现 `gen_tps > 10,000` 或 `gen_tps < 0`；`./modelbench verify` 统计 `gen_tps_invalid`/`isl_out_of_tolerance`，旧 run 的非法值计入 `legacy_gen_tps_invalid`（不导致失败），新记录同时有 `gen_tps` 与 `gen_tps_invalid` 则报 issue |
| AC-4 | 覆盖率显式展示 | 报告/Excel/Web 同时给出 `n_valid`、`gen_tps_n`、`n_length_capped`、`n_gen_tps_invalid`，不得只给均值 |
| AC-5 | actual ISL 记录 + ±10% 入带聚合 | 每条 measured 记录有 `tokenized_prompt_tokens`、`isl_tolerance`；`samples.csv` 有 `isl_band`；`overall/cells` 只对入带样本求均值，出带样本保留但不进均值；`n_isl_in_band`/`n_isl_out_of_band` 与逐条复算一致；`run.json.outcome` 不再有 `n_isl_skipped`；tokenize 失败回退声明值做窗口判定并 warning |
| AC-6 | 聚合可复算 | `aggregates.json` 的 `overall`/`cells` 可由 `records.jsonl` 逐 cell 复算（mean/median/std/min/max/p95/`ci95_low`/`ci95_high` 一致，浮点末位允许差）；1K/8K/32K 不跨档合并 |
| AC-7 | 噪声与口径标注 | 报告标注 `repeats`、`max_tokens`、缓存模式、预热；输出 `stability_cv`/`unstable`（`CV>0.2`）与 run 内 `ci95_low/high`；默认 `repeats=1` 时明确"ci95 只是 run 内抽样不确定性、无 run-to-run 噪声基线、不作显著性断言" |
| AC-8 | 环境可复现 | `run.json.env` 含硬件/OS、oMLX 版本与全局设置（脱敏）、服务端模型状态、被测模型生效设置、git commit/dirty；`MANIFEST.json` 覆盖全部产物且 `blob_refs.missing == 0` |
| AC-9 | 32K 不特判、不可测显式化 | 32K target 固定 32,768、无专属构造/阈值；仅因模型窗口不足或请求失败的行计入 `skipped` 并在报告/Excel 显示 not-testable 数量，不得把未测行当 0 或静默截断 |
| AC-10 | 同一样本集 | 所有设备/模型使用同一 testset（`testset.jsonl` SHA-256 与 manifest 一致）；不做 per-model 构造、不做运行时裁剪/补齐；比较结果必须引用同一 SHA |
| AC-11 | 样本量充分性标记 | 每个 cell/overall 报告 `n_isl_in_band`（进入均值的样本数）；in-band 样本 <8（`MIN_IN_BAND_SAMPLES`）时显式标记 `insufficient_samples`（warning/报告），不得静默当作有效结论。注意：`--samples<8` 的冒烟 run 必然触发该标记，只表示样本不足，不是数据异常 |

## 8. 风险与已知限制

1. **长上下文档位成本**：32K 档（long_context/coding/general_agent/search_agent 的 32k
   子集）冷预填充很慢，默认全量 240 请求耗时会明显高于小样本冒烟。
   建议先用 `--samples` 限流或单独运行单个场景。预构造样本超出模型窗口时**跳过而不是
   截断**（避免把 32K 档测成截断后的短文本）；legacy 样本仍头 70% + 尾 30% 截断并记录
   `truncated/truncation_ratio`。
2. **32K 可测性受模型窗口限制（不是特判）**：32K target 固定 32,768（与 1K/8K 一样普通）；
   窗口估算的逐数字推导见 [11-performance-testset.md](11-performance-testset.md) §4.5（单一事实源），
   结论：32,768 窗口模型 0/80 行可测（其余 not-testable）。运行期只按**目标模型
   actual + 50 + 128** 判定窗口；构建期验收上界只是数据集边界，**不是运行时门槛**。
   32K 没有专属构造/阈值/裁剪。
3. **单请求串行**：测的是单请求延迟/吞吐，不代表并发服务能力。
4. **默认配置差异**：oMLX 版本升级可能改变默认采样参数/引擎；`run.json.env.omlx` 记录版本，
   跨版本对比需谨慎。
5. **冷缓存依赖**：清理失败会中止 cell（不产生假冷数据）；磁盘/后台负载会抬高 TTFT 方差。
6. **24GB 内存 / oMLX 动态上限**：大模型依赖 MoE offload 才可能加载；oMLX 会按系统可用内存
   动态收缩 ceiling。内存守卫拒绝（`prefill_memory_exceeded`/HTTP 507）不重试，记录为 error，
   并在 `run.json.outcome.memory_errors`/`warnings` 中提示；失败不影响其它模型。
7. **并发与双锁**：同一 oMLX 实例上禁止并发 benchmark。双锁获取顺序与锁文件公式见
   [30-architecture.md](../design/30-architecture.md) §4.3（单一事实源：先 endpoint 锁
   （系统临时目录）后 output 根锁 `benchmark.lock`）。
   同一 `base_url` 的不同 `output` 根也互斥；不同 `base_url` 可并发。第二个 run 快速失败并
   提示持有者 PID（见 `USER_GUIDE` 排障）。
8. **口径导致的系统性偏差（不可横比）**：默认做一次模型级短预热（`warmup_per_model=1`，仿 oMLX 消除 JIT/编译），但 per-cell `warmup_requests=0`、OSL 只有 50 token（解码窗口短，TG TPS/TPOT 对短窗口更敏感）、
   `repeats=1`（单次测量，靠 240 条样本量而不是重复平均），且逐请求清 hot+SSD。
   vLLM / oMLX 官方 benchmark 通常有预热、更长生成与批处理/并发口径，因此本系统的
   TTFT/TG TPS/峰值内存**不得与官方榜或其它框架的数字直接横比**；跨 run 也必须先对齐
   `max_tokens`/缓存模式/预热/轮数。
9. **测量有效性边界（残余风险）**：`gen_tps` 在短窗口/工具调用/异常值下置 null
   （原因见 §4），覆盖率以 `n_gen_tps_invalid` 展示而不是补值；actual ISL **不触发跳过**，
   所有能测样本都 measured，只有窗口不足/请求失败才 skip；均值只使用 ±10% 入带样本，
   出带样本保留但不进均值（计数 `n_isl_in_band`/`n_isl_out_of_band`，报告显示 actual 分布）。
   入带样本过少的 cell 会标记 `insufficient_samples`，不得据此下强结论。
10. **不确定性口径（残余风险）**：默认 `repeats=1`、每 cell 10 样本，系统输出
    mean/median/std/p95 与 `CV` 标记，并附加 **run 内** `ci95_low/high`（正态近似）。
    该 CI **只描述 run 内样本均值的不确定性，不是 run-to-run 噪声/重复性区间**；系统仍
    **不做** run-to-run 重复性校准，同 run 内均值可由 `records.jsonl` 复算，但
    "模型 A 比 B 快 x%"这类结论建议对同一批 `sample_id` 做 paired 统计，或用
    `repeats=2–3` 建立噪声基线；10 样本不足以支持逐 cell p95/p99 稳定性断言。
11. **执行顺序/热漂移（残余风险）**：cells 按声明顺序执行，长 run（默认 240 请求）无随机化/交错
    或热漂移检测；跨 run 对比时应记录时间并留意漂移。
