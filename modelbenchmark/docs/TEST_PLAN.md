# 测试方案设计（性能基准测试）

> 本文定义"测什么、怎么测、如何判定"。实现与目录见 [ARCHITECTURE.md](ARCHITECTURE.md)，
> 操作步骤见 [USER_GUIDE.md](USER_GUIDE.md)，指标公式见 [METHODOLOGY.md](METHODOLOGY.md)。
>
> 版本说明：自 0.3 起，旧的多策略测试（`perf` / `accuracy` / `context` / `custom`）与
> 上下文长度矩阵**全部移除**。当前唯一的测试形式是**性能基准测试**；准确性判分代码
> 保留在 `src/modelbench/domain/evaluators/` 作为领域能力，但**不参与本测试**。

## 1. 目标与原则

在本地 oMLX 上，对选定的**测试集 × 模型**做端到端黑盒性能测试，逐样本采集：

- **TTFT**：首 token 延迟（客户端 SSE 时间戳）；
- **TG TPS**：解码窗口内的输出 token 速度（`gen_tps`）；同时记录 `tpot_ms`；
- **峰值内存**：oMLX 服务进程树在请求窗口内的 `phys_footprint` 峰值（GB/GiB，含 Metal 分配；
  与 oMLX benchmark 的 Metal 峰值 `PeakMem` 口径不同，见 `METHODOLOGY`）；
- **输入上下文大小**：每个样本的实际输入 token 数与字符数；
- 辅助指标：`e2e_ms`、`prefill_tps`、`total_tps`、峰值/均值 CPU。

对外展示**按上下文档位分别平均的整体结果**（1K / 8K / 32K 各一行，不跨档位合并），
同时**逐样本保留**结果（JSON/CSV/records）。

原则：

1. **不填充对齐、但档位是 cell 维度**：预构造性能样本自带 1K / 8K / 32K 三档
   （参考 tokenizer 计数的声明值，见 `dataset/test/performance/testset.jsonl`），
   每个 `(scenario, dataset, context_length)` 形成独立 cell；不做按目标长度的填充。
   如只需特定档位，可在 Web/CLI 勾选上下文窗口（§3），它只筛选样本，不改变
   "每个 cell 内对有效样本求平均"的统计口径。
2. **样本数可配置**：`perf.samples_per_dataset` 按**场景**（scenario）控制取多少个样本
   （字段名为历史遗留，实际每个场景各取 N；默认 `performance` 有 8 个场景，所以
   `--samples 5` = 40 个 measured 请求）；**留空（null）= 每个场景全部可用样本**
   （受数据集自身 limit 约束）。CLI/Web 的 `--samples` 帮助文案已按“每个场景”表述。
3. **冷缓存**：每个测量请求前清理 hot+SSD，校验 `cached_tokens==0`；污染记录剔除聚合。
4. **默认模型配置**：模型以 oMLX 下载/默认配置加载；**只允许**为 MoE 模型显式开启
   `moe_expert_offload_enabled`，不启用 TurboQuant / DFlash / MTP 等任何加速或压缩配置。
5. **可复现**：固定抽样、记录配置快照/硬件/oMLX 设置/模型设置/git commit/原始请求响应。
6. **先加载后测试**：每个模型加载完成后才测量；`restore_state` 默认在结束时卸载本次加载的模型。

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
  `moe_expert_offload_enabled` 允许为 true。复位动作写入 `logs/run.log`，最终生效值写入 `omlx.json`。
  说明：复位集合与可追溯字段均采用**显式白名单**（覆盖 oMLX 当前公开的加速开关）；
  白名单之外的实验性设置不复位、也不保证出现在 `omlx.json` 投影中，需要时按 oMLX 版本扩展
  `environment.py` 的 allowlist。
- 应用流程：卸载已加载实例 → `PUT /admin/api/models/<id>/settings` → 重新加载 →
  `wait_until_loaded`；失败按 `optional` 显式记录（`summary.warnings`）。
- 每次 run 采集模型的实际生效设置（oMLX admin 返回的 settings 投影）与加速特性开关，
  写入 `omlx.json`，结果可追溯"是否使用了非默认配置"。

## 3. 测试计划（唯一模式：性能基准）

| 项 | 规则 |
|---|---|
| 测试集 | `dataset/test/<name>/testset.yaml`（默认 `performance`）；场景 = 数据集 + 参数 |
| 模型 | CLI `--models` / Web 勾选；串行测试 |
| 样本数 | `perf.samples_per_dataset`；`null` = 每个场景全部可用样本 |
| 逐场景样本数 | `perf.samples_by_scenario.<scenario>`（**API 参数 `scenario_samples`**；v3 Web 界面已移除逐场景输入入口），优先于全局样本数 |
| 手选样本 | `perf.sample_ids.<scenario>`（**API 参数 `sample_ids`**；v3 Web 界面已移除手选入口），优先级最高；清空回退到样本数/全部 |
| 上下文窗口筛选 | `perf.context_windows`（Web 多选 / CLI `--context-windows`，**legacy/诊断用途**）；空 = 全部档位；**1K 接受 0～1.5K**，其他 N 接受 **N±0.5K**。新数据集的档位请用场景选择或 `perf_jsonl params.buckets`（1k/8k/32k），两套阈值不要混用（新数据集 8K 桶实测 8,023–8,201、32K 桶 32,455–32,779，与 N±0.5K 不是同一口径）。预构造样本按**声明的参考 tokenizer tokens** 筛选，legacy 样本才按目标模型 tokenizer 计数；cell 仍按 `(scenario, dataset, context_length)` 聚合 |
| 预热 | **模型级**默认 **1 次/模型**（`run.warmup_per_model=1`，加载后发一条短请求 `max_tokens=8`，仿 oMLX QUICK，丢弃且不记录/不聚合）；per-cell `run.warmup_requests` 默认 **0**，`--warmup N` 仅诊断用（每个 cell 预热 N 条，不计入 measured 预算） |
| 抽样 | 未手选时：**预构造样本按 `(context_length, lang)` 分层**分配 N（最大余数法，组内按 `sample_id` 取前 k，确定性、跨 run 可复现）；legacy 数据集仍按 `sample_id` 排序取前 N |
| 上下文 | **不设目标长度**；预构造样本原样发送（仅追加固定 decode guard 指令，见“生成长度”，实测约 +39 prompt token，预留 64）。若声明 tokens + max_tokens + decode guard 预留 + 安全边距超过模型窗口，该样本**跳过并计入 `summary.skipped`**（not-testable，不静默截断）；legacy 样本才按 `METHODOLOGY` 头尾截断。注意 32K 桶实测 32,455–32,779 token（含 tools）、8K 桶 8,023–8,201、1K 桶 978–1,072；每请求预留 `50 + 128 + 64 = 242` token，32K 档需模型窗口 **32,697–33,021**，因此 32,768 窗口模型只有 1/160 32K 行可测、其余 159 行 not-testable（当前本机模型窗口 40,960–262,144 均可容纳） |
| 生成长度 | `perf.max_tokens`（默认 **50**，CLI `--max-tokens` / Web「生成长度」可配置）；同时默认开启 `perf.decode_guard=true`，在每条请求后追加固定长输出指令，避免以 `Answer:` 结尾的样本 1 token 结束、TG TPS 为空。**默认 50 + decode guard 下绝大多数输出会 `finish_reason=length`，这是预期解码窗口而非丢弃**；达到 `max_tokens` 的样本照常参与平均，TG TPS 按实际生成 token 计算；`summary`/`report`/Excel/Web 用 `n_length_capped`（触顶数）显式展示。指令原文记录在 `records.jsonl`；`decode_guard=false` 仅用于诊断/历史对比 |
| 轮数 | `run.repeats`（默认 1）；>1 时同一 `sample_id` 产生多条记录，逐样本结果按 `sample_id` 保留 |
| 缓存 | 默认 `cold`（逐请求清理 + 禁写缓存）；`warm` 仅清理 cell 首请求并允许写缓存以测复用；`warm`/`none` 仅诊断，不得与 cold 混算 |
| 工具 | 性能 run **不执行工具循环、不评工具正确性**；agentic 预构造样本只发一次请求且 `tools` 原样保留 |

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
| 唯一性能样本 | **480 条**（8 个 scenario × 3 档 × 20） |
| 默认 8 场景 measured 请求 | **480**（1K 160 / 8K 160 / 32K 160；场景各 60） |
| 默认 cell 数（1 模型） | **24**（8 场景 × 3 档；`(scenario, context_length)` 分组） |
| 投机解码 | 不纳入默认基准（需非默认加速开关，单列专项） |
| 快速回归样本 | `dataset/test/performance/testset_10.jsonl` **240 条**（每场景 `sorted(target_id)[::2]` 取 10 个 target，三档齐全；严格子集） |
| 精度样本 | `dataset/test/accuracy/testset.jsonl` **6,589 条**（250×6 + general_agent 889 + multilingual 4,200，离线资产：`manifest.status=offline-asset`、runner 未接入、6 个 judge 仅契约）；保留在磁盘但**默认不参与**性能 run |

方案与逐场景/语言/来源明细见 [dataset/README.md](../dataset/README.md) 与
[dataset/TEST_SAMPLE_SELECTION.md](../dataset/TEST_SAMPLE_SELECTION.md)；
投机解码不纳入默认基准：需要非默认加速开关，应作为独立专项实验（不启 MTP/DFlash/SpecPrefill，不算 α/speedup）。

> **场景命名**：性能 8 场景按负载形状命名；原 `multilingual` 实际只有 LongBench v1
> `dureader` 中文长文，已更名 `chinese_long_context`；精度侧的 `multilingual` 是真正的
> 42 语种 Global-MMLU（42 × 100 配对题），两者不要混淆。
>
> **可比性**：默认模型级预热 1 次（`warmup_per_model=1`）、per-cell 预热 0、`max_tokens=50`、`repeats=1`，且逐请求冷缓存；
> 这与 vLLM / oMLX 官方 benchmark 的预热/生成长度/批处理口径不同，**不可与任何公开榜横比**。
> 提高 `--max-tokens`/`--warmup`（per-cell）/`--repeats` 可做诊断，但结果不得与默认口径混算。

## 4. 指标与统计口径

- **逐样本结果**：每个 `(model, dataset, sample_id, repeat)` 一条 `SampleRecord`，
  含 `metrics.ttft_ms / gen_tps / tpot_ms / e2e_ms / peak_cpu_pct / peak_memory_gb / peak_rss_mb`
  以及 `metrics.input_context_tokens / input_context_chars`。
- **数据单元汇总（cells）**：`model × scenario × context_length × dataset` 对有效样本求
  mean/median/std/min/max/p95（每个档位独立 cell；默认 8 场景共 24 cells / 480 measured）。
- **整体汇总（overall）**：**按 `(model, context_length)` 分别求平均**（1K/8K/32K
  各一行，绝不跨档位合并——TTFT/预填充相差一个数量级）；峰值内存同时给出该档位
  逐样本峰值的平均值与全局最大值。`overall` 行含 `context_length` 字段。
- 计数口径：`summary.n_samples` = measured 记录数（含 repeats）；
  `overall.n_samples` = 唯一样本数；`n_valid` = 参与聚合的有效记录数。
- **TG TPS 覆盖**：`gen_tps` 仅当 `completion_tokens >= 16` 时输出；报告/CSV 用
  `mean (n/有效数)` 与 `gen_tps_n` 显式展示覆盖，不做插补。
- **输出触顶不丢弃**：达到 `perf.max_tokens`（`finish_reason=length`）的样本照常参与
  TTFT/内存/TG TPS 聚合（TG TPS 用实际 `completion_tokens` 计算到停止前）；
  `n_length_capped` 在 summary.csv / report / Excel / Web 中显示触顶数。
- **有效样本**：`phase=measured` 且 `status=ok`；warmup/污染/错误不参与聚合。
- **冷缓存**：`cleared_before_request=true` 且 `cached_tokens==0`，否则污染并重试/剔除。
- 不再计算 accuracy / Wilson CI；`score` 字段恒为空。

## 5. 环境采集（每次 run）

参考 oMLX 内置 benchmark 上传到社区榜时携带的本地参数，run 开始时采集：

| 分类 | 字段（写入 bundle） |
|---|---|
| 硬件/系统 | `chip_name`、`chip_variant`、`memory_gb`、`gpu_cores`、`os_version`、CPU 核心数、磁盘 |
| oMLX | `omlx_version`、`base_url`、全局设置（cache/scheduler/memory/sampling 等，密钥脱敏） |
| 模型 | 模型 id/路径、`engine_type`、`model_type`、`quantization`、`max_context_window`、`max_tokens`、`estimated_size` |
| 模型设置 | 实际生效 settings 的 oMLX 上传白名单投影（含 `moe_expert_offload_enabled`、`turboquant_*`、`dflash_*`、`mtp_*` 等）；每次加载前后记录进程内存基线/加载后（`models[<id>].memory`） |
| 加速特性 | `feature_flags`（默认配置下应为空；若不为空说明存在非默认加速） |
| 逐请求资源 | CPU/RSS 采样窗口（写入每条记录的 metrics） |

`omlx.json` 还包含 `upload_projection`：把上述本地参数投影成 oMLX 社区榜上传载荷的形状
（chip/variant/memory/gpu/os/omlx_version + 每模型 quantization/model_settings/feature_flags），
仅写入本地 bundle，不上传任何数据。

## 6. 执行流程

```
选择模型/测试集/样本数
  → 逐个模型：卸载其它模型 → 应用 MoE 设置（若有）→ 加载模型 → 采集模型设置
  → 逐场景：
       加载并按 scenario 过滤样本 → 上下文窗口筛选 → 取前 N 个（或全部）
       → 按 (dataset, context_length) 分组为 cells
       每个 cell：
         逐样本：清 hot+SSD → 请求（禁写缓存）→ 采集 TTFT/usage/CPU/RSS/输入上下文
         → cached_tokens 校验 → 记录
  → 汇总 overall/cells/samples → summary.json / summary.csv / samples.csv / report.md
  → 卸载本次加载的模型、清缓存
```

## 7. 输出与验收

```text
output/index.json    # 全部 run + 批次分组的目录索引（可由 ./modelbench index 重建）
output/batches/<batch_id>.json  # 一次测试动作的批次元数据（多模型）
output/result_analysis/<batch_id>_<method>_<models>/summary.xlsx  # 硬件/模型平均/逐样本 Excel
output/<batch_id>_<method>_<model>/  # 每个模型一个 bundle（时间+方法+模型）
output/<run_id>/
  manifest.json      # 配置快照、硬件、oMLX 快照、模型、git commit
  hardware.json      # 硬件/OS/依赖 + oMLX device-info（对应 oMLX 上传的硬件字段）
  omlx.json          # oMLX 版本、全局设置、每模型生效设置、feature flags、模型目录
  models.json        # /v1/models/status 全量
  records.jsonl      # 逐请求权威原始记录（含输入上下文与资源采样）
  summary.json       # 按 (model, 档位) 的 overall + cells + 逐样本 samples
  summary.csv        # 整体平均值表（每模型×档位一行）
  cells.csv          # 数据集粒度表
  samples.csv        # 逐样本表（含输入上下文大小）
  report.md          # 人读摘要（含环境与逐样本结果）
  analysis/          # run.json / samples_long.csv / environment.json / README.md
  raw/samples/       # 原始测试样本（未截断）
  raw/<model>/<scenario>/<bucket>/  # 逐记录快照（native 或 1k/8k/32k）
  logs/run.log
```

验收：目标数据集全部产出记录、无 error/污染、`summary` 可由 `records.jsonl` 复算、
报告包含所选模型/测试集/样本数/硬件/oMLX 配置。

## 8. 风险与已知限制

1. **长上下文档位成本**：32K 档（long_context/coding/general_agent/search_agent 的 32k
   子集）冷预填充很慢，默认全量 480 请求耗时会明显高于小样本冒烟。
   建议先用 `--samples` 限流或单独运行单个场景。预构造样本超出模型窗口时**跳过而不是
   截断**（避免把 32K 档测成截断后的短文本）；legacy 样本仍头 70% + 尾 30% 截断并记录
   `truncated/truncation_ratio`。
2. **单请求串行**：测的是单请求延迟/吞吐，不代表并发服务能力。
3. **默认配置差异**：oMLX 版本升级可能改变默认采样参数/引擎；`manifest.json` 记录版本，
   跨版本对比需谨慎。
4. **冷缓存依赖**：清理失败会中止 cell（不产生假冷数据）；磁盘/后台负载会抬高 TTFT 方差。
5. **24GB 内存 / oMLX 动态上限**：大模型依赖 MoE offload 才可能加载；oMLX 会按系统可用内存
   动态收缩 ceiling。内存守卫拒绝（`prefill_memory_exceeded`/HTTP 507）不重试，记录为 error，
   并在 `summary.memory_errors`/`warnings` 中提示；失败不影响其它模型。
6. **并发任务**：同一 oMLX 实例上的 Web 测试任务串行排队；CLI/Web 跨进程由
   `output/.benchmark.lock` 非阻塞文件锁拦截（第二个 run 快速失败并提示持有者 PID）。
   并发 run 会互相驱逐模型、污染结果，因此禁止（AGENTS 硬不变量 10）。
7. **口径导致的系统性偏差（不可横比）**：默认做一次模型级短预热（`warmup_per_model=1`，仿 oMLX 消除 JIT/编译），但 per-cell `warmup_requests=0`、OSL 只有 50 token（解码窗口短，TG TPS/TPOT 对短窗口更敏感）、
   `repeats=1`（单次测量，靠 480 条样本量而不是重复平均），且逐请求清 hot+SSD。
   vLLM / oMLX 官方 benchmark 通常有预热、更长生成与批处理/并发口径，因此本系统的
   TTFT/TG TPS/峰值内存**不得与官方榜或其它框架的数字直接横比**；跨 run 也必须先对齐
   `max_tokens`/缓存模式/预热/轮数。
8. **不确定性口径（残余风险）**：默认 `repeats=1`、每 cell 20 样本，系统只输出 mean/median/std/p95
   与 `CV` 标记，**不做 run-to-run 重复性校准或置信区间**；同 run 内均值可由 `records.jsonl`
   复算，但"模型 A 比 B 快 x%"这类结论建议对同一批 `sample_id` 做 paired 统计，或在
   `performance_10` 上重复 2–3 次建立噪声基线；20 样本也不足以支持逐 cell p95/p99 稳定性断言。
9. **执行顺序/热漂移（残余风险）**：cells 按 `sample_id` 字典序执行（实际为 1k→32k→8k），
   长 run（默认 480 请求约数小时）无随机化/交错或热漂移检测；外部 2026-11-06 的真实 run 中
   32K 首请求与后续样本量级一致，但顺序效应未被系统验证，跨 run 对比时应记录时间并留意漂移。
