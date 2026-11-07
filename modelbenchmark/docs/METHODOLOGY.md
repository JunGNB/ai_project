# 测量方法学（METHODOLOGY）

本文档定义"测什么、怎么测、为什么这样测"，所有实现与本文档一致。
测试范围见 [TEST_PLAN.md](TEST_PLAN.md)，实现见 [ARCHITECTURE.md](ARCHITECTURE.md)。

---

## 1. 一次测量是什么

**一次测量 = 一次流式 `POST /v1/chat/completions`**（`stream=true`，
`stream_options.include_usage=true`，`temperature=0`，`max_tokens=perf.max_tokens`），
客户端使用 `time.perf_counter()` 为每个 SSE chunk 打时间戳。

- 输入 prompt 使用数据集样本的**预构造 messages/tools**（`dataset/test/performance`），
  不做渲染/拼接/填充；`RequestSpec.actual_prompt_tokens` 取样本声明的 tokens；
- 预构造样本若 `声明 tokens + max_tokens 50 + 安全边距 128 + decode_guard 预留 64`（共 242）
  超过模型窗口，该样本**跳过**并写入 `summary.skipped`（not-testable，不静默截断）；
  仅 legacy/自定义样本才按头 70% + 尾 30% 截断，并记录 `truncated/truncation_ratio`；
- `cached_tokens` 来自服务端 `usage.prompt_tokens_details.cached_tokens`，用于缓存污染检测。

---

## 2. 性能指标定义

设：
- `t0`：发出 HTTP 请求的时刻
- `t_first`：第一个携带 token 的 delta 到达（`content` 或 `reasoning_content` 或工具调用参数）
- `t_visible`：第一个 `content` delta 到达（思考型模型可能明显晚于 `t_first`）
- `t_last`：最后一个 token delta 到达
- `t_end`：SSE 流结束
- `P`：`usage.prompt_tokens`，`C`：`usage.completion_tokens`

| 指标 | 公式 | 说明 |
|---|---|---|
| `ttft_ms` | `(t_first - t0) × 1000` | 首 token 延迟（含思考 token），与 oMLX `ttft_ms` 对齐 |
| `ttft_visible_ms` | `(t_visible - t0) × 1000` | 首个可见答案 token 延迟 |
| `tpot_ms` | `(t_last - t_first) / (C - 1) × 1000` | 每输出 token 时间；`C < 16` 视为不可信 → `null`（与 oMLX 口径一致） |
| `gen_tps` | `(C - 1) / (t_last - t_first)` | **TG TPS（输出 token 速度）**，解码窗口内计算 |
| `prefill_tps` | `P / ttft_s` | 预填充速度 |
| `e2e_ms` | `(t_end - t0) × 1000` | 端到端时延 |
| `total_tps` | `(P + C) / e2e_s` | 含预填充的总吞吐 |
| `input_context_tokens` | `P`（缺失时回落到请求构建时的 token 计数） | **每个样本的输入上下文大小（token）** |
| `input_context_chars` | 请求消息内容字符数之和 | 输入上下文大小（字符），用于解释 token/字符比 |
| `peak_cpu_pct` | 请求窗口内 oMLX **进程树** CPU 占用最大值（50ms 采样） | 100% = 单核满载；12 核理论峰值 1200% |
| `mean_cpu_pct` | 窗口内 CPU 占用的采样均值 | |
| `peak_memory_gb` | oMLX 服务进程树 `phys_footprint` 峰值（GiB，含 Metal 分配） | 主口径；Apple Silicon 上 jetsam 的就是这个内核记账值 |
| `peak_rss_mb` | 进程树 RSS 峰值（MB，诊断保留） | 兼容旧 run；RSS 与 Metal 记账不一致，不再作为主展示 |
| `cpu_samples` / `cpu_sample_window_ms` | 采样点数与窗口长度 | 用于判断采样是否充分 |

**为什么需要 `max_tokens` 下限 + decode guard**：旧方案里准确性短答案（1–3 token）会让
`gen_tps/tpot` 按约定置空。性能测试不再判分，因此统一使用 `perf.max_tokens`
（默认 **50**，CLI/Web 可配置）。达到 `max_tokens`（`finish_reason=length`）**不是丢弃**：
样本照常参与 TTFT/内存/吞吐聚合，`gen_tps/tpot` 用实际生成的 `completion_tokens`
计算到停止前的解码窗口，绝不按上限外推。
但仅设 `max_tokens` 不够：性能样本常以 `Answer:` 结尾（LongBench v2 选择题），模型会
合法地只输出 1 个字母就 EOS，`C < 16` 仍会让 TG TPS 为空。因此性能模式默认开启
**decode guard**（`perf.decode_guard: true`）：在原始 messages 之后追加一条固定的长输出
指令（原文见 `records.jsonl` 的 `request.messages` 与
`request.context_meta.decode_guard_instruction`），要求模型继续写详细总结。
它是**追加的一条 user 消息**（183 字符）：会实际增加输入 prompt 约 40 token（本机实测
+39 token，随 tokenizer 略有差异；预留 64 token 足够），因此 `input_context_tokens` /
`input_context_chars` 含这部分协议开销，而 `request.actual_prompt_tokens`（数据集声明值）
不含；标题档位不变，性能测试不判分，因此不影响“测什么”的结论。
**不同 `max_tokens` 的 run 不直接可比**（解码窗口长度不同，TG TPS/TPOT 会受影响）；
跨 run 对比前先对齐该值（记录在 `manifest.config_snapshot.perf.max_tokens`）。

**TG TPS 何时为 null**：decode guard 显著降低了空值率，但不是绝对保证——模型仍可能提前
EOS，或工具调用在单个 chunk 内一次性返回（解码窗口≈0）。此时按 oMLX 口径
（`completion_tokens >= 16` 且解码窗口 > 0）置空，而不是伪造速度；
`Distribution.n` 与报告里的 `mean (n/有效数)` 会显示真实覆盖。历史 `throughput`
合成探针已不在默认测试集中，`synthetic` 适配器仅为 legacy 兼容保留。

**工具循环**：性能 run **不执行工具循环、不评工具正确性**。预构造的 agentic 场景
（general_agent/search_agent）只发送一次原始请求：`tools` 只带目标请求自身定义
（general_agent 1–3 个、search_agent 固定 search/lookup 2 个），BFCL 的大函数目录作为
messages 文本按 whole-function 前缀携带（1K 3–7 / 8K 57–61 / 32K 228–248 个函数），
不调用 `ScenarioAdapter.execute_tool`；legacy 非预构造样本才走工具循环。
记录里的 `request.tools` 保留原样供审计。

**思考模式**：默认 `thinking_mode=disabled`，请求携带
`chat_template_kwargs={"enable_thinking": false}`。这是为了在固定"直接作答"口径下
比较模型；若需测量思考型模型的真实体验，用 `--thinking-mode auto|enabled` 重跑，
并同时参考 `ttft_visible_ms`（首个可见答案 token）与 `ttft_ms`（首个 token，含思考）。

**为什么同时记录客户端与服务端数字**：服务端 `usage` 里有
`time_to_first_token / generation_tokens_per_second / prompt_tokens_per_second`，
写入原始响应（`response.usage`）用于交叉验证；主口径用客户端测量，因为它是用户真实感知，
且能覆盖网络/序列化开销。

**峰值内存口径（重要）**：主展示为 `peak_memory_gb`，取请求窗口内 oMLX 服务进程树
`phys_footprint` 的采样峰值（macOS `proc_pid_rusage` 的 `ri_phys_footprint`，单位 GiB =
1024³ bytes，与 oMLX 界面一致）。它包含 `IOAccelerator`/Metal 后备分配，但不是 oMLX 自带
benchmark 的 `PeakMem`：后者是 `mlx.core.get_peak_memory()` 的 **Metal 分配峰值**，不含
Python/tokenizer/运行时开销，本机历史观察约低 **0.4–0.9 GB（经验值，本仓库未系统实测）**。因此：

- 每次模型加载前后还记录进程内存基线/加载后（写入 `omlx.json` 的
  `models[<id>].memory = {baseline_gb, idle_gb, load_gb, ...}`，报告里展示"基线 → 加载后"）：
  基线是**模型加载前** oMLX 进程树的 `phys_footprint`，会随进程寿命、先前加载过的模型、
  allocator/缓存保留而波动（实测约 0.9～3.0 GB）；`idle_gb` 在模型加载后较稳定；
  `peak_memory_gb` 再叠加 prefill/KV 的瞬时峰值。**不要把"基线"当成固定开销**，
  也不要把它与模型权重直接相等；
- 与本系统历史 run 比较：用 `peak_memory_gb`；
- 与 oMLX 榜单/benchmark 的 PeakMem 比较：只能判断量级，不能逐位对齐；
- `peak_rss_mb` 仅作诊断，旧 run 的 Web 展示会按 `/1024` 回退到 GB。

**CPU/内存采样治理**

- 只采样请求窗口（缓存清理、模型加载不计入）；
- `psutil.Process(pid)` 进程树（含 helper 子进程）累加 CPU 时间；首个采样作为基线，
  后续按 `Δcpu_time / Δwall_time × 100` 计算；
- 服务端 PID 自动发现：`server.pid_file`（可选）→ `lsof -nP -iTCP:<port> -sTCP:LISTEN -t`；
- macOS 沙箱下 `/bin/ps`、`top` 不可用，因此用 psutil + lsof，且已实测可行。

---

## 3. 缓存治理（本系统最关键的测量前提）

### 3.1 现象（本机实测）

MiniCPM5-1B，同一 10,752 token prompt：

| 状态 | `cached_tokens` | TTFT |
|---|---|---|
| 缓存命中 | 10,752 | **0.19 s** |
| 清理 hot+SSD 缓存后 | 0 | **4.73 s** |

**约 25 倍差异。** 如果不治理，TTFT 测到的是"缓存运气"，不同模型/不同 run 不可比。

### 3.2 策略

默认 `cold` 模式：

1. 每个 **measured** 请求前调用 `POST /admin/api/hot-cache/clear` +
   `POST /admin/api/ssd-cache/clear`，等待 `cache_settle_ms`（默认 800ms）；
2. 请求头 `X-oMLX-Cache-Store: off`（oMLX 官方开关），避免基准请求写入 prefix cache；
3. 响应 `cached_tokens > 0` → 视为污染：重试一次（重新清理），仍命中则标记
   `contaminated`，**从聚合中剔除**（保留原始记录可审计）；服务端未返回
   `cached_tokens` 时同样按污染处理（fail-closed，无法证明是冷请求）；
4. 模型级预热：每个模型加载后、任何 measured cell 之前，默认发 **1 条**短请求（`run.warmup_per_model=1`，
   短 prompt、`max_tokens=8`，仿 oMLX QUICK 模式触发 JIT/Metal 编译与 KV 初始化），直接丢弃、
   不写 records、不判分、不计入 measured 预算；该请求强制 `cache_store=False`，不预热 prefix cache。
   per-cell `run.warmup_requests` 默认 0；显式配置 N 时每个 cell 再发 N 条丢弃请求（诊断用）；
5. `isolate_models: true`：切换模型前卸载其它已加载模型，避免 24GB 内存挤压；
6. run 结束后清理缓存；`restore_state=true`（默认）时卸载本次 run 加载的模型，不污染用户后续使用。

`warm` 模式用于测量**缓存复用后的稳态性能**：每个 cell 只清理一次，之后连续重复；
warm 请求**允许写入** prefix cache（`cache_store=True`，不发送 `X-oMLX-Cache-Store: off`），
否则复用无法被观测；cold 模式仍逐请求清理并禁止写缓存。
`warm` 与 `cold` 结果**分开报告，绝不混算**。

---

## 4. 重复与统计策略

### 4.1 决策

| 指标 | 处理 |
|---|---|
| TTFT / e2e / CPU / 内存 | 逐样本测量；默认每个样本 1 次（`run.repeats=1`），可配置多轮 |
| token 速度（gen_tps） | 同上；要求 `C ≥ 16`，否则该项为 null |
| 逐样本结果 | 每个 `(sample_id, repeat)` 一条记录，永不丢弃（错误记录也保留 status/error） |
| 输出触顶（finish_reason=length） | **不剔除**：达到 `max_tokens` 的样本照常参与 TTFT/内存/吞吐聚合；`gen_tps/tpot` 用实际 `completion_tokens` 计算到停止前，报告/Excel/Web 显示触顶数 `n_length_capped` |
| 计数口径 | `summary.n_samples`=measured 记录数（含 repeats）；`overall.n_samples`=唯一样本数；`n_valid`=参与聚合的有效记录数 |
| 数据集汇总 | 对有效样本求 mean/median/std/min/max/p95 |
| 整体汇总 | **按 `(model, context_length)` 分别求平均**：1K / 8K / 32K 各一行，绝不跨档位合并；峰值内存另给该档位全局最大值。旧 schema 2.0 run 的 `overall` 无档位字段，Web 标记为“全部*”，不与新口径混比 |
| 稳定性 | `CV = std(ttft)/mean(ttft)`；`CV > 0.2` 标记 cell `unstable` |

**为什么默认 1 轮**：新方案以"样本量"换稳定性——样本来自多个数据集且保留原始上下文；
逐样本结果本身可审计。需要更稳的结论时，提高 `--samples` 或 `--repeats`。

**为什么保留逐样本记录**：平均值会掩盖长尾/异常；保留每条记录（含输入上下文大小、
token 用量、错误文本）才能解释"为什么这个模型的平均值是这样"。

---

## 5. 输入上下文处理（档位来自数据集，可选窗口筛选）

- Token 语义：`tokens` = oMLX `/v1/tokenize(messages + API tools)`（chat template），即构造分桶与声明值都按**总 prompt** 计；旧版 tools 场景的双计已修正。
- 目标：使用样本自带（预构造）上下文，**不做按目标长度的对齐与填充**；1K/8K/32K 是
  数据集档位，`SampleRecord.context_length` 记录该档位并作为 cell 维度；
- **两套 token 语义（重要）**：
  1. **筛选口径（declared tokens）**：预构造样本的 `meta.tokens` 是用参考 tokenizer
     （`MiniCPM5-1B` + chat template，见 `dataset/test/performance/manifest.json`）实测的
     声明值，也是档位定义依据。`perf.context_windows` 对预构造样本**直接使用该声明值**，
     不再调用目标模型 tokenizer —— 这样筛选与数据集分桶定义一致，且避免 480 次额外
     tokenize 请求。已知偏差：目标模型 tokenizer 的计数可能与声明值相差几个 token，
     窗口边界样本可能因此被纳入/排除，属于可接受的口径偏差（legacy 样本仍按目标模型
     tokenizer 计数）。
  目标模型 `/v1/tokenize` 的计数与 `usage.prompt_tokens` 在**不含协议后缀时**相差约
  ±2 token（近似值），由 `context_safety_margin_tokens` 兜底；性能模式还会在请求末尾
  追加 decode guard（固定长输出指令，实测 +39 token），运行时窗口校验按
  `声明 tokens + max_tokens 50 + 安全边距 128 + decode_guard 预留 64 = 声明 tokens + 242`
  估算。
  2. **记录口径（服务端 usage）**：`metrics.input_context_tokens` 来自
     `usage.prompt_tokens`（缺失时回落请求声明值），代表**目标模型实际看到的**输入大小；
     因此同一条记录里 `request.actual_prompt_tokens`（声明值）与
     `metrics.input_context_tokens`（服务端值）的差值 ≈ decode guard 的 token 数
     （本机约 +39），属预期协议开销，不是 tokenizer 误差。
- **可选上下文窗口筛选**（`perf.context_windows`，**legacy/诊断用途**）：空 = 全部档位；
  **1K 特殊：接受 0～1.5K token；其他窗口 N：接受 N−0.5K～N+0.5K token**；多个窗口取并集。
  注意它与新数据集分桶是两套阈值（新数据集 1K 978–1,072、8K 8,023–8,201、32K 32,455–32,779）；新数据集请用
  场景选择或 `perf_jsonl params.buckets`（1k/8k/32k）选样，不要与 context_windows 混用。
  该筛选只决定"测哪些样本"，聚合仍按 cell（含档位）求平均。
- **窗口不足**：预构造样本原样发送、不截断；若 `声明 tokens + 242` 超过模型窗口，该行
  记 `not-testable` 并写入 `summary.skipped`。32K 档声明 32,455–32,779，需窗口
  32,697–33,021；32,768 窗口模型只有 1/160 32K 行可测。legacy 样本仅当超过模型窗口时
  才截断为头 70% + 尾 30%，问题保持在最后，并记录 `request.truncated`、
  `request.context_meta.truncation_ratio`；
- 每个样本的输入上下文大小随 `samples`/`samples.csv` 输出（含 `context_length` 档位列），
  便于按长度分层分析。

**记录字段**：`request.target_context_tokens`（性能模式恒为 null）、
`request.actual_prompt_tokens`、`request.truncated`、`request.padded`、
`request.context_meta`、`metrics.input_context_tokens`、`metrics.input_context_chars`。

---

## 6. 判分（当前不参与运行）

性能测试**不做准确性判分**：runner 不再调用 `ScenarioAdapter.score`，
`SampleRecord.score` 恒为 null，summary 不输出 accuracy/CI。

`src/modelbench/domain/evaluators/` 与各场景的 `score()` 作为领域能力保留，
供未来恢复准确性测试或离线复核使用；本测试不加载、不执行判分器（避免代码执行类判分的
额外开销与风险）。

精度数据侧已有离线资产 `dataset/test/accuracy/testset.jsonl`（6,589 行 = 250×6 + BFCL 889 + Global-MMLU 42×100），
但**不接入 run**：`manifest.status=offline-asset`、`runner=not wired`，6 个 judge 只有契约
（`implementation_status=contract-only`），没有 evaluator/runner。在接入并跑出独立 run_id 前，
不得引用任何精度分数。

---

## 7. 环境与配置采集

参考 oMLX 内置 benchmark 上传到 oMLX 社区榜时携带的本地参数，run 开始时采集：

- **硬件/系统**：`chip_name`、`chip_variant`、`memory_gb`、`gpu_cores`、`os_version`、
  可用内存/已用/swap 实时快照（`memory.available_gb/used_gb/percent/swap_used_gb`）、
  CPU 核心数、内存、磁盘、Python/依赖版本；
- **oMLX**：版本、`base_url`、全局设置（cache/scheduler/memory/sampling 等，密钥脱敏）；
- **模型**：`engine_type`、`model_type`、`quantization`（config.json 或目录名推断）、
  `max_context_window`、`max_tokens`、`estimated_size`；
- **模型设置**：oMLX admin 返回的实际生效 settings，按 oMLX 上传白名单投影
  （`moe_expert_offload_enabled`、`turboquant_*`、`dflash_*`、`mtp_*` 等）；
- **加速特性**：由 settings 推导的 `feature_flags`（默认配置下应为空）；
- **逐请求资源**：CPU/RSS 采样窗口写入每条记录；
- **内存错误**：oMLX 内存守卫拒绝（`prefill_memory_exceeded`/HTTP 507）计入 `summary.memory_errors`，
  并在 `warnings` 中给出一条可操作提示，不盲目重试；
- **默认配置复位**：加载前检测历史遗留的加速开关（DFlash/SpecPrefill/TurboQuant KV/Lightning MTP/
  VLM MTP/Qwen ANE Prefill/Qwen OQ-a8），为 true 时写回 false 并重载；只有 MoE 模型的
  `moe_expert_offload_enabled` 允许为 true。复位动作记录在 `logs/run.log`，最终生效设置记录在
  `omlx.json`（含 `upload_projection`）。settings 投影是**显式白名单**（`environment.py`），
  白名单外的实验性设置不复位、也不保证可见，需按 oMLX 版本扩展 allowlist。

落盘：`hardware.json`、`omlx.json`（含 `upload_projection`，对齐 oMLX 社区榜上传载荷的本地字段）、
`models.json`、`manifest.json`（汇总快照）。

---

## 8. 失败、超时与重试

- 传输/HTTP 错误：最多 `max_retries`（默认 2）次，**线性退避**（`retry_backoff_s × attempt`）；
- 冷缓存污染：独立计数器，最多 `max_contamination_retries`（默认 1）次，不与传输重试共用预算；
- 仍失败：`status=error`；超时（httpx timeout 文本）映射为 `status=timeout`，均保留错误文本，不静默丢弃；
- 单样本失败不影响后续样本；`fail_fast` 可配置（默认关闭）；
- 模型加载失败：整模型跳过并记录到 `summary.skipped`；
- 模型设置失败：`optional=true` 时记 warning，按默认配置继续；`optional=false` 中止该模型。

---

## 9. 可复现性

每个 run 目录包含：

- `manifest.json`：完整配置（密钥脱敏）、硬件、oMLX 快照、模型清单、git commit；
- `hardware.json`：芯片/核心数/内存/GPU 核数/macOS/oMLX 版本/依赖版本；
- `omlx.json`：oMLX 版本、全局设置、每模型生效设置与 feature flags；
- `models.json`：`/v1/models/status` 全量（上下文长度、引擎类型、大小、max_tokens）；
- `records.jsonl`：每请求的 prompt、原始响应（含 reasoning 与 usage）、指标、缓存状态；
- `summary.json` / `summary.csv` / `cells.csv` / `samples.csv`：整体/数据集/逐样本结果；
- `raw/samples/`：原始测试样本（未截断）；
- `raw/<model>/<scenario>/<bucket>/`：逐记录请求/响应/指标快照（bucket = 1k/8k/32k 或
  legacy 的 `native`）。

---

## 10. 已知限制

1. **并发**：本系统顺序发请求，测的是单请求性能；并发吞吐需扩展（如异步 worker）。
2. **准确性**：当前性能测试不判分；精度 testset 是离线资产（未接入 runner/判题），
   模型"答得好不好"不在本测试结论范围内。
3. **长文截断**：预构造样本超窗时**跳过而不是截断**；仅 legacy 样本才头尾截断，
   与原始全文结果不可直接等价。
4. **思考型模型**：默认关闭思考；若开启且 `max_tokens` 被思考耗尽，`content` 为空，
   `finish_reason=length`，性能结论需结合 `ttft_visible_ms` 与思考 token 数解读。
5. **后台负载**：macOS 无法完全锁定 GPU/电源状态；通过逐样本记录与 `unstable` 标记暴露。
6. **oMLX 版本差异**：`usage` 扩展字段、缓存清理端点依赖当前 oMLX；`manifest.json`
   记录版本，跨版本对比需注意。
7. **口径偏差（不可横比）**：默认模型级预热 1 次（`warmup_per_model=1`，per-cell 预热 0）、OSL=50、
   `repeats=1` 且逐请求冷缓存；vLLM / oMLX 官方 benchmark 的预热长度、生成长度、批处理/并发口径不同。本系统数字
   **不得与官方榜或其它框架直接横比**；跨 run 对比须先对齐 `max_tokens`/缓存模式/预热/轮数。
8. **32K 档可测性**：32K 行声明 32,455–32,779 token，加上每请求 242 token 预留后需要
   32,697–33,021 的窗口；32,768 窗口模型只有 1/160 行可测，其余记录为 not-testable。
