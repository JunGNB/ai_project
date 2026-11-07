# 可行性评估与自检（FEASIBILITY）

本文档回答两个问题：**方案能不能做成？** 以及 **怎么证明它做对了？**
所有结论都基于本机实测（macOS 15.7.7 / Apple M4 Pro / 24GB / oMLX 0.7.3.dev2），
不是纸面推演。

> 0.3 起测试方案改为**纯性能基准**（逐数据集抽样、逐样本指标、无上下文矩阵、无判分）。
> 本文记录新方案的可行性与验收方式；历史推导见 [archive/](../docs/archive/DESIGN.md)。

---

## 1. 需求 → 方案 → 验证方式

| 用户需求 | 方案 | 如何验证 |
|---|---|---|
| 删除旧测试策略 | 移除 `strategies.yaml`/`StrategyDef`/`apply_strategy`/上下文矩阵；配置只剩 `perf` + `run` | `pytest`、`validate`、docs contract |
| 指定数据集样本数（缺省=全部） | `perf.samples_per_dataset`（null=全部，整数=每个场景前 N；字段名为历史遗留）对每个场景确定性抽样 | 单测：抽样顺序/数量；dry-run 计划 |
| 不按目标长度填充/截断 | 预构造样本原样 prompt（档位来自数据集），超窗时跳过并记 `summary.skipped`；仅 legacy 样本才头尾截断 | 单测：native/not-testable/truncation；记录 `truncated` |
| 逐样本 TTFT/TG TPS/峰值内存 | 客户端 SSE 时间戳 + 进程树 50ms 采样 | fake-server 集成测试 + 真实 run |
| 对外按档位整体平均值 + 逐样本结果 | `summary.overall`（`model × context_length`）+ `summary.cells` + `summary.samples`/`samples.csv` | 复算：由 records 按档位重算 overall 一致 |
| 输入上下文大小 | `metrics.input_context_tokens/input_context_chars` | 单测 + 真实 run 抽样核对 |
| 每次采集硬件 + oMLX 配置 | `hardware.json`/`omlx.json`（版本、全局设置、模型生效设置、feature flags、量化） | 与 `sysctl`、oMLX 界面/device-info 人工核对 |
| 模型默认配置，仅 MoE 开 offload | `ModelLoadConfig` 白名单只允许 `moe_expert_offload_enabled` | 配置校验单测 + 记录 feature flags 为空 |
| 冷缓存可复现 | 逐请求清 hot+SSD、`X-oMLX-Cache-Store: off`、`cached_tokens` 校验 | 实测 25× TTFT 差异 + 记录校验 |
| 先设计、再开发、再自校验 | 本文档 → 代码 → `pytest` + `validate` + 真实 run | 见第 6 节 |

---

## 2. oMLX 接口能力实测（方案所需能力全部具备）

| 能力 | 端点 | 实测结果 |
|---|---|---|
| 服务健康 | `GET /health` | 200，无需鉴权 |
| 模型清单 | `GET /v1/models` | 11 个模型（与 `models.yaml` 同步），含 `max_model_len` |
| 模型运行态 | `GET /v1/models/status` | 每个模型 `max_context_window` / `max_tokens` / `engine_type` / `estimated_size` / `loaded` |
| 精确 token 计数 | `POST /v1/tokenize` | 支持 `messages`/`prompt`，返回 `count` + `max_model_len` |
| 流式对话 | `POST /v1/chat/completions` | SSE；`stream_options.include_usage` 返回 `prompt_tokens/completion_tokens/cached_tokens/time_to_first_token/...` |
| 缓存清理 | `POST /admin/api/hot-cache/clear`、`/admin/api/ssd-cache/clear` | 登录后 200；返回 `total_cleared/bytes_reclaimed` |
| 模型装载/卸载 | `POST /v1/models/{id}/load|unload` | 200/202；`/v1/models/status` 的 `loaded` 可轮询 |
| 设备信息 | `GET /admin/api/device-info` | `chip_name=M4, chip_variant=Pro, memory_gb=24, gpu_cores=16` |
| 服务配置 | `GET /admin/api/global-settings` | 可取到 cache/scheduler/memory/sampling 等配置（写入前脱敏） |
| 模型设置 | `GET /admin/api/models` + `PUT /admin/api/models/{id}/settings` | 每个模型含实际生效 `settings` 与 `moe_expert_offload_supported` |

鉴权：`Authorization: Bearer <key>` 或 `X-API-Key`；admin 用 `POST /admin/api/login` 换 cookie。
密钥来源：`OMLX_API_KEY` 环境变量，或回落到 `~/.omlx/settings.json` 的 `auth.api_key`。

---

## 3. 关键实测证据

### 3.1 缓存会让 TTFT 相差约 25 倍（因此必须治理）

MiniCPM5-1B，10,752 token prompt：

| 状态 | `cached_tokens` | TTFT | 备注 |
|---|---|---|---|
| 缓存命中 | 10,752 | 0.19 s | 未清理 |
| 清理 hot+SSD 后 | 0 | 4.73 s | 冷启动口径 |

结论：不治理缓存 = 测缓存而非测模型；本系统默认 `cold` 模式，逐次清理并校验。

### 3.2 CPU/内存采样可行

- `/bin/ps`、`/usr/bin/top` 在受限沙箱返回 `Operation not permitted`；
- `psutil` 读 `/proc`/`libproc` 正常：oMLX 服务进程树在 1B 模型生成时峰值约 **110%**（≈1.1 核），空闲约 1.6%；
- 服务端 PID：`lsof -nP -iTCP:8000 -sTCP:LISTEN -t` → 可用（psutil 的 `net_connections` 被沙箱拒绝）。

### 3.3 数据集可获得性

- LongBench-v2：`hf-mirror.com/datasets/zai-org/LongBench-v2`，465MB，503 题，
  上下文 48K–16M 字符（中位数约 417K 字符 ≈104K token）；超过模型窗口时按头 70% + 尾 30% 截断。
- MMLU / GSM8K / IFEval / HumanEval / MMMLU / HotpotQA：HF 镜像可下载（parquet/jsonl/csv）。
- BFCL v4：GitHub raw（`ShishirPatil/gorilla` 的 `bfcl_eval/data/`）可下载。
- 以上为 legacy 上游来源；当前运行时只读 `dataset/test/performance/testset.jsonl`
  （由 `tool/dataset` 预构造），不再在 benchmark 运行时下载/解析这些原始数据集。
  预构造路径**不截断**：样本超出模型窗口时记为 `not-testable`/`skipped`（仅 legacy 样本才头 70% + 尾 30% 截断）。

### 3.4 网络约束

- huggingface.co 直连不可用，`hf-mirror.com` 可用（1–10 MB/s）；
- PyPI 直连不可用，清华/阿里/中科大镜像可用（项目 `.venv` 已装好依赖）；
- GitHub raw 可用（偶发慢）。

---

## 4. 运行时预算（决定样本数）

默认测试集（`performance`）固定为 8 个场景 / **480 measured 请求 / 24 cells**
（1K/8K/32K 各 160 条；无投机场景）。样本与 token 预算的权威来源是
[dataset/README.md](../dataset/README.md) 与 `dataset/test/performance/manifest.json`。

> 抽样规则：预构造样本按 `(context_length, lang)` 分层分配 N（最大余数法、组内按 sample_id），
> `--samples 5` 在 `long_context` 上会同时覆盖 8K 与 32K，不再被字典序 32k 整段偏置。
> 快速冒烟仍建议 `--scenarios knowledge_qa math_logic` 以避开 32K prefill 成本。

按历史实测外推（MiniCPM5-1B；冷缓存、不单独 warmup、**OSL 256 的历史口径**；当前运行默认
`max_tokens=50`，解码更短、单请求耗时略低，下表按旧口径给出偏保守的量级）：

| 模式 | measured 请求/模型 | 1B 粗估 | 9B 粗估 |
|---|---:|---:|---:|
| 快速冒烟（2 个 1K 场景 × 1 样本） | 2 | 约 1 分钟 | 2–5 分钟 |
| `--samples 5`（8 场景） | 40 | 约 20–40 分钟 | 1–2 小时 |
| 默认全量 | 480 | 约 1.5–3 小时 | 6–12 小时 |

> 2026-11-06 外部真实 run（MiniCPM5-2B，`performance_10`=240 请求）实测 32K 单样本
> TTFT ≈ 58–60 s、1K ≈ 0.87 s；据此 2B 模型跑完默认 480 请求约 3–4 小时（32K 档占主要成本），
> 与上表同量级。实际耗时以 `logs/run.log` 的 cell 进度为准。

成本由 32K 档主导：冷预填充按 token 近似超线性增长（历史实测 129K token ≈ 348s/次）；
32K 档 160 条 measured 是主要耗时。建议先用 `--samples 1..5`
或 `--context-windows 1024 8192` 验证，再决定是否跑全量。

**磁盘**：每条记录含完整 prompt/response，32K 样本可达数百 KB；默认全量 run 约
200 MB–1 GB（`raw/<model>/<scenario>/32k/` 占大头）；`dataset/` 证据与 `output/` 已加入
`.gitignore`。

---

## 5. 风险与对策

| 风险 | 影响 | 对策（已实现/已计划） |
|---|---|---|
| 缓存污染 | TTFT 严重失真 | 冷模式清理 + 禁写头 + cached_tokens 校验 + 重试/剔除 |
| 24GB 内存装不下大模型 | 加载失败/OOM | `isolate_models` 卸载其它模型；MoE 模型开启 offload；失败显式记录 |
| 长上下文样本超窗 | 请求失败 | 预构造样本超窗跳过并记 `summary.skipped`（not-testable，不截断）；legacy 样本才头尾截断并记录 `truncated/truncation_ratio` |
| 样本量大导致 run 过长 | 阻塞 | `--samples`/`--scenarios`/`--max-requests` 限流；Web 可取消 |
| 思考型模型耗尽 max_tokens | 输出短 | `thinking_mode=disabled` 默认；可调 `--max-tokens`；记录 finish_reason |
| 模型设置被悄悄改动 | 结果不可比 | 白名单只允许 MoE offload；`omlx.json` 记录实际设置与 feature flags |
| 结果不可比（环境漂移） | 跨 run 误读 | manifest/hardware/omlx 快照 + 固定抽样 + 冷缓存口径 |
| 单次样本抖动 | 结论不稳 | 逐样本保留；提高 `--samples`/`--repeats`；CV 标记 |

**降级路径**：时间/资源不足时先用 `--samples 5` 验证系统正确性，再逐步扩大样本；
大模型（27B/35B）单独用 `--models` 跑，避免与中小模型混跑。

---

## 6. 自校验计划（开发完成后执行）

1. **静态检查**：`./modelbench validate`（import、配置、数据集、服务连通性/鉴权/tokenize）；
2. **离线单测**：`pytest test/`（metrics/padding/config/cache/writer/client mock + fake-server）；
3. **真实冒烟**：`./modelbench run --models minicpm5-1b --samples 1`，检查
   - run bundle 文件齐全（含 `hardware.json`/`omlx.json`/`samples.csv`）；
   - `summary.overall`（按 `model × context_length`）可由 `records.jsonl` 复算一致；
   - 每条 measured 记录含 `input_context_tokens` 与冷缓存字段；
   - `omlx.json` 中 `feature_flags` 为空（默认配置），MoE 模型设置符合白名单；
   - TTFT / TG TPS / 峰值内存数值量级合理；
4. **可复现性**：同一 `--samples` 连跑两次，样本集合一致（确定性抽样）；
5. **独立验证**：由 verifier 队友独立跑测试与 smoke，并对照本文档逐项检查。

---

## 7. 结论

**方案可行，且已具备落地条件：**

- 所需 oMLX 接口（流式 usage、tokenize、模型状态、缓存清理、load/unload、模型设置）全部实测可用；
- 缓存污染这一最大可信度风险已被识别、量化（25×）并有可用的治理手段；
- 逐样本 + 整体平均的产物可直接由 `records.jsonl` 复算，避免"只报平均不可审计"；
- 环境采集覆盖 oMLX 社区榜上传所需的核心本地参数（硬件/版本/模型设置/feature flags）；
- 运行时开销可控（`--samples 5` 分钟级；全样本小时级，可拆分/限流）。
