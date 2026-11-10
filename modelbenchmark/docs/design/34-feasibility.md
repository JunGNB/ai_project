# 可行性评估与实测证据（FEASIBILITY）

> **状态**：current ｜ **版本**：v2.0（2026-11-10）｜ **适用**：`modelbench` 0.3.x
> （bundle schema `3.0` / 平铺 `output/runs/<run_id>/`）｜ **读者**：维护者、评审者
> ｜ **范围**：本机（macOS 15.7.7 / Apple M4 Pro / 24GB / oMLX 0.7.3.dev2）实测的
> 可行性证据与运行时预算。方案设计见 [21-test-plan.md](../spec/21-test-plan.md)，验证方法见
> [process/50-validation-plan.md](../process/50-validation-plan.md)，历史推导见
> [archive/DESIGN.md](../archive/DESIGN.md)（已归档，命令/路径过时）。

## 1. 结论

**方案可行，且已落地。** 所需 oMLX 接口全部实测可用；缓存污染这一最大可信度风险已被量化
并有治理手段；逐样本证据可离线复算；运行时预算可控（`--samples 5` 分钟级，全量小时级）。
本文只记录"能不能做、为什么可信、要跑多久/占多少盘"；逐项验证判据见 V&V 计划。

## 2. oMLX 接口能力实测（方案所需能力全部具备）

| 能力 | 端点 | 实测结果 |
|---|---|---|
| 服务健康 | `GET /health` | 200，无需鉴权 |
| 模型清单 | `GET /v1/models` | 11 个模型（与 `models.yaml` 同步），含 `max_model_len` |
| 模型运行态 | `GET /v1/models/status` | 每个模型 `max_context_window` / `max_tokens` / `engine_type` / `estimated_size` / `loaded` |
| 目标模型 token 计数 | `POST /v1/tokenize` | 支持 `messages`/`prompt`（含 tools），返回 `count` + `max_model_len`；用于实际输入大小与窗口判定 |
| 流式对话 | `POST /v1/chat/completions` | SSE；`stream_options.include_usage` 返回 `prompt_tokens/completion_tokens/cached_tokens/time_to_first_token/...` |
| 缓存清理 | `POST /admin/api/hot-cache/clear`、`/admin/api/ssd-cache/clear` | 登录后 200；返回 `total_cleared/bytes_reclaimed` |
| 模型装载/卸载 | `POST /v1/models/{id}/load` / `unload` | 200/202；`/v1/models/status` 的 `loaded` 可轮询 |
| 设备信息 | `GET /admin/api/device-info` | `chip_name=M4, chip_variant=Pro, memory_gb=24, gpu_cores=16` |
| 服务配置 | `GET /admin/api/global-settings` | cache/scheduler/memory/sampling 等配置（落盘前脱敏） |
| 模型设置 | `GET /admin/api/models` + `PUT /admin/api/models/{id}/settings` | 每个模型含实际生效 `settings` 与 `moe_expert_offload_supported` |

鉴权：`Authorization: Bearer <key>` 或 `X-API-Key`；admin 用 `POST /admin/api/login` 换 cookie。
密钥来源：`OMLX_API_KEY` 环境变量，或回落到 `~/.omlx/settings.json` 的 `auth.api_key`。

## 3. 关键实测证据

### 3.1 缓存会让 TTFT 相差约 25 倍（因此必须治理）

MiniCPM5-1B，同一 10,752 token prompt：

| 状态 | `cached_tokens` | TTFT |
|---|---|---|
| 缓存命中 | 10,752 | **0.19 s** |
| 清理 hot+SSD 缓存后 | 0 | **4.73 s** |

不治理缓存 = 测缓存而非测模型；默认 `cold` 模式逐请求清理并校验 `cached_tokens==0`。

### 3.2 CPU/内存采样可行

- `/bin/ps`、`/usr/bin/top` 在受限沙箱返回 `Operation not permitted`；
- `psutil` 读 `/proc`/`libproc` 正常：oMLX 服务进程树在 1B 模型生成时峰值约 **110%**（≈1.1 核），空闲约 1.6%；
- 服务端 PID：`lsof -nP -iTCP:8000 -sTCP:LISTEN -t` → 可用（psutil 的 `net_connections` 被沙箱拒绝）。

### 3.3 数据与网络约束

- LongBench-v2：`hf-mirror.com/datasets/zai-org/LongBench-v2`（465MB，503 题）可下载；
  MMLU / GSM8K / IFEval / HumanEval / MMMLU / HotpotQA / BFCL v4 镜像可下载。
- 当前运行时只读预构造的 `dataset/test/performance/testset.jsonl`，不在 run 期下载/解析上游数据；
  预构造样本超窗时**跳过（not-testable）而不是截断**。
- huggingface.co 直连不可用、PyPI 直连不可用（清华/阿里/中科大镜像可用）；GitHub raw 可用（偶发慢）。

## 4. 运行时预算（决定默认样本数）

默认测试集（`performance`）固定为 8 场景 / **240 measured 请求 / 24 cells**
（1K/8K/32K 各 80 条），目标模型窗口不足时 32K 行转 `not-testable`（见 TEST_PLAN §8 第 2 条）。
样本与 token 预算的权威来源是 [dataset/README.md](../../dataset/README.md) 与
`dataset/test/performance/manifest.json`。

按历史实测外推（MiniCPM5-1B；冷缓存；OSL=50 的当前默认口径）：

| 模式 | measured 请求/模型 | 1B 粗估 | 9B 粗估 |
|---|---:|---:|---:|
| 快速冒烟（2 个 1K 场景 × 1 样本） | 2 | 约 1 分钟 | 2–5 分钟 |
| `--samples 5`（8 场景） | 40 | 约 20–40 分钟 | 1–2 小时 |
| 默认全量 | 240 | 约 45 分钟–1.5 小时 | 3–6 小时 |

> 2026-11-06 外部真实 run（MiniCPM5-2B，240 请求）实测 32K 单样本 TTFT ≈ 58–60 s、
> 1K ≈ 0.87 s，据此 2B 模型跑完默认全量约 1.5–2 小时（32K 档占主要成本）。

**磁盘（schema 3.0 平铺布局，实测）**：单 run 目录固定 7 个条目，240 样本约
**0.95 MB/run**（prompt 正文外置到 `blobs/prompts/`，全局一份 ≈13.8 MB / 240 条）；
8 个 run + 批次元数据 + 索引 ≈ **22 MB**（不含按需 `exports/` 与 `archive/`）。
历史 2.x 的 `raw/` 副本已删除，故不再有"200 MB–1 GB/run"的量级。

## 5. 自校验与验证入口

开发期/变更后的验证**不在本文重复**，统一见
[process/50-validation-plan.md](../process/50-validation-plan.md)：门禁 `harness.gate`（10 项默认检查）、
`pytest test/`、`./modelbench verify`、真实 smoke（`harness.gate --live`）、
以及"由独立 verifier 只读复核"的角色约定（`harness/prompts/verifier.md`）。
本文的本地实测数字（缓存 25×、CPU/内存量级、token 计数可用性）是那份验证计划的**输入证据**。

## 6. 结论与遗留

- 所需 oMLX 接口全部可用；缓存、内存、token 计数三条关键链路已实测；
- 剩余风险：oMLX 版本升级会改变 settings/usage/缓存端点（需按 V&V §再验证触发重跑）、
  24GB 机器装大模型需 MoE offload、长上下文预填充耗时以 32K 档为主；
- 本机实测环境：macOS 15.7.7 / M4 Pro 24GB / oMLX 0.7.3.dev2；换机的结论需重新采集。
