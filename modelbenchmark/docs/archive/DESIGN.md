# 本地模型基准测试系统设计（oMLX Benchmark Harness）

> **历史归档**：本文记录重构前的设计/审计，目录、命令与路径已过时，仅用于追溯；
> 当前设计见 [../ARCHITECTURE.md](../design/30-architecture.md)，测试方案见 [../TEST_PLAN.md](../spec/21-test-plan.md)。

> 版本：v0.1（2026-11-02）
>
> **现状说明**：本文是最初的方案设计（历史文档），其中的目录/命令已过时，仅保留设计推导与取舍背景。
> **当前实现架构以 [30-architecture.md](../design/30-architecture.md) 为准**；操作说明见
> [41-user-guide.md](../guides/41-user-guide.md)，测试方案见 [21-test-plan.md](../spec/21-test-plan.md)。
> 目标平台：macOS (Apple Silicon) + oMLX 本地推理服务（OpenAI 兼容 API）
> 项目路径：`/Users/huawei/Desktop/模型管家相关资料/modelbenchmark`

---

## 1. 背景与目标

需要一套可长期使用的本地基准测试系统，对 oMLX 中运行的不同模型，在**固定测试场景 × 固定上下文长度**下同时采集：

- **性能指标**：TTFT（首 token 延迟）、输出速度（tok/s）、峰值 CPU 占用、峰值内存、端到端时延；
- **准确性指标**：可判分任务的正确率（及置信区间）；
- **配置快照**：本机硬件配置、模型配置、oMLX 服务配置；
- **原始证据**：每个样本的完整 prompt、模型原始输出、判分明细。

结果按 `模型 × 场景 × 上下文长度` 组织，输出 JSON，便于跨模型、跨轮次对比。

### 非目标

- 不做模型训练/量化评估；不评估视觉、语音模态（当前只走文本 chat completions）。
- 不做多并发吞吐压测（本系统定位单请求时延/速率基线；并发吞吐可作为后续扩展）。
- 不替代模型自身的推理引擎内部 profiling（以客户端黑盒测量为准，服务端 usage 作为交叉验证）。

---

## 2. 业界调研（结论先行）

| 框架 | 定位 | 性能指标 | 准确性 | 对本项目的参考价值 |
|---|---|---|---|---|
| EleutherAI/lm-evaluation-harness | 学术评测事实标准 | 无 | 强（任务适配器架构） | 借鉴 `task/adapter` 扩展模式与可复现 manifest |
| open-compass/OpenCompass | 中文评测平台 | 部分 | 强 | 借鉴数据集/评测器解耦 |
| modelscope/EvalScope | 评测 + 性能压测（OpenAI 兼容） | 强（TTFT/TPOT/吞吐） | 中 | 指标定义最接近本项目；证明"性能+准确性"一体化可行 |
| huggingface/lighteval | 轻量评测 | 无 | 强 | 借鉴配置驱动 |
| vllm benchmark_serving | 服务端压测 | 强 | 无 | 指标定义参考，但绑定 vLLM |
| mlx-lm | MLX 推理脚本 | 有简易 benchmark | 无 | 参考 MLX 侧测量方式 |
| **oMLX 内置 benchmark**（`/Applications/oMLX.app/.../omlx/admin/benchmark.py`） | 进程内基准 | 强：`ttft_ms/tpot_ms/gen_tps/processing_tps/e2e_latency_s/total_throughput/peak_memory` | 内置 MMLU/GSM8K 等 | **指标定义直接对齐**，便于交叉验证；但为进程内测量、不测 CPU、不测外部数据集判分 |

**结论**：现成框架都无法同时满足"oMLX 端到端 HTTP 黑盒 + 8 类场景可插拔数据集 + 性能与准确性同 run 记录 + CPU/缓存治理"。本项目自建轻量 harness，指标定义对齐 EvalScope / oMLX 内置 benchmark（TTFT、TPOT、TPS、prefill TPS），并补上 CPU、缓存治理与判分可插拔三项关键能力。矩阵覆盖 8 个准确性场景 + 1 个性能探针场景。

---

## 3. 总体架构

```
┌──────────────────────────── CLI (bench/cli.py) ────────────────────────────┐
│  download │ run │ compare │ validate │ list-scenarios │ hardware          │
└───────┬──────────────┬──────────────┬──────────────┬──────────────┬────────┘
        │              │              │              │              │
   config loader   orchestrator    report writer   evaluators    system probes
   (settings.py)   (runner/)       (report/)       (evaluators/)  (system/)
        │              │              │              │              │
        │        ┌─────┴──────┐       │              │              │
        │        │  scenarios │───────┼──────────────┘              │
        │        │ (prompt +  │       │                             │
        │        │  padding)  │       │                             │
        │        └─────┬──────┘       │                             │
        │              │              │                             │
        │        ┌─────┴──────┐       │                             │
        │        │  datasets  │       │                             │
        │        │ (adapters) │       │                             │
        │        └────────────┘       │                             │
        │                             │                             │
        └──────────────► client/omlx.py ◄──── system/cache.py ◄────┘
                    (HTTP + SSE + admin API)   (缓存清理/模型装载)
```

**分层原则**

1. **数据层**：`datasets/` 只负责把外部数据转成统一 `Sample`；不关心 prompt 模板与判分。
2. **场景层**：`scenarios/` 负责 prompt 构造、上下文对齐、判分器选择；只依赖 `Sample` 与判分接口。
3. **判分层**：`evaluators/` 纯函数，无 IO，便于单测与复用。
4. **执行层**：`runner/` 负责矩阵展开、缓存治理、进程采样、重复轮次、统计聚合；不依赖具体数据集。
5. **客户端层**：`client/` 封装 oMLX 推理 API + 管理 API（鉴权、缓存清理、模型装载），是唯一与服务器通信的模块。
6. **报告层**：`report/` 负责 run bundle 落盘与跨 run 对比。

---

## 4. 核心数据模型（`bench/schema.py`）

| 模型 | 作用 |
|---|---|
| `Sample` | 数据集样本的统一表示：`sample_id/dataset/scenario/subtask/context/question/choices/answer/answer_type/metadata` |
| `ChatMessage` | 角色 + 内容（system/user/assistant/tool） |
| `RequestSpec` | 实际发送给模型的请求：messages、max_tokens、temperature、seed、extra |
| `RawResponse` | 原始响应：text、reasoning_text、tool_calls、finish_reason、usage、首 token 时间戳序列摘要、耗时 |
| `PerfMetrics` | `ttft_ms / ttft_visible_ms / tpot_ms / gen_tps / prefill_tps / e2e_ms / peak_cpu_pct / mean_cpu_pct / peak_rss_mb / prompt_tokens / completion_tokens / cached_tokens` 等 |
| `Score` | `correct/score/detail`（判分明细保留） |
| `SampleRecord` | 单次请求全量证据：run/model/scenario/context_length/sample/repeat/phase/request/response/metrics/score/cache_state/status |
| `CellAggregate` | 一个 `model×scenario×context_length` 单元的聚合：样本数、重复数、性能统计（mean/std/median/p95/min/max）、准确率 + Wilson 置信区间 |
| `RunManifest` | run 元信息：配置快照（内嵌 JSON）、硬件、模型配置、oMLX 版本、时间、`git_commit`/`git_dirty`（仓库在 git 下时）、矩阵定义 |

所有落盘 JSON 均带 `schema_version`，保证后续升级可迁移。

---

## 5. 场景与数据集映射

| 场景 | 数据集（本地） | 判分方式 | 备注 |
|---|---|---|---|
| `knowledge_qa` 知识问答 | MMLU（test，全学科子集） | 选项字母精确匹配 | 英文知识基准 |
| `instruction_following` 指令遵循 | IFEval | 可验证约束检查（子集实现） | prompt 级 + 指令级严格判分 |
| `long_context` 长上下文 | **LongBench-v2**（503 题，6 大类） | 4 选 1 字母匹配 | 官方格式，多文档/长对话/代码库等 |
| `math_logic` 数学与逻辑 | GSM8K（+可选自建逻辑题） | 最终数值精确匹配 | 抽 `####` 后数值，容差比较 |
| `coding` 编程 | HumanEval | 单元测试执行（子进程沙箱 + 超时） | pass@1 |
| `general_agent` 通用 Agent | BFCL v4（simple/multiple/parallel/multi_turn） | 工具调用结构 + 参数匹配 | AST 级参数比较，非文本相似 |
| `search_agent` 搜索 Agent | HotpotQA distractor（本地检索工具） | 最终答案 EM/F1 + 工具调用正确性 | 确定性本地检索，可复现 |
| `multilingual` 多语言 | MMMLU（中文/日/韩/法/德/西）+ MMLU(en) | 选项字母精确匹配 | 跨语言同构任务 |
| `custom`（任意自定义样本） | 用户任意 JSONL | 由 `answer_type` 决定 | `bench/datasets/custom.py`，无需改代码 |
| `throughput` 吞吐探针 | `synthetic`（本地生成，无需下载） | 不判分（`score=None`） | **性能专用**：要求长文输出，保证 `completion_tokens ≥ 16`，使 decode tok/s、TPOT 有效 |

> 8 个准确性场景之外额外保留 1 个**性能探针场景** `throughput`：准确性场景的答案往往只有
> 1–3 个 token，`gen_tps` 会被"样本数不足"判为无效；探针在同一 `模型 × 上下文长度`
> 网格上要求 ≥400 字长文，专门产出可解释的输出速度与峰值 CPU。其 cell 的 `accuracy`
> 恒为 `null`，不参与准确性对比。

### 思考模式（thinking）

思考型模型（如 MiniCPM5 系列）默认会先生成大量 `reasoning_content`。实测
MiniCPM5-1B 在 MMLU 题目上 1024 个 token 仍全部用于思考、`content` 为空
（`finish_reason=length`），若直接按 16 token 上限测，所有准确性都会变成 0。

因此 runner 统一发送 `chat_template_kwargs={"enable_thinking": false}`
（oMLX 0.7.3 已验证支持），使不同模型在"直接作答"模式下可比：

- `run.thinking_mode: disabled`（默认）/ `enabled` / `auto`（保留模型默认）；
- 场景可用 `defaults.thinking_mode` 覆盖；CLI `--thinking-mode` 可临时切换；
- `disabled` 时 TTFT 即首个答案 token 延迟；`enabled` 时同时记录
  `ttft_visible_ms`（首个 `content` token），两者都在 `records.jsonl` 中；
- 模式写入 `manifest.json` 与每条 `request.chat_template_kwargs`，跨 run 可审计。

数据集统一下载到 `data/raw/<dataset>/`，规范化后的 JSONL 在 `data/processed/`。下载脚本幂等、支持断点续传、记录 SHA-256。

---

## 6. 上下文长度构造（1K/4K/8K/16K/32K/64K）

定义：**目标上下文长度 = 输入 prompt 的 token 数**（含 system/user 模板，不含输出）。用服务端 `/v1/tokenize`（使用模型自身 tokenizer + chat template）做精确计数，误差控制在目标值 ±2% 内。

- **长上下文数据集（LongBench-v2）**：样本原生长度可能远超 64K（中位数约 104K token）。
  - 策略：`truncate`。按 `head_ratio`（默认 0.7）保留头部 + 尾部，保证问题出现在上下文之后；
  - 若样本原生长度 < 目标长度：用中性填充补足（见下）；
  - 同一批样本在所有上下文长度上复用 → **配对设计**，可观察"上下文变长/截断导致准确率衰减"。
- **短样本数据集（MMLU/GSM8K/IFEval/HumanEval/MMMLU/BFCL）**：`pad`。用确定性、领域中性的填充文本（本地内置 `filler.txt`，按 token 循环拼接）补足到目标 token 数，问题放在最后。填充内容对判分无影响，但会稀释注意力，属于业界常见的"上下文压力"测法。
- 每个样本记录：`target_context_tokens`、`actual_prompt_tokens`、`truncated`、`padded`、`truncation_ratio`，保证可审计。

---

## 7. 性能测量方法（客户端黑盒）

一次测量 = 一次流式 `POST /v1/chat/completions`（`stream=true, stream_options.include_usage=true`，`temperature=0`），客户端按 SSE 逐 chunk 打时间戳。

| 指标 | 定义 | 说明 |
|---|---|---|
| `ttft_ms` | 从发出请求到**第一个含 token 的 delta**（content 或 reasoning_content） | 与 oMLX `ttft_ms` 定义对齐 |
| `ttft_visible_ms` | 到**第一个可见 content**（排除 reasoning） | 思考型模型专属观测 |
| `tpot_ms` | `(末 token 时间 - 首 token 时间) / (completion_tokens - 1)` | 与 oMLX 一致；`completion_tokens < 16` 时置 null（不可信） |
| `gen_tps` | `completion_tokens / 解码窗口` | 输出 token 速度（tok/s） |
| `prefill_tps` | `prompt_tokens / ttft_s` | 预填充速度（tok/s） |
| `e2e_ms` | 请求发出到流结束 | |
| `peak_cpu_pct` | 采样窗口内服务进程树 CPU% 的峰值（单核=100%，12 核理论峰值 1200%） | psutil 按 50ms 采样进程树 |
| `peak_rss_mb` | 服务进程树 RSS 峰值 | 辅助指标 |
| `server_*` | 服务端 usage 中的 `time_to_first_token`、`generation_tokens_per_second` 等 | 交叉验证，不参与主判定 |

**采样治理**

- CPU 采样线程只在请求窗口内工作，缓存清理/模型加载窗口不计入；
- 采样间隔默认 50ms，记录采样点数与窗口长度，避免"短请求采样不足"造成假峰值；
- 服务端 PID 自动发现：配置优先 `server.pid_file`，否则通过 `lsof -nP -iTCP:<port> -sTCP:LISTEN` 获取（macOS 可用；`ps`/`top` 在受限沙箱不可用）；
- 同时记录服务进程树子进程（oMLX 可能有 helper 子进程）。

---

## 8. 缓存与资源治理（关键设计）

oMLX 具备 **prefix cache（前缀 KV 缓存）+ 热缓存 + SSD 缓存**。实测（本机 MiniCPM5-1B，10.8K token prompt）：

| 状态 | cached_tokens | TTFT |
|---|---|---|
| 缓存命中（未清理） | 10752 | **0.19 s** |
| 清理 hot+SSD 缓存后 | 0 | **4.73 s** |

差异约 **25 倍**。若不做缓存治理，TTFT/吞吐结果会被缓存命中严重污染，且不同模型/不同 run 之间不可比。

因此：

1. **默认 `cold` 模式**：每个 measured 请求前调用管理 API 清理 `hot-cache` + `ssd-cache`，并等待服务稳定；请求携带 `X-oMLX-Cache-Store: off`（oMLX 官方头，避免本系统自己的请求写入 prefix cache）；响应中 `cached_tokens > 0` 视为污染 → 重试一次，仍污染则标记 `contaminated` 并从聚合中剔除。
2. **可选 `warm` 模式**：每个 cell 清理一次后连续重复，专门测量"缓存复用后的稳态性能"，两种模式分开报告，绝不混算。
3. **模型切换隔离**：`isolate_models: true` 时在切换模型前 `POST /v1/models/{id}/unload` 卸载全部已加载模型，避免 24GB 内存下大模型互相挤占；每个模型首个请求作为 warmup 丢弃（含 tokenizer/内核编译预热）。
4. **运行后恢复**：run 结束清理缓存、按配置卸载模型，保证不污染用户后续使用。
5. **配置指纹**：manifest 记录 oMLX `settings.json` 关键项（scheduler/cache/memory guard）与服务版本，保证可比性。

---

## 9. 可信度设计

1. **多轮重复**：每个 cell 的每个样本默认重复 `repeats=3`（可配置），warmup 不计入；报告 mean/median/std/min/max/p95，标准差过大（CV>20%）的 cell 标记 `unstable`。
2. **固定环境**：temperature=0、固定 `max_tokens`、固定 prompt（含 seed 留空说明）；记录硬件、电源模式不可控项、后台负载（可选的空闲检查）。
3. **配对设计**：不同上下文长度/不同模型使用同一批 `sample_id`，准确率可按配对样本做差异分析；报告 Wilson 95% 置信区间。
4. **失败与超时策略**：单请求超时/服务错误 → 重试（默认 2 次，指数退避）；连续失败标记 `error` 并保留现场，不静默丢弃。
5. **可复现**：run bundle 保存 `manifest.json`（内嵌 `config_snapshot`）、git commit（仓库在 git 下时记录）、依赖版本、每个样本的完整 prompt 与原始响应。
6. **一轮 run 一个目录**，禁止覆盖；对比命令只读多个 run。

---

## 10. 扩展点（满足"易扩展"）

| 扩展需求 | 扩展方式 |
|---|---|
| 新增数据集 | 继承 `DatasetAdapter`（`load()` + 可选 `download()`），在 `datasets/registry.py` 注册 |
| 新增场景/判分 | 继承 `ScenarioAdapter`（`build_messages()` + `score()`），在 `scenarios/registry.py` 注册；判分纯函数放 `evaluators/` |
| 新增指标 | `runner/metrics.py` 聚合函数 + `PerfMetrics` 字段 |
| 新增被测服务 | 实现 `client/base.py` 的协议（当前 oMLX 客户端即 OpenAI 兼容实现） |
| 新增矩阵 | 新增 `configs/matrix/*.yaml`，无需改代码 |

---

## 11. 输出格式（run bundle）

```
results/<run_id>/
├── manifest.json          # 配置快照 + 硬件 + 模型配置 + 环境 + 矩阵
├── hardware.json          # 硬件/OS/oMLX/服务配置
├── models.json            # /v1/models/status 快照 + 使用的模型参数
├── records.jsonl          # 逐请求全量记录（性能+判分+缓存状态）
├── summary.json           # 按 model×scenario×context_length 聚合（核心对比文件）
├── summary.csv            # 扁平化表格（便于 Excel/可视化）
├── report.md              # 人读摘要（最佳/最差、缓存污染数、失败数）
└── raw/
    └── <model>/<scenario>/<context>k/<sample_id>/r<repeat>.json  # 原始样本+原始响应
```

`summary.json` 结构（约定）：

```json
{
  "schema_version": "1.0",
  "run_id": "20261102T193000Z",
  "cells": [
    {
      "model": "Qwen3.5-9B-MLX-4bit",
      "scenario": "long_context",
      "context_length": 8192,
      "dataset": "longbench_v2",
      "n_samples": 5, "n_repeats": 3, "n_valid": 15, "n_contaminated": 0, "n_error": 0,
      "accuracy": 0.4, "accuracy_ci95": [0.18, 0.65],
      "perf": {
        "ttft_ms": {"mean": 1234.5, "median": 1200.1, "std": 55.2, "min": 1180.0, "max": 1300.0, "p95": 1290.0},
        "gen_tps": {...}, "tpot_ms": {...}, "peak_cpu_pct": {...}, "peak_rss_mb": {...}
      }
    }
  ]
}
```

---

## 12. 目录结构

```
modelbenchmark/
├── README.md
├── pyproject.toml
├── configs/
│   ├── default.yaml        # 服务地址/密钥/输出/超时/缓存策略
│   ├── models.yaml         # 模型清单（alias + 说明）
│   ├── scenarios.yaml      # 场景→数据集→判分器→样本数→max_tokens
│   └── matrix/{smoke,standard,full}.yaml
├── bench/
│   ├── cli.py
│   ├── schema.py
│   ├── settings.py
│   ├── client/omlx.py
│   ├── system/{hardware,sampler,cache}.py
│   ├── datasets/{base,registry,longbench_v2,mmlu,gsm8k,ifeval,humaneval,mmmlu,hotpotqa,bfcl}.py
│   ├── scenarios/{base,registry,padding,knowledge,instruction,long_context,math,code,agent,search,multilingual}.py
│   ├── evaluators/{base,mc,math,code,instruction,agent,search}.py
│   ├── runner/{orchestrator,metrics}.py
│   └── report/{writer,compare}.py
├── data/{raw,processed}    # 下载数据（gitignore）
├── results/                # 运行结果（gitignore）
├── scripts/download_datasets.py
├── tests/
└── docs/{DESIGN,METHODOLOGY,DATASETS}.md
```

---

## 13. 可行性评估（详见 `docs/design/34-feasibility.md`）

- oMLX HTTP API 已实测支持：流式 usage、`/v1/tokenize` 精确计数、`/v1/models/status` 模型配置、admin 登录、hot/SSD 缓存清理、模型 load/unload。**方案所需接口全部具备**。
- 缓存污染问题已实测复现（25× TTFT 差异），治理接口有效 → 设计成立且必要。
- 主要风险：64K 上下文预填充耗时（实测 1B 模型 10.8K 约 4.7s，64K 预计 25-40s/次）→ 通过分级 repeat 策略控制总时长；大模型（16.8GB）内存压力 → 通过模型隔离与内存监控缓解；代码判分需执行模型生成代码 → 子进程 + 超时 + 无网络沙箱。
- 结论：**方案可行**；默认配置采用 smoke（快速自检）+ standard（正式对比）两档矩阵。

---

## 14. 验收标准

1. `python -m pytest` 全绿（判分/填充/聚合/配置/客户端 mock）。
2. `./modelbench download all` 完整下载 LongBench-v2 并校验 SHA-256，其他数据集可选。
3. `./modelbench validate` 离线自检通过（配置、数据、服务连通性）。
4. `./modelbench run --matrix configs/matrix/smoke.yaml` 在真实 oMLX 上产出完整 run bundle，字段齐全。
5. `modelbenchmark compare` 可对两个 run 生成对比矩阵。
