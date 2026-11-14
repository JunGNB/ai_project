# 测试样本契约与测试系统职责（跨设备/跨系统移植）

> **状态**：current ｜ **版本**：v1.0（2026-11-10）｜ **适用**：`modelbench` 0.3.x
> **读者**：维护者、测试执行者、需要复用同一套样本的其他系统 ｜ **范围**：把**同一套固定样本**
> 交给不同机器/后端/测试系统时，"哪些属于样本契约、哪些属于测试系统职责、比较准入条件是什么"。
> 样本集设计见 [11-performance-testset.md](11-performance-testset.md)（性能）与
> [12-accuracy-testset.md](12-accuracy-testset.md)（精度）；指标口径见
> [20-methodology.md](20-methodology.md)；输出契约见
> [../design/31-output-contract.md](../design/31-output-contract.md)。

## 1. 一句话结论

**同一套样本设计是唯一稳定的输入契约**；模型、tokenizer、后端、机器可以不同，但必须：
同一设计指纹（target 集合 / 抽样 seed / 构造算法 / manifest `rows_sha256`）、同一目标长度
（1,024 / 8,192 / 32,768，**容差 0**）、同一测量协议参数，并在运行前用**目标模型 tokenizer**
重建/校验对齐（`align.py` 报告随结果留存）。因为长度是对齐到目标模型的，**最终 `testset.jsonl`
的 SHA-256 允许逐模型不同**——它是"该模型上的对齐产物"，不是跨模型契约；跨模型契约是设计指纹。
精度方法另需**同一 judge 契约版本**与同一判题协议（§4.2）。

两种方法（`method=performance` / `method=accuracy`）是**两套准入条件**：性能看设计指纹 + 对齐报告 +
TTFT/TPS 协议，精度看 testset SHA + 判题契约 + 解码协议；两者之间不做横向合并。

## 2. 样本侧（设计契约，一次定义、逐模型对齐）

固定一套性能样本设计（默认 `dataset/test/performance/`，240 行 = 8 场景 × 1K/8K/32K × 10），
**所有设备/模型/后端复用同一设计，长度按目标模型对齐**：

- **不做 per-model 抽样**：target 集合、单元流、任务块、抽样 seed 由 manifest 固定；换模型只改变
  "前缀长度"，不换样本。
- **不做运行时裁剪/补齐**：对齐发生在**运行前**（`align.py` 重建同一设计并逐条校验），runner 原样发送。
- **32K target 固定 32,768，无特殊处理**：1K/8K/32K 是对称的三种 bucket，没有"32K 特殊构造"。
- **不含性能判分**：性能 testset 不携带 gold/score；精度 testset 自带判题契约（`judge` 字段），
  只用于 `method=accuracy` 的独立 run，**不接入性能 run**（见
  [12-accuracy-testset.md](12-accuracy-testset.md) 与
  [22-accuracy-test-plan.md](22-accuracy-test-plan.md)）。二者共享的只是"固定设计"原则。
- **字段契约**（性能 testset 单行，详见 [11-performance-testset.md](11-performance-testset.md) §6.2）：
  `id`、`scenario`、`bucket`（1k/8k/32k）、`target_id`、`dataset`、`tokens`（**目标 prompt token**，
  含 chat template；构建期用通用 cl100k 参考 tokenizer 定稿，生产 run 由 `align.py` 按目标模型重建，
  见 [11-performance-testset.md](11-performance-testset.md) §4）、`messages`、`tools`、
  `constructed`、`meta`（`lang`、`content_tokens`、`template_reserve_tokens`、`task_sha256`、
  `body_sha256` 等）。
- **manifest/SHA**：`dataset/test/<name>/manifest.json` 记录**设计指纹**（seed、targets、
  每档 `rows_sha256`、构造方法、`construction.tolerance_tokens=0`）与**对齐信息**
  （`tokenizer{path,sha256,vocab_size,role}`、`alignment{tolerance_tokens:0, verified_by:align.py}`）；
  每次 run 的最终 `testset.jsonl` SHA-256 与对齐报告（`dataset/.cache/performance/alignment-*.json`）
  随结果留存。仓库提交 `testset.yaml` 与 `manifest.json`，`testset.jsonl` 由 `build.py`（参考基线）
  或 `align.py`（目标模型对齐产物）确定性生成（seed 固定、无时间戳）。
- **跨设备"样本同一"的定义**：**同一设计指纹 + 同一目标长度 + 各自的对齐报告（tokenizer SHA 与
  校验结果）**。最终文件 SHA 可以不同，这是严格对齐的必然结果。

## 3. 系统侧（测试系统职责，各系统可不同实现）

同一套样本在不同测试系统上执行时，以下由**测试系统**负责，且必须显式记录：

| 职责 | 要求 | 记录位置 |
|---|---|---|
| 运行前对齐 | 用目标模型 tokenizer 按同一设计重建并逐条校验 `prompt_tokens == bucket`（0 容差）；报告含 tokenizer SHA-256、探针 reserve、每轮 attempts | `dataset/.cache/performance/alignment-<slug>.json` |
| actual token 计数 | 请求前用**目标模型** `POST /v1/tokenize` 对最终 messages 计数；strict 集必须等于 bucket | `request.tokenized_prompt_tokens`、`context_meta.isl_tolerance{target,low,high,actual,ok}` |
| 模型窗口判定 | 用 actual + `max_tokens` + `context_safety_margin` 对比模型窗口；**不足才 skip**（32K 需窗口 ≥32,946） | `run.json.outcome.skipped`、`context_meta.window_check_basis` |
| 冷缓存 | measured 请求前清 hot+SSD 并校验 `cached_tokens==0`（fail-closed） | `records[].cache` |
| 生成长度/decode guard | strict 集 `max_tokens=50`、**不追加 decode guard**；非 strict 集 `perf.max_tokens` + `perf.decode_guard` | `run.json.config.perf`、`records[].request.context_meta.decode_guard_applied` |
| warmup/repeats | 模型级 `warmup_per_model=1`、per-cell `warmup_requests=0`、`repeats=1`（默认） | `run.json.config.run` |
| 指标定义 | TTFT/TG TPS/TPOT/峰值内存/输入上下文等按 [20-methodology.md](20-methodology.md) | `records[].metrics`、`aggregates.json` |
| 聚合口径 | strict 集只用 `tokenized == bucket` 的 measured 样本，报 `n_measured/n_skipped`，无入带比例；非 strict 集按 target ±10% 入带 + 全样本双口径 | `aggregates.*.n_measured/n_skipped`（strict）、`n_isl_in_band/n_isl_out_of_band`（非 strict） |
| records/schema | 逐样本权威记录、schema 3.0、prompt 内容寻址 | `records.jsonl`、`run.json`、`MANIFEST.json` |

## 4. 跨系统/跨设备比较准入（硬条件）

### 4.1 性能（`method=performance`）

在引用/比较任何性能结果前，必须同时满足：

1. **同一设计指纹**：`manifest.json` 的 seed / target 集合 / 每档 `rows_sha256` / 构造方法一致
   （"同一套设计"的凭据）；最终 `testset.jsonl` SHA 允许逐模型不同，但必须能追溯到同一设计指纹；
2. **同一目标长度、0 容差**：1K=1,024、8K=8,192、32K=32,768，运行期 `tokenized_prompt_tokens` 必须
   精确相等（`align.py` 重建 + runner 复核，见 [11-performance-testset.md](11-performance-testset.md) §4），
   不含模型特定改造；
3. **同一对齐与协议参数**：对齐报告（目标 tokenizer SHA-256、reserve、校验通过）随结果留存；
   `max_tokens=50`、strict 集不追加 decode guard、缓存模式、warmup、repeats、采样/温度、流式口径一致；
4. **样本覆盖必须报告**：strict 集报 `n_measured`/`n_skipped`（skipped 带 expected/got）；
   非 strict 集报 `tokenized_prompt_tokens` 的 per-bucket 分布、`n_isl_in_band`/`n_isl_out_of_band` 与
   `insufficient_samples` 标记；
5. **比较范围**：
   - **跨模型（同设计、不同 tokenizer）**：可声明"同一套设计、长度已按各自 tokenizer 精确对齐到
     1K/8K/32K"，因此 TTFT/内存的差异可归因于模型/后端本身；仍不得把结果说成"官方榜同口径"。
   - **同模型跨设备/后端**：同一 tokenizer 下对齐到同一长度，可直接比较；
6. **不满足以上任一条件**：结果只能作为"该次运行"的证据，不能横向合并/平均/排名。

### 4.2 精度（`method=accuracy`）

精度比较（bf16 vs 量化、跨设备/跨系统）在引用任何 Δ/分数前，必须同时满足：

1. **同一 testset**：两次 run 的 `testset.jsonl` SHA-256 相同，且与
   `dataset/test/accuracy/manifest.json` 的 `files.testset.jsonl.sha256` 一致；manifest 的
   `status=wired` / `runner=accuracy` / `hidden=false`；
2. **同一判题契约版本**：`batch.json.inputs.testset.judge_contract_version`（由 manifest 的
   `judges` 声明派生，形如 `judges-<sha256 前 16 位>`）相同；判题族/算法/选项归一化任一变化都会
   改变该版本，等价于发布新口径；
3. **同一判题协议**：缓存 `cold`、`temperature=0`、`top_p=1`、**无 decode guard**、
   `repeats=1`、传输错误重试 1 次、生成长度取逐行 `judge.max_new_tokens`、同一
   `thinking_mode`（默认 `disabled`）；协议记录在 `run.json.config`，不一致即不可比；
4. **样本不齐显式报错**：配对按 `(scenario, sample_id)`；只在一侧存在（`unpaired`）或两侧未同时
   判分（`unscored`）的样本必须列出并使比较显式失败/带警告，**不得静默丢弃**；判题族不一致
   同样报错（`analyze --paired --method accuracy` 的行为）；
5. **Δ 口径**：场景 `Δ(pp) = 100 × (score_bf16 − score_quant)`；0/1 场景用精确 McNemar 检验，
   F1 场景用配对 bootstrap 95% CI；8 场景同时检验用 Holm 校正。`|Δ|` 还必须超过**噪声基线**
   （同一 testset、同一协议的 bf16 独立重复 run；`analyze --noise-baseline-pp` 可显式给出阈值）
   才可标记"显著退化"，否则只报点估计与 CI；
6. **随结果报告**：`n_scored`/`n_correct`/`n_judge_error`/`n_format_failure`/`n_skipped(over_window)`
   与各场景 95% CI。`--samples N` 的冒烟 run（每场景 <250）**不得**与全量 run 比较分数；
7. **不可与性能横比**：精度分数不与 TTFT/TPS/内存相乘或加权，也不与性能 run 合并。

> 不满足以上任一条件：只能作为该次运行证据，不能横向合并/排名；精度分数也不得与公开榜
> （MMLU/HumanEval 等）横比（提示词/抽样/判题实现不同）。

## 5. 交接清单（把样本交给其他系统时）

| 必须交付 | 说明 |
|---|---|
| `dataset/test/<name>/testset.jsonl` | 固定样本；SHA-256 必须与 manifest 一致 |
| `manifest.json` + `testset.yaml` | 抽样/统计/字段语义/上游 SHA；`tokens_semantics` 必须一起引用；精度侧另需 manifest 的 `judges` 声明（判题契约版本来源）与 `method: accuracy` |
| 本文 §3 的系统职责表 | 对方系统需按同一协议执行并记录 actual/窗口/缓存/参数；精度侧按 §4.2 记录判题协议与 judge 契约版本 |
| 一套样例结果（可选） | 含 `records.jsonl` 抽样、`aggregates.json`、入带/出带计数（性能）或 `records[].score` + 判题计数（精度），便于对齐 |

> **不交付**：`dataset/raw/`、`dataset/.cache/`（输入/缓存，可重建）；`output/` 是运行证据，
> 不是样本契约的一部分。`reports/` 也不是样本契约（gitignored 本地记录）。

## 6. 变更规则

- 修改 testset（重新抽样/构造/换上游版本）会改变 SHA，等价于**发布新版本样本集**：必须重新做
  难度/长度校准，不能与旧结果混比（见 [11-performance-testset.md](11-performance-testset.md) §9）。
- 修改聚合带、目标值或协议参数属于**测试系统口径变更**：同步
  [20-methodology.md](20-methodology.md) 与 [../process/50-validation-plan.md](../process/50-validation-plan.md)，
  并与旧 run 分开报告。
- **精度侧的判题契约变更**（判题族/判定算法/归一化/依赖版本语义）同样属于口径变更：会改变
  `judge_contract_version`，必须按发布新版本处理，不得把新旧精度的 Δ 直接比较；判题器迁移
  以 `src/modelbench/domain/judges/` 为单一事实源并在
  [22-accuracy-test-plan.md](22-accuracy-test-plan.md) §3 记录。
