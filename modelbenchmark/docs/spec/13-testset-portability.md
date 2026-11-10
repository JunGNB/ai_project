# 测试样本契约与测试系统职责（跨设备/跨系统移植）

> **状态**：current ｜ **版本**：v1.0（2026-11-10）｜ **适用**：`modelbench` 0.3.x
> **读者**：维护者、测试执行者、需要复用同一套样本的其他系统 ｜ **范围**：把**同一套固定样本**
> 交给不同机器/后端/测试系统时，"哪些属于样本契约、哪些属于测试系统职责、比较准入条件是什么"。
> 样本集设计见 [11-performance-testset.md](11-performance-testset.md)（性能）与
> [12-accuracy-testset.md](12-accuracy-testset.md)（精度）；指标口径见
> [20-methodology.md](20-methodology.md)；输出契约见
> [../design/31-output-contract.md](../design/31-output-contract.md)。

## 1. 一句话结论

**同一套 testset 是唯一稳定的输入契约**；模型、tokenizer、后端、机器可以不同，但必须：
同一 testset/SHA、同一 target 定义、同一 ±10% 聚合带、同一测量协议参数，并**报告 actual token 分布**
与入带/出带计数。跨模型比较只声明"跑的是同一套样本"，**不**宣称"精确相同的 token 工作量"。

## 2. 样本侧（testset 契约，一次构造、到处复用）

固定一套性能测试集（默认 `dataset/test/performance/testset.jsonl`，240 行 = 8 场景 × 1K/8K/32K × 10），
**所有设备/模型/后端复用**：

- **不做 per-model 构造**：不为某个模型/机器重新生成或挑选样本；同一 SHA 即同一输入。
- **不做运行时裁剪/补齐**：runner 原样发送样本；不截断、不填充到目标长度。
- **32K target 固定 32,768，无特殊处理**：1K/8K/32K 是对称的三种 bucket，没有"32K 特殊构造"。
- **不含判分**：性能 testset 不携带 gold/score；精度集（隐藏离线资产）自带判题契约但**不接入
  性能 run**（见 [12-accuracy-testset.md](12-accuracy-testset.md)）。
- **字段契约**（性能 testset 单行，详见 [11-performance-testset.md](11-performance-testset.md) §6.2）：
  `id`、`scenario`、`bucket`（1k/8k/32k）、`target_id`、`dataset`、`tokens`（参考 tokenizer 的
  **内容 token 声明值**）、`messages`、`tools`、`constructed`、`meta`（含 `lang` 等）。
- **manifest/SHA**：`dataset/test/<name>/manifest.json` 记录逐文件 SHA-256、抽样参数、token 统计与
  `tokens_semantics`；`testset.jsonl` 的 SHA-256 是"同一套样本"的唯一凭据。仓库提交 `testset.yaml`
  与 `manifest.json`，`testset.jsonl` 由 `tool/dataset/.../build.py` 确定性重建（seed 固定、无时间戳）。
- **跨设备"样本同一"的定义**：**同一 testset 文件 SHA-256 + 同一 manifest**。机器/模型/tokenizer/
  后端不是样本契约的一部分。

## 3. 系统侧（测试系统职责，各系统可不同实现）

同一套样本在不同测试系统上执行时，以下由**测试系统**负责，且必须显式记录：

| 职责 | 要求 | 记录位置 |
|---|---|---|
| actual token 计数 | 请求前用**目标模型** `POST /v1/tokenize` 对最终 messages+tools（含 chat template/decode guard）计数 | `request.tokenized_prompt_tokens`、`context_meta.isl_tolerance{target,low,high,actual,ok}` |
| 模型窗口判定 | 用 actual + `max_tokens` + `context_safety_margin` 对比模型窗口；**不足才 skip**（窗口不匹配不是跳过理由） | `run.json.outcome.skipped`、`context_meta.window_check_basis` |
| 冷缓存 | measured 请求前清 hot+SSD 并校验 `cached_tokens==0`（fail-closed） | `records[].cache` |
| 生成长度/decode guard | `perf.max_tokens`（默认 50）+ `perf.decode_guard`（默认开） | `run.json.config.perf`、`records[].request` |
| warmup/repeats | 模型级 `warmup_per_model=1`、per-cell `warmup_requests=0`、`repeats=1`（默认） | `run.json.config.run` |
| 指标定义 | TTFT/TG TPS/TPOT/峰值内存/输入上下文等按 [20-methodology.md](20-methodology.md) | `records[].metrics`、`aggregates.json` |
| 聚合带 | 只用 actual 在 **target ±10%**（所有档位对称，区间见 [20-methodology.md](20-methodology.md) §5 权威表）的 measured 样本求平均；出带样本保留但不进均值 | `aggregates.*.n_isl_in_band/n_isl_out_of_band`、`samples.csv.isl_band` |
| records/schema | 逐样本权威记录、schema 3.0、prompt 内容寻址 | `records.jsonl`、`run.json`、`MANIFEST.json` |

## 4. 跨系统/跨设备比较准入（硬条件）

在引用/比较任何结果前，必须同时满足：

1. **同一 testset**：`testset.jsonl` 的 SHA-256 与 manifest 一致（"同一套样本"的唯一凭据）；
2. **同一 target 定义**：1K=1,024、8K=8,192、32K=32,768（内容 token，参考 tokenizer），不含
   模型特定改造；
3. **同一聚合带**：所有档位 target ±10%（区间见 [20-methodology.md](20-methodology.md) §5 权威表），
   只对入带样本求平均；入带/出带计数必须随结果报告；
4. **同一协议参数**：`max_tokens`、`decode_guard`、缓存模式、warmup、repeats、采样/温度、流式口径；
5. **actual 分布必须报告**：至少给出 `tokenized_prompt_tokens` 的 per-bucket 分布（min/median/max 或
   直方图）、`n_isl_in_band`/`n_isl_out_of_band`，以及 `insufficient_samples` 标记（若 in-band 样本
   不足以支撑均值）。
6. **比较范围**：
   - **跨模型（同文本、不同 tokenizer）**：只能表述为"在**同一套样本**上的对比"，**不得**宣称
     "精确相同的 token 工作量"；actual 差异是解释变量，不是误差。
   - **同模型跨设备/后端**：可按 actual 对齐比较（同一 tokenizer 下 actual 相同或接近）；仍需
     报告 per-bucket actual 与入带计数。
7. **不满足以上任一条件**：结果只能作为"该次运行"的证据，不能横向合并/平均/排名。

## 5. 交接清单（把样本交给其他系统时）

| 必须交付 | 说明 |
|---|---|
| `dataset/test/<name>/testset.jsonl` | 固定样本；SHA-256 必须与 manifest 一致 |
| `manifest.json` + `testset.yaml` | 抽样/统计/字段语义/上游 SHA；`tokens_semantics` 必须一起引用 |
| 本文 §3 的系统职责表 | 对方系统需按同一协议执行并记录 actual/窗口/缓存/参数 |
| 一套样例结果（可选） | 含 `records.jsonl` 抽样、`aggregates.json`、入带/出带计数，便于对齐 |

> **不交付**：`dataset/raw/`、`dataset/.cache/`（输入/缓存，可重建）；`output/` 是运行证据，
> 不是样本契约的一部分。`reports/` 也不是样本契约（gitignored 本地记录）。

## 6. 变更规则

- 修改 testset（重新抽样/构造/换上游版本）会改变 SHA，等价于**发布新版本样本集**：必须重新做
  难度/长度校准，不能与旧结果混比（见 [11-performance-testset.md](11-performance-testset.md) §9）。
- 修改聚合带、目标值或协议参数属于**测试系统口径变更**：同步
  [20-methodology.md](20-methodology.md) 与 [../process/50-validation-plan.md](../process/50-validation-plan.md)，
  并与旧 run 分开报告。
