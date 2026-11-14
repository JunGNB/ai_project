# 2026-11-12 精度首轮全量结果审计

> **状态**：records（非规范、非证据）｜ **日期**：2026-11-12 ｜
> **关联 run**：`20261111T121245Z-b861__accuracy__{minicpm-v-4.6,minicpm-v-4.7-1b,gemma-4-e4b-oq4}`；
> 另有一次 80 样本冒烟 `20261111T115413Z-7f2b__accuracy__minicpm-v-4.6`。
> **一句话结论**：结果整体符合"8 场景能力剖面"预期，但当前 3 个 run 是**不同模型**而非同模型
> bf16/量化配对，不能据此下量化损失结论；发现 1 个确定性判题 bug（已修）和 1 个影响最大的
> 协议/模型适配问题（MiniCPM-V-4.6 的 reasoning 占满内容预算）。
> 完整审计报告（含证据命令与逐项复核）：`reports/accuracy-results-review-2026-11-12.md`。

## 1. 数据完整性

- 3 个全量 run：`n_selected=n_measured=n_ok=2000`、`n_skipped=0`、`error/timeout=0`；
  `./modelbench verify` 通过（4 batches / 10 runs / 7520 records / 2240 blobs）。
- 3 个 run 的 `config.testset.sha256` 均为 `502d827c…`（`dataset/test/accuracy/manifest.json`
  一致），判题依赖 `missing=[]`；`run_id` 方法段 `__accuracy__` 正确。
- 结论：**input/口径一致，可做同 testset 配对分析**；但 "3 个模型" 不是 bf16/量化配对。

## 2. 总分与逐场景（原始分数）

| 模型 | 宏平均 macro | 微平均 micro | 格式失败 | 说明 |
|---|---:|---:|---:|---|
| gemma-4-E4B-it-oQ4 | 0.6796 | 0.6796 | 2 | 正常输出 |
| MiniCPM-V-4.7-1B | 0.6006 | 0.6006 | 5 | 正常输出 |
| MiniCPM-V-4.6 | 0.5514 | 0.5514 | **224 (11.2%)** | 215 条 content 为空、reasoning 占满 cap |

| 场景 | gemma-oQ4 | MiniCPM-4.7-1B | MiniCPM-4.6 | 判题族/指标 |
|---|---:|---:|---:|---|
| knowledge_qa | 0.688 | 0.616 | 0.596 | mc_letter |
| instruction_following | 0.812 | 0.772 | 0.724 | ifeval |
| long_context | 0.416 | 0.416 | 0.400 | F1/ROUGE（连续） |
| math_logic | 0.832 | 0.848 | 0.684 | numeric_exact |
| coding | 0.744 | 0.468 | 0.416 | code_exec |
| general_agent | 0.600 | 0.580 | 0.584 | tool_call |
| search_agent | 0.593 | 0.441 | 0.375 | F1（连续） |
| multilingual | 0.752 | 0.664 | 0.632 | mc_letter |

## 3. 判题审计（是否代码误判）

### 3.1 已确认并修复的 bug：`mc_letter` 拒绝词优先于明确答案

- 现象：回答中包含 "not enough information / cannot determine / 无法确定" 等词、但结尾明确
  写了 "The answer is B." / "答案是 C" 时，旧实现先命中 refusal 正则直接返回 `None`，被记为
  `format_failure`（0 分）。
- 实测：1500 条 `mc_letter` 中 3 条受影响（gemma 1 条 gold=B、4.7 1 条 gold=D、4.7 1 条
  multilingual explicit=A 但 gold=B）；修复后前两条变为正确，第三条从 format_failure 变为
  正常错误，`format_failure` 计数 gemma 2→1、4.7 5→3，宏平均各 +0.0005。
- 修复：`src/modelbench/domain/judges/mc.py` 先解析明确答案短语，再回退 refusal；新增
  `test/test_judges.py` 回归用例。**现有 bundle 仍是旧判题结果**，正式引用前应重跑或重判。

### 3.2 数值判题（`normalized_match` / numeric_exact）：宽松，非确定性 bug

- 规则是"响应中任意数字与 gold 匹配"而非"最后一个数字"；这在模型把答案写在多行/验证段时
  是必要的（如 `#### 10` 后又写 `10+7=17`，取最后数字会误判），但会奖励"1 或 7"这类
  未定答案（实测 cmath/test/273 一条可疑）。
- 复核 750 条 numeric：没有发现"模型明确给出 gold 数值却被判 0"的 false negative；
  可疑 false positive 约 1–2 条/模型（<1%）。**建议保留宽松规则**并在报告说明；若追求
  更严格口径，需要新判题版本+重跑，不能与当前分数混比。

### 3.3 `tool_call`：possible-answer 严格匹配，存在少量"语义正确但格式不匹配"

- 250 条/模型中，约 29–52 条是"函数名命中但参数不匹配"被 0 分；其中一部分是真实错误
  （调用错函数/漏参数/参数值错误），另一部分是**BFCL possible-answer 的严格性**：
  - `where_to="Paris, France"` vs acceptable `"Paris"`；
  - `prompt="the current prime minister of India"` vs `"current prime minister of India"`；
  - `number_of_rooms="1"`（schema enum 是字符串）vs 数值 `1`；
  - `city="New York"` vs `"New York, NY"`。
- 这是"结构化近似 + 官方可接受值"的已知取舍（spec 12 §9.2），不是随机 bug；绝对分数
  对字符串归一化敏感，跨模型比较需固定判题契约（当前 3 个 run 一致）。
- 若后续需要提高与官方 BFCL 的一致性，建议单独做判题版本升级并重跑，不要在本次结果上
  局部修分。

### 3.4 `code_exec` / `ifeval` / F1：未发现系统性代码误判

- `code_exec`：失败以 AssertionError（91/133/37）、语法/缩进错误、NameError 为主；抽查为
  真实模型实现错误（如 `math` 未 import、函数名不符、代码被截断）；无"正确代码被判错"
  的证据。注意固定 `judge.max_new_tokens`（768/512 等）下 `finish_reason=length` 的样本
  属于协议截断，仍按 0 计。
- `ifeval`：判题异常 0；4.6 的 27 条失败均为 content 为空，其余失败是真实约束不满足
  （抽查：重复题面/未写 limerick 等），不是 checker 崩溃。
- F1/ROUGE（long_context/search）：低分主要是任务本身或表面形式差异（如 gold `Five` vs
  模型 `Five`/`5`、中文 `防城港` vs `防城港市`），属官方指标的已知局限；不是解析异常。

### 3.5 根因定位与修复：mlx_vlm processor 的 explicit-false 特殊分支

先纠正结论：**MiniCPM-V-4.6 确实是默认非思考模型**；问题不在模型，也不在
`enable_thinking` 的布尔含义，而在**运行时 processor 的显式 false 分支**。

对 MiniCPM-V-4.6 做直接 API 探针（同一短 prompt，`max_tokens=64`），并用 `/v1/tokenize`
解码实际 prompt：

| chat_template_kwargs | 实际生成 prompt 结尾 | content | reasoning |
|---|---|---|---|
| 省略 | `...<｜im_start｜>assistant\n<think>\n\n</think>\n\n` | `4` | 空（正确） |
| `{"enable_thinking": false}` | `...<｜im_start｜>assistant\n`（**没有空 think 块**） | `\n\n4` | 136 字符（错误） |
| `{"enable_thinking": true}` | `...<｜im_start｜>assistant\n<think>\n` | `\n\n4` | 136 字符（符合语义） |

本地用模型自带 `chat_template.jinja` 渲染时，no-kwargs 与 false 都得到空的
`<think></think>` 块；但运行时**打包的 mlx_vlm 0.7.10** 在
`models/minicpmv4_6/processing_minicpmv4_6.py::apply_chat_template` 中有一个特殊分支：

```python
if add_generation_prompt and enable_thinking is False:
    render_kwargs["add_generation_prompt"] = False
    rendered = self.tokenizer.apply_chat_template(*args, **render_kwargs)
    rendered = f"{rendered}<|im_start|>assistant\n"   # 丢了模板的空 think 块
```

即 false 分支只追加 assistant 前缀、丢掉了模板默认的空 `<think></think>`；模型于是从
开放 reasoning 开始，吃掉 `max_new_tokens`，content 为空。no-kwargs 走模型模板默认，
反而正确。这是 **mlx_vlm processor 的 bug**（已在上游 mlx_vlm 0.7.10 复现），不是 oMLX
服务端“key 存在即开启”的通则；本仓库用客户端 `omit` 规避。

同一探针显示 MiniCPM5-1B 相反（默认 thinking，`false` 可关闭）、4.7/gemma 两者均
no-think，所以**不能用单一全局开关**。

**修复（已实现并验证）**：

1. `ModelSpec.thinking_disable`：`explicit_false`（默认，发 `enable_thinking=false`）或
   `omit`（完全不发 chat-template kwargs）。
2. `models.yaml` 给 `MiniCPM-V-4.6` 设 `thinking_disable: omit`；MiniCPM5-1B 等保持
   `explicit_false`。
3. `thinking_control()`（`application/benchmark.py`）统一把策略写入
   `request.context_meta.thinking_control`，性能/精度共用。
4. 修复后真实 oMLX 冒烟（`--method accuracy --models MiniCPM-V-4.6 --samples 1`）：
   8/8 measured、**content 非空、reasoning 为空、format_failure=0**，
   `thinking_control=disabled:omit`，`verify` 通过；对比修复前同量样本（无 content / 格式失败）。
5. 仍需用修复后的代码重跑 4.6 全量 2,000 条（原 bundle 仍是旧行为）；登记为后续任务。

## 4. 结果是否符合预期

1. **模型间排序**：gemma-oQ4（0.68）> MiniCPM-V-4.7-1B（0.60）> MiniCPM-V-4.6（0.55）在
   原始分数上成立；但与 pilot（v1.1，4.7-1B，12 条/场景）相比，4.7 的 knowledge/instruction
   接近或更高，long_context 偏低（0.416 vs 0.566，连续 F1、样本更多），math 略低，
   coding/search 更高 —— 属合理波动，pilot 只用于难度带校准，不能当正式分数。
2. **MiniCPM-V-4.6 的 0.551 不能直接解读为"能力弱"**：其 224 条格式失败（215 条 content
   为空、reasoning_text 非空且 `enable_thinking=false` 被模型忽略）导致 math/search/instruction
   被系统性计 0；**排除格式失败后 macro≈0.620**，与 4.7（0.602）接近甚至略高。原始排序
   因此混入了"输出格式/思考预算"差异，而不是纯知识能力差异。
3. **gemma-oQ4** 在 coding/knowledge/multilingual/search 上显著高于两个 MiniCPM（配对
   McNemar + Holm）；math 与 4.7 接近；long_context 三者几乎相同（~0.41–0.42）。
4. **结论**：分数不是"全错/全对"，没有明显 floor/ceiling 泛滥；结果基本符合"8 场景能力
   剖面"的预期，但必须同时报告格式失败与条件分数，否则跨模型结论会被格式问题污染。

## 5. 数据集与测试目的匹配度

**设计目的是同模型 bf16 vs 量化损失**（spec 12）。当前 3 个 run 是三个不同模型
（MiniCPM-V-4.6 / MiniCPM-V-4.7-1B / gemma-4-E4B-it-oQ4），**没有同模型 bf16/量化配对**，
因此不能从现有结果计算量化损失 Δ；只能做模型间剖面/排序。

数据集本身对该目的**基本匹配**：

| 维度 | 判断 |
|---|---|
| 能力覆盖 | 8 场景 + 5 确定性判题族，覆盖知识/指令/长文/数学/编程/Agent/搜索/多语；与目的匹配 |
| 可比性 | 固定 testset SHA `502d827c…`；同一批 sample_id；可做配对 McNemar/bootstrap；旧 run/新 run 只要 SHA 一致可比 |
| 难度带 | 多数场景在 15%–85%；math 0.68–0.85、instruction 0.72–0.81 对强模型接近天花板 → 量化小差异可能被压缩；coding 0.34–0.79、search 0.37–0.59、knowledge 0.60–0.69 区分度更好 |
| 场景区分度 | 4.6 vs 4.7 有 129/250 coding、106/250 knowledge 等混合项；但 general_agent 三者 0.58/0.58/0.60、long_context 三者 ~0.41（连续 F1），对模型差异不敏感，不适合单独作为量化结论 |
| 语言 | 多语 25 条/语只作剖面；knowledge en/zh 各 125 对，符合设计；单语排名不可靠 |
| 统计功效 | 250/场景 95% CI 约 ±6pp；宏平均约 ±2pp；量化损失常 <5pp → 必须用配对检验 + 宏平均，不能只看单场景均值；建议同模型 bf16 重复 run 建立噪声基线 |
| 判题噪声 | tool_call 的 possible-answer 严格性、F1 表面形式、numeric 宽松规则会引入少量噪声；对同模型 bf16/量化通常同向，但需在报告中标注 |

**当前实验设计与目的不匹配的核心点**：

1. 缺同模型 bf16/量化对：可用 `gemma-4-E4B-it`（16.8GB 全权重）vs `gemma-4-E4B-it-oQ4`
   （5.2GB）或 `gemma-4-E4B-it-MLX-4bit`；三者需同一 testset SHA 各跑 2,000 条。
2. MiniCPM-V-4.6 的 reasoning 输出不符合"thinking disabled"协议；需在**同一模型对**
   上确认 bf16/量化都走同一 reasoning 行为，否则格式失败差异会污染 Δ。
3. 建议对同模型 bf16 重复跑 2 次（独立 run）估计 run-to-run 噪声；量化 run 与 bf16 用
   `analyze --paired --method accuracy` 出 Δpp/McNemar/Holm，并同时报告
   `format_failure`/条件分数，避免把格式失败当能力损失。

## 6. 建议动作

1. 用修复后的判题重跑（或实现一次显式 rejudge 并保留旧 bundle 只读）现有 3 个模型；
   本次修复影响 3 条 `mc_letter`，对宏平均约 +0.05pp，但需保持"一套判题版本对应一套分数"。
2. 若目标是量化损失：跑 `gemma-4-E4B-it` 全权重 + `gemma-4-E4B-it-oQ4` 同 testset，
   或确认 MiniCPM 侧存在同模型量化版本；不要用 4.6/4.7 代替量化对。
3. **reasoning 模型协议（已修复代码，待重跑）**：本轮已定位打包 mlx_vlm 0.7.10 的
   MiniCPM-V-4.6 processor 在 `enable_thinking=false` 时丢失空 think 块、导致模型开放
   reasoning 的 bug，并加入 `ModelSpec.thinking_disable=omit`（4.6）；修复后 8 样本冒烟
   content 非空、format_failure=0。下一步用修复后的代码重跑 4.6 全量 2,000 条，再与 4.7/
   gemma 比较；旧 4.6 bundle 不能作为最终结论。
4. 报告/Web 已显示格式失败率与分场景统计；建议展示"条件分数（排除格式失败）"作为
   辅助列，并在结论中明确 primary 口径（当前全样本 macro）。
5. 对 `tool_call` 的严格匹配保持当前契约；若要提升官方一致性，作为独立判题版本升级，
   不局部改分。
6. 修复后重跑 4.6 全量（约 2h；建议同时跑 bf16 重复样本作噪声基线），并用同一 testset/
   判题契约重新生成报告；4.6 的旧分数只保留为"修复前证据"。
