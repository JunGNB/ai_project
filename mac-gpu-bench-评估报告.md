# mac-gpu-bench 专业性与严谨性评估

> 评估对象：`mac-gpu-bench/`（`macgpu` v2.0.0，纯 Metal 的 Mac GPU 算力/带宽基准）
> 评估目的：能否作为 **TTFT / TG TPS** 评估体系里的「设备算力与带宽」指标来源
> 评估日期：2026-11-20　评估机：Apple M1 / 8 核 GPU / 16 GB / macOS 27.0.1 (26A434)
> 方法：逐行读代码 + 跑通自带验证 + **独立实现探针交叉复现**（附录 A）+ 对仓库里 22 次历史结果做统计

---

## 0. 结论

**一句话**：内核层面做得比多数 benchmark 脚本都扎实（有数值自检、防 DCE、防常量折叠、防内存压缩），
但「统计口径、跨设备可比性、到 TTFT/TG TPS 的映射」这三层还不够严谨；
**当"单机硬件上限"用没问题，当"跨设备横向对比"或"TTFT 预测器"用，现在会得到系统性偏差。**

### 评分卡

| 维度 | 分数 | 判断依据 |
|---|---|---|
| 内核正确性 / 防编译器优化 | **9 / 10** | 数值自检 + 全向量写回 + 真实 GEMM 替代整型微基准；我独立复现确认 FP32 FMA≈1.1 TFLOPS 是真上限，不是内核写坏了 |
| 计数口径 | **8 / 10** | 公式与内核逐条对得上；但 JSON 不输出 `units` / `macs_per_iteration`，外部无法复算 |
| 计时方法 | **8 / 10** | GPU 侧 `gpuStart/EndTime` + 预热丢弃 + min-of-9，方向正确；缺 `cb.error` 检查 |
| 统计与不确定性 | **5 / 10** | 只留 min/max，不留中位数/原始样本；历史语料 `--repeat` 1/2/3/10 混用，best-of-N 本身带 +2~4% 偏差 |
| 跨设备可比性 | **4 / 10** | 芯片规格表子串匹配 bug（Pro/Max/Ultra 全被当基础款）；不记录热限速/低电量模式/电源；带宽跨会话波动 20~60% |
| 到 TTFT / TG TPS 的映射 | **5 / 10** | roofline 方向对；但默认效率 0.4 与自家校准 0.78~0.83 矛盾，表格把 prefill 低估约 2 倍；无 batch、无固定开销、KV 形状不对 |
| 工程与契约 | **7 / 10** | `docs_contract` + `selfcheck` + smoke 形成闭环，纪律好；但 `notes` 与 spec 自相矛盾、同版本下字段变过（11/22 份缺 `gbps_min/max`）、MCP 摘要键冲突 |
| 可复现性 | **6 / 10** | 无 git SHA / OS build / Metal 版本；结果目录秒级时间戳会碰撞 |

**综合 6.5 / 10（等权，针对 v2.0）。**

v2.1 修复后重估约 **8.5 / 10**：内核 9、计数 9、计时 9、统计 8、跨设备 8、映射 7、工程 9、可复现 9。
剩下的扣分主要是「没有 batch/shape sweep」和「没有持续负载热稳态」，都不影响"单机上限 + 归一化基线"这个用法。

### 三种用法，三种结论

| 用法 | 现在能不能用 | 说明 |
|---|---|---|
| 单机「这颗 GPU 的算力/带宽上限」 | ✅ **可用** | 带自检、可复现（同会话 3 次独立进程 CV 0.1~2.2%），数字量级可信 |
| 多台 Mac 横向排序 | ✅ **v2.1 起可用** | 规格匹配、环境字段、中位数、并发锁都已落地；旧结果仍会把 M1 Pro 按 M1 比（见 P0-1） |
| 直接预测 TTFT / TG TPS | ⚠️ **只作数量级参考** | v2.1 默认 `eff=0.8` 已对齐 MLX；但 TTFT 仍含固定开销/排队/ANE 路径，不能当预测器 |
| 作为 TTFT/TG TPS 的**归一化基线** | ✅ **推荐** | 见第 6 节：用「实测 ÷ roofline」的比值，比直接用数值稳得多 |

### 修复状态（v2.1，2026-11-20）

本报告的 P0/P1 已在 `mac-gpu-bench` 的 v2.1.0 里落地（schema `macgpu/v2`），
下面第 3 节保留的是**修复前（v2.0）的 as-found 审计**，用于追溯问题来源。

| 问题 | 状态 | 落点 |
|---|---|---|
| P0-1 规格子串匹配 | ✅ 已修 | `tools/report.py` `lookup_spec()` 最长前缀 + 未知芯片告警 |
| P0-2 `gbps_max` | 🔁 已更正为非缺陷 | 契约写明 `gbps == gbps_max`；新增 `gbps_median` / `samples` |
| P0-3 命令缓冲错误 | ✅ 已修 | `Bench.swift` `gpuSeconds()` 检查 status/error + 物理上界 |
| P0-4 热/功耗状态 | ✅ 已修 | `host.thermal_state` / `low_power_mode` / `power_source` / `os_build`，报告独立告警节 |
| P0-5 统计口径 | ✅ 已修 | 每指标 9 个原始样本 + `value_median`；正文用中位数，峰值单列 |
| P0-6 默认 `eff=0.4` | ✅ 已修 | 默认 0.80，报告并列 0.8 / 0.4 两列 |
| P0-7 MCP 键冲突 | ✅ 已修 | `(dtype, path)` 组合键 |
| P1-1 带宽线程 65k | ✅ 已修 | 262,144 线程，线程数写入 JSON |
| P1-2 计数不可复算 | ✅ 已修 | `units` / `unit_threads` / `macs_per_iteration` / `total_macs` |
| P1-3 notes 自相矛盾 | ✅ 已修 | notes 改为与实现一致的固定文案 |
| P1-4 selfcheck 不入 JSON | ✅ 已修 | 每次 `run` 内嵌 `selfcheck`；失败退出码 2 |
| P1-5 fp16 自检贴阈值 | ✅ 已修 | 累加器清零、迭代 96→16、GEMM 用非精确输入；误差 5.6e-3（阈值 2e-2） |
| P1-6 无代码版本/OS build | ✅ 已修 | `tool.git_sha/git_dirty` + `host.os_build`（Metal 编译器随 OS，未单独记语言版本） |
| P1-9 参数校验/时间戳碰撞 | ✅ 已修 | `--tests` 白名单、GEMM 工作集上限、毫秒时间戳、`run` 并发 pid 锁 |
| P1-7/P1-8 shape sweep / 持续负载 | ⬜ 未做 | 见第 7 节，属第二阶段 |

验证：`./macgpu selfcheck` 12/12、`make docs` 通过、`make test` 7/7；
v2.1 复标定读数见 `docs/records/52-v2.1-recalibration.md`。

---

## 1. 我做了哪些验证

1. 通读 `src/*.swift`、`tools/*.py`、`tests/`、`docs/`，核对「文档口径 ↔ 内核 ↔ JSON 字段」是否一致。
2. 跑通 `./macgpu info / selfcheck / run`、`make docs`、`tests/smoke.sh` 的检查点。
3. **独立写了一个 Metal 探针**（附录 A），用同一套计时方法复现它的 FMA/带宽/MMA，并额外测了：
   显式 `fma()`、8/16/32 条独立链、标量 vs float4、65k/131k/262k 线程、8 个 MMA 累加器。
   目的：区分「硬件上限」和「内核写法造成的上限」。
4. 对 `output/results/` 里 **22 次历史运行**（去重 `latest` 软链）做统计（算力/GEMM/带宽的跨轮波动、load 相关性、`--repeat` 偏差）。
5. 用它自己的 `--repeat 5` 连跑，再用新写的跨设备采集器跑 **3 次独立进程 × 5 轮**，得到同会话重复性。

---

## 2. 做得好的地方（这些是真实优势）

1. **口径可追溯**：`docs/spec/10-methodology.md` 的指标矩阵 ↔ `src/Kernels.swift` 内核名 ↔ `ComputeCases` 计数，
   由 `tools/docs_contract.py` 双向校验，改一处不改另一处会直接失败。这在自研 benchmark 里很少见。
2. **三个历史坑都有防线**，而且是对症的：
   - 向量分量被 DCE → 峰值内核整条向量写回（`Kernels.swift:82,99`）；
   - 整型微基准被折叠成常数加法 → 放弃 int8 微基准，改测真实 int8 GEMM 并与 CPU 逐元素比对；
   - 零页/内存压缩 → 测试前用 LCG 填充不可压缩数据（`Kernels.swift:290-304`）。
3. **`selfcheck` 是硬门槛**：GEMM/int8/int4 与 CPU 参考完全相等，峰值内核逐元素比对，不过退出码 2。
   这是它比「跑个 MPS 矩阵乘看数字」专业得多的根本原因。
4. **计时口径正确**：用 `MTLCommandBuffer.gpuStartTime/gpuEndTime`，排除主机侧提交开销；
   每次测量先预热一次丢弃；min-of-9 而不是均值。对「峰值」这个定义是自洽的。
5. **零依赖、运行时编译 MSL**：`swiftc` + 系统 Metal 即可，部署到一堆 Mac 上成本极低——这是它能做 fleet 采集的最大工程优点。
6. **对结论有克制**：README 明确写「峰值是合成负载，不等于任何具体模型速度」「不测 ANE/热降频」，没有过度承诺。

### 我复现出的关键结论（说明它的数字是真的）

| 项目 | 它的读数 | 我的独立探针 | 判断 |
|---|---|---|---|
| FP32 FMA 峰值 | 1.09 TFLOPS | 1.09~1.13（8/16/32 链、标量/float4、3 种线程数） | ✅ 是硬件上限，不是内核缺陷 |
| FP16 FMA 峰值 | 1.89 TFLOPS | 未单独测，与 half2 打包一致 | ✅ 合理 |
| FP32 MMA 峰值 | 2.27 TFLOPS | 4 累加器 2.34 / 8 累加器 2.22 | ✅ 4 个累加器已够，不是靠"堆累加器"刷出来的 |
| 读带宽 | 57~59 GB/s | 同 kernel 55.2@65k 线程 → **59.8@262k 线程** | ⚠️ 数值真实，但**线程数偏少，低了约 8%**（P1-1） |

特别是 FMA：我一开始怀疑「1.1 TFLOPS 只有标称 2.6 的 42%」是 mul+add 没融合成 FMA 导致的，
于是显式调用 `fma()`、把独立链加到 32 条、换成标量写法——结果全部落在 1.09~1.13。
所以「普通着色器 FMA 只有矩阵指令的一半」这个结论**成立**；但 `notes` 里"标量只有 1/4"的解释是错的（见 P1-3）。

---

## 3. 问题清单（as-found，针对 v2.0）

> 以下按修复前的 v2.0 代码记录，行号与结论保留原样以便追溯；逐项修复状态见 0.5 节。
> 其中 P0-2 经复核**不是缺陷**（字段冗余），已就地下更正说明。

### P0 — 会直接导出错误结论，用之前必须修

#### P0-1 报告里的芯片规格是**子串匹配**，Pro / Max / Ultra 全被当成基础款

`tools/report.py:65`：

```python
spec = next((v for k, v in SPEC.items() if k in chip), None)
```

`SPEC` 的插入顺序是 `M1 → M1 Pro → M1 Max → M1 Ultra ...`，而 `"Apple M1" in "Apple M1 Pro"` 为真，
所以 **M1 Pro / M1 Max / M1 Ultra 永远命中 "Apple M1"**。我按同样逻辑验证：

```
Apple M1 Pro  -> Apple M1  (2.6 TFLOPS / 68.25 GB/s)   ← 应为 5.2 / 200
Apple M1 Max  -> Apple M1  (2.6 / 68.25)               ← 应为 10.4 / 400
Apple M4 Pro  -> Apple M4  (4.3 / 120)                 ← 应为 8.6 / 273
Apple M4 Max  -> Apple M4  (4.3 / 120)                 ← 应为 17.2 / 546
Apple M3 Pro  -> None      (静默降级为 "—")
```

后果：报告的「占规格百分比」「与公开规格对照」「评估」三节对 Pro/Max/Ultra 全部错，
而且方向很荒诞——M1 Pro（16 核）实测 ~4.5 TFLOPS 会被写成"规格 2.6 的 173%"、~170 GB/s 的读带宽
会被写成"规格 68.25 的 249%"，同一段评估却仍写着"已达到规格"。**这恰好是跨设备对比最常用的场景。**

修：最长前缀匹配 + 用 `gpu.core_count` 交叉校验，未知芯片显式告警：

```python
# 最长前缀优先：M1 Pro 会先命中 "Apple M1 Pro"，而不是 "Apple M1"
spec = next((SPEC[k] for k in sorted(SPEC, key=len, reverse=True)
             if chip == k or chip.startswith(k + " ")), None)
```

#### P0-2（已更正）`gbps_max == gbps` 不是 bug，是字段冗余

初稿把这里判成缺陷，复盘后要更正：契约把 `gbps` 定义为「9 次采样的**最优值**」，
`gbps_max` 定义为「采样中的**最好值**」，两者本来就是同一个数。`gbps_min = bytes / worst`
才是正确的对照。所以 33 条历史结果里 `gbps_max == gbps` 全部符合定义，不是"零抖动假象"。

真正的问题是**冗余且容易误读**：字段名暗示了"最大值/最小值"的对称性，实际 `gbps` 已经是最快值。
v2.1 保留该字段（兼容旧结果），在契约里写明 `gbps == gbps_max`，并补 `gbps_median`、`samples[]`。
另外 15 条历史结果没有 `gbps_min/max`，属于同版本下字段变更未升 schema（见 P2-6）。

#### P0-3 不检查 command buffer 错误，GPU 失败会变成天文数字

`timeThreads / timeGroups / blitTimed` 只 `waitUntilCompleted()`，然后
`max(gpuEndTime - gpuStartTime, 1e-9)`（`Bench.swift:294,313,608`）。
如果命令缓冲失败/超时（OOM、GPU fault、看门狗），时间可能是 0（被 1e-9 兜底）→ 算出上亿 TFLOPS，
而 `guard value.isFinite, value > 0`（`Bench.swift:457,539`）**照样通过**，`best()` 还会把它当"最快的一次"留下。
这与项目自己的信条（"数字很好看，但测的不是它声称测的东西"）正好相反。

修：

```swift
cb.waitUntilCompleted()
guard cb.status == .completed, cb.error == nil else { throw BenchError.gpuFailure(cb.error) }
let t = cb.gpuEndTime - cb.gpuStartTime
guard t > 1e-6 else { throw BenchError.invalidMetric("zero GPU time") }
```

再加物理上界（如 fp32 MMA < 1000 TFLOPS、带宽 < 10 TB/s）。

#### P0-4 不记录热/功耗状态，而现有语料里带宽波动 20~60%

先公平地说：**在受控条件下它重复得很好**——同一会话 3 次独立进程 × 5 轮，CV 只有 0.1~2.2%；
两次独立采集（16:33 与 16:36）的读带宽分别是 58.46 / 57.85 GB/s（差 1.0%），FP16 MMA 2.294 / 2.311（差 0.7%）。
问题不在随机噪声，而在**条件不受控**：仓库里 22 次结果跨了 load 1.37~3.53、`--repeat` 1~10、
不同缓冲区大小与测试子集，于是：

| 指标 | min | median | max | 跨度 |
|---|---|---|---|---|
| gpu_read | 50.1 | 53.4 | 60.3 | **+20%** |
| gpu_write | 43.1 | 49.0 | 60.5 | **+40%** |
| gpu_copy | 38.0 | 52.5 | 61.2 | **+61%** |
| fp16 mma | 2.196 | 2.279 | 2.383 | +8.5% |
| fp16 gemm | 1.104 | 1.114 | 1.250 | +13.3% |

而且 read 与 `load_avg_1m` 明显负相关：load 1.37 → 60.3 GB/s，load 3.4 → 50.1 GB/s（−17%）。
在"M1 68 → M2 100 GB/s（+47%）"的量级上，17% 的条件误差足以改变排序；同机型两次采集若一次在
空载、一次在 load 3 下做，也会得出"这台比那台慢 20%"的假结论。
JSON 里**没有** `thermal_state`、`low_power_mode`、`power_source`、GPU 利用率，
`load_avg_1m` 只是 CPU 负载，解释不了 GPU 侧争用（外接显示器、别的进程用 GPU 都不会体现在 load 上）。

修：`HostInfo` 增加

```swift
thermalState = ProcessInfo.processInfo.thermalState   // .nominal/.fair/.serious/.critical
lowPowerMode = (pmset -g | grep lowpowermode)          // 0/1
powerSource  = (pmset -g ps | AC/Battery)
```

报告中位数，并对 `load>2 / LPM=1 / 电池` 的样本打"不可比"标记。

#### P0-5 `--repeat` 语义污染了历史语料

`main.swift:337-360` 对每轮取 **max**。仓库里 `runs` 有 1、2、3、10 四种：

```
runs=1   n=5  mean=2.280
runs=2   n=2  mean=2.235
runs=3   n=9  mean=2.283
runs=10  n=1  mean=2.369   ← +4%
```

best-of-N 天然偏高，N 不同就不可比；JSON/report 只写"共 N 轮取最优"，下游无法归一化。
跨设备对比时，谁多跑几轮谁就"更快"。

修：固定协议（推荐 3 次独立进程 × 每进程 5 轮），**报中位数**；JSON 增加
`sampling: {estimator: "median", launches: 3, rounds: 5}`，报告禁止跨 estimator 比较。

#### P0-6 TTFT 默认效率 0.4 与自家校准 0.78~0.83 矛盾

`tools/report.py:299-300` 默认 `--eff 0.40`，第 5 节 `prefill = macs × eff / params`。
但 `docs/records/51-mlx-crosscheck.md` 自己测出 MLX 的 prefill 效率是 **0.78~0.83**：

| 模型 | 报告默认 0.4 预测 | 0.8 预测 | 实测（records/51） |
|---|---|---|---|
| MiniCPM5-2B (2.68B) | 171 tok/s | 343 | **321** |
| Qwen3.5-9B (9.5B) | 48 tok/s | 97 | **95.6** |

也就是说，报告第 5 节给 7B 的 **65.7 tok/s 是偏低约 2 倍**的（按自家 MLX 实现应在 ~130）。
如果直接拿这个表当 TTFT 预测：1024 token 的 TTFT 会被高估成 ~15.6 s，而实际约 ~8 s。
**给模型评估用的时候，这个默认值是错的。**

修：按运行时给 preset（`mlx/mps=0.8`、`handwritten=0.4`），表格并列两行并标注假设；
或者干脆同时输出「观测值 ÷ roofline」的比值（见第 6 节）。

#### P0-7 MCP 摘要按 dtype 建索引，mma 行被 fma 覆盖

`tools/mcp_server.py:108-112`：

```python
"compute": {m["dtype"]: {...} for m in data.get("compute", [])}
```

同一 dtype 有 `mma` 和 `fma` 两条，后写的 **fma 覆盖 mma**。我直接 import 这个函数跑了一遍
`latest/result.json`，MCP 返回 `fp16 = 1.929 TFLOPS`（FMA 通路），而同一次运行的矩阵指令峰值是
**2.369**——Agent 通过 MCP 完全看不到矩阵指令这一行。
契约 `docs/design/21-output-contract.md` 明确写了"**不能按 dtype 建索引，必须用 (dtype, path)**"，
`report.py` 做对了，MCP 没做。

修：`f"{m['dtype']}/{m['path']}"` 作 key。

---

### P1 — 影响精度/可复现性，建议修

| # | 问题 | 证据 / 位置 | 影响 | 修法 |
|---|---|---|---|---|
| P1-1 | 带宽读测试线程数固定 65,536，偏少 | `Bench.swift:569`；我的探针同 kernel：55.2 → **59.8 GB/s @262k 线程** | 读带宽低约 8%，decode 预测系统性偏低；也解释了 records/51 里"decode 折算带宽 111% 超过实测"的怪现象 | `threads = max(262144, cores*8192)`，或扫 2~3 档取最好，并把线程数写进 JSON |
| P1-2 | JSON 缺 `units` / `macs_per_iteration`，无法复算 | `Bench.swift:50-74`；契约却声称 `MACs = units × iterations × macs_per_iter × reps` | 外部只能验 `value=macs/seconds`，无法独立验证"数了多少次乘加"——而这正是核心口径 | 加 `units`、`unit_threads`、`macs_per_iteration`、`total_macs` |
| P1-3 | `notes` 与 spec 自相矛盾且过期 | `main.swift:315`"ALU 4-wide、标量只有 1/4" vs `docs/spec/10-methodology.md:30-31`"不要再引用 4-wide"；`main.swift:318`"重复 5 次" vs 代码 9 次 | `notes` 是机器可读 provenance，比没有更糟；我的探针证明 scalar32≈float4（1.12 vs 1.10） | `notes` 由常量生成，并纳入 `docs_contract` 检查 |
| P1-4 | `selfcheck` 结果不进 `result.json` | `main.swift:296-320` | 可以不跑自检就入库；对"曾虚高 4 倍"的工具，最该固化的证据反而没固化 | run 前自动自检（或默认 `--selfcheck on`），把 `{passed, checks[]}` 写进结果，失败拒绝落盘或标 `valid:false` |
| P1-5 | fp16 自检阈值贴线；mma 参考假设 C 初值为 0 | 实测 fp16 mma 误差 **3.8e-2** / 阈值 5e-2；`Kernels.swift:29-30` 从 C 读初值，`Bench.swift:274-276` 用未初始化内存 | 换 GPU/换 macOS 编译器可能假失败；新分配页碰巧为 0 才通过 | 显式清零 C；参考值用 double + 误差界；iters 降到 16；或比对 A·B 乘积而非累加和 |
| P1-6 | 结果不可归因 | 只有 `version`，无 git SHA / OS build / `MTLLanguageVersion` | Metal 着色器运行时编译，编译器随 macOS 变；无法回答"两个数是不是同一版工具测的" | 加 `tool.git_sha`、`host.os_build`、`metal.language_version` |
| P1-7 | GEMM 只有单一 N=2048³，无厂商实现对照 | `main.swift:63`；`tools/report.py:160` | TTFT 关心 M=prompt 长度的 GEMM，TG TPS 关心 batch=1 的 GEMV；2048³ 两者都不代表；fp16 时 A+B=16 MB，大 SLC 机型会整块驻留 | 加 shape sweep（M∈{1,8,32,128,512,2048}, N=K=4096）；可选调用 MPS/MLX GEMM 做上界对照 |
| P1-8 | 无持续负载/热降频模式 | README「局限」自认 | 比 MacBook Air（无风扇）和 Mac Studio（有风扇）时，10 分钟后的差距可达 20%+ | `--mode sustained --seconds 600`，输出前后两段与降频比 |
| P1-9 | 参数校验缺口 | `--tests foo` 静默跳过（`main.swift:82-83`）；`--gemm-size 16384` 可申请 3×1 GB（`main.swift:91-93`）；`selfcheck` 全用 `try!`（`main.swift:413-433`） | 空结果/ OOM / 崩溃而非退出码 2 | 校验 tests 枚举；按 `recommendedMaxWorkingSetSize` 限制 GEMM；改 do/catch |
| P1-10 | 时间戳目录秒级碰撞 | `main.swift:381-385` | 同秒两次 run 互相覆盖，`latest` 指向被覆盖的那次；自动化采集容易撞 | 加毫秒或 PID |

---

### P2 — 细节/文档

- **规格表 GB/GiB 混用**：200/400/800 被写成 204.8/409.6/819.2（+2.4%）；M1 的 68.25 又是对的。
- **规格表缺 M3 Ultra / M5 系列**：未知芯片静默输出 "—"，报告三节降级但不报错。
- **KV 头数按 MHA 口径**：`report.py:35-38` 7B/13B 用 32/40 头，对 Qwen3/Llama3 这类 GQA（8 头）会把 KV 字节高估 4 倍，长上下文 decode 预测偏低。
- **int4 每参数字节用 0.5**：`report.py:39` 忽略 group scale/zero；自家 records/51 用的是 0.53，应统一。
- **报告参数不进 JSON**：`--eff/--bw-eff` 不在结果里，同一份 JSON 能生成不同结论，不可复现。
- **`spread()` 的 `max(w,b)` 恒等于 w**（`Bench.swift:332`）；`precondition` 用在用户输入上；`try?` 吞掉软链错误。
- **`bw_fill` 把同一 buffer 绑到 index 0 和 1**（`Bench.swift:565-568`），kernel 只用 1，语义混乱。
- **冒烟测试上下界过宽**：`tests/smoke.sh` 用 `0.05 < value < 500`，抓不到 2 倍级错误；建议加"与基线 ±30%"。
- **Intel Mac 直接 exit 1**：`simdgroup_matrix` 编译失败没有优雅降级（对纯 Apple Silicon fleet 无影响）。
- **`docs_contract` 不检查 `notes` / 报告数值**：只检查字段名和内核名，所以 P1-3 的过期文案能一直过关。
- **同一 `version`/`schema` 下字段已经变过**：22 份结果全是 `2.0.0` + `macgpu/v1`，但其中 11 份的
  `bandwidth[]` 没有 `gbps_min/gbps_max`（契约第 2 节声明为必有字段）。"改字段不升 schema"已经实际发生，
  下游按契约硬解析会 `KeyError`。

---

## 4. 本机实测数据（Apple M1 / 8 核 GPU / 16 GB）

### 4.1 本次受控读数

数值为 3 次独立进程（每次 `--repeat 5`）的中位数，CV 为这 3 次的轮间变异系数；测试时 load 2.0。
下表是 **v2.0（as-found）** 的峰值口径读数，用于审计；**v2.1 复标定**（中位数，无并发进程）为
FP32/FP16 矩阵指令 2.28 / 2.32 TFLOPS、FMA 1.10 / 1.90、读 58.8 / 写 61.8 / 拷贝 60.0 GB/s，
CV 0.3~1.9%，详见 `mac-gpu-bench/docs/records/52-v2.1-recalibration.md`。

| 指标 | 本次读数 | 3×独立进程 CV | 规格参考 | 达成率 |
|---|---|---|---|---|
| FP32 矩阵指令 (MMA) | **2.27 TFLOPS** | 0.09% | 2.6 | 87% |
| FP16 矩阵指令 (MMA) | **2.29 TFLOPS** | 0.67% | 2.6 | 88% |
| FP32 普通 FMA | **1.09 TFLOPS** | 0.22% | — | 42%（相对标称） |
| FP16 普通 FMA | **1.89 TFLOPS** | 0.48% | — | — |
| GEMM fp32 2048³ | 0.80 TFLOPS | 0.32% | — | 73% of FMA |
| GEMM fp16 2048³ | **1.24 TFLOPS** | 0.30% | — | 65% of FMA |
| GEMM int8 2048³ | 0.31 TOPS | 0.33% | — | 13% of FP16 MMA |
| GEMM int4 2048³ | 0.30 TOPS | 0.58% | — | 13% of FP16 MMA |
| GPU 读 | **58.5 GB/s**（min 55.6） | 0.41% | 68.25 | 86% |
| GPU 写 | 51.9 GB/s（min 47.5） | 2.21% | 68.25 | 76% |
| GPU 拷贝（blit r+w） | 59.3 GB/s（min 57.9） | 0.75% | — | — |

**重要观察**：同会话 3 次独立进程的 CV 只有 0.1~2.2%——**工具本身的重复性很好**。
两次独立采集（相隔 3 分钟）对比：

| 指标 | 会话 1（16:33） | 会话 2（16:36） | 差异 |
|---|---|---|---|
| FP16 MMA | 2.294 | 2.311 | +0.7% |
| FP32 FMA | 1.088 | 1.091 | +0.3% |
| GEMM fp16 | 1.237 | 1.234 | −0.2% |
| 读带宽 | 58.46 | 57.85 | −1.0% |
| 写带宽 | 51.90 | 51.88 | 0.0% |

所以 P0-4 里 20~60% 的跨度**不是测量噪声，而是条件差异**（负载 1.4~3.5、`--repeat` 1~10、尺寸不同）。
这意味着：只要协议固定 + 记录环境，它完全够格做 fleet 基线；反过来，不记录环境的 fleet 汇总基本不可用。

### 4.2 历史 22 次运行的分布（同一台 M1）

见 P0-4 表。结论：`fp16 mma` 稳定在 2.20~2.38（±4%），`read` 50~60（±20%），
所以**算力可以做 1~2% 精度的横向比较，带宽必须多次测量并记录环境**，否则 M1 与 M2 的差距会被噪声吃掉。

### 4.3 与我独立探针的差异

| 项目 | macgpu | 我的探针 | 差异原因 |
|---|---|---|---|
| FP32 FMA | 1.09 | 1.09~1.13 | 一致（显式 fma / 16 链 / 标量 32 链都一样） |
| FP32 MMA | 2.27 | 2.34（4 累加器）/ 2.22（8 累加器） | 一致，4 累加器已饱和 |
| 读带宽 | 58.5 | **59.8** @262k 线程（@65k 仅 54~55） | 线程数不同（P1-1） |
| GEMV 型读（decode 形态） | 未测 | 58.4~59.4 | 流式读可作为 decode 代理，我的结论是**可以** |

---

## 5. 如果要拿它评估不同 Mac 设备的「实际算力与带宽」

### 5.1 现成可用的一键采集

我在 `macgpu-fleet/collect.py` 写了一个采集器（已在本机跑通）：

```bash
python3 macgpu-fleet/collect.py collect --bench ./mac-gpu-bench \
        --launches 3 --repeat 5 --out ./macgpu-fleet/results

# 汇总所有机器的 summary.json：
python3 macgpu-fleet/collect.py table macgpu-fleet/results
```

它做的事（正是 macgpu 缺的那部分）：

1. 采集前记录 `hw.model`、芯片、GPU 核数、内存、`sw_vers` build、`pmset` 电源/低电量/热限速、loadavg
   （本机 `pmset -g` 不暴露 `lowpowermode`，此时该字段留空而不是当成"已关闭"）；
2. 先跑 `selfcheck`，失败直接标记该机不可用；
3. 跑 **3 次独立进程**（不是一次进程 `--repeat 3`），每次 5 轮；
4. 每列取**中位数**并算轮间 **CV**，写 `summary.json` + 追加 `fleet.csv`；
5. 自动告警：selfcheck 失败、read CV > 5%、fp16 mma CV > 3%、load > 2、LPM 未关、电池供电；
6. `collect.py compare` 逐项检查协议（工作集/GEMM 规模/线程/repeat/schema）、OS build、binned 核数，
   不一致直接退出码 3，并给出按 GPU 核数归一的读数。

本机输出（示例）：

```
Apple M1 / Macmini9,1 / GPU 8 核 / macOS 27.0.1 (26A434)
  selfcheck: PASS  launches=3 repeat=5
  FP32 MMA      2.27 TFLOPS CV=0.09%
  FP16 MMA      2.29 TFLOPS CV=0.67%
  FP32 FMA      1.09 TFLOPS CV=0.22%
  FP16 FMA      1.89 TFLOPS CV=0.48%
  GEMM fp16     1.24 TFLOPS CV=0.3%
  GEMM int8     0.31 TOPS   CV=0.33%
  read         58.46 GB/s   CV=0.41%
  write        51.90 GB/s   CV=2.21%
  copy         59.27 GB/s   CV=0.75%
  ⚠ 测试时 load_avg > 2.0
```

### 5.2 采集协议（每台机器照做）

**前置条件**

- 接 **AC 电源**；关闭**低电量模式**（`pmset -g` 留档）；显示器可以休眠，但不要开省电降频模式。
- 退出所有 GPU 相关应用（浏览器视频、Xcode、Docker、模型服务），静置 5 分钟。
- 环境温度 20~25 ℃，不要放在被子/支架上；笔记本不要合盖。
- 外接显示器会占用 GPU 带宽，**同组对比必须一致**（要么都不接，要么都接）。

**执行**

1. `./macgpu selfcheck` → 必须 PASS。
2. `python3 macgpu-fleet/collect.py collect --launches 3 --repeat 5`。
3. 记录 `hw.model`（如 `Mac14,6`）与 **GPU 核数**——同一芯片有 14/18 核、30/40 核等 binned 版本，
   只写"Apple M3 Pro"无法区分，而它们带宽/算力差 20%+。
4. 有条件再跑一次 **10 分钟持续负载**（`--mode sustained` 未实现，可用 `--repeat 20` 近似），
   记录后段相对前段的降频比——这决定长上下文 TTFT 与持续 TG TPS。

**验收（不满足就重测或标记 unreliable）**

| 检查 | 阈值 |
|---|---|
| `selfcheck` | PASS |
| 读带宽轮间 CV | ≤ 3% |
| fp16 MMA 轮间 CV | ≤ 2% |
| 测试时 load_avg_1m | ≤ 2.0 |
| 电源/低电量 | AC / LPM=0 |
| 拷贝带宽 vs 读带宽 | 0.9~1.1（明显偏离说明 blit 路径异常） |

### 5.3 设备清单与规格参考

下表**不是实测值**，是公开规格（带宽改为十进制 GB/s；TFLOPS 为 128 lane × 2 × 频率的推算），
**仅作量级参考，具体以 Apple 官方规格页为准**。
实测列请用上面的采集器填。注意：仓库自带的 `SPEC` 表把 M1 Pro/Max/Ultra 写成 204.8/409.6/819.2，
是把 200/400/800 GB/s 按 1.024 放大，且子串匹配有 P0-1 的 bug——**不要直接引用它的"占规格"**。

| 芯片 | GPU 核（binned） | FP32 标称 TFLOPS（满配） | 带宽 GB/s | 实测 FP16 MMA | 实测读 GB/s | 实测写 GB/s | 备注 |
|---|---|---|---|---|---|---|---|
| M1 | 7 / 8 | 2.6 | 68.25 | 2.29（8 核，本机） | 58.5 | 51.9 | 基线 |
| M1 Pro | 14 / 16 | 5.2 | 200 | 待测 | 待测 | 待测 | |
| M1 Max | 24 / 32 | 10.4 | 400 | 待测 | 待测 | 待测 | |
| M1 Ultra | 48 / 64 | 20.8 | 800 | 待测 | 待测 | 待测 | 双 die |
| M2 | 8 / 10 | 3.6 | 100 | 待测 | 待测 | 待测 | |
| M2 Pro | 16 / 19 | 6.8 | 200 | 待测 | 待测 | 待测 | |
| M2 Max | 30 / 38 | 13.6 | 400 | 待测 | 待测 | 待测 | |
| M2 Ultra | 60 / 76 | 27.2 | 800 | 待测 | 待测 | 待测 | |
| M3 | 8 / 10 | 4.1 | 100 | 待测 | 待测 | 待测 | |
| M3 Pro | 14 / 18 | 7.4 | 150 | 待测 | 待测 | 待测 | 两种 bin |
| M3 Max | 30 / 40 | 16.4 | 300 / 400 | 待测 | 待测 | 待测 | 30 核版带宽 300 |
| M4 | 8 / 10 | 4.3 | 120 | 待测 | 待测 | 待测 | |
| M4 Pro | 16 / 20 | 8.6 | 273 | 待测 | 待测 | 待测 | |
| M4 Max | 32 / 40 | 17.2 | 410 / 546 | 待测 | 待测 | 待测 | 32 核版 410 |
| M3 Ultra | 60 / 80 | — | 约 819 | 待测 | 待测 | 待测 | SPEC 表缺失 |
| M5 系列 | — | — | — | 待测 | 待测 | 待测 | SPEC 表缺失 |

> 这些标称值只用来判断"实测是否量级合理"（一般 MMA 达 85~90%、读带宽达 80~90%）；
> **横向排序必须用实测中位数**，且同芯片不同 bin 要按 GPU 核数分组。

### 5.4 跨代可信度：M1~M4 与 M5 之后

**结论分三层：测量可信、比较有前提、外推有红线。**

| 比较场景 | 可信度 | 原因 / 必须先确认 |
|---|---|---|
| 同机、同 OS、同协议，跨进程重跑 | **高** | 本机两次独立进程：FP32 MMA 2.275 vs 2.27、读 58.8 vs 58.9（差 <0.5%）；FP16 2.32 vs 2.26（2.5%，仍在轮间噪声内） |
| 同代、同 GPU 核数、不同机型 | **较高** | 架构与编译器相同；仍需 AC、热状态、负载一致 |
| 同代、不同核数（binned） | **中** | 算力按核数折算有效；**带宽不随核数线性缩放**，不要折算 |
| M1 ↔ M2 ↔ M3 ↔ M4 | **中** | 读数本身可比（同一套内核/协议/自检），但 **FP16/FP32、MMA/FMA、每核带宽逐代变化**，M1 的比例不能当常数 |
| M1~M4 ↔ M5+ | **低~中** | 测量仍有效，但 M5 在 GPU 核内加入神经加速器，MMA 峰值与 FP16/FP32 关系可能整体改写；规格表未收录，工具会跳过规格对照而不是硬套 M1 |
| Air ↔ Pro/Studio | **低（持续负载）** | 无风扇机型十几秒就可能降频；短时峰值可比，持续吞吐不可比 |

为此这一版做了三件事，避免"跨代继续输出错误结论"：

1. 报告里所有比例（FP16/FP32、MMA/FMA、FMA 占矩阵指令）改为**按本机实测输出**，
   不再写死「FP16 与 FP32 同速」「FMA 只有一半」；M5 上若 FP16 是 2×，报告会如实写 2×。
2. 规格表未收录的芯片（M5 及以后）显式告警并跳过第 6 节；binned 核数不一致时按核数折算算力并告警。
3. `compare` 把协议差异变成硬失败，防止"工作集/线程/OS build 不同"的数字被放在一张表里排序。
4. `report.py --raw` 只输出实测表（算力 / GEMM / 带宽），跳过规格对照、roofline 与结论；
   如果你只用实测数据，这是推荐入口。终端入口也收敛为一条：`./macgpu`（无参数即全套），
   跑完直接打印「推荐数据」，不依赖报告生成；带宽线程按机器自动调优。

**"只保留实测"能去掉换算误差，但不能替代协议约束。** 同一组数字如果不锁定
`--size-mb` / `--bw-threads` / `--repeat`、不做 selfcheck、不排除并发，仍然只是
"某个参数下的实测"。这一版把这四件事都变成了 JSON 字段和自动检查，而不是靠人记。

**仍然不可信、需要补测的部分**：持续负载热稳态（`--mode sustained` 未实现）、
大核数 GPU 的带宽线程数（262,144 只在 M1 8 核验证过，40/76 核需重新扫）、
M5+ 的规格与 roofline 效率系数（必须用真实模型重标，不能沿用 0.8）。

---

## 6. 作为 TTFT / TG TPS 指标：正确用法与错误用法

### 6.1 它到底提供了什么

- `C_d`：设备 FP16 矩阵指令 MAC/s（**算力上限**）
- `B_d`：设备读带宽 GB/s（**decode 带宽上限**）
- `G_d`：手写 GEMM 的 fp16 TFLOPS（**"不用矩阵指令"时的现实上限**）
- `I8/I4`：int8/int4 GEMM（**量化不提算力**的证据）

它**不提供**：TTFT、排队、batch 调度、KV 管理、ANE 加速、模型加载、tokenizer、网络。

### 6.2 错误用法（现在报告默认就在这么干）

```text
prefill tok/s = C_d × 0.4 / params          # 0.4 与自家 0.78~0.83 矛盾
decode  tok/s = B_d × 0.9 / (params×bpp + KV)   # 无 batch，KV 按 MHA
```

问题：
1. `0.4` 把 MLX/MPS 的 prefill 低估 ~2 倍（P0-6）；
2. TTFT = 预填充时间 + 固定开销 + 排队；上式的 "tok/s" 还要除以 prompt 长度才是 TTFT，
   而固定开销（首 token 采样、调度、图编译）在短 prompt 上占比很大；
3. decode 只在 **batch=1 且带宽受限**时成立；batch 一大就变成算力受限（见 6.4）；
4. KV 按 MHA 算，GQA 模型会高估 4 倍；MoE 模型的"每 token 权重字节"根本不是全量。

### 6.3 推荐用法一：归一化比值（最能发挥它价值的方式）

对每一次模型性能 run，同时记录设备读数，计算：

```text
prefill_util = observed_prefill_TFLOPS / C_d      # 期望：MLX 0.75~0.85，手写 0.3~0.6
decode_util  = observed_decode_bytes_per_s / B_d  # 期望：0.85~1.00
```

- `prefill_util > 1`：说明你的参数量/提示 token 数算错了，或用了 ANE；
- `decode_util > 1`：说明每 token 字节数被高估（GQA/MoE/量化 scale），或权重被缓存；
- 同一运行时在不同设备上的 util 应当接近——**util 稳定、绝对 tok/s 变化**，才是"硬件差异"；
  util 也大幅波动，说明是"实现/模型差异"。
- 用 util 做跨设备归一化，比拿 TTFT 绝对值排序稳得多（TTFT 还受 prompt 长度、cache、并发影响）。

### 6.4 推荐用法二：带 batch 和固定开销的 roofline（做容量规划时）

```text
每步字节      = W + b × T × K                    # W 权重；K 每 token KV 字节；b batch；T 上下文
每步算力      = b × 2 × P                        # FLOP

t_step = max( (W + b·T·K) / (B_d·η_b),  b·2P / (C_d·η_c) ) + t_overhead
TG TPS = b / t_step
TTFT  ≈ max( 2·P·T_prompt / (C_d·η_c),  W/(B_d·η_b) ) + t_fixed
```

用本机数（M1，7B、INT4 3.5 GB、2k 上下文）代入：

- `W/B = 3.5 GB / 52 GB/s ≈ 67 ms`；
- 单序列算力：`2×7e9 / 1.84e12 ≈ 7.6 ms`；
- **batch=1**：67 ms → **~15 tok/s**（与报告 11.4 的差别来自 KV：2k MHA KV 1 GB，加上后 ~11.4）；
- **batch=8**：KV 8 GB → 11.5 GB / 52 ≈ 221 ms → 36 tok/s（仍带宽受限）；
- 但如果是 **GQA（8 KV 头）**：KV 只有 0.26 GB/seq，batch=32 时
  `W+32×0.268≈12.1 GB → 233 ms` vs `32×7.6=243 ms` → **开始算力受限**。

结论：**"decode 只受带宽限制"只在 MHA + 小 batch 下成立**；GQA + batching 后算力重新变成瓶颈。
这正是单看一个 `read GB/s` 数字不够、必须配合 `C_d` 与模型形状的原因。

### 6.5 推荐用法三：只做 sanity bound

拿到任何 TTFT/TG TPS 后，先算一下它对应的 util：

- util > 1.0 → **一定是测量或假设错了**（不可能超过物理上限）；
- util < 0.3 → 实现还有 3 倍空间，或模型参数/KV 估算错了；
- 0.3~0.6 → 手写内核/非矩阵路径的正常水平；
- 0.75~0.9 → MLX/MPS/llama.cpp 这类成熟实现的正常水平。

`docs/records/51` 里 Qwen "decode 折算带宽 111%" 就是 util > 1 的实例。
作者用"测试时 load 2.5、空载读到 59.8"解释；但我的探针显示**同一进程内把读线程从 65k 提到 262k
就能 +8%（54~55 → 59.8）**，所以至少有一部分偏差来自基准本身，不必全部归因于负载。
这类"超过 100%"的记录不应该被解释掉，而应该当成基准有 bug 的信号去排查。

---

## 7. 修复路线图（按投入产出排序）

### 第 1 档：先把"会错"变成"不错"（P0）——✅ 已全部落地

1. ✅ 规格匹配改最长前缀 + 未知芯片告警（P0-1）。
2. 🔁 `gbps_max` 经复核是冗余字段而非 bug，改为在契约里写明并补中位数（P0-2）。
3. ✅ 命令缓冲 status/error 检查 + 物理上界（P0-3）。
4. ✅ `HostInfo` 加 thermal/LPM/power/os_build，报告对异常样本打标（P0-4）。
5. ✅ 每指标 9 个原始样本 + 中位数，JSON 写 `sampling`，正文用中位数（P0-5）。
6. ✅ 报告默认 `eff=0.8`，并列 0.4 手写档（P0-6）。
7. ✅ MCP 摘要键改 `(dtype,path)`（P0-7）。

### 第 2 档：把"能比"变成"好比"——大部分已落地

8. ✅ 带宽线程 65,536 → 262,144，线程数写入 JSON（P1-1）。
9. ✅ `units / unit_threads / macs_per_iteration / total_macs`（P1-2）；`notes` 改成与实现一致的固定文案（P1-3）。
10. ✅ `selfcheck` 内嵌进 `run`，失败退出码 2（P1-4）；累加器清零、fp16 迭代 96→16、GEMM 用非精确输入（P1-5）。
11. ✅ `git_sha / git_dirty / os_build`（P1-6）。Metal 语言版本没有公开 API 可读，用 OS build 归因。
12. ⬜ GEMM shape sweep + MPS/MLX 对照（P1-7）——未做。
13. ✅ 毫秒时间戳、`--tests` 白名单、GEMM 工作集上限、`try!` 改 do/catch、并发 pid 锁（P1-9/10）。

### 第 3 档：把"设备读数"变成"评估体系的一部分"

14. ⬜ `--mode sustained`（10 分钟热稳态）（P1-8）。
15. ⬜ 新增 `gemv_batch`（batch 1/4/16/64）与 `gemm_sweep`，直接对应 TG TPS/TTFT 的形状；
    这两个比 N=2048³ 有用得多。
16. ⬜ 在 `modelbenchmark` 的 perf run 里存 `roofline_ratio` 与设备 JSON 的 hash，
    让"设备能力"与"模型表现"在同一张表里可追溯。

---

## 8. 最终建议

- **可以用**：把 `macgpu` 当"设备算力/带宽的标定器"，每台机器采一次，存 JSON，作为模型 TTFT/TG TPS 的**分母**。
- **不要直接用**：不要用报告第 5 节的表当 TTFT 预测；不要用它的"占规格百分比"做机型结论；
  不要把 `--repeat` 不同的历史结果放在一起比。
- **最值得补的两个数**：① 带宽读测试线程数修到 262k 级（+8%）；② GEMM 按 M=1/32/128/512 做 shape sweep。
  这两个补上之后，它对 TTFT/TG TPS 的解释力会从"参考"变成"能定位瓶颈"。
- **它的最大价值不是数字本身，而是"设备侧解释变量"**：同一个模型在 M1 和 M4 Pro 上差 2 倍，
  到底是算力差还是带宽差？`C_d` 与 `B_d` 的比值直接回答这个问题——这比任何一个 tok/s 都更有信息量。

---

## 附录 A：独立探针（关键代码与结果）

探针不依赖 `mac-gpu-bench`，在同一台机器上单独编译运行；核心是与它**同名同结构**的内核加变体，
计时同样用 `gpuStartTime/gpuEndTime`，先预热再 9 次取 min。

```metal
// 它的内核：a*x+y，8 条 float4 链，整条写回
// 变体 1：显式 fma(a, xv, yv)
// 变体 2：16 条链
// 变体 3：32 条标量链（同样 32 MAC/iter，但 32 条指令）
// 变体 4：4 累加器 vs 8 累加器 的 simdgroup MMA
// 变体 5：read1（1 次 load/iter）vs read4（4 次独立 load/iter），线程数 65k/131k/262k
```

结果（M1 / 8 核，load ≈ 1.9）：

```
_fma8           131k  1.11 TFLOPS      _fma8_explicit  131k  1.11
_fma8           262k  1.11              _fma8_explicit  262k  1.11
_fma8           524k  1.10              _fma8_explicit  524k  1.11
_fma16          131k  1.13              _fma_scalar32   131k  1.13
_fma16          262k  1.12              _fma_scalar32   262k  1.12
_fma16          524k  1.12              _fma_scalar32   524k  1.12

_read1  65k  54.3~55.2 GB/s   _read4  65k  56.0
_read1 131k  55.1             _read4 131k  58.8
_read1 262k  59.8             _read4 262k  59.8
_gemv   262k 59.4             （decode 形态读 ≈ 流式读）

_mma8_f32（8 累加器/warp，4096 warps） 2.22 TFLOPS
```

三条结论：
1. **FP32 FMA ≈ 1.1 TFLOPS 是硬件上限**：显式 fma、16 链、32 标量链、3 种线程数全部落在 1.09~1.13。
   同时说明「标量只有 1/4」是错的——总吞吐一样，只是每线程 MAC 数不同。
2. **MMA 用 4 个累加器已经饱和**（8 个反而略低），它的峰值内核写法没问题。
3. **读带宽随线程数上升 8%**：它的 65k 线程不是最优；decode 预测因此偏低。

## 附录 B：复现命令

```bash
cd mac-gpu-bench
./macgpu info --json
./macgpu selfcheck
./macgpu run --repeat 5 --json --out /tmp/m1.json
python3 tools/report.py /tmp/m1.json --eff 0.8 --bw-eff 1.0

# 跨设备采集（本仓库新增，已在本机验证）
cd ..
python3 macgpu-fleet/collect.py collect --bench ./mac-gpu-bench --launches 3 --repeat 5
python3 macgpu-fleet/collect.py table macgpu-fleet/results
```

> 说明：本报告里的"实测"均为本机（Macmini9,1 / M1 / 8 核 / 16 GB / macOS 27.0.1）读数；
> 其余机型只给了公开规格参考，需按第 5.2 节协议在对应机器上采集后填入。
