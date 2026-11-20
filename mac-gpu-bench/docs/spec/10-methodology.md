# 测量方法 spec / Methodology

> 状态：**current**。本文件定义「测什么、怎么算」。改测量项或计数口径必须先改这里，
> 并同步 `src/Kernels.swift` 的内核与 `src/Bench.swift` 的 `ComputeCases`。

## 1. 指标定义

| 指标 | 单位 | 定义 |
|---|---|---|
| 算力峰值 | TFLOPS（fp32/fp16）、TOPS（int8） | 该通路单位时间完成的浮点/整数运算数，1 MAC = 2 OPS |
| MAC 速率 | TMAC/s | `运算数 / 2 / 秒`，消除「乘加算几次」的歧义，跨精度可比 |
| 矩阵乘算力 | TFLOPS / TOPS | `2 × N³ × reps / 秒`，N = GEMM 边长 |
| 带宽 | GB/s | `每轮搬运字节数 × reps / 秒`。读/写各算一次，拷贝算读写两次 |

算力统一按 **实测内核实跑的乘加次数** 计数：

```
MACs = units × iterations × macs_per_iteration × reps
FLOPS = MACs × 2
value = FLOPS / seconds / 1e12
```

## 2. 通路（path）与内核

Metal 上有两条完全不同的执行通路，同一颗 GPU 的吞吐差一倍：

- `mma` —— 矩阵指令。只有它能逼近芯片标称算力，也是 GEMM 库（MLX、MPS）实际走的路。
- `fma` —— 普通着色器 ALU。手写算子、非矩阵形状的算子都落在这一档。

> 历史坑：`fma` 通路曾按「向量 4-wide」解释，实测证明是错的——标量 ILP 与 float4 同速，
> 差异来自**发射率**而不是向量宽度。不要再引用「4-wide」说法。

## 3. 指标矩阵（内核 ↔ 计数）

`make docs` 会校验：下表出现的每个内核名都能在 `src/Kernels.swift` 里找到，
且 `src/Kernels.swift` 里每个 `peak_*` / `gemm_*` 内核都在下表出现过。

| Path | dtype | kernel | MACs / iter | units | unit |
|---|---|---|---|---|---|
| Matrix | fp32 | `peak_mma_f32` | 2048 / warp | 4096 warps | TFLOPS |
| Matrix | fp16 | `peak_mma_f16` | 2048 / warp | 4096 warps | TFLOPS |
| Scalar FMA | fp32 | `peak_f32` | 32 / thread | 262144 threads | TFLOPS |
| Scalar FMA | fp16 | `peak_f16` | 32 / thread | 262144 threads | TFLOPS |
| Integer ALU | int8 | `peak_i8` | 32 / thread | 262144 threads | TOPS（int32 IMAD，无 tensor 通路） |
| GEMM | fp32 | `gemm_f32` | N³ / 矩阵 | — | TFLOPS |
| GEMM | fp16 | `gemm_f16` | N³ / 矩阵 | — | TFLOPS |
| GEMM | int8 | `gemm_i8` | N³ / 矩阵 | — | TOPS |
| GEMM | int4 | `gemm_i4` | N³ / 矩阵 | — | TOPS（每字节 2 个 4-bit，解包计入耗时） |

计数依据（必须与内核逐条对上）：

- `peak_mma_*`：每 warp 4 个 8×8 累加器，每个累加器每轮 64 MAC → `4 × 64 = 256`，
  而 `simdgroup_multiply_accumulate` 对 8×8 是 `8×8×8 = 512` MAC/指令，
  因此 4 个累加器 = `4 × 512 = 2048` MAC/warp/iter。
- `peak_f32` / `peak_f16`：8 条向量链 × 4 分量 = 32 MAC/thread/iter。
- `peak_i8`：同结构的 8 条 uint4 链，把 FMA 换成 int32 乘加（IMAD），用来量整型 ALU 的真实上限；
  结果与 CPU 逐元素必须**完全相等**（整数无舍入），CPU 对不上就说明循环被优化掉了。
- `gemm_*`：每个输出元素 N 次乘加，共 N² 个输出 → N³ MAC；real GEMM 不可被编译器折叠。

## 4. 为什么不用微基准测 INT8 / INT4

整型微基准的乘数是循环不变量，编译器会把整段折叠成「每轮加一个常数」，甚至直接算成
`c += iters × k`。实测虚高 **100 倍**（84 TOPS）。因此量化算力改测**真实 int8 GEMM**
（与 CPU 整数结果逐元素完全一致），INT4 不单独出算力数字：没有 tensor 指令时它与 int8
走同一条 ALU 路径，收益只体现在带宽。

## 5. 带宽

- 读：`float4` 网格步长累加，只读不写。
- 写：`float4` 网格步长写入，只写不读。
- 拷贝：`MTLBlitCommandEncoder`，由 DMA 完成，读写各计一次。
- 缓冲区在测试前用 LCG 填充**不可压缩数据**——零页会被 macOS 内存压缩，实测虚高到 89~121 GB/s。
- 缓冲区默认 256 MB，远大于 SLC，避免命中缓存。
- 派发 **262,144 线程**（8192 warps）。M1 上实测：65,536 线程读 54~55 GB/s，
  262,144 线程 58~60 GB/s，**旧默认值低估约 8%**；网格步长循环保证线程多于元素时不会越界。
- 记录**中位数**（`gbps_median`，稳定可达）与**最快一次**（`gbps`，≈硬件上限）。
  `gbps_min/gbps_max` 同时落盘，用来判断本次读数是否被后台任务污染；
  按定义 `gbps == gbps_max`（同一个「最快」值），两个字段并存只为兼容旧结果。
- 缓冲区大小会影响读数（64 MB 可能部分命中 SLC，实测比 256 MB 高 5~10%），
  **跨设备比较必须用同一 `--size-mb`**。
- **线程饱和检查（每台机自动做）**：默认线程测完后，再用 2× 线程复测一遍，把读/写/拷贝的
  中位数增益写进 `bandwidth_saturation`。任一增益 ≥ 3% 即判为**未饱和**，脚本会**自动改用
  2× 线程跑后面的 3 轮**（不需要手工加参数），选择结果写进 JSON；显式传 `--bw-threads` 则尊重用户值。
  这样同一份脚本在大核数 GPU 上也能自己找到 saturated 的线程数。
  默认起点 262,144 是在 M1 8 核上标定的。

## 6. 计时与统计

- 用 `MTLCommandBuffer.gpuStartTime/gpuEndTime`，只算 GPU 执行时间，不含提交开销。
- 每个指标先用 1 次派发校准迭代次数，使单次批处理落在 20~120 ms。
- 每个指标先**预热一次**并丢弃（首次派发有页错误、时钟爬升，曾观测到 22 GB/s 的假低值）。
- 同一指标再连续采样 9 次，**全部原始秒数落盘**（`samples`），同时给出：
  - `value` = 最快一次（峰值，与历史数据兼容）；
  - `value_median` = 中位数（跨设备比较、roofline 估算用这个）。
- `--repeat N` 表示整套测试跑 N 轮，轮间取峰值；`--repeat` 不同会带来 ~4% 的偏差
  （M1 上 1 轮 2.28、10 轮 2.37 TFLOPS），**跨设备比较固定用 3 轮 × 每轮 5 次采样**。
- 命令缓冲必须 `status == .completed` 且 `error == nil` 才计入；耗时 < 1 µs 视为未真正执行。
  旧实现只做 `max(dt, 1e-9)`，一次失败的派发能算出上亿 TFLOPS，还会被「取最快」保留。
- 每个指标还有物理上界（算力 < 1000 TFLOPS、带宽 < 10 TB/s），越界直接报错而不是写进结果。
- 结果记录 `host.load_avg_1m`、`thermal_state`、`low_power_mode`、`power_source`、`os_build`、`git_sha`；
  报告在负载 > 2.0、热状态非 nominal、低电量、电池供电时给出告警。

## 7. 自检（内嵌进每次 run）

`run` 在测性能之前先跑完整自检（`--no-selfcheck` 可跳过），结果写入 `result.json` 的
`selfcheck` 字段；`passed == false` 时 `run` 以退出码 2 结束。自检失败的数据不要用于对比。

| 检查 | 内容 | 判据 |
|---|---|---|
| `gemm_i8` / `gemm_i4` | 128³ GEMM vs CPU 整数参考 | 逐元素完全相等 |
| `gemm_f32` / `gemm_f16` | 128³ GEMM vs CPU 浮点参考（输入用非二进制精确数） | 相对误差 ≤ 2e-3 / 2e-2 |
| `*_mma_*` | 峰值内核输出 vs CPU 参考（累加器显式清零） | 相对误差 ≤ 1e-4 / 2e-2 |
| `*_fma_*` | 同上，检查整条向量都写回（防 DCE） | 相对误差 ≤ 1e-4 / 2e-2 |
| `bandwidth` | 读/写/拷贝 | 5 ~ 5000 GB/s |

## 8. 跨代与跨设备可比性

工具测的是「这台机器的 Metal 通路能跑多快」，**不是「这颗芯片的架构上限」**。跨设备比较前先过一遍：

### 8.1 必须一致的协议参数

| 参数 | 写入位置 | 不一致的后果 |
|---|---|---|
| `bandwidth[].bytes_per_rep`（默认 256 MB） | JSON | 小于 SLC 会测到缓存带宽，M1 上 64 MB 与 256 MB 差 5~10% |
| `bandwidth[].threads`（默认 262,144） | JSON | 线程不足会测不到峰值；大核数 GPU 需重新验证 |
| `gemm[].size`（默认 2048） | JSON | 工作集不同，GEMM 达成率不可比 |
| `runs` / `sampling.samples_per_metric` | JSON | 峰值与中位数混用会引入 +2~4% 偏差 |
| `host.os_build` | JSON | Metal 编译器随 OS 变，同一份 MSL 可能生成不同指令 |

上表各项不一致时，读数**不得放进同一张表排序**：这些字段都写在 `result.json` 里，跨机对比前
逐项比对。同代不同核数（binned）的机器按 GPU 核数折算算力，但**带宽不按核数折算**。

### 8.2 架构代差：哪些结论不能外推

- **FP16/FP32 的矩阵指令比例**：M1~M4 上是 1:1；M5 起 GPU 核内加入神经加速器，
  比例可能变成 2:1 或更高。报告已改为按实测比值输出，不再写死「同速」。
- **MMA/FMA 比例**：反映每代矩阵单元的相对宽度，M1 约 2:1，M3/M4 每核 FP32 更高，
  M5 可能再变。这个比值必须逐代实测。
- **内存带宽与 GPU 核数的关系**：不是线性的（binned Max 的带宽随内存控制器减少），
  所以 `compare` 只按核数折算算力，不折算带宽。
- **Neural Engine / GPU 神经加速器**：不通过 Metal 暴露，本工具不测；不要用 Metal 数字推断 ANE。

### 8.3 机型形态

同样标称 M4 Pro 的 MacBook Air 与 Mac Studio，持续负载下的差距远大于一代架构差：
Air 无风扇，`--repeat 3` 的十几秒就可能开始降频。**跨形态只比「短时峰值」，
比持续吞吐要用 `--mode sustained`（尚未实现）。**

### 8.4 binned SKU

同一芯片名有不同 GPU 核数（如 M4 Max 32/40 核、M3 Pro 14/18 核）。
`result.json` 的 `gpu.core_count` 是实测值；报告发现核数与规格表不一致时，
会按核数折算算力并显式告警。**横向排序请按「芯片 + GPU 核数」分组，不要只按芯片名。**
