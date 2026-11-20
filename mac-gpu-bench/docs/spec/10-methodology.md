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
| GEMM | fp32 | `gemm_f32` | N³ / 矩阵 | — | TFLOPS |
| GEMM | fp16 | `gemm_f16` | N³ / 矩阵 | — | TFLOPS |
| GEMM | int8 | `gemm_i8` | N³ / 矩阵 | — | TOPS |
| GEMM | int4 | `gemm_i4` | N³ / 矩阵 | — | TOPS（每字节 2 个 4-bit，解包计入耗时） |

计数依据（必须与内核逐条对上）：

- `peak_mma_*`：每 warp 4 个 8×8 累加器，每个累加器每轮 64 MAC → `4 × 64 = 256`，
  而 `simdgroup_multiply_accumulate` 对 8×8 是 `8×8×8 = 512` MAC/指令，
  因此 4 个累加器 = `4 × 512 = 2048` MAC/warp/iter。
- `peak_f32` / `peak_f16`：8 条向量链 × 4 分量 = 32 MAC/thread/iter。
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
- 记录的是**短脉冲最优值**（9 次采样取最快）≈ 硬件上限；持续满载约低 10%（DVFS 降频）。
  `gbps_min/gbps_max` 同时落盘，用来判断本次读数是否被后台任务污染。

## 6. 计时与统计

- 用 `MTLCommandBuffer.gpuStartTime/gpuEndTime`，只算 GPU 执行时间，不含提交开销。
- 每个指标先用 1 次派发校准迭代次数，使单次批处理落在 20~120 ms。
- 每个指标先**预热一次**并丢弃（首次派发有页错误、时钟爬升，曾观测到 22 GB/s 的假低值）。
- 同一指标再连续采样 9 次取**最小值**（最接近无干扰状态）。
- `--repeat N` 表示整套测试跑 N 轮，各指标仍取全局最优。
- 结果记录 `host.load_avg_1m`；负载 > 1.5 时报告会给出带宽偏低的警告。
