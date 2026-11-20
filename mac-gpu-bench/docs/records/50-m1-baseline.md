# 基线记录：Apple M1 / 8 核 GPU / 16 GB

> 状态：**record**（历史读数，不随代码更新；新增机型另起一篇）。
> 复现：`cd mac-gpu-bench && make run && make report`

## 环境

| 项 | 值 |
|---|---|
| 机型 | Macmini9,1（Apple M1） |
| GPU | 8 核，`maxThreadsPerThreadgroup` = 1024 |
| 内存 | 16 GB 统一内存，`recommendedMaxWorkingSetSize` ≈ 12 GB |
| 系统 | macOS 27.0.1 |
| 工具 | macgpu v2.0.0，`--repeat 3` |
| 测试时系统负载 | 2.2 ~ 2.7（后台有未知任务，`load_avg_1m`） |

## 读数

| 指标 | 实测 | 备注 |
|---|---|---|
| FP32 矩阵指令 | 2.28 TFLOPS | 4096 warps，4 × `simdgroup_float8x8` |
| FP16 矩阵指令 | 2.24 TFLOPS | 与 FP32 同速，**无 2 倍** |
| FP32 普通 FMA | 1.10 TFLOPS | 8 × float4 链，整条写回 |
| FP16 普通 FMA | 1.88 TFLOPS | 8 × half4 链，half2 打包 |
| fp32 GEMM 2048³ | 0.74 TFLOPS | 峰值的 32% |
| fp16 GEMM 2048³ | 1.11 TFLOPS | 峰值的 49% |
| int8 GEMM 2048³ | 0.30 TOPS | 无 tensor 指令，ALU 硬算 |
| GPU 读 | 52.5 GB/s | 规格 68.25 的 77% |
| GPU 写 | 47.7 GB/s | 70% |
| GPU 拷贝（blit） | 52.0 GB/s | 76% |

## 结论

1. 标称 2.6 TFLOPS 只在矩阵指令通路成立（85%）；普通 FMA 只有一半。
2. FP16 与 FP32 矩阵指令同速，流传的「FP16 = 2× FP32 = 5.2 TFLOPS」在本机不成立。
3. 量化不提算力：int8 GEMM 仅为 FP16 峰值的 13%，收益在带宽。
4. 带宽 47~53 GB/s，decode 阶段 7B FP16 权重 14 GB → 理论上限约 3 tok/s；
   INT4 权重 3.5 GB → 约 13 tok/s。

## 与公开数据对照

| 来源 | FP16 矩阵指令 | FP32 FMA | 带宽读 |
|---|---|---|---|
| 本次实测 | 2.24 | 1.10 | 52.5 GB/s |
| 同机其他公开测量（本机 `m1-fp16-bench` 记录） | 2.49 | 1.10 | 60.0 GB/s |

差异来源：本记录的带宽是在负载 2.2~2.7 下测的；矩阵指令取 4096 warps，与对方的 warp 数/累加器
组织不同。差 10% 属正常范围，**量级与排序一致**。

## 历史

- **v1.x 的数字全部作废**：峰值内核只写回向量 `.x`，编译器删掉了 3/4 的 FMA，FP32/FP16 虚高约 4 倍
  （曾出现 FP16 = 6.42 TFLOPS），int8 微基准被折叠成常数加法（虚高 100 倍）。
  修复方式见 `docs/process/40-validation.md`；从 v2.0.0 起所有峰值内核都过数值自检。
