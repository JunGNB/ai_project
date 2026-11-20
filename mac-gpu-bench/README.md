# macgpu —— Mac GPU Compute & Bandwidth Benchmark

在 Mac（Apple Silicon，M1~M5 通用）上实测 **GPU 算力**与**内存带宽**，
直接给出评估 **prefill / decode** 需要的实测数据。零第三方依赖：只用系统自带的 `swiftc` + Metal。

**测什么**：同一颗芯片上有两条完全不同的执行通路，本项目分开测——

| Path | 含义 | 本机（M1 / 8 核 GPU） |
|---|---|---|
| **Matrix** | `simdgroup_matrix` 矩阵指令，GEMM 库实际走的路 | 2.2 ~ 2.3 TFLOPS |
| **Scalar FMA** | 普通着色器 ALU，手写算子落在这里 | 1.0 TFLOPS（fp32）/ 1.8 TFLOPS（fp16） |
| **Real GEMM** | 手写 64×64 分块矩阵乘，看“真实内核”达成率 | fp32 0.8 / fp16 1.2 / int8 0.3 / int4 0.3 |
| **Bandwidth** | GPU 读 / 写 / 拷贝 | 55~60 GB/s（规格 68.25） |

数值口径：正文用 **9 次采样的中位数**（跨设备可比），同时给出峰值（最快一次）。
每次 `run` 会先跑数值自检，结果写进 `selfcheck` 字段；失败则退出码 2、数据不采用。

## 快速开始（只有一个入口）

```bash
cd mac-gpu-bench
./macgpu              # 默认 3 轮
./macgpu --repeat 10  # 想更稳就多跑几轮（每轮约 10 秒）
```

就这一条。它会：编译 → 数值自检 → 算力 / GEMM / 带宽实测（3 轮）→
在终端打印 **[4] 推荐数据**，并把完整 JSON 写到 `output/results/<时间戳>/result.json`。

```
[4] 推荐数据 / Recommended — 跨设备对比只用这一节（每轮 9 次采样取中位数）
    ★ prefill   FP16 Matrix      2.25 TFLOPS
    ★ decode    GPU read         58.0 GB/s
    环境：热状态 nominal · 电源 ac · 负载 1.44 · 自检 12/12 PASS · 带宽已饱和
```

**不需要记别的命令。** 下面这些只有排查或自动化时才用：

| 命令 | 什么时候用 |
|---|---|
| `./macgpu info` | 只想看芯片 / GPU 核数 / 内存 / OS |
| `./macgpu selfcheck` | 怀疑数值不对，单独验证内核 |
| `make report` | 想把 JSON 转成 Markdown（可选，不影响测量） |
| `make test` / `make docs` | 改代码后自检 |

第一次运行自动编译（约 3~5 秒），全程约 40 秒。带宽对后台负载敏感：
跑之前退掉浏览器视频 / Xcode / Docker；`./macgpu` 已内置并发锁。

## 三条关键结论（M1 实测）

**1. 只有矩阵指令能跑满 GPU。** `simdgroup_matrix` 实测 2.2~2.4 TFLOPS，是规格 2.6 的 85~90%；
**FP16 与 FP32 同速，没有“FP16 双倍”**。

**2. 不用矩阵指令：fp32 掉一半，fp16 还能到八成。** fp32 1.0 TFLOPS（矩阵指令的 48%）/
fp16 1.8 TFLOPS（85%，half2 打包带来的 1.7 倍）。不调矩阵指令的算子，prefill 天花板就在这里。

**3. 量化不提算力，只省带宽。** Metal 没有 int8/int4 tensor 指令：纯整型 ALU 实测 0.61 TOPS
（FP16 的 1/3），int8/int4 GEMM 都只有 0.3 TOPS（FP16 峰值的 1/7）。收益在 decode：
权重 2 → 0.5 字节/参数，token/s 直接翻倍。

## 为什么结果可信

| 坑 | 现象 | 防线 |
|---|---|---|
| 向量分量被当死代码删除 | 只写 `.x` → FP32 虚高 4 倍（3.86 vs 0.97） | 整条向量写回 + 与 CPU 参考逐元素比对 |
| 循环被折叠成常数加法 | int8 微基准虚高 100 倍（84 TOPS） | 放弃微基准，改测真实 int8 GEMM |
| 缓冲区被内存压缩 | 带宽虚高到 89~121 GB/s | 测试前填充不可压缩数据 |
| 命令缓冲执行失败 | 0 秒 → 上亿 TFLOPS，且被“取最快”保留 | 检查 `status/error`、耗时下限、物理上界 |
| 环境不同却直接比 | 低电量/电池/热降频让读数差 10~40% | 记录 OS build/热状态/电源/负载/git sha，报告告警 |
| 固定线程喂不饱大 GPU | 大核数机器被当成“带宽低” | 每次 run 自动用 2× 线程复测；未饱和就自动切换，`bandwidth_saturation` 记录依据 |

`./macgpu selfcheck` 覆盖以上全部：fp32/fp16/int8 GEMM 与 CPU 参考的误差、每个峰值内核的
数值、带宽量级。`make docs` 再把「文档里写的口径」和「代码里的内核/字段」对齐。

## 目录

```
mac-gpu-bench/
├── macgpu             # 入口：按需编译后执行
├── src/               # Swift + MSL 内核（运行时编译）
├── tools/             # report.py / mcp_server.py / docs_contract.py
├── tests/             # 冒烟测试 + 文档契约测试
├── docs/              # spec / design / guides / process / records
├── AGENT.md           # 在本仓库工作的约定（给人也给 Agent）
└── output/            # 全部本机产物（不进库）：build/ results/ reports/
```

## 文档

| 想知道 | 看 |
|---|---|
| 测什么、公式怎么算 | [docs/spec/10-methodology.md](docs/spec/10-methodology.md) |
| 怎么实现、内核怎么防优化 | [docs/design/20-architecture.md](docs/design/20-architecture.md) |
| JSON 字段、目录契约、版本策略 | [docs/design/21-output-contract.md](docs/design/21-output-contract.md) |
| 命令与结果解读 | [docs/guides/30-quickstart.md](docs/guides/30-quickstart.md) |
| 怎么验证、验收清单 | [docs/process/40-validation.md](docs/process/40-validation.md) |
| M1 基线读数 | [docs/records/50-m1-baseline.md](docs/records/50-m1-baseline.md) |

## 给大模型 / Agent

MCP（stdio）零依赖服务：

```json
{
  "mcpServers": {
    "macgpu": {
      "command": "python3",
      "args": ["/绝对路径/mac-gpu-bench/tools/mcp_server.py"]
    }
  }
}
```

工具：`machine_info`、`run_benchmark`、`get_latest_result`、`list_results`、`generate_report`、`selfcheck`。
Agent 改动本仓库前请先读 [AGENT.md](AGENT.md)。

## 局限

- 峰值是合成负载，代表通路上限，不等于任何具体模型速度。
- 参考 GEMM 未做矩阵指令/双缓冲优化，只代表“普通内核”的达成率。
- 不测 Neural Engine（不经 Metal 暴露）、多 GPU 并发、长时间满载的热降频。
- 带宽随系统负载可差 10~20%，横向比较请在空载时复测。
