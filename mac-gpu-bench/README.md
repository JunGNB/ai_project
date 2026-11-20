# macgpu —— Mac GPU Compute & Bandwidth Benchmark

在 Mac（Apple Silicon）上实测 **GPU 算力**与**内存带宽**，并据此推算模型的
**prefill / decode** 上限。零第三方依赖：只用系统自带的 `swiftc` + Metal。

**测什么**：同一颗芯片上有两条完全不同的执行通路，本项目分开测——

| Path | 含义 | 本机（M1 / 8 核 GPU） |
|---|---|---|
| **Matrix** | `simdgroup_matrix` 矩阵指令，GEMM 库实际走的路 | 2.2 ~ 2.3 TFLOPS |
| **Scalar FMA** | 普通着色器 ALU，手写算子落在这里 | 1.0 TFLOPS（fp32）/ 1.8 TFLOPS（fp16） |
| **Real GEMM** | 手写 64×64 分块矩阵乘，看“真实内核”达成率 | fp32 0.8 / fp16 1.1 / int8 0.3 / int4 0.3 |
| **Bandwidth** | GPU 读 / 写 / 拷贝 | ~50 GB/s（规格 68.25） |

## 快速开始

```bash
cd mac-gpu-bench

./macgpu run --repeat 3   # 全套测试（约 10~15 秒）
make report               # 生成 output/reports/REPORT.md
./macgpu selfcheck        # 数值自检，不过则退出码 2
make test                 # 冒烟测试（编译→实测→自检→文档一致→报告）
make docs                 # 文档与实现一致性校验
```

第一次运行自动编译（3~5 秒）。机器上有其他任务时用 `--repeat 3`：带宽对负载很敏感。

## 三条关键结论（M1 实测）

**1. 只有矩阵指令能跑满 GPU。** `simdgroup_matrix` 实测 2.2~2.3 TFLOPS，是规格 2.6 的 85%；
**FP16 与 FP32 同速，没有“FP16 双倍”**。

**2. 普通 FMA 着色器只有一半。** fp32 1.0 TFLOPS / fp16 1.8 TFLOPS（half2 打包带来的 1.7 倍）。
不调矩阵指令的算子，prefill 天花板就在这里。

**3. 量化不提算力，只省带宽。** Metal 没有 int8/int4 tensor 指令，int8/int4 GEMM 都只有
0.3 TOPS（FP16 峰值的 1/7）。收益在 decode：权重 2 → 0.5 字节/参数，token/s 直接翻倍。

## 为什么结果可信

| 坑 | 现象 | 防线 |
|---|---|---|
| 向量分量被当死代码删除 | 只写 `.x` → FP32 虚高 4 倍（3.86 vs 0.97） | 整条向量写回 + 与 CPU 参考逐元素比对 |
| 循环被折叠成常数加法 | int8 微基准虚高 100 倍（84 TOPS） | 放弃微基准，改测真实 int8 GEMM |
| 缓冲区被内存压缩 | 带宽虚高到 89~121 GB/s | 测试前填充不可压缩数据 |

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
