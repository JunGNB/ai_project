# 快速上手 / Quickstart

> 状态：**current**。改 CLI、参数、输出时同步更新。

## 1. 环境

- macOS 11+，Apple Silicon（Intel Mac 会退化为单线程 Metal，数值无参考价值）。
- 只需 **Xcode Command Line Tools**（提供 `swiftc`），不需要 Xcode 工程、不需要 pip 包：
  `xcode-select --install`
- 验证：`swiftc --version` 有输出即可。

## 2. 三条命令

```bash
cd mac-gpu-bench
./macgpu
```

唯一入口。跑完会在终端打印 **推荐数据**（实测中位数），JSON 在 `output/results/<时间戳>/result.json`。
可选：`make report` 把最新 JSON 转成 Markdown；`python3 tools/report.py --raw` 只要实测三张表。

第一次运行会自动编译（约 3~5 秒），产物在 `output/build/`。

## 3. 常用参数

| 参数 | 说明 |
|---|---|
| `--tests compute,gemm,bandwidth` | 只跑其中几组 |
| `--dtypes fp32,fp16,int8,int4` | 只测某些精度（int8/int4 只有 GEMM，没有峰值） |
| `--gemm-size 2048` | GEMM 边长，必须是 64 的倍数；1024 更快，2048 更稳 |
| `--size-mb 256` | 带宽缓冲区大小；小于 64 可能命中缓存 |
| `--repeat N` | 整套跑 N 轮取最优（1~20，默认 3）；机器有后台负载时建议加大 |
| `--no-selfcheck` | 跳过内嵌自检（不建议；跳过后结果标注 `selfcheck.skipped`） |
| `--out PATH` | 指定 JSON 路径 |
| `--json` | 把 JSON 打到 stdout（给脚本/Agent 用） |

```bash
# 想更稳：跑 10 轮（每轮约 10 秒）
./macgpu --repeat 10

# 只想知道 fp16 的上限，输出 JSON 给 jq
./macgpu --json --tests compute --dtypes fp16 | jq '.compute[] | {path, value, unit}'
```

## 4. 怎么读结果

```
[1] Compute peak — per execution path
    FP32 Matrix (simdgroup_float8x8)  FP32   2.21 TFLOPS   1.08 TMAC/s
    FP32 Scalar FMA (shader ALU)      FP32   1.03 TFLOPS   0.51 TMAC/s
```

- 同一 dtype 出现两行不是重复：**Matrix 是矩阵指令通路，Scalar FMA 是普通着色器通路**。
  fp32 的 FMA 通常只有矩阵指令的一半，fp16 能到八成（half2 打包）。
  写算子时能不能用上 `simdgroup_matrix`，直接决定你拿到 1.0 还是 2.2。
- **TMAC/s 列**用于跨精度比较（fp16 的 2 FLOP 和 int8 的 1 次乘加不是一回事）。
- `[2] Real GEMM` 的 `vs ALU peak` 是「手写 GEMM / 同精度的 Scalar FMA 峰值」——GEMM 走的是
  普通 ALU 通路，跟矩阵指令峰值比不公平。这个比值说明分块/访存损失了多少；再乘上
  「矩阵指令峰值 ÷ FMA 峰值」（约 2 倍）才是用上 tensor 通路的理论上限。
- `[3] Memory bandwidth`：读/写/拷贝三个数，decode 阶段基本只取决于读。

## 5. 直接用：[4] 推荐数据块

每次 `run` 的最后会打印一块 **[4] Recommended**，把评估 prefill / decode 该用的行标出来：

```
[4] 推荐数据 / Recommended — 评估 prefill 与 decode 直接用这几行
    用途            指标                      数值           怎么用
    ★ prefill       FP16 Matrix peak          2.11 TFLOPS    模型算力上限（MLX / MPS 走这条；中位数）
      prefill       FP16 Scalar FMA           1.79 TFLOPS    自己写的内核没调 simdgroup_matrix 时用它
      prefill       INT8 GEMM                 0.29 TOPS      量化不提算力：prefill 仍按 FP16 算
    ★ decode        GPU read                  54.5 GB/s      decode 几乎只受它限制（中位数）
```

- **★ 标的两行就是默认推荐值**：`FP16 Matrix peak` 估 prefill，`GPU read` 估 decode。
- 自己写内核（没用 `simdgroup_matrix`）就把 prefill 换成 `Scalar FMA` 那一行。
- INT8/INT4 的算力行只是告诉你「量化不提算力」，估 prefill 时仍用 FP16。
- 块尾会用本机实测值代 7B 算一遍（prefill 给 0.8 / 0.4 两档效率、decode eff=90%），可直接对数量级。

## 6. 关键判据

| 现象 | 含义 |
|---|---|
| 矩阵指令 ≈ 规格 × 0.85 | 正常，说明标称算力是真的 |
| FP32 FMA ≈ 矩阵指令 × 0.5 | 正常，发射率限制；FP16 FMA 约 × 0.8 |
| 读带宽 ≈ 规格 × 0.75~0.9 | 正常；低于 0.7 先查系统负载 |
| 读带宽 > 规格 | 异常，多半是缓冲区被压缩或命中了缓存 |
| 同一台机器两次差 > 10% | 看 `load_avg_1m` 与 `9 次采样区间`；用 `--repeat 3` 重测 |
| `selfcheck.passed == false` | 先修内核；`run` 本身会以退出码 2 结束 |
| 采样区间很宽（如 35–51） | 后台任务在抢统一内存带宽；绝对值以最优点为准，比值仍可用 |

## 7. 给 Agent / 大模型接入

MCP（stdio）：

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
Agent 侧的协作约定见仓库根 [AGENT.md](../../AGENT.md)。

## 8. 排障

| 现象 | 处理 |
|---|---|
| `find: swiftc` / 编译失败 | `xcode-select --install`；看 `output/build/build.log` |
| `selfcheck` 失败 | 说明内核被编译器改写或计数不对，不要采信本次 `run` 的数字 |
| 带宽忽高忽低 | 后台任务；`--repeat 3`，或空载重测 |
| 二进制与源码不同步 | `make clean && ./macgpu info` |
