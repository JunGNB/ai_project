# 架构设计 / Architecture

> 状态：**current**。改模块边界、内核组织、计时方式时同步更新。

## 1. 分层

```
macgpu (bash)            # 入口：按需编译，产物落 output/build
  └── src/main.swift     # CLI：参数、调度、摘要输出、JSON 落盘
        └── src/Bench.swift     # 设备探测、计时器、Compute/Gemm/Bandwidth 三组基准、SelfCheck
              └── src/Kernels.swift  # MSL 源码字符串（运行时 device.makeLibrary(source:)）
tools/report.py          # 消费 result.json → Markdown
tools/mcp_server.py      # 同一批能力暴露为 MCP 工具（stdio，零依赖）
tools/docs_contract.py   # 文档/实现一致性校验
```

**没有 Xcode 工程、没有第三方依赖。** 只用 `swiftc` + Metal 运行时；
MSL 在进程内编译，改内核不用重启工具链。

## 2. 数据流

```
CLI 参数 → Options →（compute / gemm / bandwidth 三选或全选）
   → 每组基准返回 Metric 结构体
   → BenchResult（Codable）→ output/results/<时间戳>/result.json（+ latest 软链接）
   → report.py 渲染 Markdown / mcp_server.py 直接返回 JSON
```

`result.json` 是唯一的产品契约：报告和 MCP 都只读它，不再回查进程状态。

## 3. 计时器（GPU 类）

三个关键方法，所有基准共用：

| 方法 | 作用 |
|---|---|
| `timeThreads(...)` / `timeGroups(...)` | 一次或 `reps` 次派发，返回 GPU 侧墙钟 |
| `best(_:body:)` | 连续采样 9 次取最小值 |
| 校准步骤 | 先跑 1 次估算单次耗时，再决定 `reps` 使批处理落在 20~120 ms |

校准避免了两类误差：派发太短被固定开销淹没，太长被热降频和后台任务污染。

## 4. 内核组织与防优化

MSL 内核按用途分组，命名前缀即语义：

- `peak_mma_f32` / `peak_mma_f16`：矩阵指令峰值。每 warp 4 个累加器，**全部写回**。
- `peak_f32` / `peak_f16`：着色器 ALU 峰值。8 条向量链，**整条向量写回**。
- `gemm_f32` / `gemm_f16` / `gemm_i8`：64×64 线程块、每线程 4×4 的真实矩阵乘。
- `bw_read` / `bw_write` / `bw_fill`：带宽三件套。

三条硬规则（违反任何一条，数字立刻虚高）：

1. **写回整条向量**。只写 `.x` 会让编译器删掉另外 3/4 的 FMA（旧版虚高 4 倍）。
2. **循环体内要有循环携带依赖**。否则整段可能被折叠成常数加法（整型微基准虚高 100 倍）。
3. **计数 = 实跑**。`macs_per_iteration` 必须能逐条对上内核语句，并由 `selfcheck` 兜底。

## 5. 自检（SelfCheck）

`./macgpu selfcheck` 不走「数值看起来合理」，而是**逐元素与 CPU 参考比对**：

- `gemm_{f32,f16}`：128³ 与 CPU 参考比最大误差。
- `gemm_i8`：与 CPU 整数结果**完全相等**。
- `peak_mma_*`：迭代若干次后与 CPU 复算的 A·B 比对。
- `peak_f32` / `peak_f16`：按同样的 FMA 链在 CPU 上复算后比对。
- 带宽：量级合理性（5 ~ 5000 GB/s）。

任一项不通过 → 退出码 2。CI/脚本据此阻断。

## 6. 已知取舍

- 只测 GPU，不测 Neural Engine（ANE 不经 Metal 暴露）。
- 不测热降频：单次测试约 5 秒，远短于触发降频的时间。
- 参考 GEMM 不追求极致性能，用来展示「普通内核 vs 峰值」的差距。
