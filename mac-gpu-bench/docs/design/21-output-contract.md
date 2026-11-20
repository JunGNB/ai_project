# 产物契约 / Output contract

> 状态：**current**。JSON 字段、`output/` 布局、报告章节都是对外契约，改之前先改本文件。

## 1. 目录布局

所有本机产物都在 `output/` 下，且**整目录不进版本库**（见仓库根 `.gitignore`）：

```
output/
├── build/            # 编译中间件：macgpu 二进制、build.log
├── results/
│   ├── <YYYYMMDD-HHMMSS>/result.json   # 每次运行一份，保留历史
│   └── latest -> <最近一次运行目录>     # 软链接，供脚本/文档引用稳定路径
└── reports/
    └── REPORT.md     # 由 tools/report.py 生成
```

约定：

- **源码 / 文档 / 测试 / 契约** 进库；**二进制 / 结果 / 报告** 不进库。
- 报告是可再生的：任何时刻都能用 `make report` 从 `output/results/latest` 重建。
- 需要长期留存的读数，抄进 `docs/records/`（人工撰写、可追溯），而不是提交 JSON。

## 2. result.json

顶层结构：

```schema
top-level : schema, version, generated_at, runs, host, gpu, compute, gemm, bandwidth, notes
compute[] : dtype, path, label, label_en, value, unit, macs_per_sec, threads, iterations, dispatches, seconds, method, method_en
gemm[]    : dtype, size, tflops, unit, seconds, method
bandwidth[] : name, gbps, gbps_min, gbps_max, seconds, bytes_per_rep, reps, method
```

字段语义：

| 字段 | 说明 |
|---|---|
| `schema` | 固定 `macgpu/v1`，结构性变更才升版本 |
| `version` | 工具版本，语义化版本号 |
| `runs` | `--repeat` 的轮数，最终各指标取最优 |
| `host.chip/model/macOS/memory_gb/load_avg_1m` | 机器信息；负载用于判断本次读数可信度 |
| `gpu.core_count/max_threads_per_threadgroup/recommended_working_set_mb` | GPU 能力 |
| `compute[].path` | `mma`（矩阵指令）/ `fma`（着色器 ALU），是区分通路的关键字段 |
| `compute[].macs_per_sec` | 每秒乘加数，跨精度比较用这个 |
| `compute[].threads/iterations/dispatches` | 复算口径：`MACs = units × iterations × macs_per_iter × reps` |
| `gemm[].tflops` | 实际吞吐；`unit` 区分 TFLOPS / TOPS |
| `bandwidth[].bytes_per_rep` | 每轮搬运字节数，用于核对 GB/s 口径 |
| `bandwidth[].gbps_min` / `gbps_max` | 9 次采样的最差/最好值，用来判断这次读数稳不稳 |

`compute` 中每个 `dtype` 最多两条（`mma` + `fma`），因此**不能按 dtype 建索引**，
必须用 `(dtype, path)` 组合键——这一点在 `tools/report.py` 和冒烟测试里都有体现。

## 3. 报告结构

`tools/report.py` 输出固定 8 节，标题中英并列，顺序不变：

1. 结论速览 / Executive summary
2. 算力峰值 / Compute peak by execution path
3. 实际矩阵乘 / Real GEMM
4. 内存带宽 / Memory bandwidth
5. roofline / Prefill & decode upper bound
6. 与公开规格对照 / Vs. public specs
7. 结果评估 / Assessment
8. 方法学与局限 / Method and limits

任何一节缺失（例如没跑 GEMM）时整节省略，不输出空表；第 5 节在缺 fp16 时回退用 fp32 峰值。

## 4. 版本策略

- **patch**：文档、阈值、错误信息。
- **minor**：新增指标/字段（向后兼容），如新增一条通路。
- **major**：改字段语义、改目录布局、改计数口径——必须同步升 `schema`。

当前：`version = 2.0.0`，`schema = macgpu/v1`（2.0 相比 1.x：项目更名、产物迁入 `output/`、
`compute` 增加 `path` / `label_en` / `method_en`、量化算力改为 int8 GEMM）。
