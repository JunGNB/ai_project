# 产物契约 / Output contract

> 状态：**current**。JSON 字段、`output/` 布局、报告章节都是对外契约，改之前先改本文件。

## 1. 目录布局

所有本机产物都在 `output/` 下，且**整目录不进版本库**（见仓库根 `.gitignore`）：

```
output/
├── build/            # 编译中间件：macgpu 二进制、build.log
├── results/
│   ├── <YYYYMMDD-HHMMSS-SSS>/result.json   # 每次运行一份，保留历史
│   └── latest -> <最近一次运行目录>         # 软链接，供脚本/文档引用稳定路径
└── reports/
    └── REPORT.md     # 由 tools/report.py 生成
```

约定：

- **源码 / 文档 / 测试 / 契约** 进库；**二进制 / 结果 / 报告** 不进库。
- 报告是可再生的：任何时刻都能用 `make report` 从 `output/results/latest` 重建。
- 需要长期留存的读数，抄进 `docs/records/`（人工撰写、可追溯），而不是提交 JSON。
- 时间戳带毫秒（`-SSS`），避免同一秒内多次运行互相覆盖。

## 2. result.json

顶层结构：

```schema
top-level   : schema, version, generated_at, runs, sampling, host, gpu, tool, selfcheck, compute, gemm, bandwidth, bandwidth_saturation?, notes
compute[]   : dtype, path, label, label_en, value, value_median, unit, macs_per_sec, macs_per_sec_median, threads, units, unit_threads, macs_per_iteration, total_macs, iterations, dispatches, seconds, seconds_median, samples, method, method_en
gemm[]      : dtype, size, tflops, tflops_median, unit, seconds, seconds_median, samples, method
bandwidth[] : name, gbps, gbps_min, gbps_max, gbps_median, seconds, seconds_median, bytes_per_rep, reps, threads, samples, method
bandwidth_saturation? : primary_threads, alt_threads, read_gain_pct, write_gain_pct, copy_gain_pct, saturated
```

字段语义：

| 字段 | 说明 |
|---|---|
| `schema` | `macgpu/v2`。改字段语义、改计数口径才升；新增字段走 minor，也要升 schema 才能与旧结果区分 |
| `version` | 工具版本，语义化版本号（当前 `2.1.0`） |
| `runs` | `--repeat` 的轮数，轮间各指标取最优 |
| `sampling` | 采样口径：`rounds`、`samples_per_metric=9`、`value_estimator=min`、`median_estimator=median` |
| `host.chip/model/macOS/os_build/arch/memory_gb` | 机器信息；`os_build` 决定 Metal 运行时编译器的版本 |
| `host.uptime_hours/load_avg_1m` | 测试时环境；`load_avg_1m` 用于判断读数可信度 |
| `host.thermal_state` | `nominal/fair/serious/critical/unknown`（`ProcessInfo.thermalState`） |
| `host.low_power_mode` | `0/1/unknown`（`pmset -g`；台式机/部分机型不暴露该键时为 `unknown`） |
| `host.power_source` | `ac/battery/unknown`（`pmset -g ps`） |
| `gpu.core_count/max_threads_per_threadgroup/recommended_working_set_mb` | GPU 能力 |
| `tool.git_sha/git_dirty` | 代码版本；不同 commit 的数字不保证可比 |
| `selfcheck.passed/skipped/checks[]` | 每次 `run` 内嵌的数值自检结果；`passed=false` 时不要采用本次数据 |
| `compute[].path` | `mma`（矩阵指令）/ `fma`（着色器 ALU），是区分通路的关键字段 |
| `compute[].value` / `value_median` | 峰值（9 次最快）/ 中位数（9 次排序取中）；**跨设备比较用 `value_median`** |
| `compute[].macs_per_sec` | 每秒乘加数，跨精度比较用这个（也有 `_median` 版本） |
| `compute[].units/unit_threads/macs_per_iteration/total_macs` | 复算口径：`MACs = units × iterations × macs_per_iter × reps`；`total_macs` 已算好，可直接核对 `value × seconds` |
| `compute[].samples` | 9 次采样的原始秒数；`min(samples) == seconds`，`sorted(samples)[4] == seconds_median` |
| `gemm[].tflops` / `tflops_median` | 峰值 / 中位数；`unit` 区分 TFLOPS / TOPS |
| `bandwidth[].gbps` / `gbps_median` | 峰值 / 中位数 |
| `bandwidth[].gbps_min` / `gbps_max` | 9 次采样中的最差 / 最好；按定义 `gbps == gbps_max`（冗余字段，保留兼容旧结果） |
| `bandwidth[].bytes_per_rep` / `threads` | 每轮搬运字节数、派发线程数（自动调优，起点 262,144；可用 `--bw-threads` 强制），用于核对口径与复现 |
| `bandwidth_saturation`（可选） | 用 2× 线程复测的饱和证据：`read/write/copy_gain_pct` 是"翻倍线程后的中位数增益"，`saturated = max(gain) < 3%`。只用于判断默认线程数是否喂饱这台 GPU，**不改变上报的带宽值** |

`output/results/latest` 只指向**全套**运行（`compute` + `gemm` + `bandwidth`）；
用 `--tests` 只跑一部分时不会更新它，避免 `make report` 从残缺结果生成。

`compute` 中每个 `dtype` 最多两条（`mma` + `fma`），因此**不能按 dtype 建索引**，
必须用 `(dtype, path)` 组合键。v2.0 的 MCP `summarize` 就因为按 dtype 建键，
让 `fma` 覆盖了 `mma`，Agent 看到的 FP16 峰值实际是普通 FMA 通路。

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
环境告警（selfcheck 失败、热降频、低电量、电池、负载 > 2）以独立小节插在第 1 节之前，
不占用 8 节编号；`--strict` 时出现告警以退出码 3 结束。

报告正文用中位数（稳定可达），表格同时给峰值；第 5 节给出 `--eff`（默认 0.80，对应 MLX/MPS）
与 0.40（手写内核）两列，避免把两种实现混在一起比较。

## 4. 版本策略

- **patch**：文档、阈值、错误信息。
- **minor**：新增指标/字段（向后兼容），如新增一条通路；schema 同步升位以便区分旧结果。
- **major**：改字段语义、改目录布局、改计数口径——必须同步升 `schema`。

当前：`version = 2.1.0`，`schema = macgpu/v2`。

- **v1 → v2.0**：项目更名、产物迁入 `output/`、`compute` 增加 `path` / `label_en` / `method_en`、
  量化算力改为 int8 GEMM。
- **v2.0 → v2.1**：新增 `sampling` / `tool` / `selfcheck` 与全部 `*_median` / `samples` 字段；
  新增可选的 `bandwidth_saturation`（线程饱和证据）与 `--bw-threads`；
  `host` 增加 `os_build` / `thermal_state` / `low_power_mode` / `power_source`；
  带宽线程数 65,536 → 262,144；`run` 内嵌自检。以上均为新增字段，`value` 语义不变（仍是最快一次）。
