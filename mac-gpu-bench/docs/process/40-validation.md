# 验证与治理 / Validation

> 状态：**current**。本项目出过错，所以「怎么证明数字是对的」和「怎么测」同等重要。

## 1. 三个已经踩过的坑

| 坑 | 现象 | 现在的防线 |
|---|---|---|
| 向量分量被当死代码删除 | 内核只写回 `.x`，FP32 虚高 4 倍（3.86 vs 0.97） | 峰值内核整条向量写回 + 数值自检 |
| 循环被折叠成常数加法 | 整型微基准虚高 100 倍（84 TOPS） | 放弃 int8/int4 微基准，改测真实 int8 GEMM |
| 缓冲区被内存压缩 | 带宽虚高到 89~121 GB/s | 测试前用 LCG 填充不可压缩数据 |

共同点：**程序没报错、数字很好看、但测的不是它声称测的东西**。所以验证不靠“看起来合理”。

## 2. selfcheck 覆盖范围

`./macgpu selfcheck` 逐项与 CPU 参考比对，不通过则退出码 2；`run` 也会先跑同一套自检，
结果写进 `result.json` 的 `selfcheck` 字段，失败时 `run` 以退出码 2 结束（`--no-selfcheck` 可跳过）。

| 检查项 | 判据 |
|---|---|
| `gemm_i8` / `gemm_i4` 128³ | 与 CPU 整数矩阵乘**完全相等**（误差 0） |
| `gemm_f32` / `gemm_f16` 128³ | 输入用非二进制精确数（0.0731/0.0419），相对误差 ≤ 2e-3 / 2e-2 |
| `peak_mma_f32` / `peak_mma_f16` | 累加器**显式清零**后与 CPU 复算的 A·B 比对（相对误差 ≤ 1e-4 / 2e-2） |
| `peak_f32` / `peak_f16` | 按同样的 FMA 链在 CPU 复算后比对（≤ 1e-4 / 2e-2） |
| 带宽三项 | 量级落在 5~5000 GB/s |

阈值取「数值精度」和「实现错误」之间的空隙：相对误差 1e-6 是 float32 的正常水平，
一旦计数错 4 倍，误差会变成 0.75 量级，不可能漏过。

v2.1 之前这三处会**假通过**，已修：

- 峰值内核的累加器缓冲区未清零，CPU 参考却假设初值为 0；
- fp16 自检跑 96 次迭代，半精度累加误差贴到 5e-2 阈值（实测 3.8e-2）；
- fp32/fp16 GEMM 用 0.25/0.5 这类二进制精确值，误差恒为 0，等于没检。

现在 fp16 自检的典型误差是 2e-3 ~ 6e-3（阈值 2e-2），留了 3 倍以上余量。

## 3. 文档一致性（make docs）

`tools/docs_contract.py` 检查四件事，任何一条不平就失败：

1. `docs/` 下每个文件都被 `docs/README.md` 收录。
2. 指标矩阵里出现的每个内核名都能在 `src/Kernels.swift` 找到；
   反过来，源码里每个 `peak_*` / `gemm_*` 内核都在指标矩阵里出现过。
3. `design/21-output-contract.md` 声明的 JSON 字段与最近一次 `result.json` 的实际字段一致。
4. 仓库里没有残留旧名字（`macbench`、根目录 `results/`、根目录 `REPORT.md`）。

**这就是「方案与实现保持一致」的机制**：改测量项时，文档不跟着改，`make docs` 立刻报错。

## 4. 冒烟测试（make test）

`tests/smoke.sh` 在临时目录里跑一遍最小闭环，约 20 秒：

1. 从零编译通过。
2. `info --json` 字段完整、GPU 核心数 > 0。
3. 小规模实测（GEMM 256³、带宽 32 MB）数值落在合理区间，`schema` 与 `compute` 结构正确。
4. `selfcheck` 全通过（含 `run` 内嵌的自检与 `selfcheck.passed == true`）。
5. MCP 握手 + `tools/list` + `machine_info`。
6. 报告生成且包含关键章节。

## 5. 验收清单（改代码后逐条过）

- [ ] `make test` 全绿
- [ ] `./macgpu selfcheck` 输出 PASS
- [ ] `make docs` 通过（文档已同步）
- [ ] `make run && make report` 的数字与 `docs/records/` 里的基线同量级
- [ ] 新增/修改的指标在 `docs/spec/10-methodology.md` 的指标矩阵里有对应行
- [ ] 改动涉及字段/目录时，`docs/design/21-output-contract.md` 与 `schema` 版本已更新

## 6. 什么时候数字不可信

- `selfcheck.passed == false`：先修内核，不要看 `run` 的结果（`run` 本身会以退出码 2 结束）。
- `host.thermal_state != nominal`：已降频，绝对值偏低。
- `host.low_power_mode == "1"` 或 `host.power_source == "battery"`：频率/功耗受限。
- `host.load_avg_1m` > 2.0：带宽可能偏低 10~20%，算力受影响较小。
- 单次 `run`（未 `--repeat`）：取最优次数不足，M1 上 `--repeat 1` 比 `--repeat 10` 低约 4%。
- 与另一台机器比较：必须同为 Apple Silicon、同一 `--size-mb` / `--gemm-size` / `--repeat`，
  且按 **GPU 核数** 分组（同芯片有 binned 版本）。Intel Mac 的 Metal 后端差异过大，直接退出。
