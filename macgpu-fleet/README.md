# macgpu-fleet —— 跨 Mac 设备采集与可比性检查

配套 [`../mac-gpu-bench`](../mac-gpu-bench/README.md)。目标不是「多跑几台」，而是**让几台机器的数字真的能放在一张表里**。

## 用法

```bash
# 在每台机器上（先 build 好 macgpu）
python3 macgpu-fleet/collect.py collect --bench ./mac-gpu-bench \
    --launches 3 --repeat 5 --out ./macgpu-fleet/results

# 汇总表
python3 macgpu-fleet/collect.py table macgpu-fleet/results

# 可比性检查（协议、OS build、binned、按核数归一）
python3 macgpu-fleet/collect.py compare macgpu-fleet/results
```

`compare` 发现 schema / 带宽工作集 / GEMM 规模 / 带宽线程 / repeat 不一致时**退出码 3**，
并明确写出「数值不能直接排序」；只有环境告警（负载、热状态）时仍返回 0。

## 采集口径

- 每台机跑 **3 次独立进程**，每次进程内 `--repeat 5`：既采到进程内抖动，也采到时钟/热状态漂移。
- 每个指标先取**轮内中位数**（v2.1 起；旧结果自动退回峰值），再在 3 个进程间取中位数并给 CV。
- 记录 AC/电池、低电量模式、热状态、OS build、`hw.model`、GPU 核数、`git_sha`。
- `selfcheck` 不通过 → 该机数据标记不可用。

## 跨代比较的红线

| 代际 | 能直接比什么 | 不能直接比什么 |
|---|---|---|
| 同代同核数（如两台 M4 Pro 18 核） | 全部指标 | —— |
| 同代不同核数（binned，如 M4 Max 32/40 核） | 按核数归一的 MMA/核、读/核 | 绝对算力、带宽（带宽不随核数线性缩放） |
| M1 ↔ M2 ↔ M3 ↔ M4 | 微基准绝对读数（同一套内核与协议） | FP16/FP32 比例、MMA/FMA 比例、每核带宽 |
| M1~M4 ↔ M5+ | 只有「测出来的数」本身 | 任何从 M1 外推的系数：M5 的 GPU 神经加速器会改变 MMA 峰值与 FP16/FP32 关系 |
| Air ↔ Pro/Studio | 短时峰值 | 持续吞吐（Air 无风扇，十几秒就可能降频） |

规格表（`mac-gpu-bench/tools/report.py` 的 `SPEC`）目前只收录到 M4。M5 及以后会被显式标为
「未收录」，不会套用错误规格；补充新芯片时请同时验证 8.2 节里的三个比例。
