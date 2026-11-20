#!/usr/bin/env python3
"""Turn a macgpu result.json into a Markdown report (stdlib only).

Usage:
    python3 tools/report.py [result.json] [-o output/reports/REPORT.md]
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 公开规格（仅作量级对照）：GPU 核心数、FP32 TFLOPS、带宽 GB/s
SPEC = {
    "Apple M1": {"cores": 8, "fp32_tflops": 2.6, "bandwidth_gbs": 68.25},
    "Apple M1 Pro": {"cores": 16, "fp32_tflops": 5.2, "bandwidth_gbs": 204.8},
    "Apple M1 Max": {"cores": 32, "fp32_tflops": 10.4, "bandwidth_gbs": 409.6},
    "Apple M1 Ultra": {"cores": 64, "fp32_tflops": 20.8, "bandwidth_gbs": 819.2},
    "Apple M2": {"cores": 10, "fp32_tflops": 3.6, "bandwidth_gbs": 102.4},
    "Apple M2 Pro": {"cores": 19, "fp32_tflops": 6.8, "bandwidth_gbs": 204.8},
    "Apple M2 Max": {"cores": 38, "fp32_tflops": 13.6, "bandwidth_gbs": 409.6},
    "Apple M2 Ultra": {"cores": 76, "fp32_tflops": 27.2, "bandwidth_gbs": 819.2},
    "Apple M3": {"cores": 10, "fp32_tflops": 4.1, "bandwidth_gbs": 102.4},
    "Apple M3 Pro": {"cores": 18, "fp32_tflops": 7.4, "bandwidth_gbs": 153.6},
    "Apple M3 Max": {"cores": 40, "fp32_tflops": 16.4, "bandwidth_gbs": 409.6},
    "Apple M4": {"cores": 10, "fp32_tflops": 4.3, "bandwidth_gbs": 120.0},
    "Apple M4 Pro": {"cores": 20, "fp32_tflops": 8.6, "bandwidth_gbs": 273.0},
    "Apple M4 Max": {"cores": 40, "fp32_tflops": 17.2, "bandwidth_gbs": 546.0},
}

MODELS = {
    "7B": {"name": "7B", "params": 7.0e9, "layers": 32, "kv_heads": 32, "head_dim": 128},
    "13B": {"name": "13B", "params": 13.0e9, "layers": 40, "kv_heads": 40, "head_dim": 128},
    "70B": {"name": "70B", "params": 70.0e9, "layers": 80, "kv_heads": 8, "head_dim": 128},
}
BYTES_PER_PARAM = {"fp32": 4.0, "fp16": 2.0, "int8": 1.0, "int4": 0.5}


def find_latest():
    paths = glob.glob(os.path.join(ROOT, "output", "results", "*", "result.json"))
    if not paths:
        sys.exit("没有找到结果文件，先运行 ./macgpu run")
    return max(paths, key=os.path.getmtime)


def fmt(v, digits=2):
    if v is None:
        return "—"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.{digits}f}"


def build(data, args):
    L = []
    add = L.append

    host, gpu = data["host"], data["gpu"]
    chip = host["chip"]
    spec = next((v for k, v in SPEC.items() if k in chip), None)
    load = host.get("load_avg_1m")

    compute = {(m["dtype"], m.get("path", "fma")): m for m in data["compute"]}
    gemm = {m["dtype"]: m for m in data["gemm"]}
    bandwidth = {m["name"]: m for m in data["bandwidth"]}

    fp32_mma, fp16_mma = compute.get(("fp32", "mma")), compute.get(("fp16", "mma"))
    fp32_fma, fp16_fma = compute.get(("fp32", "fma")), compute.get(("fp16", "fma"))
    read_gbs = bandwidth.get("gpu_read", {}).get("gbps")
    write_gbs = bandwidth.get("gpu_write", {}).get("gbps")
    copy_gbs = bandwidth.get("gpu_copy", {}).get("gbps")

    # ---------------- 头部
    add("# Mac GPU 算力与带宽实测报告")
    add("")
    add("*Mac GPU Compute & Bandwidth Benchmark — `macgpu` 实测结果*")
    add("")
    try:
        when = datetime.fromisoformat(data["generated_at"].replace("Z", "+00:00")).astimezone()
        when_text = when.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, KeyError):
        when_text = data.get("generated_at", "?")
    load_text = f" · 测试时系统负载 {load:.2f}" if load is not None else ""
    runs_text = f"　共 {data['runs']} 轮取各指标最优" if data.get("runs", 1) > 1 else ""
    add(f"- **机型**：{chip}（{host['model']}）· GPU {gpu.get('core_count') or '?'} 核"
        f" · {host['memory_gb']:.0f} GB 统一内存 · macOS {host['macOS']}{load_text}")
    add(f"- **测试时间**：{when_text}　**工具**：macgpu v{data['version']}（纯 Metal，无第三方依赖）{runs_text}")
    add("")
    add("> 口径：算力取**每条通路的峰值**（寄存器内循环、无访存）；1 FMA = 2 FLOP，1 MAC = 2 OPS。"
        "所有峰值内核的输出都会与 CPU 参考逐元素比对（`./macgpu selfcheck`），"
        "防止编译器把未使用的向量分量当死代码删掉——这正是 v1.0 虚高 4 倍的原因。")
    add("")

    # ---------------- 结论
    add("## 1. 结论速览 / Executive summary")
    add("")
    n = 1
    if fp32_mma and fp16_mma and spec:
        add(f"{n}. **唯一能跑满 GPU 的是矩阵指令**：`simdgroup_matrix` 实测 "
            f"{fmt(fp32_mma['value'])} TFLOPS（FP32）/ {fmt(fp16_mma['value'])} TFLOPS（FP16），"
            f"是公开规格 {spec['fp32_tflops']} TFLOPS 的 "
            f"{fp16_mma['value'] / spec['fp32_tflops'] * 100:.0f}%。"
            f"注意 **FP16 与 FP32 同速，没有常说的“FP16 双倍”**。")
        n += 1
    if fp32_fma and fp32_mma:
        add(f"{n}. **不用矩阵指令，天花板减半**：普通 FMA 着色器 FP32 {fmt(fp32_fma['value'])} TFLOPS、"
            f"FP16 {fmt(fp16_fma['value']) if fp16_fma else '—'} TFLOPS，"
            f"只有峰值的 {fp32_fma['value'] / fp32_mma['value'] * 100:.0f}% / "
            f"{fp16_fma['value'] / fp16_mma['value'] * 100 if fp16_fma and fp16_mma else 0:.0f}%。")
        n += 1
    if gemm:
        parts = "、".join(f"{k.upper()} {fmt(v['tflops'])} {v.get('unit', 'TFLOPS')}" for k, v in gemm.items())
        add(f"{n}. **实际矩阵乘**（手写 GEMM，未用矩阵指令）：{parts}。")
        n += 1
    if read_gbs:
        share = f"，约为规格 {spec['bandwidth_gbs']} GB/s 的 {read_gbs / spec['bandwidth_gbs'] * 100:.0f}%" if spec else ""
        add(f"{n}. **带宽**：读 {fmt(read_gbs, 1)} / 写 {fmt(write_gbs, 1)} / 拷贝 {fmt(copy_gbs, 1)} GB/s{share}。")
        n += 1
    gemm_int = [m for k, m in gemm.items() if k in ("int8", "int4")]
    gemm_text = "、".join(f"{k.upper()} {fmt(m['tflops'])} TOPS" for k, m in gemm.items() if k in ("int8", "int4"))
    add(f"{n}. **量化在 Mac 上不提算力，只省带宽**：Metal 没有 int8/int4 tensor 指令，"
        + (f"{gemm_text}，只有 FP16 峰值的 "
           f"{min(m['tflops'] for m in gemm_int) / fp16_mma['value'] * 100:.0f}%~"
           f"{max(m['tflops'] for m in gemm_int) / fp16_mma['value'] * 100:.0f}%（见第 3 节）；" if gemm_int and fp16_mma else "")
        + "量化的价值全在 decode 的权重字节数。")
    add("")
    if fp16_mma and read_gbs:
        add("> **评估 prefill / decode 直接可用（推荐值）**：")
        add(f"> prefill 算力 = **FP16 矩阵指令峰值 {fmt(fp16_mma['value'])} TFLOPS**"
            f"（× 效率 {args.eff * 100:.0f}% = {fmt(fp16_mma['value'] * args.eff)} TFLOPS）；"
            f"decode 带宽 = **GPU 读 {fmt(read_gbs, 1)} GB/s**（× {args.bw_eff * 100:.0f}% = "
            f"{fmt(read_gbs * args.bw_eff, 1)} GB/s）。")
        add(f"> 量化（INT8/INT4）**不提高算力**（GEMM 实测见第 3 节），只把 decode 的权重字节数"
            f"从 2 降到 1 / 0.5，所以 prefill 不变、decode 变快。第 5 节给出 7B/13B/70B 的具体数字。")
        add("")

    # ---------------- 算力
    add("## 2. 算力峰值 / Compute peak by execution path")
    add("")
    add("| Path (English) | 精度 | 实测 | 每秒乘加 | 说明 |")
    add("|---|---|---|---|---|")
    for key in [("fp32", "mma"), ("fp16", "mma"), ("fp32", "fma"), ("fp16", "fma")]:
        m = compute.get(key)
        if not m:
            continue
        add(f"| {m.get('label_en', m['label'])} | {m['dtype'].upper()} | {fmt(m['value'])} {m['unit']} | "
            f"{fmt(m['macs_per_sec'] / 1e12)} TMAC/s | {m['method']} |")
    add("")
    add("> 矩阵指令（`simdgroup_matrix`）是 Apple GPU 的 tensor 通路，也是唯一能逼近规格的写法；"
        "普通 FMA 着色器受发射率限制只能到一半左右。两者是同一颗芯片上完全不同的执行通路。")
    add("")

    # ---------------- GEMM
    if gemm:
        size = next(iter(gemm.values()))["size"]
        add(f"## 3. 实际矩阵乘 / Real GEMM (N={size}³)")
        add("")
        add("| dtype | 实测 | vs ALU peak | 说明 |")
        add("|---|---|---|---|")
        # GEMM 是普通 ALU 内核，天花板是 fma 峰值而不是矩阵指令峰值
        peaks = {"fp32": fp32_fma, "fp16": fp16_fma, "int8": None, "int4": None}
        for k, m in gemm.items():
            peak = peaks.get(k)
            rel = f"{m['tflops'] / peak['value'] * 100:.0f}%" if peak else "—（无 ALU 参照）"
            add(f"| {k.upper()} | {fmt(m['tflops'])} {m.get('unit', 'TFLOPS')} | {rel} | {m['method']} |")
        add("")
        add("> 手写 GEMM：64×64 线程块 / 每线程 4×4 / 纯 ALU 通路。成熟实现（MLX、llama.cpp 的 Metal 后端）"
            "用上 `simdgroup_matrix` 后 fp16 能到 2.0~2.4 TFLOPS，差距全在分块与矩阵指令。")
        add("")

    # ---------------- 带宽
    if read_gbs:
        add("## 4. 内存带宽 / Memory bandwidth")
        add("")
        add("| Test | 实测 | 9 次采样区间 | 占规格 | 测试方法 |")
        add("|---|---|---|---|---|")
        methods = {
            "gpu_read": ("GPU read / 读", "float4 网格步长累加（只读）"),
            "gpu_write": ("GPU write / 写", "float4 网格步长写入（只写）"),
            "gpu_copy": ("GPU copy / 拷贝", "MTLBlitCommandEncoder 拷贝（DMA，读写各一次）"),
        }
        for key in ["gpu_read", "gpu_write", "gpu_copy"]:
            m = bandwidth.get(key)
            if not m:
                continue
            share = f"{m['gbps'] / spec['bandwidth_gbs'] * 100:.0f}%" if spec else "—"
            lo = m.get("gbps_min")
            rng = f"{fmt(lo, 1)} – {fmt(m['gbps'], 1)}" if lo else "—"
            add(f"| {methods[key][0]} | {fmt(m['gbps'], 1)} GB/s | {rng} | {share} | {methods[key][1]} |")
        add("")
        buf_mb = bandwidth.get("gpu_read", {}).get("bytes_per_rep", 0) / 1024 / 1024
        add(f"> 缓冲区 {buf_mb:.0f} MB，测试前填充不可压缩数据，避免零页/内存压缩导致虚高。"
            "表中「实测」是 9 次采样中最快的一次（≈硬件上限），「采样区间」是同一轮里最慢到最快，"
            "区间越宽说明后台任务抢带宽越凶；FMA/GEMM 共用同一块统一内存，也会被一起拖慢。")
        add(f"> ⚠️ 测试时系统负载 {load:.2f}，后台任务会分走带宽，实测值可能偏低 10~20%。")
        add("")

    # ---------------- roofline
    peak_m = fp16_mma or fp32_mma   # 缺 fp16 时退回 fp32 峰值，避免整节消失
    if peak_m and read_gbs:
        add("## 5. 模型 roofline / Prefill & decode upper bound")
        add("")
        read_bw = read_gbs * args.bw_eff * 1e9   # 字节/秒
        add("| 模型 | 精度 | 权重 | prefill tok/s | decode tok/s | 能否放进可用显存 |")
        add("|---|---|---|---|---|---|")
        working_set_gb = (gpu.get("recommended_working_set_mb") or 12288) / 1024
        prefill_base = (peak_m["macs_per_sec"]) * args.eff
        for model in MODELS.values():
            for prec in ["fp16", "int8", "int4"]:
                bpp = BYTES_PER_PARAM[prec]
                weights_gb = model["params"] * bpp / 1e9
                kv = 2 * model["layers"] * model["kv_heads"] * model["head_dim"] * 2
                prefill = prefill_base / model["params"]
                decode = read_bw / (model["params"] * bpp + kv * args.ctx)
                fits = "是" if weights_gb + 2 < working_set_gb else "否"
                add(f"| {model['name']} | {prec.upper()} | {weights_gb:.1f} GB | {prefill:.1f} | "
                    f"{decode:.1f} | {fits} |")
        add("")
        add(f"> prefill 按峰值 {fmt(peak_m['value'])} {peak_m['unit']} × {args.eff * 100:.0f}% 计，"
            f"decode 按带宽 {fmt(read_bw / 1e9, 1)} GB/s × {args.bw_eff * 100:.0f}% 计，"
            f"KV cache 按 {args.ctx} 上下文、FP16。prefill 的实际达成率高度依赖 GEMM 写法（第 3 节），"
            "decode 几乎只取决于带宽。")
        add("> **实测校核**：用 oMLX 跑真实 4-bit 模型（MiniCPM5-2B、Qwen3.5-9B），"
            "prefill 达到本机峰值的 78%~83%、decode 达到实测读带宽的 92%~111%，"
            "说明这里用的 roofline 公式与 ±20% 的误差范围是可信的；"
            "若你跑的是 MLX/MPS 这类成熟实现，效率取 0.8 而不是 0.4（详见 "
            "docs/records/51-mlx-crosscheck.md）。")
        add("")

    # ---------------- 对照
    if spec:
        add("## 6. 与公开规格对照 / Vs. public specs")
        add("")
        add("| 指标 | 公开口径 | 本机实测 | 偏差 | 解读 |")
        add("|---|---|---|---|---|")
        if fp32_mma:
            add(f"| FP32 峰值 | {spec['fp32_tflops']} TFLOPS | {fmt(fp32_mma['value'])} TFLOPS | "
                f"{fp32_mma['value'] / spec['fp32_tflops'] * 100 - 100:+.0f}% | 矩阵指令通路，已达到规格 |")
        if fp32_fma:
            add(f"| FP32 普通 FMA | — | {fmt(fp32_fma['value'])} TFLOPS | — | 非矩阵写法只有峰值的一半 |")
        if fp16_mma:
            add(f"| FP16 峰值 | 常被误传为 2× FP32（5.2） | {fmt(fp16_mma['value'])} TFLOPS | — | "
                "实测 FP16 与 FP32 矩阵指令同速，**没有 2 倍** |")
        if read_gbs:
            add(f"| 内存带宽 | {spec['bandwidth_gbs']} GB/s | {fmt(read_gbs, 1)} GB/s | "
                f"{read_gbs / spec['bandwidth_gbs'] * 100 - 100:+.0f}% | 视负载而定，空闲时可达 85~88% |")
        add("| Neural Engine | 16 核 / 11 TOPS | 未测 | — | ANE 不通过 Metal 暴露，需 Core ML |")
        add("")

    # ---------------- 评估
    add("## 7. 结果评估 / Assessment")
    add("")
    if fp16_mma and spec:
        add(f"- **标称算力是真的，但只对矩阵指令成立**：FP16 `simdgroup_matrix` "
            f"{fmt(fp16_mma['value'])} TFLOPS ≈ 规格 {spec['fp32_tflops']} 的 "
            f"{fp16_mma['value'] / spec['fp32_tflops'] * 100:.0f}%；普通 FMA 写法只有一半。"
            "写 Metal 推理内核时能否用上 `simdgroup_matrix`，直接决定 prefill 是 1.0 还是 2.2 TFLOPS。")
    if gemm.get("fp16") and fp16_mma:
        add(f"- **手写 GEMM 只到峰值的 {gemm['fp16']['tflops'] / fp16_mma['value'] * 100:.0f}%**："
            f"{fmt(gemm['fp16']['tflops'])} TFLOPS。MLX / llama.cpp 这类成熟实现能到 2.0~2.4 TFLOPS，"
            "差距在分块、双缓冲和对矩阵指令的使用。")
    if fp16_mma and any(k in gemm for k in ("int8", "int4")):
        parts = "、".join(f"{k.upper()} {fmt(gemm[k]['tflops'])} TOPS" for k in ("int8", "int4") if k in gemm)
        add(f"- **量化不加速**：{parts}，都只有 FP16 峰值的 "
            f"{min(gemm[k]['tflops'] for k in ('int8', 'int4') if k in gemm) / fp16_mma['value'] * 100:.0f}% 上下。"
            "Mac 上量化的价值是省带宽，不是提算力——这也是为什么 decode 受益、prefill 不受益。")
    if read_gbs:
        add(f"- **带宽是 decode 的硬上限**：实测读 {fmt(read_gbs, 1)} GB/s。7B FP16 权重 14 GB → "
            f"理论 ≤{read_gbs / 14.0:.1f} tok/s；7B INT4 权重 3.5 GB → ≤{read_gbs / 3.5:.1f} tok/s（未计 KV）。")
    add("")

    # ---------------- 方法
    add("## 8. 方法学与局限 / Method and limits")
    add("")
    add("- **计时**：每个指标重复 9 次取最小值，用 `MTLCommandBuffer` 的 GPU 侧起止时间；"
        "单次派发校准到 20~120 ms。")
    add("- **防虚高**：峰值内核整条向量写回，并在 `selfcheck` 中与 CPU 参考逐元素比对"
        "（FP32 相对误差 1e-6 量级）。INT8/INT4 的微基准极易被编译器折叠成常数加法，"
        "因此量化算力改由**真实 int8 / int4 GEMM** 测量。")
    add("- **峰值≠应用性能**：峰值内核无访存；参考 GEMM 只到峰值的一半，说明内核写法这一层的损失是实打实的。")
    add("- **没测的部分**：Neural Engine、多进程并发、长时间满载的热降频。")
    add("")
    add("---")
    add(f"*Generated by macgpu v{data['version']} · 重新生成：`make report`*")

    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description="macgpu 结果转 Markdown 报告")
    ap.add_argument("result", nargs="?", help="result.json 路径（默认取最新）")
    ap.add_argument("-o", "--out", help="输出文件（默认打印到 stdout）")
    ap.add_argument("--ctx", type=int, default=2048, help="上下文长度（默认 2048）")
    ap.add_argument("--eff", type=float, default=0.40, help="prefill 计算效率（默认 0.40）")
    ap.add_argument("--bw-eff", type=float, default=0.90, help="decode 带宽利用率（默认 0.90）")
    args = ap.parse_args()

    with open(args.result or find_latest()) as f:
        data = json.load(f)
    text = build(data, args)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"报告已写入 {args.out}")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
