#!/usr/bin/env python3
"""Turn a macgpu result.json into a Markdown report (stdlib only).

Usage:
    python3 tools/report.py [result.json] [-o output/reports/REPORT.md]

口径约定：
  - 正文/roofline 用中位数（value_median / tflops_median / gbps_median），代表"稳定可达"；
  - "峰值"列与规格对照用最快一次（value），代表"硬件上限"；
  - 旧结果（schema macgpu/v1）没有 *_median 字段时自动退回 value。
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 公开规格（仅作量级对照）。带宽按十进制 GB/s（1e9），与 macgpu 的 gbps 口径一致。
# 同一芯片有 binned 版本（GPU 核数/带宽不同），跨设备比较请以实测 GPU 核数为准。
SPEC = {
    "Apple M1": {"cores": 8, "fp32_tflops": 2.6, "bandwidth_gbs": 68.25},
    "Apple M1 Pro": {"cores": 16, "fp32_tflops": 5.2, "bandwidth_gbs": 200.0},
    "Apple M1 Max": {"cores": 32, "fp32_tflops": 10.4, "bandwidth_gbs": 400.0},
    "Apple M1 Ultra": {"cores": 64, "fp32_tflops": 20.8, "bandwidth_gbs": 800.0},
    "Apple M2": {"cores": 10, "fp32_tflops": 3.6, "bandwidth_gbs": 100.0},
    "Apple M2 Pro": {"cores": 19, "fp32_tflops": 6.8, "bandwidth_gbs": 200.0},
    "Apple M2 Max": {"cores": 38, "fp32_tflops": 13.6, "bandwidth_gbs": 400.0},
    "Apple M2 Ultra": {"cores": 76, "fp32_tflops": 27.2, "bandwidth_gbs": 800.0},
    "Apple M3": {"cores": 10, "fp32_tflops": 4.1, "bandwidth_gbs": 100.0},
    "Apple M3 Pro": {"cores": 18, "fp32_tflops": 7.4, "bandwidth_gbs": 150.0},
    "Apple M3 Max": {"cores": 40, "fp32_tflops": 16.4, "bandwidth_gbs": 400.0},
    "Apple M4": {"cores": 10, "fp32_tflops": 4.3, "bandwidth_gbs": 120.0},
    "Apple M4 Pro": {"cores": 20, "fp32_tflops": 8.6, "bandwidth_gbs": 273.0},
    "Apple M4 Max": {"cores": 40, "fp32_tflops": 17.2, "bandwidth_gbs": 410.0},
}

MODELS = {
    "7B": {"name": "7B", "params": 7.0e9, "layers": 32, "kv_heads": 32, "head_dim": 128},
    "13B": {"name": "13B", "params": 13.0e9, "layers": 40, "kv_heads": 40, "head_dim": 128},
    "70B": {"name": "70B", "params": 70.0e9, "layers": 80, "kv_heads": 8, "head_dim": 128},
}
BYTES_PER_PARAM = {"fp32": 4.0, "fp16": 2.0, "int8": 1.0, "int4": 0.5}


def find_latest():
    latest = os.path.join(ROOT, "output", "results", "latest", "result.json")
    if os.path.isfile(latest):
        return latest
    paths = glob.glob(os.path.join(ROOT, "output", "results", "*", "result.json"))
    if not paths:
        sys.exit("没有找到结果文件，先运行 ./macgpu run")
    return max(paths, key=os.path.getmtime)


def lookup_spec(chip):
    """最长前缀匹配。

    旧实现用 `k in chip`：'Apple M1' 会命中 'Apple M1 Pro'，导致 Pro/Max/Ultra
    全部按基础款规格算；'Apple M3 Pro' 反而一个都命中不了。
    """
    for name in sorted(SPEC, key=len, reverse=True):
        if chip.startswith(name):
            return name, SPEC[name]
    return None, None


def fmt(v, digits=2):
    if v is None:
        return "—"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.{digits}f}"


def stable(m, field):
    """中位数优先；旧结果没有该字段时退回峰值。"""
    return m.get(field + "_median", m[field])


def peak(m, field):
    return m[field]


def build(data, args):
    L = []
    add = L.append

    host, gpu = data["host"], data["gpu"]
    chip = host["chip"]
    spec_name, spec = lookup_spec(chip)
    load = host.get("load_avg_1m")
    selfcheck = data.get("selfcheck") or {}
    has_median = (any("value_median" in m for m in data.get("compute", []))
                  or any("tflops_median" in m for m in data.get("gemm", []))
                  or any("gbps_median" in m for m in data.get("bandwidth", [])))
    env_issues = []

    if selfcheck.get("skipped") is False and not selfcheck.get("passed", True):
        env_issues.append("**selfcheck 失败**：本次数值不可信，不要用于任何对比")
    if host.get("thermal_state") not in (None, "nominal"):
        env_issues.append(f"热状态为 `{host['thermal_state']}`，GPU 可能已降频")
    if host.get("low_power_mode") == "1":
        env_issues.append("低电量模式开启，GPU 频率受限")
    if host.get("power_source") == "battery":
        env_issues.append("电池供电，可能触发功耗限制")
    if load is not None and load > 2.0:
        env_issues.append(f"测试时系统负载 {load:.2f}，后台任务可能分走带宽与算力，读数可能偏低")
    if spec_name is None:
        env_issues.append(f"规格表里没有匹配 `{chip}` 的条目，第 6 节会跳过（M5 及以后尚未收录）")
    cores = gpu.get("core_count")
    binned = bool(spec and cores and spec.get("cores") and cores != spec["cores"])
    if binned:
        ratio = cores / spec["cores"]
        spec = dict(spec, fp32_tflops=round(spec["fp32_tflops"] * ratio, 2),
                    cores=cores, binned_from=spec["cores"])
        env_issues.append(f"规格表原为 {spec['binned_from']} 核，本机实测 {cores} 核（binned 版本）；"
                          "第 6 节的算力按核数折算，带宽不折算（带宽与核数不严格成比例）")
    if not has_median:
        env_issues.append("**旧结果（schema `macgpu/v1`）**：没有中位数与原始样本，"
                          "下表数值是峰值；跨设备比较请用 v2.1 重跑")

    compute = {(m["dtype"], m.get("path", "fma")): m for m in data["compute"]}
    gemm = {m["dtype"]: m for m in data["gemm"]}
    bandwidth = {m["name"]: m for m in data["bandwidth"]}

    fp32_mma, fp16_mma = compute.get(("fp32", "mma")), compute.get(("fp16", "mma"))
    fp32_fma, fp16_fma = compute.get(("fp32", "fma")), compute.get(("fp16", "fma"))
    read_gbs = bandwidth.get("gpu_read") and stable(bandwidth["gpu_read"], "gbps")
    write_gbs = bandwidth.get("gpu_write") and stable(bandwidth["gpu_write"], "gbps")
    copy_gbs = bandwidth.get("gpu_copy") and stable(bandwidth["gpu_copy"], "gbps")

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
    load_text = f" · 负载 {load:.2f}" if load is not None else ""
    runs_text = f"　共 {data['runs']} 轮" if data.get("runs", 1) > 1 else ""
    add(f"- **机型**：{chip}（{host.get('model', '?')}）· GPU {gpu.get('core_count') or '?'} 核"
        f" · {host['memory_gb']:.0f} GB 统一内存 · macOS {host['macOS']}"
        f"（{host.get('os_build', '?')}）{load_text}")
    add(f"- **测试时间**：{when_text}　**工具**：macgpu v{data['version']}（纯 Metal，无第三方依赖）{runs_text}")
    git = data.get("tool") or {}
    if git.get("git_sha"):
        add(f"- **代码版本**：`{git['git_sha']}`{'（有未提交改动）' if git.get('git_dirty') else ''}"
            f"　**结果 schema**：`{data.get('schema', '?')}`")
    add("")
    add("> 口径：表格并列**中位数**（9 次排序取中，跨设备比较用）与**峰值**（9 次最快，硬件上限）；"
        "正文与 roofline 用中位数。1 FMA = 2 FLOP，1 MAC = 2 OPS。"
        "所有峰值内核的输出都会与 CPU 参考逐元素比对（自检已内嵌进 result.json 的 `selfcheck` 字段），"
        "防止编译器把未使用的向量分量当死代码删掉——这正是 v1.0 虚高 4 倍的原因。")
    add("")

    if env_issues:
        add("### ⚠️ 本次测量的环境告警")
        add("")
        for issue in env_issues:
            add(f"- {issue}")
        add("")
        add("> 跨设备对比前，先确认两边都是 AC 供电、未开低电量模式、`thermal_state == nominal`、"
            "`selfcheck.passed == true`，否则排序可能只是在排电源管理策略。")
        add("")

    if selfcheck.get("checks"):
        passed = sum(1 for c in selfcheck["checks"] if c["passed"])
        total = len(selfcheck["checks"])
        add(f"- **数值自检**：{'PASS' if selfcheck.get('passed') else 'FAIL'}"
            f"（{passed}/{total}），逐项明细见 result.json。")
        add("")

    # ---------------- 结论
    add("## 1. 结论速览 / Executive summary")
    add("")
    n = 1
    if fp32_mma and fp16_mma:
        ratio = stable(fp16_mma, "value") / stable(fp32_mma, "value")
        relation = ("FP16 与 FP32 同速，没有常说的“FP16 双倍”" if 0.9 <= ratio <= 1.1
                    else f"FP16 是 FP32 的 {ratio:.2f} 倍（不同代 GPU 的矩阵单元比例不同，不能外推）")
        spec_part = ""
        if spec:
            spec_part = (f"是公开规格 {spec['fp32_tflops']} TFLOPS 的 "
                         f"{stable(fp32_mma, 'value') / spec['fp32_tflops'] * 100:.0f}%（中位数口径）")
        tail = f"{spec_part}；{relation}" if spec_part else relation
        add(f"{n}. **唯一能跑满 GPU 的是矩阵指令**：`simdgroup_matrix` 实测中位数 "
            f"{fmt(stable(fp32_mma, 'value'))} TFLOPS（FP32）/ {fmt(stable(fp16_mma, 'value'))} TFLOPS（FP16），"
            f"{tail}。")
        n += 1
    if fp32_fma and fp32_mma:
        r32 = stable(fp32_fma, "value") / stable(fp32_mma, "value")
        r16 = (stable(fp16_fma, "value") / stable(fp16_mma, "value")) if fp16_fma and fp16_mma else None
        pct = f"{r32 * 100:.0f}% / {r16 * 100:.0f}%" if r16 is not None else f"{r32 * 100:.0f}%"
        add(f"{n}. **不用矩阵指令，只剩零头**：普通 FMA 着色器 "
            f"FP32 {fmt(stable(fp32_fma, 'value'))} TFLOPS、"
            f"FP16 {fmt(stable(fp16_fma, 'value')) if fp16_fma else '—'} TFLOPS，"
            f"分别是同精度矩阵指令的 {pct}。")
        n += 1
    if gemm:
        parts = "、".join(f"{k.upper()} {fmt(stable(v, 'tflops'))} {v.get('unit', 'TFLOPS')}"
                          for k, v in gemm.items())
        add(f"{n}. **实际矩阵乘**（手写 GEMM，未用矩阵指令）：{parts}。")
        n += 1
    if read_gbs:
        share = (f"，约为规格 {spec['bandwidth_gbs']} GB/s 的 "
                 f"{read_gbs / spec['bandwidth_gbs'] * 100:.0f}%") if spec else ""
        add(f"{n}. **带宽**：读 {fmt(read_gbs, 1)} / 写 {fmt(write_gbs, 1)} / 拷贝 {fmt(copy_gbs, 1)} GB/s"
            f"（中位数）{share}。")
        n += 1
    gemm_int = [m for k, m in gemm.items() if k in ("int8", "int4")]
    gemm_text = "、".join(f"{k.upper()} {fmt(stable(m, 'tflops'))} TOPS"
                          for k, m in gemm.items() if k in ("int8", "int4"))
    add(f"{n}. **量化在 Mac 上不提算力，只省带宽**：Metal 没有 int8/int4 tensor 指令，"
        + (f"{gemm_text}，只有 FP16 峰值的 "
           + (lambda lo, hi: f"{lo:.0f}%" if abs(lo - hi) < 1 else f"{lo:.0f}%~{hi:.0f}%")(
               min(stable(m, 'tflops') for m in gemm_int) / stable(fp16_mma, 'value') * 100,
               max(stable(m, 'tflops') for m in gemm_int) / stable(fp16_mma, 'value') * 100)
           + "（见第 3 节）；" if gemm_int and fp16_mma else "")
        + "量化的价值全在 decode 的权重字节数。")
    add("")
    if fp16_mma and read_gbs:
        add("> **评估 prefill / decode 直接可用（推荐值，均取中位数）**：")
        add(f"> prefill 算力 = **FP16 矩阵指令 {fmt(stable(fp16_mma, 'value'))} TFLOPS**"
            f"（× 效率 {args.eff * 100:.0f}% = {fmt(stable(fp16_mma, 'value') * args.eff)} TFLOPS）；"
            f"decode 带宽 = **GPU 读 {fmt(read_gbs, 1)} GB/s**（× {args.bw_eff * 100:.0f}% = "
            f"{fmt(read_gbs * args.bw_eff, 1)} GB/s）。")
        add("> 量化（INT8/INT4）**不提高算力**（GEMM 实测见第 3 节），只把 decode 的权重字节数"
            "从 2 降到 1 / 0.5，所以 prefill 不变、decode 变快。第 5 节给出 7B/13B/70B 的具体数字。")
        add("")

    # ---------------- 算力
    add("## 2. 算力峰值 / Compute peak by execution path")
    add("")
    add("| Path (English) | 精度 | 中位数 | 峰值 | 每秒乘加（中位数） | 说明 |" if has_median
        else "| Path (English) | 精度 | 实测（峰值） | 每秒乘加 | 说明 |")
    add("|---|---|---|---|---|---|")
    for key in [("fp32", "mma"), ("fp16", "mma"), ("fp32", "fma"), ("int8", "alu"), ("fp16", "fma")]:
        m = compute.get(key)
        if not m:
            continue
        if has_median:
            add(f"| {m.get('label_en', m['label'])} | {m['dtype'].upper()} | "
                f"{fmt(stable(m, 'value'))} {m['unit']} | {fmt(peak(m, 'value'))} {m['unit']} | "
                f"{fmt(stable(m, 'macs_per_sec') / 1e12)} TMAC/s | {m['method']} |")
        else:
            add(f"| {m.get('label_en', m['label'])} | {m['dtype'].upper()} | "
                f"{fmt(m['value'])} {m['unit']} | {fmt(m['macs_per_sec'] / 1e12)} TMAC/s | {m['method']} |")
    add("")
    add("> 矩阵指令（`simdgroup_matrix`）是 Apple GPU 的 tensor 通路，也是唯一能逼近规格的写法；"
        "普通 FMA 着色器受发射率限制只能到一半左右。两者是同一颗芯片上完全不同的执行通路。"
        "「中位数」是 9 次采样的中位，「峰值」是最快一次；跨设备比较用中位数。")
    add("")

    # ---------------- GEMM
    if gemm:
        size = next(iter(gemm.values()))["size"]
        add(f"## 3. 实际矩阵乘 / Real GEMM (N={size}³)")
        add("")
        add("| dtype | 实测（中位数） | 峰值 | vs FMA peak | 说明 |" if has_median
            else "| dtype | 实测（峰值） | vs FMA peak | 说明 |")
        add("|---|---|---|---|---|")
        # GEMM 是普通 ALU 内核，天花板是 fma 峰值而不是矩阵指令峰值
        int8_alu = compute.get(("int8", "alu"))
        peaks = {"fp32": fp32_fma, "fp16": fp16_fma, "int8": int8_alu, "int4": None}
        for k, m in gemm.items():
            ref = peaks.get(k)
            rel = f"{stable(m, 'tflops') / stable(ref, 'value') * 100:.0f}%" if ref else "—（无 ALU 参照）"
            if has_median:
                add(f"| {k.upper()} | {fmt(stable(m, 'tflops'))} {m.get('unit', 'TFLOPS')} | "
                    f"{fmt(peak(m, 'tflops'))} {m.get('unit', 'TFLOPS')} | {rel} | {m['method']} |")
            else:
                add(f"| {k.upper()} | {fmt(m['tflops'])} {m.get('unit', 'TFLOPS')} | {rel} | {m['method']} |")
        add("")
        add("> 手写 GEMM：64×64 线程块 / 每线程 4×4 / 纯 ALU 通路。成熟实现（MLX、llama.cpp 的 Metal 后端）"
            "用上 `simdgroup_matrix` 后 fp16 能到 2.0~2.4 TFLOPS，差距全在分块与矩阵指令。")
        add("")

    # ---------------- 带宽
    if read_gbs:
        add("## 4. 内存带宽 / Memory bandwidth")
        add("")
        add("| Test | 中位数 | 峰值 | 采样区间（最慢–最快） | 占规格 | 线程数 | 测试方法 |" if has_median
            else "| Test | 实测（峰值） | 采样区间 | 占规格 | 线程数 | 测试方法 |")
        add("|---|---|---|---|---|---|---|")
        methods = {
            "gpu_read": ("GPU read / 读", "float4 网格步长累加（只读）"),
            "gpu_write": ("GPU write / 写", "float4 网格步长写入（只写）"),
            "gpu_copy": ("GPU copy / 拷贝", "MTLBlitCommandEncoder 拷贝（DMA，读写各一次）"),
        }
        for key in ["gpu_read", "gpu_write", "gpu_copy"]:
            m = bandwidth.get(key)
            if not m:
                continue
            share = f"{stable(m, 'gbps') / spec['bandwidth_gbs'] * 100:.0f}%" if spec else "—"
            lo = m.get("gbps_min")
            hiref = m.get("gbps_max", peak(m, "gbps"))
            rng = f"{fmt(lo, 1)} – {fmt(hiref, 1)}" if lo is not None else "—"
            if has_median:
                add(f"| {methods[key][0]} | {fmt(stable(m, 'gbps'), 1)} GB/s | {fmt(peak(m, 'gbps'), 1)} GB/s "
                    f"| {rng} | {share} | {m.get('threads', '—')} | {methods[key][1]} |")
            else:
                add(f"| {methods[key][0]} | {fmt(m['gbps'], 1)} GB/s | {rng} | {share} | "
                    f"{m.get('threads', '—')} | {methods[key][1]} |")
        add("")
        buf_mb = bandwidth.get("gpu_read", {}).get("bytes_per_rep", 0) / 1024 / 1024
        add(f"> 缓冲区 {buf_mb:.0f} MB，测试前填充不可压缩数据，避免零页/内存压缩导致虚高。"
            "「中位数」用于跨设备比较，「峰值」是 9 次采样中最快的一次（≈硬件上限）。"
            "`gbps == gbps_max`（同一个「最快」值），保留两个字段是为了兼容旧结果。")
        add("> 注意：缓冲区大小会影响读数（64 MB 可能部分命中 SLC）。跨设备比较必须用同一 `--size-mb`。")
        sat = data.get("bandwidth_saturation")
        if sat:
            verdict = ("已饱和（线程翻倍后提升 <3%）" if sat.get("saturated")
                       else "**未饱和**，说明这台 GPU 需要更多线程，请提高 `--bw-threads` 后复测")
            add(f"> 线程饱和检查：{sat.get('alt_threads')} vs {sat.get('primary_threads')} 线程，"
                f"读 {sat.get('read_gain_pct', 0):+.1f}% / 写 {sat.get('write_gain_pct', 0):+.1f}% / "
                f"拷贝 {sat.get('copy_gain_pct', 0):+.1f}%，{verdict}。")
        add("")

    # ---------------- roofline
    peak_m = fp16_mma or fp32_mma   # 缺 fp16 时退回 fp32 峰值，避免整节消失
    if peak_m and read_gbs:
        add("## 5. 模型 roofline / Prefill & decode upper bound")
        add("")
        read_bw = read_gbs * args.bw_eff * 1e9   # 字节/秒
        base = stable(peak_m, "macs_per_sec")
        hand_eff = 0.4
        show_hand = abs(args.eff - hand_eff) > 1e-9
        head = "| 模型 | 精度 | 权重 | prefill tok/s @%.1f |" % args.eff
        if show_hand:
            head += " prefill @%.1f（手写） |" % hand_eff
        add(head + " decode tok/s | 能否放进可用显存 |")
        add("|---" * (6 + (1 if show_hand else 0)) + "|")
        working_set_gb = (gpu.get("recommended_working_set_mb") or 12288) / 1024
        for model in MODELS.values():
            for prec in ["fp16", "int8", "int4"]:
                bpp = BYTES_PER_PARAM[prec]
                weights_gb = model["params"] * bpp / 1e9
                kv = 2 * model["layers"] * model["kv_heads"] * model["head_dim"] * 2
                prefill = base * args.eff / model["params"]
                cell = f"| {model['name']} | {prec.upper()} | {weights_gb:.1f} GB | {prefill:.1f} |"
                if show_hand:
                    cell += f" {base * hand_eff / model['params']:.1f} |"
                decode = read_bw / (model["params"] * bpp + kv * args.ctx)
                fits = "是" if weights_gb + 2 < working_set_gb else "否"
                add(cell + f" {decode:.1f} | {fits} |")
        add("")
        add(f"> prefill 按峰值 {fmt(stable(peak_m, 'value'))} {peak_m['unit']}（中位数）× 效率 "
            f"{args.eff * 100:.0f}% 计；decode 按带宽 {fmt(read_bw / 1e9, 1)} GB/s × "
            f"{args.bw_eff * 100:.0f}% 计；KV cache 按 {args.ctx} 上下文、FP16。"
            "prefill 的实际达成率高度依赖 GEMM 写法（第 3 节），decode 在 MHA + 小 batch 下几乎只取决于带宽。")
        add("> **效率别用错**：MLX / MPS / llama.cpp 这类成熟实现实测 0.78~0.83，手写内核 0.3~0.6。"
            "旧版报告默认 0.4，会把手写实现和 MLX 混在一起、并低估 MLX 约 2 倍。"
            "**decode 公式在 GQA + 大 batch 下会转为算力受限，不能只看带宽。**"
            "（校核见 docs/records/51-mlx-crosscheck.md）")
        add("")

    # ---------------- 对照
    if spec:
        add("## 6. 与公开规格对照 / Vs. public specs")
        add("")
        add("| 指标 | 公开口径 | 本机峰值 | 偏差 | 解读 |")
        add("|---|---|---|---|---|")
        if fp32_mma:
            add(f"| FP32 峰值 | {spec['fp32_tflops']} TFLOPS | {fmt(peak(fp32_mma, 'value'))} TFLOPS | "
                f"{peak(fp32_mma, 'value') / spec['fp32_tflops'] * 100 - 100:+.0f}% | "
                f"矩阵指令通路，约为规格的 {peak(fp32_mma, 'value') / spec['fp32_tflops'] * 100:.0f}%"
                "（差额来自发射率与规格口径） |")
        if fp32_fma:
            add(f"| FP32 普通 FMA | — | {fmt(peak(fp32_fma, 'value'))} TFLOPS | — | 非矩阵写法只有峰值的一半 |")
        if fp16_mma and fp32_mma:
            ratio = peak(fp16_mma, "value") / peak(fp32_mma, "value")
            note = ("FP16 与 FP32 矩阵指令同速，**没有 2 倍**" if 0.9 <= ratio <= 1.1
                    else f"FP16 是 FP32 的 {ratio:.2f} 倍；跨代 GPU 的比例不同，不能外推")
            add(f"| FP16 峰值 | 同代不同架构 | {fmt(peak(fp16_mma, 'value'))} TFLOPS | — | {note} |")
        if read_gbs:
            add(f"| 内存带宽 | {spec['bandwidth_gbs']} GB/s | {fmt(peak(bandwidth['gpu_read'], 'gbps'), 1)} GB/s | "
                f"{peak(bandwidth['gpu_read'], 'gbps') / spec['bandwidth_gbs'] * 100 - 100:+.0f}% | "
                "视负载而定，空闲时可达 85~88% |")
        add("| Neural Engine | 各代不同 | 未测 | — | ANE 不通过 Metal 暴露，需 Core ML；M5 起 GPU 内也有神经加速器 |")
        add("")
        if spec.get("binned_from"):
            add(f"> 规格匹配：`{chip}` → `{spec_name}`，按实测 {spec['cores']} 核折算"
                f"（原表 {spec['binned_from']} 核）。带宽未折算。")
        else:
            add(f"> 规格匹配：`{chip}` → `{spec_name}`。同芯片有 binned 版本（GPU 核数不同），"
                "若核数对不上，请以实测核数为准，不要直接套规格。")
        add("")

    # ---------------- 评估
    add("## 7. 结果评估 / Assessment")
    add("")
    if fp16_mma and spec:
        ratio = (stable(fp16_mma, "value") / stable(fp32_mma, "value")) if fp32_mma else 1.0
        add(f"- **标称算力是真的，但只对矩阵指令成立**：FP32 `simdgroup_matrix` "
            f"{fmt(stable(fp32_mma, 'value'))} TFLOPS ≈ 规格 {spec['fp32_tflops']} 的 "
            f"{stable(fp32_mma, 'value') / spec['fp32_tflops'] * 100:.0f}%；"
            f"FP16 {fmt(stable(fp16_mma, 'value'))} TFLOPS（{ratio:.2f}× FP32）；"
            "普通 FMA 明显低于同精度矩阵指令（比例见第 1 节）。"
            "写 Metal 推理内核时能否用上 `simdgroup_matrix`，直接决定 prefill 能到矩阵峰值的几成。")
    if gemm.get("fp16") and fp16_mma:
        add(f"- **手写 GEMM 只到 FMA 峰值的 "
            f"{stable(gemm['fp16'], 'tflops') / stable(fp16_fma, 'value') * 100 if fp16_fma else 0:.0f}%**："
            f"{fmt(stable(gemm['fp16'], 'tflops'))} TFLOPS。MLX / llama.cpp 这类成熟实现通常能到"
            "同精度矩阵峰值的 85~95%，差距在分块、双缓冲与对矩阵指令的使用；"
            "**这个比例也要在每代上重新验证，不能直接套 M1 的系数**。")
    if fp16_mma and any(k in gemm for k in ("int8", "int4")):
        parts = "、".join(f"{k.upper()} {fmt(stable(gemm[k], 'tflops'))} TOPS"
                          for k in ("int8", "int4") if k in gemm)
        add(f"- **整型没有快通路**：纯 int32 ALU 峰值 {fmt(int8_alu['value'])} TOPS，只有 FP16 ALU 峰值的 "
            f"{int8_alu['value'] / fp16_fma['value'] * 100:.0f}%；int8/int4 GEMM 又只有该 ALU 峰值的 "
            f"{gemm['int8']['tflops'] / int8_alu['value'] * 100:.0f}%——先把 char 转成 int32 再累加，"
            "转换与分块访存吃掉另一半。" if int8_alu and fp16_fma and gemm.get("int8") else "")
        add(f"- **量化不加速**：{parts}，都只有 FP16 峰值的 "
            f"{min(stable(gemm[k], 'tflops') for k in ('int8', 'int4') if k in gemm) / stable(fp16_mma, 'value') * 100:.0f}% 上下。"
            "Mac 上量化的价值是省带宽，不是提算力——这也是为什么 decode 受益、prefill 不受益。")
    if read_gbs:
        add(f"- **带宽是 decode 的硬上限**：实测读 {fmt(read_gbs, 1)} GB/s。7B FP16 权重 14 GB → "
            f"理论 ≤{read_gbs / 14.0:.1f} tok/s；7B INT4 权重 3.5 GB → ≤{read_gbs / 3.5:.1f} tok/s（未计 KV）。")
    add("")

    # ---------------- 方法
    add("## 8. 方法学与局限 / Method and limits")
    add("")
    add("- **计时**：每个指标采样 9 次（先丢弃 1 次预热），保留全部原始样本；"
        "`value` 为最快一次（峰值），`value_median` 为中位数。计时用 `MTLCommandBuffer` 的 GPU 侧起止时间，"
        "单次派发校准到 20~120 ms。命令缓冲失败/超时会直接报错，不会静默产出 0 秒。")
    add("- **防虚高**：峰值内核整条向量写回，并在每次 `run` 内嵌自检（GPU vs CPU 参考逐元素比对）；"
        "自检失败时 `run` 以退出码 2 结束。INT8/INT4 的微基准极易被编译器折叠成常数加法，"
        "因此量化算力改由**真实 int8 / int4 GEMM** 测量。")
    add("- **环境可归因**：结果记录 OS build、热状态、低电量模式、电源、系统负载、git sha；"
        "带宽默认 262,144 线程；每台机都会用 2× 线程复测一次做饱和判断，未饱和时请按提示提高线程数。")
    add("- **峰值≠应用性能**：峰值内核无访存；手写 GEMM 只到 FMA 峰值的 "
        f"{stable(gemm['fp32'], 'tflops') / stable(fp32_fma, 'value') * 100:.0f}%（FP32）/ "
        f"{stable(gemm['fp16'], 'tflops') / stable(fp16_fma, 'value') * 100:.0f}%（FP16），"
        "说明内核写法这一层的损失是实打实的。" if gemm.get('fp32') and fp32_fma and fp16_fma and gemm.get('fp16')
        else "- **峰值≠应用性能**：峰值内核无访存，实际应用会低于峰值。")
    add("- **没测的部分**：Neural Engine、多进程并发、长时间满载的热降频（`--repeat` 只有几秒，不足以触发）。")
    add("")
    add("---")
    add(f"*Generated by macgpu v{data['version']} · 重新生成：`make report`*")

    return "\n".join(L) + "\n"


def raw_view(text, data):
    """只保留实测部分（算力 / GEMM / 带宽），去掉规格对照、roofline 与结论。"""
    lines = text.split("\n")
    idx = next((i for i, l in enumerate(lines) if l.startswith("## ")), len(lines))
    preamble = lines[:idx]
    sections, cur = [], None
    for line in lines[idx:]:
        if line.startswith("## "):
            if cur:
                sections.append(cur)
            cur = [line]
        elif cur is not None:
            cur.append(line)
    if cur:
        sections.append(cur)
    def has_rows(sec):
        rows = [l for l in sec if l.startswith("|") and "---" not in l]
        return len(rows) > 1

    kept = [s for s in sections
            if s[0].startswith(("## 2.", "## 3.", "## 4.")) and has_rows(s)]
    out = preamble + ["> 本报告只含实测值：不做规格对照、不换算 roofline、不给效率系数。"]
    for sec in kept:
        out.extend(sec)
    notes = data.get("notes") or []
    if notes:
        out += ["## 备注 / Notes", ""] + [f"- {n}" for n in notes] + [""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="macgpu 结果转 Markdown 报告")
    ap.add_argument("result", nargs="?", help="result.json 路径（默认取最新）")
    ap.add_argument("-o", "--out", help="输出文件（默认打印到 stdout）")
    ap.add_argument("--ctx", type=int, default=2048, help="上下文长度（默认 2048）")
    ap.add_argument("--eff", type=float, default=0.80,
                    help="prefill 计算效率（默认 0.80，对应 MLX/MPS；手写内核约 0.4）")
    ap.add_argument("--bw-eff", type=float, default=0.90, help="decode 带宽利用率（默认 0.90）")
    ap.add_argument("--strict", action="store_true", help="能量环境告警时以退出码 3 结束")
    ap.add_argument("--raw", action="store_true", help="只输出实测值，跳过规格/roofline/结论")
    args = ap.parse_args()

    with open(args.result or find_latest()) as f:
        data = json.load(f)
    text = build(data, args)
    if args.raw:
        text = raw_view(text, data)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"报告已写入 {args.out}")
    else:
        sys.stdout.write(text)
    if args.strict and "环境告警" in text:
        sys.exit(3)


if __name__ == "__main__":
    main()
