#!/usr/bin/env python3
"""跨 Mac 设备采集 macgpu 读数，汇总成可比较的表（仅标准库）。

用法：
  python3 macgpu-fleet/collect.py collect --bench ./mac-gpu-bench
  python3 macgpu-fleet/collect.py collect --bench ./mac-gpu-bench --launches 3 --repeat 5
  python3 macgpu-fleet/collect.py table macgpu-fleet/results

设计要点（针对 macgpu 现存的抖动问题）：
  * 每台机跑 N 次**独立进程**（默认 3），而不是一次进程里 --repeat N；
    进程级重复会把「时钟/热状态/后台负载」也采样进来。
  * 只取每轮的 min-of-9（macgpu 自身口径），在轮间取**中位数**，并给出 CV。
  * 采集 AC/电池、低电量模式、热限速、OS build、hw.model 与 GPU 核数——
    这些是横向比较时最容易漏掉、又足以造成 10~40% 偏差的变量。
  * selfcheck 不通过直接标记该机数据不可用。
"""
import argparse
import csv
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import time

COMPUTE_KEYS = [("fp32", "mma"), ("fp16", "mma"), ("fp32", "fma"), ("fp16", "fma")]
GEMM_KEYS = ["fp32", "fp16", "int8", "int4"]
BW_KEYS = ["gpu_read", "gpu_write", "gpu_copy"]


def sh(cmd, timeout=120):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, "", str(exc)


def env_snapshot():
    env = {}
    env["collected_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    env["python"] = platform.python_version()
    for key, cmd in [
        ("os_version", ["sw_vers", "-productVersion"]),
        ("os_build", ["sw_vers", "-buildVersion"]),
        ("hw_model", ["sysctl", "-n", "hw.model"]),
        ("chip", ["sysctl", "-n", "machdep.cpu.brand_string"]),
        ("cpu_cores", ["sysctl", "-n", "hw.ncpu"]),
        ("memory_bytes", ["sysctl", "-n", "hw.memsize"]),
    ]:
        rc, out, _ = sh(cmd)
        env[key] = out if rc == 0 else None
    for cmd in (["pmset", "-g"], ["pmset", "-g", "custom"]):
        rc, out, _ = sh(cmd)
        if rc == 0:
            for line in out.splitlines():
                if "lowpowermode" in line:
                    env["low_power_mode"] = line.split()[-1]
    rc, out, _ = sh(["pmset", "-g", "therm"])
    if rc == 0:
        env["thermal"] = out
    rc, out, _ = sh(["pmset", "-g", "ps"])
    if rc == 0:
        env["power_source_detail"] = out
        if "AC Power" in out:
            env["power_source"] = "AC"
        elif "Battery Power" in out:
            env["power_source"] = "Battery"
    rc, out, _ = sh(["sysctl", "-n", "vm.loadavg"])
    env["loadavg"] = out if rc == 0 else None
    return env


def row_from(result):
    row = {"host": result["host"], "gpu": result["gpu"],
           "runs": result.get("runs"), "generated_at": result.get("generated_at")}
    def med(m, field):
        """v2.1 起优先中位数；v2.0 旧结果没有 *_median 就退回峰值。"""
        return m.get(field + "_median", m[field]) if m else None

    for dtype, path in COMPUTE_KEYS:
        m = next((m for m in result.get("compute", [])
                  if m["dtype"] == dtype and m["path"] == path), None)
        row[f"{dtype}_{path}"] = med(m, "value")
    for dtype in GEMM_KEYS:
        m = next((m for m in result.get("gemm", []) if m["dtype"] == dtype), None)
        row[f"gemm_{dtype}"] = med(m, "tflops")
    for name in BW_KEYS:
        m = next((m for m in result.get("bandwidth", []) if m["name"] == name), None)
        row[f"bw_{name}"] = med(m, "gbps")
        row[f"bw_{name}_min"] = m["gbps_min"] if m else None
    return row


def aggregate(rows):
    """对每列取中位数与 CV（轮间）。"""
    med, cv = {}, {}
    if not rows:
        return med, cv
    for key in rows[0]:
        if key in ("host", "gpu", "runs", "generated_at"):
            continue
        vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
        if not vals:
            continue
        med[key] = round(statistics.median(vals), 4)
        cv[key] = (round(statistics.stdev(vals) / statistics.mean(vals) * 100, 2)
                   if len(vals) > 1 and statistics.mean(vals) else None)
    return med, cv


def collect(args):
    bench = os.path.abspath(args.bench)
    binary = os.path.join(bench, "macgpu")
    if not os.access(binary, os.X_OK):
        sys.exit(f"找不到可执行入口: {binary}")
    env = env_snapshot()
    host = (env.get("hw_model") or "mac").replace(",", "-")
    out_root = os.path.abspath(args.out)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(out_root, f"{host}-{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 1) 数值自检：不过就意味着这台的数字不可信
    rc, out, err = sh([binary, "selfcheck"], timeout=300)
    selfcheck = {"passed": rc == 0, "exit": rc, "output": (out + "\n" + err).strip()}
    with open(os.path.join(run_dir, "selfcheck.txt"), "w") as fh:
        fh.write(selfcheck["output"] + "\n")

    # 2) 多进程重复
    launches = []
    for i in range(1, args.launches + 1):
        result_path = os.path.join(run_dir, f"run{i}.json")
        rc, out, err = sh([binary, "run", "--repeat", str(args.repeat),
                           "--json", "--out", result_path], timeout=900)
        if rc != 0:
            print(f"  launch {i} 失败: {err[:300]}", file=sys.stderr)
            continue
        with open(result_path) as fh:
            launches.append(json.load(fh))
        print(f"  launch {i}/{args.launches} ok", file=sys.stderr)

    if not launches:
        sys.exit("没有任何成功的 run")

    rows = [row_from(r) for r in launches]
    med, cv = aggregate(rows)
    first = launches[0]
    bw0 = (first.get("bandwidth") or [{}])[0]
    summary = {
        "env": env,
        "bench": {"path": bench, "version": first.get("version"),
                  "schema": first.get("schema")},
        "protocol": {
            "schema": first.get("schema"),
            "tool_version": first.get("version"),
            "size_mb": round(bw0.get("bytes_per_rep", 0) / 1e6) if bw0.get("bytes_per_rep") else None,
            "bw_threads": bw0.get("threads"),
            "gemm_size": (first.get("gemm") or [{}])[0].get("size"),
            "repeat": args.repeat,
            "launches": len(launches),
            "samples_per_metric": (first.get("sampling") or {}).get("samples_per_metric"),
            "os_build": first.get("host", {}).get("os_build"),
            "thermal_state": first.get("host", {}).get("thermal_state"),
            "power_source": first.get("host", {}).get("power_source"),
            "low_power_mode": first.get("host", {}).get("low_power_mode"),
            "gpu_cores": first.get("gpu", {}).get("core_count"),
        },
        "selfcheck": selfcheck,
        "launches": len(launches),
        "repeat": args.repeat,
        "rows": rows,
        "median": med,
        "cv_pct": cv,
        "accept": {
            "selfcheck_ok": selfcheck["passed"],
            "gpu_cores": launches[0]["gpu"].get("core_count"),
            "load_avg_1m": max((r["host"].get("load_avg_1m") or 0) for r in launches),
            "read_cv_pct": cv.get("bw_gpu_read"),
            "fp16_mma_cv_pct": cv.get("fp16_mma"),
            "warnings": [
                w for w in [
                    None if selfcheck["passed"] else "selfcheck 未通过",
                    None if (cv.get("bw_gpu_read") or 0) <= 5 else f"读带宽轮间 CV={cv.get('bw_gpu_read')}% > 5%",
                    None if (cv.get("fp16_mma") or 0) <= 3 else f"FP16 MMA 轮间 CV={cv.get('fp16_mma')}% > 3%",
                    None if max((r["host"].get("load_avg_1m") or 0) for r in launches) <= 2.0 else "测试时 load_avg > 2.0",
                    None if (env.get("low_power_mode") in (None, "0")) else "低电量模式未关闭",
                    None if env.get("power_source") in (None, "AC") else "未接电源（电池供电可能降频）",
                ] if w
            ],
        },
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "env.json"), "w") as fh:
        json.dump(env, fh, ensure_ascii=False, indent=2)

    # 3) 追加到总表
    csv_path = os.path.join(out_root, "fleet.csv")
    write_header = not os.path.exists(csv_path)
    flat = {"run_dir": os.path.relpath(run_dir, out_root),
            "chip": env.get("chip"), "hw_model": env.get("hw_model"),
            "gpu_cores": launches[0]["gpu"].get("core_count"),
            "memory_gb": launches[0]["host"].get("memory_gb"),
            "os": env.get("os_version"), "os_build": env.get("os_build"),
            "power": env.get("power_source"), "lpm": env.get("low_power_mode"),
            "selfcheck": selfcheck["passed"], "launches": len(launches),
            "repeat": args.repeat}
    flat.update({k: v for k, v in med.items()})
    flat.update({f"cv_{k}": v for k, v in cv.items() if v is not None})
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(flat.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(flat)

    print(f"\n结果目录: {run_dir}")
    print(f"汇总表:   {csv_path}")
    print_summary(summary)
    return run_dir


def print_summary(s):
    m, env = s["median"], s["env"]
    print(f"\n{s['env'].get('chip')} / {env.get('hw_model')} / GPU "
          f"{s['accept']['gpu_cores']} 核 / {env.get('memory_bytes')} B / "
          f"macOS {env.get('os_version')} ({env.get('os_build')})")
    print(f"  selfcheck: {'PASS' if s['selfcheck']['passed'] else 'FAIL'}"
          f"  launches={s['launches']} repeat={s['repeat']}")
    for label, key, unit in [
        ("FP32 MMA", "fp32_mma", "TFLOPS"), ("FP16 MMA", "fp16_mma", "TFLOPS"),
        ("FP32 FMA", "fp32_fma", "TFLOPS"), ("FP16 FMA", "fp16_fma", "TFLOPS"),
        ("GEMM fp16", "gemm_fp16", "TFLOPS"), ("GEMM int8", "gemm_int8", "TOPS"),
        ("read", "bw_gpu_read", "GB/s"), ("write", "bw_gpu_write", "GB/s"),
        ("copy", "bw_gpu_copy", "GB/s"),
    ]:
        if key in m:
            cv = s["cv_pct"].get(key)
            print(f"  {label:10s} {m[key]:7.2f} {unit:6s} CV={cv if cv is not None else '—'}%")
    if s["accept"]["warnings"]:
        print("  ⚠ " + "；".join(s["accept"]["warnings"]))


def load_spec(bench):
    """从 mac-gpu-bench/tools/report.py 读规格表，避免两处维护。"""
    path = os.path.join(os.path.abspath(bench), "tools", "report.py")
    if not os.path.exists(path):
        return {}
    try:
        spec = importlib.util.spec_from_file_location("bench_report", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "SPEC", {})
    except Exception:
        return {}


def compare(dirs, bench):
    """检查多台设备的结果是否真的可比，并给出按核数归一的读数。"""
    summaries = []
    for root in dirs:
        for base, _dirs, files in os.walk(root):
            if "summary.json" in files:
                with open(os.path.join(base, "summary.json")) as fh:
                    summaries.append(json.load(fh))
    if not summaries:
        sys.exit("没有找到 summary.json")
    SPEC = load_spec(bench)
    base = summaries[0]
    bp = base.get("protocol") or {}
    hard, rows = [], []
    for s in summaries:
        p = s.get("protocol") or {}
        env, m, acc = s["env"], s["median"], s["accept"]
        flags = list(acc.get("warnings") or [])
        if not p:
            flags.append("旧会话（无协议字段），请用 v2.1 重跑")
            hard.append(f"{env.get('chip')}: 缺协议字段")
        for key, label in [("schema", "schema"), ("size_mb", "带宽工作集"),
                           ("gemm_size", "GEMM 规模"), ("bw_threads", "带宽线程"),
                           ("repeat", "repeat")]:
            if p and bp and p.get(key) != bp.get(key):
                flags.append(f"{label}不同（{p.get(key)} vs {bp.get(key)}）")
                hard.append(f"{env.get('chip')}: {label}不一致")
        if p and bp and p.get("os_build") != bp.get("os_build"):
            flags.append("OS build 不同（Metal 编译器不同）")
        if p.get("thermal_state") not in (None, "nominal"):
            flags.append(f"热状态 {p['thermal_state']}")
        cores = acc.get("gpu_cores") or p.get("gpu_cores")
        name = env.get("chip") or ""
        spec = next((v for k, v in sorted(SPEC.items(), key=lambda kv: -len(kv[0]))
                     if name == k or name.startswith(k + " ")), None)
        if spec and cores and spec.get("cores") and cores != spec["cores"]:
            flags.append(f"binned（{spec['cores']} 核规格，实测 {cores} 核）")
        rows.append((s, p, cores, spec, flags))

    hdr = ("芯片", "GPU核", "OS build", "带宽集", "GEMM", "线程", "repeat", "状态")
    print("## 协议一致性\n")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for s, p, cores, spec, flags in rows:
        env = s["env"]
        ok = not flags
        print("| " + " | ".join(str(x) for x in [
            (env.get("chip") or "?") + ("" if ok else " ⚠"),
            cores or "—", p.get("os_build") or "—", p.get("size_mb") or "—",
            p.get("gemm_size") or "—", p.get("bw_threads") or "—",
            p.get("repeat") or "—", "可比" if ok else "；".join(flags),
        ]) + " |")

    print("\n## 归一化读数（按实测 GPU 核数）\n")
    hdr2 = ("芯片", "核", "MMA/核 TFLOPS", "读/核 GB/s", "FP16/FP32", "MMA/FMA", "占折算规格")
    print("| " + " | ".join(hdr2) + " |")
    print("|" + "---|" * len(hdr2))
    for s, p, cores, spec, _flags in rows:
        m = s["median"]
        cores = cores or 1
        fp32 = m.get("fp32_mma")
        f16 = m.get("fp16_mma")
        fma = m.get("fp32_fma")
        rd = m.get("bw_gpu_read")
        pct = "—"
        if spec and fp32:
            effective = spec["fp32_tflops"] * (cores / spec["cores"])
            pct = f"{fp32 / effective * 100:.0f}%"
        print("| " + " | ".join(str(x) for x in [
            s["env"].get("chip") or "?", cores,
            f"{fp32 / cores:.3f}" if fp32 else "—",
            f"{rd / cores:.2f}" if rd else "—",
            f"{f16 / fp32:.2f}×" if f16 and fp32 else "—",
            f"{fp32 / fma:.2f}×" if fp32 and fma else "—",
            pct,
        ]) + " |")

    if hard:
        print("\n> 存在硬性不一致（协议/schema/缺字段），**数值不能直接排序**：")
        for h in dict.fromkeys(hard):
            print(f"> - {h}")
        return 3
    print("\n> 协议一致；跨代比较仍需注意散热（Air vs Studio）、binned 核数与 OS 版本。")
    return 0


def table(dirs):
    summaries = []
    for root in dirs:
        for base, _dirs, files in os.walk(root):
            if "summary.json" in files:
                with open(os.path.join(base, "summary.json")) as fh:
                    summaries.append(json.load(fh))
    if not summaries:
        sys.exit("没有找到 summary.json")
    summaries.sort(key=lambda s: (s["env"].get("chip") or "", s["env"].get("hw_model") or ""))
    cols = [("chip", "芯片"), ("hw_model", "机型"), ("gpu_cores", "GPU核"),
            ("os", "macOS"), ("fp32_mma", "FP32 MMA"), ("fp16_mma", "FP16 MMA"),
            ("fp32_fma", "FP32 FMA"), ("fp16_fma", "FP16 FMA"),
            ("gemm_fp16", "GEMM fp16"), ("gemm_int8", "GEMM int8"),
            ("bw_gpu_read", "读 GB/s"), ("bw_gpu_write", "写 GB/s"),
            ("bw_gpu_copy", "拷贝 GB/s"), ("read_cv", "读CV%")]
    print("| " + " | ".join(c[1] for c in cols) + " |")
    print("|" + "---|" * len(cols))
    for s in summaries:
        env, m = s["env"], s["median"]
        gpu_cores = s["accept"]["gpu_cores"]
        vals = {
            "chip": env.get("chip"), "hw_model": env.get("hw_model"),
            "gpu_cores": gpu_cores, "os": env.get("os_version"),
            "fp32_mma": f"{m.get('fp32_mma', 0):.2f}",
            "fp16_mma": f"{m.get('fp16_mma', 0):.2f}",
            "fp32_fma": f"{m.get('fp32_fma', 0):.2f}",
            "fp16_fma": f"{m.get('fp16_fma', 0):.2f}",
            "gemm_fp16": f"{m.get('gemm_fp16', 0):.2f}",
            "gemm_int8": f"{m.get('gemm_int8', 0):.2f}",
            "bw_gpu_read": f"{m.get('bw_gpu_read', 0):.1f}",
            "bw_gpu_write": f"{m.get('bw_gpu_write', 0):.1f}",
            "bw_gpu_copy": f"{m.get('bw_gpu_copy', 0):.1f}",
            "read_cv": f"{s['accept'].get('read_cv_pct') or 0:.1f}",
        }
        flag = "" if s["selfcheck"]["passed"] and not s["accept"]["warnings"] else " ⚠"
        print("| " + " | ".join(str(vals[c[0]] if c[0] != "chip" else f"{vals['chip']}{flag}") for c in cols) + " |")
    print("\n注：数值为多轮独立进程的中位数；⚠ 表示 selfcheck 失败或轮间抖动超阈值。")


def main():
    ap = argparse.ArgumentParser(description="macgpu 跨设备采集与汇总")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--bench", default="./mac-gpu-bench", help="mac-gpu-bench 目录")
    c.add_argument("--out", default="./macgpu-fleet/results", help="产物目录")
    c.add_argument("--launches", type=int, default=3, help="独立进程重复次数（默认 3）")
    c.add_argument("--repeat", type=int, default=5, help="每次进程内 --repeat（默认 5）")
    t = sub.add_parser("table")
    t.add_argument("dirs", nargs="+", help="包含 summary.json 的目录")
    c2 = sub.add_parser("compare")
    c2.add_argument("dirs", nargs="+", help="包含 summary.json 的目录")
    c2.add_argument("--bench", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "..", "mac-gpu-bench"),
                    help="用于读规格表的 mac-gpu-bench 目录")
    args = ap.parse_args()
    if args.cmd == "compare":
        sys.exit(compare(args.dirs, args.bench))
    if args.cmd == "collect":
        collect(args)
    else:
        table(args.dirs)


if __name__ == "__main__":
    main()
