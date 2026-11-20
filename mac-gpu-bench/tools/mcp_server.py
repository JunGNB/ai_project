#!/usr/bin/env python3
"""macgpu (Mac GPU Compute & Bandwidth Benchmark) 的 MCP 服务（stdio，零依赖标准库）。

让大模型 / Agent 可以直接调用本机的 GPU 算力与带宽测试：
    {"command": "python3", "args": ["/path/to/mac-gpu-bench/tools/mcp_server.py"]}

提供的工具：
    machine_info      机器与 GPU 基本信息
    run_benchmark     跑算力/GEMM/带宽测试，返回结果 JSON
    get_latest_result 读取最近一次完整结果
    list_results      列出历史结果
    generate_report   生成 Markdown 报告
    selfcheck         数值正确性自检

协议：JSON-RPC 2.0 over stdio（MCP 2024-11-05 风格，按行分隔）。
"""

import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "macgpu")
RESULTS = os.path.join(ROOT, "output", "results")
SERVER_VERSION = "2.0.0"

TOOLS = [
    {
        "name": "machine_info",
        "description": "读取当前 Mac 的芯片、CPU/GPU 核心数、内存与 Metal 工作集上限。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "run_benchmark",
        "description": (
            "在当前 Mac 上实测 GPU 算力（FP32/FP16 的矩阵指令与普通 FMA 峰值、fp32/fp16/int8 GEMM）和内存带宽。"
            "返回指标摘要与结果文件路径，完整数据可用 get_latest_result 读取。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tests": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["compute", "gemm", "bandwidth"]},
                    "description": "要跑的测试集，默认全部",
                },
                "dtypes": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["fp32", "fp16", "int8", "int4"]},
                    "description": "精度子集，默认全部",
                },
                "gemm_size": {"type": "integer", "description": "GEMM 边长，64 的倍数，默认 2048"},
                "size_mb": {"type": "integer", "description": "带宽缓冲区 MB，默认 256"},
                "timeout_s": {"type": "integer", "description": "超时秒数，默认 300"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_latest_result",
        "description": "读取最近一次 run_benchmark 的完整 JSON（含每个指标的线程数、迭代数、测试方法）。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "list_results",
        "description": "列出历史测试结果（时间、机型、关键指标）。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "generate_report",
        "description": "把某次结果汇总成 Markdown 报告（含 prefill/decode roofline 估算与评估）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "result_path": {"type": "string", "description": "result.json 路径，默认最新一次"},
                "ctx": {"type": "integer", "description": "decode 估算的上下文长度，默认 2048"},
                "eff": {"type": "number", "description": "计算效率假设，默认 0.4"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "selfcheck",
        "description": "运行数值自检：GEMM 与 CPU 参考实现比对、峰值内核输出有限性、带宽合理性。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]


def call_cli(args, timeout=600):
    proc = subprocess.run([BIN] + args, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip()[:2000])
    return proc.stdout


def result_dirs():
    return sorted(d for d in glob.glob(os.path.join(RESULTS, "*")) if os.path.isdir(d))


def summarize(data):
    return {
        "host": f"{data['host']['chip']} / {data['host']['model']} / {data['host']['memory_gb']:.0f}GB",
        "gpu_cores": data["gpu"].get("core_count"),
        "compute": {
            m["dtype"]: {"value": round(m["value"], 3), "unit": m["unit"],
                         "tmacs": round(m["macs_per_sec"] / 1e12, 3)}
            for m in data.get("compute", [])
        },
        "gemm": {m["dtype"]: {"tflops": round(m["tflops"], 3), "size": m["size"]}
                 for m in data.get("gemm", [])},
        "bandwidth": {m["name"]: round(m["gbps"], 2) for m in data.get("bandwidth", [])},
        "generated_at": data["generated_at"],
    }


def tool_run_benchmark(arguments):
    tests = arguments.get("tests") or ["compute", "gemm", "bandwidth"]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(RESULTS, stamp, "result.json")
    cmd = ["run", "--json", "--tests", ",".join(tests), "--out", out_path]
    if arguments.get("dtypes"):
        cmd += ["--dtypes", ",".join(arguments["dtypes"])]
    if arguments.get("gemm_size"):
        cmd += ["--gemm-size", str(arguments["gemm_size"])]
    if arguments.get("size_mb"):
        cmd += ["--size-mb", str(arguments["size_mb"])]
    stdout = call_cli(cmd, timeout=int(arguments.get("timeout_s", 300)))
    data = json.loads(stdout)
    return json.dumps({"result_path": out_path, **summarize(data)}, ensure_ascii=False, indent=2)


def latest_result_path():
    dirs = [d for d in result_dirs() if os.path.exists(os.path.join(d, "result.json"))]
    if not dirs:
        raise RuntimeError("还没有测试结果，先调用 run_benchmark")
    return os.path.join(dirs[-1], "result.json")


def tool_list_results(_arguments):
    rows = []
    for d in result_dirs():
        path = os.path.join(d, "result.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        rows.append({
            "path": path,
            "generated_at": data["generated_at"],
            "machine": f"{data['host']['chip']} {data['host']['memory_gb']:.0f}GB",
            "fp32_tflops": next((round(m["value"], 2) for m in data["compute"] if m["dtype"] == "fp32"), None),
            "bandwidth_gbs": next((round(m["gbps"], 1) for m in data["bandwidth"] if m["name"] == "gpu_read"), None),
        })
    return json.dumps(rows, ensure_ascii=False, indent=2)


def tool_generate_report(arguments):
    cmd = [sys.executable, os.path.join(ROOT, "tools", "report.py")]
    if arguments.get("result_path"):
        cmd.append(arguments["result_path"])
    if arguments.get("ctx"):
        cmd += ["--ctx", str(arguments["ctx"])]
    if arguments.get("eff"):
        cmd += ["--eff", str(arguments["eff"])]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip()[:2000])
    return proc.stdout


HANDLERS = {
    "machine_info": lambda _a: call_cli(["info", "--json"]).strip(),
    "run_benchmark": tool_run_benchmark,
    "get_latest_result": lambda _a: open(latest_result_path()).read().strip(),
    "list_results": tool_list_results,
    "generate_report": tool_generate_report,
    "selfcheck": lambda _a: call_cli(["selfcheck"]),
}


def response(msg_id, result=None, error=None):
    body = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        body["error"] = error
    else:
        body["result"] = result
    return body


def handle(msg):
    method = msg.get("method")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    if method == "initialize":
        return response(msg_id, {
            "protocolVersion": params.get("protocolVersion", "2024-11-05"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "macgpu", "version": SERVER_VERSION},
            "instructions": "苹果 Mac GPU 算力/带宽实测工具。先 machine_info，再 run_benchmark，最后 generate_report。",
        })
    if method in ("notifications/initialized", "initialized"):
        return None
    if method == "ping":
        return response(msg_id, {})
    if method == "tools/list":
        return response(msg_id, {"tools": TOOLS})
    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        handler = HANDLERS.get(name)
        if handler is None:
            return response(msg_id, error={"code": -32602, "message": f"未知工具: {name}"})
        try:
            text = handler(arguments)
            return response(msg_id, {"content": [{"type": "text", "text": text}], "isError": False})
        except Exception as exc:  # noqa: BLE001 - 工具错误要回给模型而不是崩掉服务
            return response(msg_id, {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                                     "isError": True})
    if method in ("resources/list", "prompts/list"):
        return response(msg_id, {"resources": []} if method == "resources/list" else {"prompts": []})
    if msg_id is None:
        return None
    return response(msg_id, error={"code": -32601, "message": f"未实现的方法: {method}"})


def main():
    if "--list-tools" in sys.argv:
        print(json.dumps(TOOLS, ensure_ascii=False, indent=2))
        return
    if "--selftest" in sys.argv:
        for probe in (
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "machine_info", "arguments": {}}},
        ):
            print(json.dumps(handle(probe), ensure_ascii=False)[:400])
        return

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.stdout.write(json.dumps(response(None, error={"code": -32700, "message": str(exc)})) + "\n")
            sys.stdout.flush()
            continue
        reply = handle(msg)
        if reply is not None:
            sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
