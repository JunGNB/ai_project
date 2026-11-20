#!/usr/bin/env bash
# 冒烟测试：编译 → info → 小规模实测 → 文档一致性 → 自检 + MCP → 报告
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export MACGPU_HOME="$ROOT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

pass=0
fail=0
check() {
  if [[ "$2" == "0" ]]; then
    echo "  ok   $1"; pass=$((pass + 1))
  else
    echo "  FAIL $1"; fail=$((fail + 1))
  fi
}

echo "[1/6] 编译"
make -s all >/dev/null 2>&1
check "swiftc 编译通过" $?

echo "[2/6] 机器信息"
./macgpu info --json > "$TMP/info.json" 2>/dev/null
python3 - "$TMP/info.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["host"]["chip"], "缺少 chip"
assert d["gpu"]["core_count"] > 0, "GPU 核心数异常"
assert d["host"]["memory_gb"] > 0
assert d["gpu"]["max_threads_per_threadgroup"] >= 256
PY
check "info --json 字段完整" $?

echo "[3/6] 小规模实测"
./macgpu run --json --tests compute,gemm,bandwidth --dtypes fp32,fp16,int8 \
  --gemm-size 256 --size-mb 32 --out "$TMP/result.json" > "$TMP/run.json" 2>/dev/null
python3 - "$TMP/run.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["schema"].startswith("macgpu/"), d["schema"]
comp = {(m["dtype"], m["path"]): m for m in d["compute"]}
assert 0.05 < comp[("fp32", "mma")]["value"] < 500, comp[("fp32", "mma")]["value"]
assert 0.05 < comp[("fp32", "fma")]["value"] < 500, comp[("fp32", "fma")]["value"]
gemm = {m["dtype"]: m for m in d["gemm"]}
assert 0.01 < gemm["int8"]["tflops"] < 5000, gemm["int8"]["tflops"]
assert gemm["int8"]["unit"] == "TOPS"
bw = {m["name"]: m for m in d["bandwidth"]}
assert 1 < bw["gpu_read"]["gbps"] < 5000, bw["gpu_read"]["gbps"]
assert 1 < bw["gpu_copy"]["gbps"] < 5000
assert bw["gpu_copy"]["bytes_per_rep"] > 0
assert d["gemm"][0]["tflops"] > 0
assert d["notes"], "缺少方法说明"
PY
check "实测 JSON 合法且数值在合理区间" $?

echo "[4/6] 文档一致性"
python3 tools/docs_contract.py >/dev/null 2>&1
check "make docs（内核/字段/命名与实现一致）" $?

echo "[5/6] 自检 + MCP"
./macgpu selfcheck >/dev/null 2>&1
check "selfcheck 通过" $?

printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"machine_info","arguments":{}}}' \
  | python3 tools/mcp_server.py > "$TMP/mcp.jsonl"
python3 - "$TMP/mcp.jsonl" <<'PY'
import json, sys
lines = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
assert len(lines) == 3
assert lines[0]["result"]["serverInfo"]["name"] == "macgpu"
names = [t["name"] for t in lines[1]["result"]["tools"]]
assert "run_benchmark" in names and "generate_report" in names
assert "Apple" in lines[2]["result"]["content"][0]["text"]
PY
check "MCP 握手与工具调用" $?

echo "[6/6] 报告生成"
python3 tools/report.py "$TMP/result.json" -o "$TMP/report.md" >/dev/null
grep -q "结论速览" "$TMP/report.md"
grep -q "prefill" "$TMP/report.md"
grep -q "Compute peak" "$TMP/report.md"
check "Markdown 报告生成" $?

echo ""
echo "通过 $pass 项，失败 $fail 项"
[[ "$fail" == "0" ]]
