#!/usr/bin/env python3
"""文档与实现的一致性校验（stdlib only）。

检查四件事：
1. docs/ 下每个文件都被 docs/README.md 收录；
2. 指标矩阵里的内核名与 src/Kernels.swift 里的 `kernel void` 一一对应；
3. design/21-output-contract.md 声明的 JSON 字段与最近一次 result.json 一致；
4. 仓库里没有残留旧项目名 / 旧路径。

用法: python3 tools/docs_contract.py [-v]
退出码: 0 全部通过, 1 有问题。
"""
import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
INDEX = os.path.join(DOCS, "README.md")
METHOD = os.path.join(DOCS, "spec", "10-methodology.md")
CONTRACT = os.path.join(DOCS, "design", "21-output-contract.md")
KERNELS = os.path.join(ROOT, "src", "Kernels.swift")
LATEST = os.path.join(ROOT, "output", "results", "latest", "result.json")

# 旧名字残留：文档/代码里出现即视为未同步（历史记录目录豁免）
# 旧引用用正则匹配：output/results/... 是现行路径，不能误判
STALE = [
    (re.compile(r"\bmacbench\b"), "旧项目名 macbench"),
    (re.compile(r"(?<!output/)results/<"), "旧结果路径（应为 output/results/）"),
    (re.compile(r"build/macbench"), "旧二进制路径"),
]
# 记录与治理文档会刻意提到历史命名，豁免（其余文档/代码不得残留）
STALE_EXEMPT = (
    "docs/records/",
    "docs/process/40-validation.md",
    "tools/docs_contract.py",
)

KERNEL_RE = re.compile(r"kernel\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
BACKTICK_KERNEL_RE = re.compile(r"`((?:peak|gemm|bw)_[A-Za-z0-9_]+)`")
SCHEMA_LINE_RE = re.compile(r"^(\w[\w-]*)(\[\])?\s*:\s*(.+)$")


def iter_docs():
    out = []
    for base, _dirs, files in os.walk(DOCS):
        for name in sorted(files):
            if name.endswith(".md"):
                out.append(os.path.join(base, name))
    return sorted(out)


def rel(path):
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def check_index():
    issues = []
    index_text = open(INDEX, encoding="utf-8").read()
    for path in iter_docs():
        if path == INDEX:
            continue
        r = os.path.relpath(path, DOCS).replace(os.sep, "/")
        if r not in index_text:
            issues.append(f"docs/README.md 未收录 {r}")
    return issues


def check_kernels():
    issues = []
    src = open(KERNELS, encoding="utf-8").read()
    implemented = {m.group(1) for m in KERNEL_RE.finditer(src)}
    implemented = {k for k in implemented if k.startswith(("peak_", "gemm_"))}

    documented = set()
    for path in iter_docs():
        if "/records/" in path:
            continue
        documented |= set(BACKTICK_KERNEL_RE.findall(open(path, encoding="utf-8").read()))
    documented = {k for k in documented if k.startswith(("peak_", "gemm_"))}

    for k in sorted(implemented - documented):
        issues.append(f"内核 {k} 已在源码实现，但指标矩阵/文档里没有记载")
    for k in sorted(documented - implemented):
        issues.append(f"文档引用了内核 {k}，但 src/Kernels.swift 里不存在")
    return issues


def check_schema():
    """契约文档声明的字段集合 == 真实 result.json 的字段集合。

    字段或顶层键末尾加 `?` 表示可选：存在时必须带上声明的子字段，不存在不算错。
    """
    issues = []
    if not os.path.isfile(LATEST):
        return issues  # 还没有跑过：跳过（make run 后再校验）
    text = open(CONTRACT, encoding="utf-8").read()
    block = re.search(r"```schema\n(.*?)```", text, re.S)
    if not block:
        return ["design/21-output-contract.md 缺少 ```schema 字段声明块"]

    declared, optional_keys, optional_fields = {}, set(), {}
    for line in block.group(1).splitlines():
        m = SCHEMA_LINE_RE.match(line.strip())
        if not m:
            continue
        name, _is_list, fields = m.group(1), m.group(2), m.group(3)
        if name.endswith("?"):
            name = name[:-1]
            optional_keys.add(name)
        req, opt = set(), set()
        for f in (x.strip() for x in fields.split(",")):
            if not f:
                continue
            if f.endswith("?"):
                opt.add(f[:-1])
            else:
                req.add(f)
        declared[name] = req | opt
        optional_fields[name] = opt | (declared[name] if name in optional_keys else set())

    data = json.load(open(LATEST, encoding="utf-8"))
    actual = set(data)
    for name in declared:
        if name.endswith("[]") or name == "top-level" or name in optional_keys:
            continue
        if name not in actual:
            issues.append(f"契约声明了顶层字段 {name}，但 result.json 里没有")
    top = declared.get("top-level")
    if top is not None:
        for k in sorted(actual - top):
            issues.append(f"result.json 有未声明的顶层字段 {k}")
        for k in sorted((top - optional_fields.get("top-level", set())) - actual):
            issues.append(f"契约声明了顶层字段 {k}，实际缺失")

    for name, payload in [("compute", data.get("compute") or []),
                          ("gemm", data.get("gemm") or []),
                          ("bandwidth", data.get("bandwidth") or [])]:
        want = declared.get(name + "[]")
        if want is None or not payload:
            continue
        got = set(payload[0])
        for k in sorted(got - want):
            issues.append(f"{name}[] 有未声明字段 {k}")
        for k in sorted((want - optional_fields.get(name + "[]", set())) - got):
            issues.append(f"{name}[] 声明了字段 {k}，实际缺失")

    # 可选对象：存在时校验其字段
    for key in optional_keys:
        if key not in data:
            continue
        got = set(data[key] if isinstance(data[key], dict) else {})
        want = declared.get(key, set())
        for k in sorted(got - want):
            issues.append(f"{key} 有未声明字段 {k}")
        for k in sorted(want - got):
            issues.append(f"{key} 声明了字段 {k}，实际缺失")
    return issues


def check_stale():
    issues = []
    targets = [KERNELS, os.path.join(ROOT, "src", "Bench.swift"),
               os.path.join(ROOT, "src", "main.swift"),
               os.path.join(ROOT, "Makefile"),
               os.path.join(ROOT, "macgpu"),
               os.path.join(ROOT, "AGENT.md")]
    targets += [p for p in iter_docs() if not any(e in rel(p) for e in STALE_EXEMPT)]
    targets += glob.glob(os.path.join(ROOT, "tools", "*.py"))
    targets += glob.glob(os.path.join(ROOT, "tests", "*.sh"))

    for path in targets:
        if not os.path.isfile(path):
            continue
        if any(e in rel(path) for e in STALE_EXEMPT):
            continue
        for i, line in enumerate(open(path, encoding="utf-8", errors="ignore"), 1):
            for pattern, desc in STALE:
                if pattern.search(line):
                    issues.append(f"{rel(path)}:{i} 残留旧引用（{desc}）")
    return issues


def check_all():
    issues = []
    for fn in (check_index, check_kernels, check_schema, check_stale):
        issues += fn()
    return {"ok": not issues, "issues": issues}


def main():
    ap = argparse.ArgumentParser(description="文档/实现一致性校验")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    report = check_all()
    if args.verbose or not report["ok"]:
        for issue in report["issues"]:
            print("  ✗", issue)
    if report["ok"]:
        print(f"docs contract OK — {len(iter_docs())} 篇文档，内核/字段/命名均一致")
        return 0
    print(f"docs contract FAILED — {len(report['issues'])} 处不一致")
    return 1


if __name__ == "__main__":
    sys.exit(main())
