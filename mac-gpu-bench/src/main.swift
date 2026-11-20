import Foundation
import Metal

// macgpu —— Mac GPU Compute & Bandwidth Benchmark
// Apple Silicon 算力（矩阵指令 / 普通 FMA 两条通路）与内存带宽实测
//
// 用法:
//   macgpu info                       机器与 GPU 信息
//   macgpu run                        完整测试并输出 JSON
//   macgpu selfcheck                  数值正确性自检
//
// run 选项:
//   --tests compute,gemm,bandwidth    选择测试集（默认全部）
//   --dtypes fp32,fp16,int8,int4      选择精度（默认全部；每项测矩阵指令 + 普通 FMA 通路）
//   --gemm-size 2048                  GEMM 边长（64 的倍数）
//   --size-mb 256                     带宽测试缓冲区大小
//   --out <path>                      JSON 输出路径（默认 output/results/<时间戳>/result.json）
//   --json                            结果以 JSON 打印到 stdout
//   --quiet                           不打印人类可读摘要

let version = "2.1.0"

func displayWidth(_ s: String) -> Int {
    s.unicodeScalars.reduce(0) { $0 + ($1.value > 0x2E80 ? 2 : 1) }
}

func pad(_ s: String, _ width: Int) -> String {
    s + String(repeating: " ", count: max(0, width - displayWidth(s)))
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(("错误: " + message + "\n").data(using: .utf8)!)
    exit(1)
}

func usage() -> String {
    return """
    macgpu \(version) — Mac GPU Compute & Bandwidth Benchmark
    Apple Silicon 算力（矩阵指令 / 普通 FMA 两条通路）与内存带宽实测

    Usage:
      ./macgpu [options]                ← 就这一条：跑全套（默认 3 轮）+ 打印推荐数据
      ./macgpu --repeat 10              ← 同一条命令，指定轮数（1~20）
      ./macgpu info                     machine / GPU info
      ./macgpu run [options]            same as above, with options
      ./macgpu selfcheck                numeric self-check only

    Options:
      --tests compute,gemm,bandwidth    pick test groups (default: all)
      --dtypes fp32,fp16,int8           pick precisions (default: all)
      --gemm-size N                     GEMM edge length, multiple of 64 (default 2048)
      --size-mb N                       bandwidth buffer size in MB (default 256)
      --bw-threads N                    bandwidth grid threads (default: auto-tuned per machine)
      --repeat N                        run N rounds, keep the best of each (suggest 3)
      --no-selfcheck                    skip the built-in numeric self-check
      --out PATH                        JSON output path (default output/results/<stamp>/result.json)
      --json                            print result JSON to stdout
      --quiet                           no human-readable summary
    """
}

// MARK: - 参数解析

struct Options {
    var tests: Set<String> = ["compute", "gemm", "bandwidth"]
    var dtypes: [String] = ["fp32", "fp16", "int8", "int4"]
    var gemmSize = 2048
    var sizeMB = 256
    var repeatCount = 1
    var bwThreads: Int?
    var out: String?
    var json = false
    var quiet = false
    var selfcheck = true
}

func parseOptions(_ args: [String]) -> Options {
    var o = Options()
    var i = 0
    while i < args.count {
        let a = args[i]
        func nextValue() -> String {
            guard i + 1 < args.count else { fail("选项 \(a) 缺少参数") }
            i += 1
            return args[i]
        }
        switch a {
        case "--tests":
            let names = Set(nextValue().split(separator: ",").map { String($0).trimmingCharacters(in: .whitespaces) })
            let known: Set<String> = ["compute", "gemm", "bandwidth"]
            if names.isEmpty || !names.isSubset(of: known) {
                fail("未知测试集: \(names.sorted().joined(separator: ","))（可选 compute/gemm/bandwidth）")
            }
            o.tests = names
        case "--dtypes":
            let names = nextValue().split(separator: ",").map { String($0).trimmingCharacters(in: .whitespaces) }
            let known = Set(ComputeCases.all.map { $0.dtype } + GemmDType.allCases.map { $0.rawValue })
            for n in names where !known.contains(n) {
                fail("未知精度: \(n)（可选 \(known.sorted().joined(separator: "/"))）")
            }
            o.dtypes = names
        case "--gemm-size":
            guard let v = Int(nextValue()), v >= 64, v % 64 == 0 else { fail("--gemm-size 必须是 >=64 且能被 64 整除") }
            o.gemmSize = v
        case "--size-mb":
            guard let v = Int(nextValue()), v >= 16, v <= 4096 else { fail("--size-mb 需在 16~4096 之间") }
            o.sizeMB = v
        case "--bw-threads":
            guard let v = Int(nextValue()), v >= 16_384, v <= 1_048_576, v % 256 == 0 else {
                fail("--bw-threads 需在 16384~1048576 之间且是 256 的倍数")
            }
            o.bwThreads = v
        case "--out":
            o.out = nextValue()
        case "--repeat":
            guard let v = Int(nextValue()), v >= 1, v <= 20 else { fail("--repeat 需在 1~20 之间") }
            o.repeatCount = v
        case "--json":
            o.json = true
        case "--quiet":
            o.quiet = true
        case "--no-selfcheck":
            o.selfcheck = false
        default:
            fail("未知选项: \(a)\n\n" + usage())
        }
        i += 1
    }
    return o
}

// MARK: - 输出

let isoFormatter: ISO8601DateFormatter = {
    let f = ISO8601DateFormatter()
    f.formatOptions = [.withInternetDateTime]
    return f
}()

func printSummary(_ r: BenchResult) {
    let cores = r.gpu.coreCount.map { "\($0)-core GPU" } ?? "GPU"
    print("macgpu \(r.version) — Mac GPU Compute & Bandwidth Benchmark")
    let load = r.host.loadAvg1m.map { String(format: " · load %.2f", $0) } ?? ""
    print("  \(r.host.chip) · \(cores) · \(String(format: "%.1f", r.host.memoryGB)) GB unified memory · macOS \(r.host.macOS) (\(r.host.osBuild))\(load)")
    print("  power \(r.host.powerSource) · low-power \(r.host.lowPowerMode) · thermal \(r.host.thermalState)"
          + " · git \(r.tool.gitSha)\(r.tool.gitDirty ? "+dirty" : "")")
    if !r.selfcheck.skipped {
        print("  selfcheck \(r.selfcheck.passed ? "PASS" : "FAIL") "
              + "(\(r.selfcheck.checks.filter { $0.passed }.count)/\(r.selfcheck.checks.count))")
    }
    print("")

    if !r.compute.isEmpty {
        print("[1] Compute peak — per execution path, no memory traffic")
        print("    " + pad("path", 34) + pad("dtype", 7) + pad("FLOPS", 14) + pad("MAC/s", 14) + "kernel")
        for m in r.compute {
            print("    " + pad(m.labelEn, 34) + pad(m.dtype.uppercased(), 7)
                  + pad(String(format: "%.2f %@", m.value, m.unit), 14)
                  + pad(String(format: "%.2f TMAC/s", m.macsPerSec / 1e12), 14)
                  + m.methodEn)
        }
        print("    峰值 = 9 次采样最快；中位数（跨设备比较用）见 value_median / 马尔可夫报告。")
        print("    INT8 row above is the plain int32 ALU (imad) ceiling — no tensor path involved.")
        print("    INT4 has no 4-bit ALU in Metal, so it only exists as real GEMM, see [2].")
        print("")
    }

    if !r.gemm.isEmpty {
        print("[2] Real GEMM — hand-written, no simdgroup_matrix (N\(r.gemm[0].size)³)")
        print("    " + pad("dtype", 7) + pad("throughput", 16) + "vs ALU peak (same dtype)")
        // GEMM 走的是普通 ALU 通路，和矩阵指令峰值比没有意义；
        // 同通路的 fma 峰值才是它的天花板。
        var peaks: [String: Double] = [:]
        for m in r.compute where m.path == "fma" || m.path == "alu" { peaks[m.dtype] = m.value }
        for m in r.gemm {
            let rel = peaks[m.dtype].map { String(format: "%.0f%%", m.tflops / $0 * 100) } ?? "—"
            print("    " + pad(m.dtype.uppercased(), 7) + pad(String(format: "%.2f %@", m.tflops, m.unit), 16) + rel)
        }
        print("")
    }

    if !r.bandwidth.isEmpty {
        let bufMB = (r.bandwidth.first?.bytesPerRep ?? 0) / 1024 / 1024
        print("[3] Memory bandwidth — \(bufMB) MB buffer, incompressible data")
        for m in r.bandwidth {
            let name: String
            switch m.name {
            case "gpu_read": name = "GPU read (load only)"
            case "gpu_write": name = "GPU write (store only)"
            default: name = "GPU copy (blit DMA, r+w)"
            }
            let spread = m.gbpsMin > 0 && m.gbpsMin < m.gbps
                ? String(format: "  (9 samples: %.1f–%.1f)", m.gbpsMin, m.gbps)
                : ""
            print("    " + pad(name, 30) + String(format: "%6.1f GB/s", m.gbps) + spread)
        }
        print("")
    }

    print("Note: only simdgroup_matrix (Matrix rows) reaches ~2.2 TFLOPS on this GPU;")
    print("      plain shader FMA tops out at about half of that. INT8/INT4 have no")
    print("      tensor path in Metal, so quantization saves bandwidth, not FLOPs.")
    print("")
    printRecommended(r)
}

/// 最后一块：把「评估 prefill / decode 该用哪几个数」一次性列清楚，带推荐标记。
func printRecommended(_ r: BenchResult) {
    let fp32Mma = r.compute.first { $0.dtype == "fp32" && $0.path == "mma" }
    let fp16Mma = r.compute.first { $0.dtype == "fp16" && $0.path == "mma" }
    let fp32Fma = r.compute.first { $0.dtype == "fp32" && $0.path == "fma" }
    let fp16Fma = r.compute.first { $0.dtype == "fp16" && $0.path == "fma" }
    let read = r.bandwidth.first { $0.name == "gpu_read" }
    let write = r.bandwidth.first { $0.name == "gpu_write" }
    let copy = r.bandwidth.first { $0.name == "gpu_copy" }
    let i8 = r.gemm.first { $0.dtype == "int8" }
    let i4 = r.gemm.first { $0.dtype == "int4" }

    print("[4] 推荐数据 / Recommended — 跨设备对比只用这一节（每轮 9 次采样取中位数）")
    print("    " + pad("用途", 16) + pad("指标", 26) + pad("数值", 15) + "怎么用")
    func row(_ use: String, _ name: String, _ value: String, _ hint: String) {
        print("    " + pad(use, 16) + pad(name, 26) + pad(value, 15) + hint)
    }
    if let m = fp16Mma {
        row("★ prefill", "FP16 Matrix", String(format: "%.2f TFLOPS", m.valueMedian),
            "模型算力上限（MLX / MPS 走这条）")
    }
    if let m = fp32Mma {
        row("  prefill", "FP32 Matrix", String(format: "%.2f TFLOPS", m.valueMedian),
            "只有当模型是 FP32 权重时才用它")
    }
    if let m = fp16Fma {
        row("  prefill", "FP16 Scalar FMA", String(format: "%.2f TFLOPS", m.valueMedian),
            "自己写的内核没调 simdgroup_matrix 时用它")
    }
    if let m = fp32Fma {
        row("  prefill", "FP32 Scalar FMA", String(format: "%.2f TFLOPS", m.valueMedian),
            "同上，FP32 版本")
    }
    let i8Alu = r.compute.first { $0.dtype == "int8" && $0.path == "alu" }
    if let m = i8Alu {
        let ratio = i8.map { String(format: "；GEMM 跑到它的 %.0f%%", $0.tflopsMedian / m.valueMedian * 100) } ?? ""
        row("  prefill", "INT8 Integer ALU", String(format: "%.2f TOPS", m.valueMedian),
            "纯整型算力天花板" + ratio)
    }
    if let m = i8 {
        row("  prefill", "INT8 GEMM", String(format: "%.2f TOPS", m.tflopsMedian),
            "量化不提算力：prefill 仍按 FP16 算")
    }
    if let m = i4 {
        row("  prefill", "INT4 GEMM", String(format: "%.2f TOPS", m.tflopsMedian),
            "同上；4-bit 只影响 decode 的权重字节数")
    }
    if let m = read {
        row("★ decode", "GPU read", String(format: "%.1f GB/s", m.gbpsMedian),
            "decode 几乎只受它限制（权重逐 token 读一遍）")
    }
    if let m = write {
        row("  decode", "GPU write", String(format: "%.1f GB/s", m.gbpsMedian),
            "写 KV cache / 输出，通常不是瓶颈")
    }
    if let m = copy {
        row("  transfer", "GPU copy (blit)", String(format: "%.1f GB/s", m.gbpsMedian),
            "加载权重、搬运张量时参考")
    }
    print("")
    let checks = r.selfcheck.checks
    let okCount = checks.filter { $0.passed }.count
    let checkPart = r.selfcheck.skipped ? "自检 跳过" : "自检 \(okCount)/\(checks.count) \(r.selfcheck.passed ? "PASS" : "FAIL")"
    let satPart = r.bandwidthSaturation.map { $0.saturated ? "带宽已饱和" : "带宽未饱和" } ?? "带宽未测"
    let load = r.host.loadAvg1m.map { String(format: "%.2f", $0) } ?? "—"
    print("    环境：热状态 \(r.host.thermalState) · 电源 \(r.host.powerSource) · 负载 \(load) · \(checkPart) · \(satPart)")
    print("    本工具只输出实测值：不做规格对照、不假设效率系数。")
}

// MARK: - 并发锁

/// 并发跑两个 benchmark 会互相抢 GPU/带宽（实测读带宽从 55 掉到 24 GB/s）。
/// 用 pid 锁挡住第二个实例；pid 已死则视为陈旧锁直接覆盖。
func acquireLockOrFail() -> String {
    let home = ProcessInfo.processInfo.environment["MACGPU_HOME"]
        ?? FileManager.default.currentDirectoryPath
    let path = "\(home)/output/.run.lock"
    try? FileManager.default.createDirectory(atPath: "\(home)/output",
                                             withIntermediateDirectories: true)
    if let text = try? String(contentsOfFile: path, encoding: .utf8),
       let other = Int32(text.trimmingCharacters(in: .whitespacesAndNewlines)),
       other != getpid(), kill(other, 0) == 0 {
        fail("已有另一个 macgpu 进程在跑测量（pid \(other)）。并发测量会互相污染，请等它结束。")
    }
    try? "\(getpid())".write(toFile: path, atomically: true, encoding: .utf8)
    return path
}

// MARK: - 主流程

var args = Array(CommandLine.arguments.dropFirst())
// 唯一入口：不带参数（或直接跟 --选项）就跑全套，等价于 run --repeat 3
let meta = ["--help", "-h", "help", "--version", "version"]
if args.isEmpty {
    args = ["run", "--repeat", "3"]
} else if args[0].hasPrefix("--"), !meta.contains(args[0]) {
    // 单入口直接跟参数：用户给了 --repeat 就用用户的，否则默认 3 轮
    var opts = args
    if !opts.contains("--repeat") { opts = ["--repeat", "3"] + opts }
    args = ["run"] + opts
}
guard let command = args.first else { print(usage()); exit(0) }

switch command {
case "info":
    let gpu: GPU
    do { gpu = try GPU() } catch { fail("\(error)") }
    let host = SysInfo.host()
    let info = GPUInfo(
        name: gpu.device.name,
        coreCount: SysInfo.gpuCoreCount(),
        unifiedMemory: gpu.device.hasUnifiedMemory,
        maxThreadsPerThreadgroup: gpu.device.maxThreadsPerThreadgroup.width,
        recommendedWorkingSetMB: (Double(gpu.device.recommendedMaxWorkingSetSize) / 1024 / 1024).rounded(),
        registryID: gpu.device.registryID
    )
    let payload: [String: Any] = [
        "host": ["chip": host.chip, "model": host.model, "macOS": host.macOS,
                 "os_build": host.osBuild, "arch": host.arch,
                 "cpu_cores": host.cpuCores, "p_cores": host.pCores ?? 0, "e_cores": host.eCores ?? 0,
                 "memory_gb": host.memoryGB, "thermal_state": host.thermalState,
                 "low_power_mode": host.lowPowerMode, "power_source": host.powerSource],
        "gpu": ["name": info.name, "core_count": info.coreCount ?? 0, "unified_memory": info.unifiedMemory,
                "max_threads_per_threadgroup": info.maxThreadsPerThreadgroup,
                "recommended_working_set_mb": info.recommendedWorkingSetMB]
    ]
    let data = try! JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
    if args.contains("--json") {
        print(String(data: data, encoding: .utf8)!)
        exit(0)
    }
    print("\(pad("芯片", 12))\(host.chip)  (\(host.model))")
    print("\(pad("CPU", 12))\(host.cpuCores) 核 (P\(host.pCores ?? 0)+E\(host.eCores ?? 0))")
    print("\(pad("内存", 12))\(host.memoryGB) GB 统一内存")
    print("\(pad("GPU", 12))\(info.name)  \(info.coreCount.map { "\($0) 核" } ?? "未知")")
    print("\(pad("工作集上限", 12))\(Int(info.recommendedWorkingSetMB)) MB")
    print("\(pad("系统", 12))macOS \(host.macOS) (\(host.osBuild))  \(host.arch)")
    print("\(pad("电源", 12))\(host.powerSource)  低电量=\(host.lowPowerMode)  热状态=\(host.thermalState)")

case "run":
    var options = parseOptions(Array(args.dropFirst()))
    if options.json { options.quiet = true }   // --json 时只输出 JSON，便于程序解析
    let gpu: GPU
    do { gpu = try GPU() } catch { fail("\(error)") }

    let lock = acquireLockOrFail()
    defer { try? FileManager.default.removeItem(atPath: lock) }

    let host = SysInfo.host()
    let git = SysInfo.gitInfo()

    // 自检先行：数值不对，后面的数字都不能用来做结论
    var checkReport = SelfCheckReport(passed: true, skipped: true, checks: [])
    if options.selfcheck {
        let items = SelfCheck(gpu: gpu).runAll()
        checkReport = SelfCheckReport(passed: items.allSatisfy { $0.passed },
                                      skipped: false, checks: items)
        if !options.quiet {
            let okCount = items.filter { $0.passed }.count
            print("[0/3] selfcheck \(checkReport.passed ? "PASS" : "FAIL") (\(okCount)/\(items.count))")
            for item in items where !item.passed {
                print("      ✗ \(item.name): \(item.detail)")
            }
        }
    }

    // GEMM 需要 A/B/C 三块 n² 缓冲，超出工作集会被系统换页，测出来没有意义
    let gemmBytes = Double(options.gemmSize) * Double(options.gemmSize) * 16
    let workingSet = Double(gpu.device.recommendedMaxWorkingSetSize)
    if gemmBytes > workingSet / 2 {
        fail("--gemm-size \(options.gemmSize) 约需 \(Int(gemmBytes / 1024 / 1024)) MB，"
             + "超过建议工作集的一半（\(Int(workingSet / 1024 / 1024)) MB）")
    }

    var result = BenchResult(
        schema: "macgpu/v2",
        version: version,
        runs: options.repeatCount,
        sampling: SamplingInfo(rounds: options.repeatCount, samplesPerMetric: 9,
                               valueEstimator: "min", medianEstimator: "median"),
        generatedAt: isoFormatter.string(from: Date()),
        host: host,
        gpu: GPUInfo(
            name: gpu.device.name,
            coreCount: SysInfo.gpuCoreCount(),
            unifiedMemory: gpu.device.hasUnifiedMemory,
                maxThreadsPerThreadgroup: gpu.device.maxThreadsPerThreadgroup.width,
            recommendedWorkingSetMB: (Double(gpu.device.recommendedMaxWorkingSetSize) / 1024 / 1024).rounded(),
            registryID: gpu.device.registryID
        ),
        tool: ToolInfo(version: version, gitSha: git.sha, gitDirty: git.dirty),
        selfcheck: checkReport,
        compute: [],
        gemm: [],
        bandwidth: [],
        notes: [
            "峰值算力为寄存器内压测，代表该精度在 Metal 上可达到的算力上限。",
            "矩阵指令（simdgroup_matrix）是 Apple GPU 的专用通路；普通着色器 FMA 明显低于它，具体比例以本机实测为准（不同代 GPU 不同）。",
            "INT8/INT4 在 Metal 上没有公开的 tensor 指令，测量的是 ALU 模拟路径的真实吞吐。",
            "带宽测试使用 \(options.sizeMB) MB 私有缓冲区、\(options.bwThreads ?? BandwidthBench.defaultThreads) 线程；拷贝走 blit DMA 路径。",
            "每个指标采样 9 次：value / gbps / tflops 为最快一次（峰值），*_median 为中位数；--repeat 多轮时轮间仍取峰值。",
            "跨设备比较请用 *_median，并确认 selfcheck.passed == true、thermal_state == nominal、low_power_mode == 0。"
        ]
    )

    do {
        var computeBest: [String: ComputeMetric] = [:]
        var gemmBest: [String: GemmMetric] = [:]
        var bwBest: [String: BandwidthMetric] = [:]
        let cases = ComputeCases.selected(Set(options.dtypes))

        // 带宽线程按机器自动选：先用默认值，再用 2× 复测，未饱和就自动切过去
        var bwThreads = options.bwThreads ?? BandwidthBench.defaultThreads
        if options.tests.contains("bandwidth") {
            let picked = try BandwidthBench.choose(gpu: gpu, sizeMB: options.sizeMB,
                                                   preferred: bwThreads,
                                                   force: options.bwThreads != nil)
            bwThreads = picked.threads
            result.bandwidthSaturation = picked.saturation
            if !options.quiet {
                let s = picked.saturation
                let note = s.saturated ? "已饱和" : "未饱和，已自动提高"
                print("[0/3] 带宽线程 \(s.primaryThreads) → \(picked.threads)（\(note)）")
            }
        }

        for round in 0..<options.repeatCount {
            if options.repeatCount > 1 && !options.quiet {
                print("-- 第 \(round + 1)/\(options.repeatCount) 轮")
            }
            if options.tests.contains("compute") {
                let bench = ComputeBench(gpu: gpu)
                for c in cases {
                    let metric = try bench.run(c)
                    let key = metric.dtype + "/" + metric.path
                    if computeBest[key] == nil || metric.value > computeBest[key]!.value {
                        computeBest[key] = metric
                    }
                    if !options.quiet {
                        print("  compute \(pad(c.dtype + " " + c.path, 12)) \(String(format: "%.2f %@", metric.value, metric.unit))")
                    }
                }
            }
            if options.tests.contains("gemm") {
                let bench = GemmBench(gpu: gpu)
                for dtype in GemmDType.allCases where options.dtypes.contains(dtype.rawValue) {
                    let metric = try bench.run(dtype, size: options.gemmSize)
                    if gemmBest[metric.dtype] == nil || metric.tflops > gemmBest[metric.dtype]!.tflops {
                        gemmBest[metric.dtype] = metric
                    }
                    if !options.quiet { print("  gemm    \(pad(dtype.rawValue, 12)) \(String(format: "%.2f %@", metric.tflops, metric.unit))") }
                }
            }
            if options.tests.contains("bandwidth") {
                let bench = BandwidthBench(gpu: gpu)
                for metric in try bench.run(sizeMB: options.sizeMB, threads: bwThreads) {
                    if bwBest[metric.name] == nil || metric.gbps > bwBest[metric.name]!.gbps {
                        bwBest[metric.name] = metric
                    }
                    if !options.quiet { print("  bw      \(pad(metric.name, 12)) \(String(format: "%.1f GB/s", metric.gbps))") }
                }
            }
        }

        result.compute = cases.compactMap { computeBest[$0.dtype + "/" + $0.path] }
        result.gemm = GemmDType.allCases.map { $0.rawValue }.compactMap { gemmBest[$0] }
        result.bandwidth = ["gpu_read", "gpu_write", "gpu_copy"].compactMap { bwBest[$0] }
        if let sat = result.bandwidthSaturation, !sat.saturated {
            result.notes.append("带宽线程已自动提高到 \(sat.altThreads)（比 \(sat.primaryThreads) 快 "
                                + "\(String(format: "%.1f", max(sat.readGainPct, sat.writeGainPct, sat.copyGainPct)))%）；"
                                + "跨设备比较请确认对方也是同一线程数。")
        }
        if options.repeatCount > 1 {
            result.notes.append("重复 \(options.repeatCount) 轮，各指标取最优，以降低后台负载带来的抖动。")
        }
        if result.host.lowPowerMode == "1" {
            result.notes.append("警告：测试时低电量模式开启，GPU 频率受限，绝对值偏低。")
        }
        if result.host.powerSource == "battery" {
            result.notes.append("警告：测试时使用电池供电，可能触发功耗限制。")
        }
        if result.host.thermalState != "nominal" {
            result.notes.append("警告：测试时热状态为 \(result.host.thermalState)，可能已降频。")
        }
    } catch {
        fail("\(error)")
    }

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    guard let jsonData = try? encoder.encode(result) else { fail("JSON 编码失败") }

    let home = ProcessInfo.processInfo.environment["MACGPU_HOME"] ?? FileManager.default.currentDirectoryPath
    let stamp: String = {
        let f = DateFormatter()
        f.dateFormat = "yyyyMMdd-HHmmss-SSS"
        return f.string(from: Date())
    }()
    let outPath = options.out ?? "\(home)/output/results/\(stamp)/result.json"
    let url = URL(fileURLWithPath: outPath)
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    do { try jsonData.write(to: url) } catch { fail("写入 \(outPath) 失败: \(error)") }
    // output/results/latest -> 最新一次**全套**运行，方便脚本和文档引用稳定路径。
    // 只跑 compute/gemm 的部分测试不更新 latest，避免报告从残缺结果生成。
    let dir = url.deletingLastPathComponent()
    let isFullRun = options.tests == Set(["compute", "gemm", "bandwidth"])
    if isFullRun, dir.path.hasPrefix("\(home)/output/results/"), dir.lastPathComponent != "latest" {
        let link = URL(fileURLWithPath: "\(home)/output/results/latest")
        try? FileManager.default.removeItem(at: link)
        try? FileManager.default.createSymbolicLink(at: link, withDestinationURL: dir)
    } else if !isFullRun, !options.quiet {
        print("（部分测试，未更新 latest）")
    }

    if options.json {
        print(String(data: jsonData, encoding: .utf8)!)
    } else if !options.quiet {
        print("")
        printSummary(result)
        print("结果已保存: \(outPath)")
    }

    if !checkReport.passed {
        FileHandle.standardError.write(
            "selfcheck 失败：本次数据不可用于对比（已写入 \(outPath) 供排查）\n".data(using: .utf8)!)
        exit(2)
    }

case "selfcheck":
    let lock = acquireLockOrFail()
    defer { try? FileManager.default.removeItem(atPath: lock) }
    let gpu: GPU
    do { gpu = try GPU() } catch { fail("\(error)") }
    print("Numeric self-check — GPU output vs CPU reference")
    let items = SelfCheck(gpu: gpu).runAll()
    for item in items {
        print("  \(pad(item.name, 24))\(pad(item.detail, 22))\(item.passed ? "通过" : "失败")")
    }
    let ok = items.allSatisfy { $0.passed }
    print(ok ? "PASS — all checks passed" : "FAIL — see above")
    exit(ok ? 0 : 2)

case "--help", "-h", "help":
    print(usage())

case "--version", "version":
    print(version)

default:
    print(usage())
    exit(1)
}
