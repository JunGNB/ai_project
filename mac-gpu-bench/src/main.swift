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

let version = "2.0.0"

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
      macgpu info                       machine / GPU info
      macgpu run [options]              run all tests and save JSON
      macgpu selfcheck                  numeric self-check (GPU vs CPU reference)

    Options:
      --tests compute,gemm,bandwidth    pick test groups (default: all)
      --dtypes fp32,fp16,int8           pick precisions (default: all)
      --gemm-size N                     GEMM edge length, multiple of 64 (default 2048)
      --size-mb N                       bandwidth buffer size in MB (default 256)
      --repeat N                        run N rounds, keep the best of each (suggest 3)
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
    var out: String?
    var json = false
    var quiet = false
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
            o.tests = Set(nextValue().split(separator: ",").map { String($0).trimmingCharacters(in: .whitespaces) })
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
        case "--out":
            o.out = nextValue()
        case "--repeat":
            guard let v = Int(nextValue()), v >= 1, v <= 20 else { fail("--repeat 需在 1~20 之间") }
            o.repeatCount = v
        case "--json":
            o.json = true
        case "--quiet":
            o.quiet = true
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
    print("  \(r.host.chip) · \(cores) · \(String(format: "%.1f", r.host.memoryGB)) GB unified memory · macOS \(r.host.macOS)\(load)")
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
        print("    INT8 / INT4: no matrix path in Metal, so there is no honest ALU peak —")
        print("                 they are measured as real int8/int4 GEMM, see [2].")
        print("")
    }

    if !r.gemm.isEmpty {
        print("[2] Real GEMM — hand-written, no simdgroup_matrix (N\(r.gemm[0].size)³)")
        print("    " + pad("dtype", 7) + pad("throughput", 16) + "vs ALU peak (same dtype)")
        // GEMM 走的是普通 ALU 通路，和矩阵指令峰值比没有意义；
        // 同通路的 fma 峰值才是它的天花板。
        var peaks: [String: Double] = [:]
        for m in r.compute where m.path == "fma" { peaks[m.dtype] = m.value }
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

    print("[4] 推荐数据 / Recommended — 评估 prefill 与 decode 直接用这几行")
    print("    " + pad("用途", 16) + pad("指标", 26) + pad("数值", 15) + "怎么用")
    func row(_ use: String, _ name: String, _ value: String, _ hint: String) {
        print("    " + pad(use, 16) + pad(name, 26) + pad(value, 15) + hint)
    }
    if let m = fp16Mma {
        row("★ prefill", "FP16 Matrix peak", String(format: "%.2f TFLOPS", m.value),
            "模型算力上限（MLX / MPS 走这条）")
    }
    if let m = fp32Mma {
        row("  prefill", "FP32 Matrix peak", String(format: "%.2f TFLOPS", m.value),
            "只有当模型是 FP32 权重时才用它")
    }
    if let m = fp16Fma {
        row("  prefill", "FP16 Scalar FMA", String(format: "%.2f TFLOPS", m.value),
            "自己写的内核没调 simdgroup_matrix 时用它")
    }
    if let m = fp32Fma {
        row("  prefill", "FP32 Scalar FMA", String(format: "%.2f TFLOPS", m.value),
            "同上，FP32 版本")
    }
    if let m = i8 {
        row("  prefill", "INT8 GEMM", String(format: "%.2f TOPS", m.tflops),
            "量化不提算力：prefill 仍按 FP16 算")
    }
    if let m = i4 {
        row("  prefill", "INT4 GEMM", String(format: "%.2f TOPS", m.tflops),
            "同上；4-bit 只影响 decode 的权重字节数")
    }
    if let m = read {
        row("★ decode", "GPU read", String(format: "%.1f GB/s", m.gbps),
            "decode 几乎只受它限制（权重逐 token 读一遍）")
    }
    if let m = write {
        row("  decode", "GPU write", String(format: "%.1f GB/s", m.gbps),
            "写 KV cache / 输出，通常不是瓶颈")
    }
    if let m = copy {
        row("  transfer", "GPU copy (blit)", String(format: "%.1f GB/s", m.gbps),
            "加载权重、搬运张量时参考")
    }
    print("")
    print("    估算式（eff 为效率：手写内核 0.3~0.6，MLX/MPS 实测 0.8；带宽 eff 0.9~1.0）：")
    print("      prefill tok/s = 算力(MAC/s) × eff ÷ 参数量")
    print("      decode  tok/s = 读带宽(GB/s) × eff ÷ (权重字节 + KV字节)")
    print("      本机用 oMLX 实测校核过：MiniCPM5-2B 78%/92%，Qwen3.5-9B 83%/111%（见 docs/records/51）")

    guard let peak = fp16Mma, let bw = read else { return }
    let macs = peak.macsPerSec
    let prefill = macs * 0.4 / 7.0e9
    let bwEff = bw.gbps * 0.9
    print("")
    print(String(format: "    本机实测值代入（7B、prefill eff=40%%、decode eff=90%%、KV 忽略）："))
    print(String(format: "      FP16: prefill ≈ %5.1f tok/s   decode ≈ %5.1f tok/s", prefill, bwEff / 14.0))
    print(String(format: "      INT8: prefill ≈ %5.1f tok/s   decode ≈ %5.1f tok/s", prefill, bwEff / 7.0))
    print(String(format: "      INT4: prefill ≈ %5.1f tok/s   decode ≈ %5.1f tok/s", prefill, bwEff / 3.5))
}

// MARK: - 主流程

let args = Array(CommandLine.arguments.dropFirst())
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
        "host": ["chip": host.chip, "model": host.model, "macOS": host.macOS, "arch": host.arch,
                 "cpu_cores": host.cpuCores, "p_cores": host.pCores ?? 0, "e_cores": host.eCores ?? 0,
                 "memory_gb": host.memoryGB],
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
    print("\(pad("系统", 12))macOS \(host.macOS)  \(host.arch)")

case "run":
    var options = parseOptions(Array(args.dropFirst()))
    if options.json { options.quiet = true }   // --json 时只输出 JSON，便于程序解析
    let gpu: GPU
    do { gpu = try GPU() } catch { fail("\(error)") }

    var result = BenchResult(
        schema: "macgpu/v1",
        version: version,
        runs: options.repeatCount,
        generatedAt: isoFormatter.string(from: Date()),
        host: SysInfo.host(),
        gpu: GPUInfo(
            name: gpu.device.name,
            coreCount: SysInfo.gpuCoreCount(),
            unifiedMemory: gpu.device.hasUnifiedMemory,
                maxThreadsPerThreadgroup: gpu.device.maxThreadsPerThreadgroup.width,
            recommendedWorkingSetMB: (Double(gpu.device.recommendedMaxWorkingSetSize) / 1024 / 1024).rounded(),
            registryID: gpu.device.registryID
        ),
        compute: [],
        gemm: [],
        bandwidth: [],
        notes: [
            "峰值算力为寄存器内 ALU 压测，代表该精度在 Metal 上可达到的算力上限。",
            "Apple GPU 的 ALU 指令是 4-wide（float4/half4），标量代码只能达到约 1/4 峰值。",
            "INT8/INT4 在 Metal 上没有公开的 tensor 指令，测量的是 ALU 模拟路径的真实吞吐。",
            "带宽测试使用 \(options.sizeMB) MB 私有缓冲区；拷贝走 blit DMA 路径。",
            "所有数值为 GPU 侧命令耗时最小值（重复 5 次），已排除主机侧调度抖动。"
        ]
    )

    do {
        var computeBest: [String: ComputeMetric] = [:]
        var gemmBest: [String: GemmMetric] = [:]
        var bwBest: [String: BandwidthMetric] = [:]
        let cases = ComputeCases.selected(Set(options.dtypes))

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
                for metric in try bench.run(sizeMB: options.sizeMB) {
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
        if options.repeatCount > 1 {
            result.notes.append("重复 \(options.repeatCount) 轮，各指标取最优，以降低后台负载带来的抖动。")
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
        f.dateFormat = "yyyyMMdd-HHmmss"
        return f.string(from: Date())
    }()
    let outPath = options.out ?? "\(home)/output/results/\(stamp)/result.json"
    let url = URL(fileURLWithPath: outPath)
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    do { try jsonData.write(to: url) } catch { fail("写入 \(outPath) 失败: \(error)") }
    // output/results/latest -> 最新一次运行，方便脚本和文档引用稳定路径
    let dir = url.deletingLastPathComponent()
    if dir.path.hasPrefix("\(home)/output/results/"), dir.lastPathComponent != "latest" {
        let link = URL(fileURLWithPath: "\(home)/output/results/latest")
        try? FileManager.default.removeItem(at: link)
        try? FileManager.default.createSymbolicLink(at: link, withDestinationURL: dir)
    }

    if options.json {
        print(String(data: jsonData, encoding: .utf8)!)
    } else if !options.quiet {
        print("")
        printSummary(result)
        print("结果已保存: \(outPath)")
    }

case "selfcheck":
    let gpu: GPU
    do { gpu = try GPU() } catch { fail("\(error)") }
    var ok = true
    let check = SelfCheck(gpu: gpu)
    print("Numeric self-check — GPU output vs CPU reference")

    for (name, result) in [("int8", try! check.gemmI8Error(n: 128)), ("int4", try! check.gemmI4Error(n: 128))] {
        let (err, passed) = result
        ok = ok && passed
        print("  \(pad(name + " GEMM 128³", 20))最大误差 \(String(format: "%.0f", err))  \(passed ? "通过" : "失败")")
    }

    for (dtype, tolerance) in [(GemmDType.fp32, 1e-3), (.fp16, 2e-1)] {
        let (err, passed) = try! check.gemmError(dtype: dtype, n: 128, tolerance: tolerance)
        ok = ok && passed
        print("  \(pad(dtype.rawValue + " GEMM 128³", 20))最大误差 \(String(format: "%.2e", err))  \(passed ? "通过" : "失败")")
    }
    for c in ComputeCases.all {
        let (err, passed) = try! check.peakCheck(c)
        ok = ok && passed
        print("  \(pad(c.dtype + " " + c.path + " 数值", 22))相对误差 \(String(format: "%.1e", err))  \(passed ? "通过" : "失败")")
    }
    let band = try! BandwidthBench(gpu: gpu).run(sizeMB: 64)
    for m in band {
        let passed = m.gbps > 5 && m.gbps < 5000
        ok = ok && passed
        print("  \(pad(m.name, 20))\(String(format: "%.1f GB/s", m.gbps))  \(passed ? "通过" : "失败")")
    }
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
