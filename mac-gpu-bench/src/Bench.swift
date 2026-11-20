import Foundation
import Metal
import Darwin

// MARK: - 数据模型（直接序列化为 JSON）

struct HostInfo: Codable {
    var chip: String
    var model: String
    var macOS: String
    /// 系统 build（如 26A434）。Metal 着色器是运行时编译的，编译器随 OS 变，
    /// 因此跨设备/跨时间比较必须能确认 OS build 一致。
    var osBuild: String
    var arch: String
    var cpuCores: Int
    var pCores: Int?
    var eCores: Int?
    var memoryGB: Double
    var uptimeHours: Double?
    var loadAvg1m: Double?
    /// nominal / fair / serious / critical / unknown（ProcessInfo.thermalState）
    var thermalState: String
    /// "0" / "1" / "unknown"（pmset -g；部分机型不暴露该键）
    var lowPowerMode: String
    /// "ac" / "battery" / "unknown"
    var powerSource: String

    enum CodingKeys: String, CodingKey {
        case chip, model, macOS, arch
        case osBuild = "os_build"
        case cpuCores = "cpu_cores"
        case pCores = "p_cores"
        case eCores = "e_cores"
        case memoryGB = "memory_gb"
        case uptimeHours = "uptime_hours"
        case loadAvg1m = "load_avg_1m"
        case thermalState = "thermal_state"
        case lowPowerMode = "low_power_mode"
        case powerSource = "power_source"
    }
}

/// 采样口径：每个指标 9 次（min 为峰值，median 为稳定值），
/// `--repeat` 轮在轮间仍取峰值，但每轮都保留自己的 samples。
struct SamplingInfo: Codable {
    var rounds: Int
    var samplesPerMetric: Int
    var valueEstimator: String
    var medianEstimator: String

    enum CodingKeys: String, CodingKey {
        case rounds
        case samplesPerMetric = "samples_per_metric"
        case valueEstimator = "value_estimator"
        case medianEstimator = "median_estimator"
    }
}

/// 工具自身的可归因信息：同一台机器换一版代码，数字不可直接比。
struct ToolInfo: Codable {
    var version: String
    var gitSha: String
    var gitDirty: Bool

    enum CodingKeys: String, CodingKey {
        case version
        case gitSha = "git_sha"
        case gitDirty = "git_dirty"
    }
}

struct SelfCheckItem: Codable {
    var name: String
    var passed: Bool
    var detail: String
}

struct SelfCheckReport: Codable {
    var passed: Bool
    var skipped: Bool
    var checks: [SelfCheckItem]
}

struct GPUInfo: Codable {
    var name: String
    var coreCount: Int?
    var unifiedMemory: Bool
    var maxThreadsPerThreadgroup: Int
    var recommendedWorkingSetMB: Double
    var registryID: UInt64

    enum CodingKeys: String, CodingKey {
        case name
        case coreCount = "core_count"
        case unifiedMemory = "unified_memory"
        case maxThreadsPerThreadgroup = "max_threads_per_threadgroup"
        case recommendedWorkingSetMB = "recommended_working_set_mb"
        case registryID = "registry_id"
    }
}

/// 一项算力测量。value 的单位由 unit 决定（TFLOPS / TOPS），
/// macs_per_sec 是跨精度可比较的归一化指标（每秒多少次乘加）。
struct ComputeMetric: Codable {
    var dtype: String
    /// 计算通路：mma（矩阵指令）/ fma（普通着色器 FMA）
    var path: String
    var label: String
    /// 英文描述，供英文报告 / 跨团队沟通
    var labelEn: String
    /// 峰值：9 次采样中最快一次的吞吐
    var value: Double
    /// 稳定值：9 次采样的中位数吞吐（跨设备比较用这个）
    var valueMedian: Double
    var unit: String
    var macsPerSec: Double
    var macsPerSecMedian: Double
    var threads: Int
    /// 计数单元数（mma 为 warp 数，fma 为线程数），复算 MACs 必需
    var units: Int
    var unitThreads: Int
    var macsPerIteration: Double
    var totalMacs: Double
    var iterations: Int
    var dispatches: Int
    var seconds: Double
    var secondsMedian: Double
    /// 9 次采样的原始秒数（min = seconds，排序后取中位 = secondsMedian）
    var samples: [Double]
    var method: String
    var methodEn: String

    enum CodingKeys: String, CodingKey {
        case dtype, path, label
        case labelEn = "label_en"
        case value, unit, threads, units, iterations, dispatches, seconds, samples, method
        case valueMedian = "value_median"
        case macsPerSec = "macs_per_sec"
        case macsPerSecMedian = "macs_per_sec_median"
        case unitThreads = "unit_threads"
        case macsPerIteration = "macs_per_iteration"
        case totalMacs = "total_macs"
        case secondsMedian = "seconds_median"
        case methodEn = "method_en"
    }
}

/// 矩阵乘用的精度（只测 fp32 / fp16 两条）
enum GemmDType: String, CaseIterable {
    case fp32, fp16, int8, int4
    var kernel: String {
        switch self {
        case .fp32: return "gemm_f32"
        case .fp16: return "gemm_f16"
        case .int8: return "gemm_i8"
        case .int4: return "gemm_i4"
        }
    }
    /// 打包后每 N×N 矩阵占多少字节
    func matrixBytes(_ n: Int) -> Int {
        switch self {
        case .fp32: return n * n * 4
        case .fp16: return n * n * 2
        case .int8: return n * n
        case .int4: return n * n / 2      // 每字节 2 个 4-bit 值
        }
    }
    /// 输出元素字节数（整型累加到 int32）
    var outputBytes: Int { self == .int8 || self == .int4 ? 4 : matrixBytes(1) }
    var unit: String { self == .int8 || self == .int4 ? "TOPS" : "TFLOPS" }
}

struct GemmMetric: Codable {
    var dtype: String
    var size: Int
    /// 峰值：fp32/fp16 是 TFLOPS，int8/int4 是 TOPS
    var tflops: Double
    var tflopsMedian: Double
    var unit: String
    var seconds: Double
    var secondsMedian: Double
    var samples: [Double]
    var method: String

    enum CodingKeys: String, CodingKey {
        case dtype, size, tflops, unit, seconds, samples, method
        case tflopsMedian = "tflops_median"
        case secondsMedian = "seconds_median"
    }
}

struct BandwidthMetric: Codable {
    var name: String
    /// 9 次采样中的最优值（= gbps_max，字段保留是为了兼容旧结果）
    var gbps: Double
    /// 9 次采样中的最差值，用于判断这次读数稳不稳
    var gbpsMin: Double
    var gbpsMax: Double
    /// 9 次采样的中位数（跨设备比较用这个；`gbps` 是峰值）
    var gbpsMedian: Double
    var seconds: Double
    var secondsMedian: Double
    var bytesPerRep: Int
    var reps: Int
    var threads: Int
    var samples: [Double]
    var method: String

    enum CodingKeys: String, CodingKey {
        case name, gbps, seconds, reps, threads, samples, method
        case gbpsMin = "gbps_min"
        case gbpsMax = "gbps_max"
        case gbpsMedian = "gbps_median"
        case secondsMedian = "seconds_median"
        case bytesPerRep = "bytes_per_rep"
    }
}

/// 带宽线程数是否已经饱和：用 2× 线程复测一遍，若明显更快，说明默认线程数喂不饱这台 GPU。
struct BandwidthSaturation: Codable {
    var primaryThreads: Int
    var altThreads: Int
    var readGainPct: Double
    var writeGainPct: Double
    var copyGainPct: Double
    var saturated: Bool

    enum CodingKeys: String, CodingKey {
        case saturated
        case primaryThreads = "primary_threads"
        case altThreads = "alt_threads"
        case readGainPct = "read_gain_pct"
        case writeGainPct = "write_gain_pct"
        case copyGainPct = "copy_gain_pct"
    }
}

struct BenchResult: Codable {
    var schema: String
    var version: String
    var runs: Int
    var sampling: SamplingInfo
    var generatedAt: String
    var host: HostInfo
    var gpu: GPUInfo
    var tool: ToolInfo
    var selfcheck: SelfCheckReport
    var compute: [ComputeMetric]
    var gemm: [GemmMetric]
    var bandwidth: [BandwidthMetric]
    var bandwidthSaturation: BandwidthSaturation?
    var notes: [String]

    enum CodingKeys: String, CodingKey {
        case schema, version, runs, sampling, host, gpu, tool, selfcheck
        case compute, gemm, bandwidth, notes
        case bandwidthSaturation = "bandwidth_saturation"
        case generatedAt = "generated_at"
    }
}

// MARK: - 系统信息

enum SysInfo {
    static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buf = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return nil }
        return String(cString: buf)
    }

    static func sysctlUInt64(_ name: String) -> UInt64? {
        var value: UInt64 = 0
        var size = MemoryLayout<UInt64>.size
        guard sysctlbyname(name, &value, &size, nil, 0) == 0 else { return nil }
        return value
    }

    static func sysctlInt(_ name: String) -> Int? {
        guard let v = sysctlUInt64(name) else { return nil }
        return Int(v)
    }

    /// GPU 核心数没有公开 API，从 IORegistry 读（约 30ms）
    static func gpuCoreCount() -> Int? {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/sbin/ioreg")
        proc.arguments = ["-rc", "IOAccelerator", "-d", "1"]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = Pipe()
        do { try proc.run() } catch { return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        proc.waitUntilExit()
        guard let text = String(data: data, encoding: .utf8) else { return nil }
        guard let range = text.range(of: "gpu-core-count") else { return nil }
        let tail = text[range.upperBound...].drop { $0 == " " || $0 == "\"" || $0 == "=" }
        let digits = tail.prefix { $0.isNumber }
        return Int(digits)
    }

    /// 系统 1 分钟平均负载：用于判断测试时是否被后台任务干扰
    static func loadAverage() -> Double? {
        var load = [Double](repeating: 0, count: 3)
        guard getloadavg(&load, 3) > 0 else { return nil }
        return (load[0] * 100).rounded() / 100
    }

    /// 跑一个短命令并返回 stdout（只有系统工具，无第三方依赖）
    static func run(_ path: String, _ args: [String]) -> String? {
        guard FileManager.default.isExecutableFile(atPath: path) else { return nil }
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: path)
        proc.arguments = args
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = Pipe()
        do { try proc.run() } catch { return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        proc.waitUntilExit()
        return String(data: data, encoding: .utf8)
    }

    /// OS build：Metal 着色器运行时编译，编译器随 OS 变，跨时间比较必须对齐 build
    static func osBuild() -> String {
        sysctlString("kern.osversion") ?? "unknown"
    }

    static func thermalState() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    /// 低电量模式会显著降低 GPU 频率；部分机型/台式机不暴露该键，此时记 unknown
    static func lowPowerMode() -> String {
        for args in [["-g"], ["-g", "custom"]] {
            guard let out = run("/usr/bin/pmset", args) else { continue }
            for line in out.split(separator: "\n") where line.contains("lowpowermode") {
                if let v = line.split(separator: " ").last { return String(v) }
            }
        }
        return "unknown"
    }

    static func powerSource() -> String {
        guard let out = run("/usr/bin/pmset", ["-g", "ps"]) else { return "unknown" }
        if out.contains("AC Power") { return "ac" }
        if out.contains("Battery Power") { return "battery" }
        return "unknown"
    }

    static func gitInfo() -> (sha: String, dirty: Bool) {
        let root = ProcessInfo.processInfo.environment["MACGPU_HOME"]
            ?? FileManager.default.currentDirectoryPath
        func git(_ args: [String]) -> String? {
            run("/usr/bin/git", ["-C", root] + args)?.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        let sha = git(["rev-parse", "--short", "HEAD"]) ?? "unknown"
        let dirty = !(git(["status", "--porcelain"]) ?? "").isEmpty
        return (sha, dirty)
    }

    static func host() -> HostInfo {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        let memBytes = sysctlUInt64("hw.memsize") ?? 0
        return HostInfo(
            chip: sysctlString("machdep.cpu.brand_string") ?? "unknown",
            model: sysctlString("hw.model") ?? "unknown",
            macOS: "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)",
            osBuild: osBuild(),
            arch: sysctlString("hw.machine") ?? "unknown",
            cpuCores: sysctlInt("hw.ncpu") ?? ProcessInfo.processInfo.activeProcessorCount,
            pCores: sysctlInt("hw.perflevel0.physicalcpu"),
            eCores: sysctlInt("hw.perflevel1.physicalcpu"),
            memoryGB: (Double(memBytes) / 1024 / 1024 / 1024 * 10).rounded() / 10,
            uptimeHours: (ProcessInfo.processInfo.systemUptime / 3600 * 10).rounded() / 10,
            loadAvg1m: loadAverage(),
            thermalState: thermalState(),
            lowPowerMode: lowPowerMode(),
            powerSource: powerSource()
        )
    }
}

// MARK: - Metal 运行器

enum BenchError: Error, CustomStringConvertible {
    case noDevice
    case shaderCompile(String)
    case missingKernel(String)
    case invalidMetric(String)
    case unknownCase(String)
    case gpuFailure(String)

    var description: String {
        switch self {
        case .noDevice: return "找不到 Metal 设备"
        case .shaderCompile(let m): return "着色器编译失败: \(m)"
        case .missingKernel(let k): return "缺少内核: \(k)"
        case .invalidMetric(let m): return "测量结果异常: \(m)"
        case .unknownCase(let c): return "未注册的测量项: \(c)"
        case .gpuFailure(let m): return "GPU 执行失败: \(m)"
        }
    }
}

final class GPU {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw BenchError.noDevice }
        guard let queue = device.makeCommandQueue() else { throw BenchError.noDevice }
        let options = MTLCompileOptions()
        do {
            self.library = try device.makeLibrary(source: mslSource, options: options)
        } catch {
            throw BenchError.shaderCompile(error.localizedDescription)
        }
        self.device = device
        self.queue = queue
    }

    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        if let p = pipelines[name] { return p }
        guard let fn = library.makeFunction(name: name) else { throw BenchError.missingKernel(name) }
        let p = try device.makeComputePipelineState(function: fn)
        pipelines[name] = p
        return p
    }

    func buffer<T>(_ values: [T], shared: Bool = true) -> MTLBuffer? {
        values.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return nil }
            return device.makeBuffer(bytes: base, length: raw.count,
                                     options: shared ? .storageModeShared : .storageModePrivate)
        }
    }

    func emptyBuffer(_ bytes: Int, shared: Bool = false) -> MTLBuffer {
        device.makeBuffer(length: max(bytes, 16), options: shared ? .storageModeShared : .storageModePrivate)!
    }

    /// GPU 侧耗时的唯一入口：检查 command buffer 状态，拒绝 0 时间/NaN。
    /// 旧实现只取 gpuEnd-gpuStart 并用 max(...,1e-9) 兜底：一旦 command buffer 失败，
    /// 会算出上亿 TFLOPS，而且会被"取最快一次"当成有效样本保留。
    private func gpuSeconds(_ cb: MTLCommandBuffer) throws -> Double {
        cb.waitUntilCompleted()
        guard cb.status == .completed, cb.error == nil else {
            throw BenchError.gpuFailure(cb.error?.localizedDescription
                                        ?? "command buffer status \(cb.status.rawValue)")
        }
        let t = cb.gpuEndTime - cb.gpuStartTime
        // 只挡"0 或负值"这种显然没执行的情况；8µs 级的合法短派发不能误杀，
        // 数值是否离谱由各 benchmark 的物理上界（<1000 TFLOPS 等）兜底。
        guard t.isFinite, t > 1e-6 else {
            throw BenchError.invalidMetric("GPU 时间异常（\(t)s），可能是命令缓冲未真正执行")
        }
        return t
    }

    /// 一次 command buffer 内重复 reps 次线程网格派发，返回 GPU 侧耗时（秒）
    func timeThreads(_ pso: MTLComputePipelineState,
                     buffers: [(Int, MTLBuffer)],
                     threads: Int,
                     threadgroup: Int,
                     reps: Int) throws -> Double {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        for (index, buf) in buffers { enc.setBuffer(buf, offset: 0, index: index) }
        let grid = MTLSize(width: threads, height: 1, depth: 1)
        let tg = MTLSize(width: threadgroup, height: 1, depth: 1)
        for _ in 0..<max(reps, 1) { enc.dispatchThreads(grid, threadsPerThreadgroup: tg) }
        enc.endEncoding()
        cb.commit()
        return try gpuSeconds(cb)
    }

    /// 线程块派发（GEMM 用），返回 GPU 侧耗时（秒）
    func timeGroups(_ pso: MTLComputePipelineState,
                    buffers: [(Int, MTLBuffer)],
                    groups: (Int, Int),
                    threadgroup: (Int, Int),
                    reps: Int) throws -> Double {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        for (index, buf) in buffers { enc.setBuffer(buf, offset: 0, index: index) }
        let grid = MTLSize(width: groups.0, height: groups.1, depth: 1)
        let tg = MTLSize(width: threadgroup.0, height: threadgroup.1, depth: 1)
        for _ in 0..<max(reps, 1) { enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg) }
        enc.endEncoding()
        cb.commit()
        return try gpuSeconds(cb)
    }

    /// 预热一次丢弃，再采样 9 次，返回全部原始耗时（秒）。
    /// 保留原始样本是为了让下游能自己算中位数/CV，而不是只拿到一个 min。
    func sample(_ count: Int = 9, _ body: () throws -> Double) rethrows -> [Double] {
        _ = try body()                      // 预热：首次派发有页错误/时钟爬升，必须丢弃
        var out: [Double] = []
        out.reserveCapacity(count)
        for _ in 0..<count { out.append(try body()) }
        return out
    }
}

// MARK: - 算力峰值

/// 一项算力测量：dtype × 计算通路。
/// 关键约束：macsPerIteration 必须与内核实际执行的乘加次数一致——
/// v1.0 就是因为内核只写回向量的 .x 分量（其余被编译器当死代码删除），
/// 而计数按 4 个分量算，导致 FP32/FP16/INT8 虚高 4 倍。
struct ComputeCase {
    let dtype: String
    let path: String
    let kernel: String
    let label: String
    let labelEn: String
    /// 每个 unit 每次迭代的乘加次数（unit = warp for mma，thread 其他）
    let macsPerIteration: Double
    /// warp 数（mma）或线程数（fma/alu）
    let units: Int
    /// 一个 unit 包含多少线程
    let unitThreads: Int
    let unit: String
    let method: String
    let methodEn: String

    var threads: Int { units * unitThreads }
    var threadgroup: Int { path == "mma" ? 128 : 256 }
    var id: String { "\(dtype)_\(path)" }
}

enum ComputeCases {
    static let fmaThreads = 262_144
    static let mmaWarps = 4_096

    static let all: [ComputeCase] = [
        ComputeCase(dtype: "fp32", path: "mma", kernel: "peak_mma_f32",
                    label: "FP32 · 矩阵指令",
                    labelEn: "FP32 Matrix (simdgroup_float8x8)",
                    macsPerIteration: 2048, units: mmaWarps, unitThreads: 32, unit: "TFLOPS",
                    method: "每 warp 4 × simdgroup_float8x8，寄存器内循环，无访存",
                    methodEn: "4 × simdgroup_float8x8 per warp, registers only"),
        ComputeCase(dtype: "fp16", path: "mma", kernel: "peak_mma_f16",
                    label: "FP16 · 矩阵指令",
                    labelEn: "FP16 Matrix (simdgroup_half8x8)",
                    macsPerIteration: 2048, units: mmaWarps, unitThreads: 32, unit: "TFLOPS",
                    method: "每 warp 4 × simdgroup_half8x8，寄存器内循环，无访存",
                    methodEn: "4 × simdgroup_half8x8 per warp, registers only"),
        ComputeCase(dtype: "fp32", path: "fma", kernel: "peak_f32",
                    label: "FP32 · 普通 FMA",
                    labelEn: "FP32 Scalar FMA (shader ALU)",
                    macsPerIteration: 32, units: fmaThreads, unitThreads: 1, unit: "TFLOPS",
                    method: "8 条 float4 FMA 链（整条写回），不进矩阵单元",
                    methodEn: "8 × float4 FMA chains, full-vector store"),
        ComputeCase(dtype: "int8", path: "alu", kernel: "peak_i8",
                    label: "INT8 · 整数 ALU",
                    labelEn: "INT8 Integer ALU (int32 imad)",
                    macsPerIteration: 32, units: fmaThreads, unitThreads: 1, unit: "TOPS",
                    method: "8 条 uint4 整数乘加链（整条写回），不进矩阵单元",
                    methodEn: "8 × uint4 int32 imad chains, full-vector store"),
        ComputeCase(dtype: "fp16", path: "fma", kernel: "peak_f16",
                    label: "FP16 · 普通 FMA",
                    labelEn: "FP16 Scalar FMA (shader ALU)",
                    macsPerIteration: 32, units: fmaThreads, unitThreads: 1, unit: "TFLOPS",
                    method: "8 条 half4 FMA 链（整条写回），half2 打包",
                    methodEn: "8 × half4 FMA chains, half2 packed"),
    ]

    static func selected(_ dtypes: Set<String>) -> [ComputeCase] {
        all.filter { dtypes.contains($0.dtype) }
    }
}

struct ComputeBench {
    let gpu: GPU

    /// 按用例准备 buffer，并返回迭代次数所在槽位。
    /// 注意不能假设 iters 一定在索引 1：MMA 用例的索引 1 是矩阵 B。
    func makeBuffers(_ c: ComputeCase, units: Int) throws -> (buffers: [(Int, MTLBuffer)], itersIndex: Int) {
        switch c.kernel {
        case "peak_mma_f32":
            var a = [Float](repeating: 0, count: 64), b = a
            for i in 0..<64 { a[i] = Float(i % 7) * 0.1; b[i] = Float(i % 5) * 0.2 }
            return ([(0, gpu.buffer(a)!), (1, gpu.buffer(b)!),
                     (2, gpu.emptyBuffer(units * 256 * 4, shared: true)), (3, gpu.buffer([UInt32(1)])!)], 3)
        case "peak_mma_f16":
            var a = [UInt16](repeating: 0, count: 64), b = a
            for i in 0..<64 {
                a[i] = Float16(Float(i % 7) * 0.1).bitPattern
                b[i] = Float16(Float(i % 5) * 0.2).bitPattern
            }
            return ([(0, gpu.buffer(a)!), (1, gpu.buffer(b)!),
                     (2, gpu.emptyBuffer(units * 256 * 2, shared: true)), (3, gpu.buffer([UInt32(1)])!)], 3)
        case "peak_f32":
            return ([(0, gpu.emptyBuffer(units * 16, shared: true)),
                     (1, gpu.buffer([UInt32(1024)])!),
                     (2, gpu.buffer([Float(1.0 + 1.1920929e-7), Float(1.1920929e-7)])!)], 1)
        case "peak_i8":
            return ([(0, gpu.emptyBuffer(units * 16, shared: true)),
                     (1, gpu.buffer([UInt32(1024)])!),
                     (2, gpu.buffer([UInt32(0x9e3779b9), UInt32(0x85ebca6b)])!)], 1)
        case "peak_f16":
            return ([(0, gpu.emptyBuffer(units * 8, shared: true)),
                     (1, gpu.buffer([UInt32(1024)])!),
                     (2, gpu.buffer([UInt16(0x3C01), UInt16(0x1400)])!)], 1)
        default:
            throw BenchError.unknownCase(c.id)
        }
    }

    func run(_ c: ComputeCase) throws -> ComputeMetric {
        let pso = try gpu.pipeline(c.kernel)
        let (buffers, itersIndex) = try makeBuffers(c, units: c.units)
        let itersBuf = buffers.first { $0.0 == itersIndex }?.1

        func measure(_ iters: Int, reps: Int) throws -> Double {
            itersBuf?.contents().copyMemory(from: [UInt32(iters)], byteCount: 4)
            return try gpu.timeThreads(pso, buffers: buffers, threads: c.threads,
                                       threadgroup: c.threadgroup, reps: reps)
        }

        // 单次派发校准到 20~120 ms，跨机型都稳定
        var iters = c.path == "mma" ? 64 : 1024
        for _ in 0..<10 {
            let t = try measure(iters, reps: 1)
            if t < 0.02 { iters = min(iters * 4, 1 << 20) }
            else if t > 0.12 { iters = max(iters / 2, 32) }
            else { break }
        }
        let single = try measure(iters, reps: 1)
        let reps = max(1, min(16, Int((0.08 / single).rounded())))
        let samples = try gpu.sample(9) { try measure(iters, reps: reps) }
        let sorted = samples.sorted()
        let best = sorted.first ?? single
        let median = sorted[sorted.count / 2]

        let macs = Double(c.units) * Double(iters) * c.macsPerIteration * Double(reps)
        let value = macs * 2 / best / 1e12
        let valueMedian = macs * 2 / median / 1e12
        // 物理上界：任何单颗 GPU 的着色器/矩阵通路都不可能到 1000 TFLOPS；
        // 超界通常意味着计时器异常或命令缓冲没真正执行。
        guard value.isFinite, value > 0, value < 1000 else {
            throw BenchError.invalidMetric("\(c.id) 峰值 \(value) TFLOPS")
        }
        guard valueMedian.isFinite, valueMedian > 0 else {
            throw BenchError.invalidMetric("\(c.id) 中位数异常")
        }

        return ComputeMetric(
            dtype: c.dtype,
            path: c.path,
            label: c.label,
            labelEn: c.labelEn,
            value: value,
            valueMedian: valueMedian,
            unit: c.unit,
            macsPerSec: macs / best,
            macsPerSecMedian: macs / median,
            threads: c.threads,
            units: c.units,
            unitThreads: c.unitThreads,
            macsPerIteration: c.macsPerIteration,
            totalMacs: macs,
            iterations: iters,
            dispatches: reps,
            seconds: best,
            secondsMedian: median,
            samples: samples,
            method: c.method,
            methodEn: c.methodEn
        )
    }
}

// MARK: - GEMM（参考实现）

struct GemmBench {
    let gpu: GPU

    private func fill(_ n: Int, dtype: GemmDType) -> MTLBuffer {
        let count = n * n
        var state: UInt32 = 12345
        if dtype == .int4 {
            // 低 nibble 在前：k 为偶数时取低 4 位
            var packed = [UInt8](repeating: 0, count: count / 2)
            for i in 0..<packed.count {
                state = state &* 1664525 &+ 1013904223
                let lo = UInt8(Int32(bitPattern: state >> 24) & 0xF)
                state = state &* 1664525 &+ 1013904223
                let hi = UInt8(Int32(bitPattern: state >> 24) & 0xF)
                packed[i] = lo | (hi << 4)
            }
            return gpu.buffer(packed)!
        }
        if dtype == .int8 {
            var values = [Int8](repeating: 0, count: count)
            for i in 0..<count {
                state = state &* 1664525 &+ 1013904223
                values[i] = Int8(Int32(state >> 24) % 15 - 7)
            }
            return gpu.buffer(values)!
        }
        if dtype == .fp16 {
            var values = [UInt16](repeating: 0, count: count)
            for i in 0..<count {
                state = state &* 1664525 &+ 1013904223
                let v = Float(Int32(bitPattern: state >> 16) & 0x3FF) / 1024.0 - 0.5
                values[i] = Float16(v).bitPattern
            }
            return gpu.buffer(values)!
        }
        var values = [Float](repeating: 0, count: count)
        for i in 0..<count {
            state = state &* 1664525 &+ 1013904223
            values[i] = Float(Int32(bitPattern: state >> 16) & 0x3FF) / 1024.0 - 0.5
        }
        return gpu.buffer(values)!
    }

    func run(_ dtype: GemmDType, size n: Int) throws -> GemmMetric {
        guard n >= 64, n % 64 == 0 else {
            throw BenchError.invalidMetric("GEMM 尺寸必须是 >=64 且能被 64 整除（收到 \(n)）")
        }
        let pso = try gpu.pipeline(dtype.kernel)
        let a = fill(n, dtype: dtype)
        let b = fill(n, dtype: dtype)
        let c = gpu.emptyBuffer(n * n * dtype.outputBytes, shared: true)
        let nb = gpu.buffer([UInt32(n)])!
        let buffers: [(Int, MTLBuffer)] = [(0, a), (1, b), (2, c), (3, nb)]
        let groups = (n / 64, n / 64)

        let single = try gpu.timeGroups(pso, buffers: buffers, groups: groups,
                                        threadgroup: (16, 16), reps: 1)
        let reps = max(1, min(8, Int((0.08 / single).rounded())))
        let samples = try gpu.sample(9) {
            try gpu.timeGroups(pso, buffers: buffers, groups: groups,
                               threadgroup: (16, 16), reps: reps)
        }
        let sorted = samples.sorted()
        let best = sorted.first ?? single
        let median = sorted[sorted.count / 2]
        let ops = 2.0 * Double(n) * Double(n) * Double(n) * Double(reps)
        let tflops = ops / best / 1e12
        let tflopsMedian = ops / median / 1e12
        guard tflops.isFinite, tflops > 0, tflops < 1000 else {
            throw BenchError.invalidMetric("gemm_\(dtype.rawValue) = \(tflops) TFLOPS")
        }
        let method: String
        switch dtype {
        case .fp32: method = "64x64 线程块 / 每线程 4x4 / float4 累加，纯 FMA 通路"
        case .fp16: method = "64x64 线程块 / 每线程 4x4 / half 存储 + float4 累加，纯 FMA 通路"
        case .int8: method = "64x64 线程块 / 每线程 4x4 / int8 输入 int32 累加（无 tensor 指令）"
        case .int4: method = "64x64 线程块 / 每线程 4x4 / 4-bit 打包解包 + int32 累加（无 tensor 指令）"
        }
        return GemmMetric(dtype: dtype.rawValue, size: n,
                          tflops: tflops, tflopsMedian: tflopsMedian, unit: dtype.unit,
                          seconds: best, secondsMedian: median, samples: samples, method: method)
    }
}

// MARK: - 带宽

struct BandwidthBench {
    let gpu: GPU

    /// 262,144 线程 = 8192 warps：M1 上实测比 65,536 线程高约 8%，
    /// 大 GPU 也不会过载（网格步长循环，线程多于元素时自动退化为 1 次）。
    static let defaultThreads = 262_144

    /// 自动选线程数：先用默认值测一遍，再用 2× 线程测一遍。
    /// 增益 ≥3% 说明默认值喂不饱这台 GPU，自动切到 2×；显式传了 --bw-threads 则尊重用户值。
    static func choose(gpu: GPU, sizeMB: Int, preferred: Int, force: Bool) throws
        -> (threads: Int, saturation: BandwidthSaturation) {
        let base = try BandwidthBench(gpu: gpu).run(sizeMB: sizeMB, threads: preferred)
        let alt = min(preferred * 2, 1_048_576)
        guard alt > preferred else {
            return (preferred, BandwidthSaturation(primaryThreads: preferred, altThreads: preferred,
                                                   readGainPct: 0, writeGainPct: 0, copyGainPct: 0,
                                                   saturated: true))
        }
        let other = try BandwidthBench(gpu: gpu).run(sizeMB: sizeMB, threads: alt)
        func value(_ list: [BandwidthMetric], _ name: String) -> Double {
            list.first { $0.name == name }?.gbpsMedian ?? 0
        }
        func gain(_ name: String) -> Double {
            let a = value(base, name), b = value(other, name)
            guard a > 0 else { return 0 }
            return (b / a - 1) * 100
        }
        let gains = [gain("gpu_read"), gain("gpu_write"), gain("gpu_copy")]
        let worst = gains.max() ?? 0
        let sat = BandwidthSaturation(primaryThreads: preferred, altThreads: alt,
                                      readGainPct: gains[0], writeGainPct: gains[1],
                                      copyGainPct: gains[2], saturated: worst < 3.0)
        return (force || worst < 3.0 ? preferred : alt, sat)
    }

    func run(sizeMB: Int, threads: Int = BandwidthBench.defaultThreads) throws -> [BandwidthMetric] {
        let bytes = sizeMB * 1024 * 1024
        let n4 = bytes / 16
        let inBuf = gpu.emptyBuffer(bytes)
        let outBuf = gpu.emptyBuffer(bytes)
        let nb = gpu.buffer([UInt32(n4)])!
        let threadgroup = 256
        // 先用不可压缩的伪随机数据填充，避免零页/内存压缩导致带宽虚高
        let fill = try gpu.pipeline("bw_fill")
        _ = try gpu.timeThreads(fill, buffers: [(0, inBuf), (1, inBuf), (2, nb)],
                                threads: threads, threadgroup: threadgroup, reps: 1)
        _ = try gpu.timeThreads(fill, buffers: [(0, outBuf), (1, outBuf), (2, nb)],
                                threads: threads, threadgroup: threadgroup, reps: 1)

        func timed(_ pso: MTLComputePipelineState, reps: Int) throws -> Double {
            try gpu.timeThreads(pso, buffers: [(0, inBuf), (1, outBuf), (2, nb)],
                                threads: threads, threadgroup: threadgroup, reps: reps)
        }

        func measureReadWrite(_ name: String,
                              pso: MTLComputePipelineState,
                              trafficPerRep: Int) throws -> BandwidthMetric {
            let single = try timed(pso, reps: 1)
            let reps = max(1, min(40, Int((0.08 / single).rounded())))
            let samples = try gpu.sample(9) { try timed(pso, reps: reps) }
            let sorted = samples.sorted()
            let best = sorted.first ?? single
            let worst = sorted.last ?? single
            let median = sorted[sorted.count / 2]
            let traffic = Double(trafficPerRep) * Double(reps)
            let gbps = traffic / best / 1e9
            guard gbps.isFinite, gbps > 0, gbps < 10_000 else {
                throw BenchError.invalidMetric("\(name) = \(gbps) GB/s")
            }
            return BandwidthMetric(
                name: name,
                gbps: gbps,
                // gbps 是"最快一次"的吞吐，因此 gbps_max 与 gbps 同值（字段保留兼容）
                gbpsMin: traffic / worst / 1e9,
                gbpsMax: gbps,
                gbpsMedian: traffic / median / 1e9,
                seconds: best,
                secondsMedian: median,
                bytesPerRep: trafficPerRep,
                reps: reps,
                threads: threads,
                samples: samples,
                method: name == "gpu_read" ? "float4 网格步长累加（只读）" : "float4 网格步长写入（只写）"
            )
        }

        var results: [BandwidthMetric] = []
        results.append(try measureReadWrite("gpu_read", pso: try gpu.pipeline("bw_read"),
                                            trafficPerRep: bytes))
        results.append(try measureReadWrite("gpu_write", pso: try gpu.pipeline("bw_write"),
                                            trafficPerRep: bytes))

        // 拷贝走 blit（DMA 路径），比计算 kernel 拷贝更接近真实上限
        func blitTimed(_ reps: Int) throws -> Double {
            let cb = gpu.queue.makeCommandBuffer()!
            let enc = cb.makeBlitCommandEncoder()!
            for _ in 0..<max(reps, 1) {
                enc.copy(from: inBuf, sourceOffset: 0, to: outBuf, destinationOffset: 0, size: bytes)
            }
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            guard cb.status == .completed, cb.error == nil else {
                throw BenchError.gpuFailure(cb.error?.localizedDescription ?? "blit status \(cb.status.rawValue)")
            }
            let t = cb.gpuEndTime - cb.gpuStartTime
            guard t.isFinite, t > 1e-5 else { throw BenchError.invalidMetric("blit 时间异常") }
            return t
        }
        let singleBlit = try blitTimed(1)
        let blitReps = max(1, min(40, Int((0.08 / singleBlit).rounded())))
        let blitSamples = try gpu.sample(9) { try blitTimed(blitReps) }
        let blitSorted = blitSamples.sorted()
        let blitBest = blitSorted.first ?? singleBlit
        let blitWorst = blitSorted.last ?? singleBlit
        let blitMedian = blitSorted[blitSorted.count / 2]
        let copyBytes = Double(bytes) * 2 * Double(blitReps)
        results.append(BandwidthMetric(
            name: "gpu_copy",
            gbps: copyBytes / blitBest / 1e9,
            gbpsMin: copyBytes / blitWorst / 1e9,
            gbpsMax: copyBytes / blitBest / 1e9,
            gbpsMedian: copyBytes / blitMedian / 1e9,
            seconds: blitBest,
            secondsMedian: blitMedian,
            bytesPerRep: bytes * 2,
            reps: blitReps,
            threads: threads,
            samples: blitSamples,
            method: "MTLBlitCommandEncoder 拷贝（DMA，读写各一次）"
        ))

        return results
    }

}

// MARK: - 自检（数值正确性）

struct SelfCheck {
    let gpu: GPU

    /// CPU 参考 GEMM，返回最大绝对误差
    func gemmError(dtype: GemmDType, n: Int, tolerance: Double) throws -> (Double, Bool) {
        let isHalf = dtype == .fp16
        let pso = try gpu.pipeline(dtype.kernel)
        var a = [Float](repeating: 0, count: n * n)
        var b = [Float](repeating: 0, count: n * n)
        for i in 0..<(n * n) {
            // 0.0731 / 0.0419 不是二进制精确数：误差才不会被舍入抹平，
            // 否则 0.25/0.5 这类值做累加会得到"误差恒为 0"的假通过。
            a[i] = Float((i % 13) - 6) * 0.0731 + 0.0017
            b[i] = Float((i % 7) - 3) * 0.0419 - 0.0023
        }
        let aBuf: MTLBuffer
        let bBuf: MTLBuffer
        if isHalf {
            aBuf = gpu.buffer(a.map { Float16($0).bitPattern })!
            bBuf = gpu.buffer(b.map { Float16($0).bitPattern })!
        } else {
            aBuf = gpu.buffer(a)!
            bBuf = gpu.buffer(b)!
        }
        let cBytes = n * n * (isHalf ? 2 : 4)
        let cBuf = gpu.emptyBuffer(cBytes, shared: true)
        // makeBuffer 的内容未定义；参考实现假设 C 初值为 0，必须显式清零
        memset(cBuf.contents(), 0, cBytes)
        let nb = gpu.buffer([UInt32(n)])!
        _ = try gpu.timeGroups(pso, buffers: [(0, aBuf), (1, bBuf), (2, cBuf), (3, nb)],
                               groups: (n / 64, n / 64), threadgroup: (16, 16), reps: 1)

        var refs = [Float](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                var sum: Float = 0
                for k in 0..<n { sum += a[i * n + k] * b[k * n + j] }
                refs[i * n + j] = sum
            }
        }
        let scale = max(refs.map { abs($0) }.max() ?? 1, 1)
        var maxErr: Double = 0
        if isHalf {
            let ptr = cBuf.contents().bindMemory(to: UInt16.self, capacity: n * n)
            for i in 0..<(n * n) {
                let got = Double(Float(Float16(bitPattern: ptr[i])))
                maxErr = max(maxErr, abs(got - Double(refs[i])))
            }
        } else {
            let ptr = cBuf.contents().bindMemory(to: Float.self, capacity: n * n)
            for i in 0..<(n * n) { maxErr = max(maxErr, abs(Double(ptr[i]) - Double(refs[i]))) }
        }
        return (maxErr, maxErr <= tolerance * Double(scale))
    }

    /// int8 GEMM 与 CPU 整数参考逐元素比对（应当完全相等）
    func gemmI8Error(n: Int) throws -> (Double, Bool) {
        let pso = try gpu.pipeline(GemmDType.int8.kernel)
        var a = [Int8](repeating: 0, count: n * n)
        var b = [Int8](repeating: 0, count: n * n)
        for i in 0..<(n * n) {
            a[i] = Int8((i % 13) - 6)
            b[i] = Int8((i % 7) - 3)
        }
        let aBuf = gpu.buffer(a)!, bBuf = gpu.buffer(b)!
        let cBuf = gpu.emptyBuffer(n * n * 4, shared: true)
        memset(cBuf.contents(), 0, n * n * 4)
        let nb = gpu.buffer([UInt32(n)])!
        _ = try gpu.timeGroups(pso, buffers: [(0, aBuf), (1, bBuf), (2, cBuf), (3, nb)],
                               groups: (n / 64, n / 64), threadgroup: (16, 16), reps: 1)
        var worst = 0
        let ptr = cBuf.contents().bindMemory(to: Int32.self, capacity: n * n)
        for i in 0..<n {
            for j in 0..<n {
                var sum = 0
                for k in 0..<n { sum += Int(a[i * n + k]) * Int(b[k * n + j]) }
                worst = max(worst, abs(Int(ptr[i * n + j]) - sum))
            }
        }
        return (Double(worst), worst == 0)
    }

    /// int4 GEMM 与 CPU 整数参考逐元素比对（解包口径必须一致，否则会有 2 倍误差）
    func gemmI4Error(n: Int) throws -> (Double, Bool) {
        let pso = try gpu.pipeline(GemmDType.int4.kernel)
        // 与 GemmBench.fill 完全相同的 LCG 与打包顺序（低 nibble 在前）
        var state: UInt32 = 12345
        var packed = [UInt8](repeating: 0, count: n * n / 2)
        for i in 0..<packed.count {
            state = state &* 1664525 &+ 1013904223
            let lo = UInt8(Int32(bitPattern: state >> 24) & 0xF)
            state = state &* 1664525 &+ 1013904223
            let hi = UInt8(Int32(bitPattern: state >> 24) & 0xF)
            packed[i] = lo | (hi << 4)
        }
        state = 999
        var packedB = [UInt8](repeating: 0, count: n * n / 2)
        for i in 0..<packedB.count {
            state = state &* 1664525 &+ 1013904223
            let lo = UInt8(Int32(bitPattern: state >> 24) & 0xF)
            state = state &* 1664525 &+ 1013904223
            let hi = UInt8(Int32(bitPattern: state >> 24) & 0xF)
            packedB[i] = lo | (hi << 4)
        }
        func unpack(_ p: [UInt8], _ idx: Int) -> Int {
            let byte = p[idx / 2]
            let nib = idx % 2 == 0 ? (byte & 0xF) : (byte >> 4)
            return nib >= 8 ? Int(nib) - 16 : Int(nib)
        }
        let a = gpu.buffer(packed)!, b = gpu.buffer(packedB)!
        let c = gpu.emptyBuffer(n * n * 4, shared: true)
        memset(c.contents(), 0, n * n * 4)
        let nb = gpu.buffer([UInt32(n)])!
        _ = try gpu.timeGroups(pso, buffers: [(0, a), (1, b), (2, c), (3, nb)],
                               groups: (n / 64, n / 64), threadgroup: (16, 16), reps: 1)
        var worst = 0
        let ptr = c.contents().bindMemory(to: Int32.self, capacity: n * n)
        for i in 0..<n {
            for j in 0..<n {
                var sum = 0
                for k in 0..<n { sum += unpack(packed, i * n + k) * unpack(packedB, k * n + j) }
                worst = max(worst, abs(Int(ptr[i * n + j]) - sum))
            }
        }
        return (Double(worst), worst == 0)
    }

    /// 峰值内核数值校验：用 CPU 复算同样的运算再逐元素比对。
    /// 这一项专门针对“向量分量没被使用 → 被编译器当死代码删掉 → 实测虚高”的坑。
    func peakCheck(_ c: ComputeCase) throws -> (Double, Bool) {
        let bench = ComputeBench(gpu: gpu)
        let units = c.path == "mma" ? 4 : 512
        let threads = units * c.unitThreads
        // 16 次就够：half 累加的舍入误差随 iters 线性增长，96 次会贴到 5e-2 阈值上
        let iters = 16
        let (buffers, itersIndex) = try bench.makeBuffers(c, units: units)
        // 参考实现假设累加器初值为 0；makeBuffer 不保证清零。
        // 只能清输出（mma 在 index 2，fma 在 index 0），不能清 A/B。
        let outIndex = c.path == "mma" ? 2 : 0
        if let outBuf = buffers.first(where: { $0.0 == outIndex })?.1 {
            memset(outBuf.contents(), 0, outBuf.length)
        }
        buffers.first { $0.0 == itersIndex }?.1.contents().copyMemory(from: [UInt32(iters)], byteCount: 4)
        let pso = try gpu.pipeline(c.kernel)
        _ = try gpu.timeThreads(pso, buffers: buffers, threads: threads, threadgroup: c.threadgroup, reps: 1)

        func rel(_ got: Double, _ ref: Double) -> Double {
            abs(got - ref) / max(abs(ref), 1e-9)
        }

        switch c.kernel {
        case "peak_mma_f32", "peak_mma_f16":
            let isHalf = c.kernel.hasSuffix("f16")
            var a = [Double](repeating: 0, count: 64), b = a
            for i in 0..<64 {
                a[i] = isHalf ? Double(Float16(Float(i % 7) * 0.1)) : Double(Float(i % 7) * 0.1)
                b[i] = isHalf ? Double(Float16(Float(i % 5) * 0.2)) : Double(Float(i % 5) * 0.2)
            }
            var worst = 0.0
            let ptr = buffers[2].1.contents()   // 索引 2 才是 C
            for i in 0..<8 {
                for j in 0..<8 {
                    var ref = 0.0
                    for k in 0..<8 { ref += a[i * 8 + k] * b[k * 8 + j] }
                    ref *= Double(iters)
                    let idx = i * 8 + j
                    let got: Double = isHalf
                        ? Double(Float16(bitPattern: ptr.load(fromByteOffset: idx * 2, as: UInt16.self)))
                        : Double(ptr.load(fromByteOffset: idx * 4, as: Float.self))
                    worst = max(worst, rel(got, ref))
                }
            }
            return (worst, worst < (isHalf ? 2e-2 : 1e-3))

        case "peak_i8":
            let k1: UInt32 = 0x9e37_79b9, k2: UInt32 = 0x85eb_ca6b
            func mad(_ x: SIMD4<UInt32>, _ k: UInt32, _ c: UInt32) -> SIMD4<UInt32> {
                x &* SIMD4(repeating: k) &+ SIMD4(repeating: c)
            }
            var worst = 0.0
            let ptr = buffers[0].1.contents()
            for gid in 0..<8 {
                let g = UInt32(gid)
                var a0 = SIMD4<UInt32>(repeating: g &+ 1)
                var a1 = SIMD4<UInt32>(repeating: g &+ 2)
                var a2 = SIMD4<UInt32>(repeating: g &+ 3)
                var a3 = SIMD4<UInt32>(repeating: g &+ 5)
                var b0 = SIMD4<UInt32>(repeating: g &* 3 &+ 1)
                var b1 = SIMD4<UInt32>(repeating: g &* 5 &+ 1)
                var b2 = SIMD4<UInt32>(repeating: g &* 7 &+ 1)
                var b3 = SIMD4<UInt32>(repeating: g &* 11 &+ 1)
                for _ in 0..<iters {
                    a0 = mad(a0, k1, k2); a1 = mad(a1, k1, k2)
                    a2 = mad(a2, k1, k2); a3 = mad(a3, k1, k2)
                    b0 = mad(b0, k2, k1); b1 = mad(b1, k2, k1)
                    b2 = mad(b2, k2, k1); b3 = mad(b3, k2, k1)
                }
                let sum = a0 &+ a1 &+ a2 &+ a3 &+ b0 &+ b1 &+ b2 &+ b3
                for lane in 0..<4 {
                    let got = ptr.load(fromByteOffset: gid * 16 + lane * 4, as: UInt32.self)
                    worst = max(worst, got == sum[lane] ? 0 : 1)
                }
            }
            return (worst, worst == 0)

        case "peak_f32", "peak_f16":
            let isHalf = c.kernel.hasSuffix("f16")
            let gid = 511
            var worst = 0.0
            let ptr = buffers[0].1.contents()
            for k in 0..<4 {
                var ref: Double
                if isHalf {
                    var chains = (0..<8).map { Float16(Float16(Float(gid & 1023)) * Float16(bitPattern: 0x1400)) + Float16(Float($0)) }
                    let x = Float16(bitPattern: 0x3C01), y = Float16(bitPattern: 0x1400)
                    for _ in 0..<iters { for j in 0..<8 { chains[j] = chains[j] * x + y } }
                    var sum = Float16(0)
                    for j in 0..<8 { sum += chains[j] }
                    ref = Double(sum)
                } else {
                    let x = Float(1.0 + 1.1920929e-7), y = Float(1.1920929e-7)
                    var chains = (0..<8).map { Float(gid & 1023) * y + Float($0) }
                    for _ in 0..<iters { for j in 0..<8 { chains[j] = chains[j] * x + y } }
                    var sum: Float = 0
                    for j in 0..<8 { sum += chains[j] }
                    ref = Double(sum)
                }
                let got: Double = isHalf
                    ? Double(Float16(bitPattern: ptr.load(fromByteOffset: (gid * 4 + k) * 2, as: UInt16.self)))
                    : Double(ptr.load(fromByteOffset: (gid * 4 + k) * 4, as: Float.self))
                worst = max(worst, rel(got, ref))
            }
            return (worst, worst < (isHalf ? 2e-2 : 1e-4))

        default:
            throw BenchError.unknownCase(c.id)
        }
    }

    /// 跑完整自检并返回结构化结果，供 `run` 内嵌进 result.json。
    /// 只返回数据、不打印——打印由 CLI 层决定。
    func runAll() -> [SelfCheckItem] {
        var items: [SelfCheckItem] = []

        func add(_ name: String, _ body: () throws -> (Double, Bool)) {
            do {
                let (err, passed) = try body()
                items.append(SelfCheckItem(name: name, passed: passed,
                                           detail: String(format: "最大误差 %.2e", err)))
            } catch {
                items.append(SelfCheckItem(name: name, passed: false, detail: "\(error)"))
            }
        }

        add("int8 GEMM 128³") { try gemmI8Error(n: 128) }
        add("int4 GEMM 128³") { try gemmI4Error(n: 128) }
        add("fp32 GEMM 128³") { try gemmError(dtype: .fp32, n: 128, tolerance: 2e-3) }
        add("fp16 GEMM 128³") { try gemmError(dtype: .fp16, n: 128, tolerance: 2e-2) }
        for c in ComputeCases.all {
            add("\(c.dtype) \(c.path) 数值") { try peakCheck(c) }
        }

        do {
            let metrics = try BandwidthBench(gpu: gpu).run(sizeMB: 64)
            for m in metrics {
                let passed = m.gbps.isFinite && m.gbps > 5 && m.gbps < 5000
                items.append(SelfCheckItem(name: m.name, passed: passed,
                                           detail: String(format: "%.1f GB/s", m.gbps)))
            }
        } catch {
            items.append(SelfCheckItem(name: "bandwidth", passed: false, detail: "\(error)"))
        }
        return items
    }
}

