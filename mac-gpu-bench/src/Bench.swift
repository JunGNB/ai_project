import Foundation
import Metal
import Darwin

// MARK: - 数据模型（直接序列化为 JSON）

struct HostInfo: Codable {
    var chip: String
    var model: String
    var macOS: String
    var arch: String
    var cpuCores: Int
    var pCores: Int?
    var eCores: Int?
    var memoryGB: Double
    var uptimeHours: Double?
    var loadAvg1m: Double?

    enum CodingKeys: String, CodingKey {
        case chip, model, macOS, arch
        case cpuCores = "cpu_cores"
        case pCores = "p_cores"
        case eCores = "e_cores"
        case memoryGB = "memory_gb"
        case uptimeHours = "uptime_hours"
        case loadAvg1m = "load_avg_1m"
    }
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
    /// 计算通路：mma（矩阵指令）/ fma（普通着色器 FMA）/ alu（整型模拟）
    var path: String
    var label: String
    /// 英文描述，供英文报告 / 跨团队沟通
    var labelEn: String
    var value: Double
    var unit: String
    var macsPerSec: Double
    var threads: Int
    var iterations: Int
    var dispatches: Int
    var seconds: Double
    var method: String
    var methodEn: String

    enum CodingKeys: String, CodingKey {
        case dtype, path, label
        case labelEn = "label_en"
        case value, unit, threads, iterations, dispatches, seconds, method
        case methodEn = "method_en"
        case macsPerSec = "macs_per_sec"
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
    /// fp32/fp16 是 TFLOPS，int8 是 TOPS
    var tflops: Double
    var unit: String
    var seconds: Double
    var method: String

    enum CodingKeys: String, CodingKey {
        case dtype, size, tflops, unit, seconds, method
    }
}

struct BandwidthMetric: Codable {
    var name: String
    /// 9 次采样的最优值
    var gbps: Double
    /// 9 次采样中的最差值，用于判断这次读数稳不稳
    var gbpsMin: Double
    var gbpsMax: Double
    var seconds: Double
    var bytesPerRep: Int
    var reps: Int
    var method: String

    enum CodingKeys: String, CodingKey {
        case name, gbps, seconds, reps, method
        case gbpsMin = "gbps_min"
        case gbpsMax = "gbps_max"
        case bytesPerRep = "bytes_per_rep"
    }
}

struct BenchResult: Codable {
    var schema: String
    var version: String
    var runs: Int
    var generatedAt: String
    var host: HostInfo
    var gpu: GPUInfo
    var compute: [ComputeMetric]
    var gemm: [GemmMetric]
    var bandwidth: [BandwidthMetric]
    var notes: [String]

    enum CodingKeys: String, CodingKey {
        case schema, version, runs, host, gpu, compute, gemm, bandwidth, notes
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

    static func host() -> HostInfo {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        let memBytes = sysctlUInt64("hw.memsize") ?? 0
        return HostInfo(
            chip: sysctlString("machdep.cpu.brand_string") ?? "unknown",
            model: sysctlString("hw.model") ?? "unknown",
            macOS: "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)",
            arch: sysctlString("hw.machine") ?? "unknown",
            cpuCores: sysctlInt("hw.ncpu") ?? ProcessInfo.processInfo.activeProcessorCount,
            pCores: sysctlInt("hw.perflevel0.physicalcpu"),
            eCores: sysctlInt("hw.perflevel1.physicalcpu"),
            memoryGB: (Double(memBytes) / 1024 / 1024 / 1024 * 10).rounded() / 10,
            uptimeHours: (ProcessInfo.processInfo.systemUptime / 3600 * 10).rounded() / 10,
            loadAvg1m: loadAverage()
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

    var description: String {
        switch self {
        case .noDevice: return "找不到 Metal 设备"
        case .shaderCompile(let m): return "着色器编译失败: \(m)"
        case .missingKernel(let k): return "缺少内核: \(k)"
        case .invalidMetric(let m): return "测量结果异常: \(m)"
        case .unknownCase(let c): return "未注册的测量项: \(c)"
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

    /// 一次 command buffer 内重复 reps 次线程网格派发，返回 GPU 侧耗时（秒）
    func timeThreads(_ pso: MTLComputePipelineState,
                     buffers: [(Int, MTLBuffer)],
                     threads: Int,
                     threadgroup: Int,
                     reps: Int) -> Double {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        for (index, buf) in buffers { enc.setBuffer(buf, offset: 0, index: index) }
        let grid = MTLSize(width: threads, height: 1, depth: 1)
        let tg = MTLSize(width: threadgroup, height: 1, depth: 1)
        for _ in 0..<max(reps, 1) { enc.dispatchThreads(grid, threadsPerThreadgroup: tg) }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        return max(cb.gpuEndTime - cb.gpuStartTime, 1e-9)
    }

    /// 线程块派发（GEMM 用），返回 GPU 侧耗时（秒）
    func timeGroups(_ pso: MTLComputePipelineState,
                    buffers: [(Int, MTLBuffer)],
                    groups: (Int, Int),
                    threadgroup: (Int, Int),
                    reps: Int) -> Double {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        for (index, buf) in buffers { enc.setBuffer(buf, offset: 0, index: index) }
        let grid = MTLSize(width: groups.0, height: groups.1, depth: 1)
        let tg = MTLSize(width: threadgroup.0, height: threadgroup.1, depth: 1)
        for _ in 0..<max(reps, 1) { enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg) }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        return max(cb.gpuEndTime - cb.gpuStartTime, 1e-9)
    }

    /// 重复 9 次取最小，减少调度抖动与后台负载干扰
    func best(_ reps: Int, _ body: () -> Double) -> Double {
        _ = body()                      // 预热：首次派发有页错误/时钟爬升，必须丢弃
        var bestValue = Double.infinity
        for _ in 0..<9 { bestValue = min(bestValue, body()) }
        return bestValue
    }

    /// 采样 9 次，同时给出最快与最慢，用来暴露抖动（带宽对系统负载很敏感）
    func spread(_ reps: Int, _ body: () -> Double) -> (best: Double, worst: Double) {
        _ = body()                      // 预热同上
        var b = Double.infinity, w = 0.0
        for _ in 0..<9 {
            let t = body()
            b = min(b, t); w = max(w, t)
        }
        return (b, max(w, b))
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

        func measure(_ iters: Int, reps: Int) -> Double {
            itersBuf?.contents().copyMemory(from: [UInt32(iters)], byteCount: 4)
            return gpu.timeThreads(pso, buffers: buffers, threads: c.threads,
                                   threadgroup: c.threadgroup, reps: reps)
        }

        // 单次派发校准到 20~120 ms，跨机型都稳定
        var iters = c.path == "mma" ? 64 : 1024
        for _ in 0..<10 {
            let t = measure(iters, reps: 1)
            if t < 0.02 { iters = min(iters * 4, 1 << 20) }
            else if t > 0.12 { iters = max(iters / 2, 32) }
            else { break }
        }
        let single = measure(iters, reps: 1)
        let reps = max(1, min(16, Int((0.08 / single).rounded())))
        let seconds = gpu.best(reps) { measure(iters, reps: reps) }

        let macs = Double(c.units) * Double(iters) * c.macsPerIteration * Double(reps)
        let value = macs * 2 / seconds / 1e12
        guard value.isFinite, value > 0 else { throw BenchError.invalidMetric(c.id) }

        return ComputeMetric(
            dtype: c.dtype,
            path: c.path,
            label: c.label,
            labelEn: c.labelEn,
            value: value,
            unit: c.unit,
            macsPerSec: macs / seconds,
            threads: c.threads,
            iterations: iters,
            dispatches: reps,
            seconds: seconds,
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
        precondition(n % 64 == 0, "GEMM 尺寸必须是 64 的倍数")
        let pso = try gpu.pipeline(dtype.kernel)
        let a = fill(n, dtype: dtype)
        let b = fill(n, dtype: dtype)
        let c = gpu.emptyBuffer(n * n * dtype.outputBytes, shared: true)
        let nb = gpu.buffer([UInt32(n)])!
        let buffers: [(Int, MTLBuffer)] = [(0, a), (1, b), (2, c), (3, nb)]
        let groups = (n / 64, n / 64)

        let single = gpu.timeGroups(pso, buffers: buffers, groups: groups, threadgroup: (16, 16), reps: 1)
        let reps = max(1, min(8, Int((0.08 / single).rounded())))
        let seconds = gpu.best(reps) {
            gpu.timeGroups(pso, buffers: buffers, groups: groups, threadgroup: (16, 16), reps: reps)
        }
        let ops = 2.0 * Double(n) * Double(n) * Double(n) * Double(reps)
        let tflops = ops / seconds / 1e12
        guard tflops.isFinite, tflops > 0 else { throw BenchError.invalidMetric("gemm_\(dtype.rawValue)") }
        let method: String
        switch dtype {
        case .fp32: method = "64x64 线程块 / 每线程 4x4 / float4 累加，纯 FMA 通路"
        case .fp16: method = "64x64 线程块 / 每线程 4x4 / half 存储 + float4 累加，纯 FMA 通路"
        case .int8: method = "64x64 线程块 / 每线程 4x4 / int8 输入 int32 累加（无 tensor 指令）"
        case .int4: method = "64x64 线程块 / 每线程 4x4 / 4-bit 打包解包 + int32 累加（无 tensor 指令）"
        }
        return GemmMetric(dtype: dtype.rawValue, size: n, tflops: tflops, unit: dtype.unit,
                          seconds: seconds, method: method)
    }
}

// MARK: - 带宽

struct BandwidthBench {
    let gpu: GPU

    func run(sizeMB: Int) throws -> [BandwidthMetric] {
        let bytes = sizeMB * 1024 * 1024
        let n4 = bytes / 16
        let inBuf = gpu.emptyBuffer(bytes)
        let outBuf = gpu.emptyBuffer(bytes)
        let nb = gpu.buffer([UInt32(n4)])!
        // 先用不可压缩的伪随机数据填充，避免零页/内存压缩导致带宽虚高
        let fill = try gpu.pipeline("bw_fill")
        _ = gpu.timeThreads(fill, buffers: [(0, inBuf), (1, inBuf), (2, nb)],
                            threads: 65_536, threadgroup: 256, reps: 1)
        _ = gpu.timeThreads(fill, buffers: [(0, outBuf), (1, outBuf), (2, nb)],
                            threads: 65_536, threadgroup: 256, reps: 1)
        let threads = 65_536
        let threadgroup = 256

        func timed(_ pso: MTLComputePipelineState, reps: Int) -> Double {
            gpu.timeThreads(pso, buffers: [(0, inBuf), (1, outBuf), (2, nb)],
                            threads: threads, threadgroup: threadgroup, reps: reps)
        }

        func measureReadWrite(_ name: String, pso: MTLComputePipelineState, trafficPerRep: Int) -> BandwidthMetric {
            let single = timed(pso, reps: 1)
            let reps = max(1, min(40, Int((0.08 / single).rounded())))
            let (best, worst) = gpu.spread(reps) { timed(pso, reps: reps) }
            let bytes = Double(trafficPerRep) * Double(reps)
            return BandwidthMetric(
                name: name,
                gbps: bytes / best / 1e9,
                gbpsMin: bytes / worst / 1e9,
                gbpsMax: bytes / best / 1e9,
                seconds: best,
                bytesPerRep: trafficPerRep,
                reps: reps,
                method: name == "gpu_read" ? "float4 网格步长累加（只读）" : "float4 网格步长写入（只写）"
            )
        }

        var results: [BandwidthMetric] = []
        results.append(measureReadWrite("gpu_read", pso: try gpu.pipeline("bw_read"), trafficPerRep: bytes))
        results.append(measureReadWrite("gpu_write", pso: try gpu.pipeline("bw_write"), trafficPerRep: bytes))

        // 拷贝走 blit（DMA 路径），比计算 kernel 拷贝更接近真实上限
        func blitTimed(_ reps: Int) -> Double {
            let cb = gpu.queue.makeCommandBuffer()!
            let enc = cb.makeBlitCommandEncoder()!
            for _ in 0..<max(reps, 1) {
                enc.copy(from: inBuf, sourceOffset: 0, to: outBuf, destinationOffset: 0, size: bytes)
            }
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            return max(cb.gpuEndTime - cb.gpuStartTime, 1e-9)
        }
        let singleBlit = blitTimed(1)
        let blitReps = max(1, min(40, Int((0.08 / singleBlit).rounded())))
        let (blitBest, blitWorst) = gpu.spread(blitReps) { blitTimed(blitReps) }
        let copyBytes = Double(bytes) * 2 * Double(blitReps)
        results.append(BandwidthMetric(
            name: "gpu_copy", gbps: copyBytes / blitBest / 1e9,
            gbpsMin: copyBytes / blitWorst / 1e9,
            gbpsMax: copyBytes / blitBest / 1e9,
            seconds: blitBest, bytesPerRep: bytes * 2, reps: blitReps,
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
            a[i] = Float((i % 13) - 6) * 0.25
            b[i] = Float((i % 7) - 3) * 0.5
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
        let cBuf = gpu.emptyBuffer(n * n * (isHalf ? 2 : 4), shared: true)
        let nb = gpu.buffer([UInt32(n)])!
        _ = gpu.timeGroups(pso, buffers: [(0, aBuf), (1, bBuf), (2, cBuf), (3, nb)],
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
        let nb = gpu.buffer([UInt32(n)])!
        _ = gpu.timeGroups(pso, buffers: [(0, aBuf), (1, bBuf), (2, cBuf), (3, nb)],
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
        let nb = gpu.buffer([UInt32(n)])!
        _ = gpu.timeGroups(pso, buffers: [(0, a), (1, b), (2, c), (3, nb)],
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
        let iters = 96
        let (buffers, itersIndex) = try bench.makeBuffers(c, units: units)
        let pso = try gpu.pipeline(c.kernel)
        buffers.first { $0.0 == itersIndex }?.1.contents().copyMemory(from: [UInt32(iters)], byteCount: 4)
        _ = gpu.timeThreads(pso, buffers: buffers, threads: threads, threadgroup: c.threadgroup, reps: 1)

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
            return (worst, worst < (isHalf ? 5e-2 : 1e-3))

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
            return (worst, worst < (isHalf ? 5e-2 : 1e-4))

        default:
            throw BenchError.unknownCase(c.id)
        }
    }
}

