// Metal 着色器源码（运行时编译，不依赖 xcrun metal / Xcode 工程）
//
// 两条写内核的硬规则（都是踩过坑换来的）：
// 1) 累加结果必须**整条向量写回**。若只取 `.x`，编译器会把 .y/.z/.w 判为死代码删掉，
//    实测值会虚高 4 倍（本工具 v1.0 就栽在这里）。
// 2) 所有迭代次数、输入数据都从 buffer 来，防止常量折叠。
//
// 另外 M1 上有三条性质完全不同的计算通路：
//   - 矩阵指令（simdgroup_matrix MMA）：唯一能跑到 ~2.5 TFLOPS 的通路，fp32/fp16 同速
//   - 普通 FMA 着色器：fp32 ≈ 1.0 TFLOPS，fp16（half2 打包）≈ 1.8 TFLOPS
//   - 整型 ALU：没有 tensor 指令，INT8/INT4 只能逐元素模拟

let mslSource = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ---------------------------------------------------------------- 矩阵指令（MMA）
// 每个 warp 每轮 4 × (8x8x8) = 4 × 512 = 2048 MAC；4 个累加器全部写回
kernel void peak_mma_f32(device const float* A [[buffer(0)]],
                         device const float* B [[buffer(1)]],
                         device float* C [[buffer(2)]],
                         constant uint& iters [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    simdgroup_float8x8 a, b;
    simdgroup_load(a, A, 8);
    simdgroup_load(b, B, 8);
    simdgroup_float8x8 c0, c1, c2, c3;
    simdgroup_load(c0, C, 8); simdgroup_load(c1, C, 8);
    simdgroup_load(c2, C, 8); simdgroup_load(c3, C, 8);
    for (uint i = 0; i < iters; ++i) {
        simdgroup_multiply_accumulate(c0, a, b, c0);
        simdgroup_multiply_accumulate(c1, a, b, c1);
        simdgroup_multiply_accumulate(c2, a, b, c2);
        simdgroup_multiply_accumulate(c3, a, b, c3);
    }
    uint w = gid / 32u;
    simdgroup_store(c0, C + w * 256u + 0u, 8);
    simdgroup_store(c1, C + w * 256u + 64u, 8);
    simdgroup_store(c2, C + w * 256u + 128u, 8);
    simdgroup_store(c3, C + w * 256u + 192u, 8);
}

kernel void peak_mma_f16(device const half* A [[buffer(0)]],
                         device const half* B [[buffer(1)]],
                         device half* C [[buffer(2)]],
                         constant uint& iters [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    simdgroup_half8x8 a, b;
    simdgroup_load(a, A, 8);
    simdgroup_load(b, B, 8);
    simdgroup_half8x8 c0, c1, c2, c3;
    simdgroup_load(c0, C, 8); simdgroup_load(c1, C, 8);
    simdgroup_load(c2, C, 8); simdgroup_load(c3, C, 8);
    for (uint i = 0; i < iters; ++i) {
        simdgroup_multiply_accumulate(c0, a, b, c0);
        simdgroup_multiply_accumulate(c1, a, b, c1);
        simdgroup_multiply_accumulate(c2, a, b, c2);
        simdgroup_multiply_accumulate(c3, a, b, c3);
    }
    uint w = gid / 32u;
    simdgroup_store(c0, C + w * 256u + 0u, 8);
    simdgroup_store(c1, C + w * 256u + 64u, 8);
    simdgroup_store(c2, C + w * 256u + 128u, 8);
    simdgroup_store(c3, C + w * 256u + 192u, 8);
}

// ---------------------------------------------------------------- FP32 FMA
// 8 条独立 float4 链 = 32 MAC/iter；整条 float4 写回（不能只取 .x）
kernel void peak_f32(device float4* out [[buffer(0)]],
                     constant uint& iters [[buffer(1)]],
                     device const float* seed [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    float x = seed[0], y = seed[1];
    float4 a0 = float4(float(gid & 1023u) * y);
    float4 a1 = a0 + 1.0f, a2 = a0 + 2.0f, a3 = a0 + 3.0f;
    float4 b0 = a0 + 4.0f, b1 = a0 + 5.0f, b2 = a0 + 6.0f, b3 = a0 + 7.0f;
    for (uint i = 0; i < iters; ++i) {
        a0 = a0 * x + y; a1 = a1 * x + y; a2 = a2 * x + y; a3 = a3 * x + y;
        b0 = b0 * x + y; b1 = b1 * x + y; b2 = b2 * x + y; b3 = b3 * x + y;
    }
    out[gid] = a0 + a1 + a2 + a3 + b0 + b1 + b2 + b3;
}

// ---------------------------------------------------------------- FP16 FMA
// 8 条独立 half4 链 = 32 MAC/iter；整条 half4 写回
kernel void peak_f16(device half4* out [[buffer(0)]],
                     constant uint& iters [[buffer(1)]],
                     device const half* seed [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    half x = seed[0], y = seed[1];
    half4 a0 = half4(half(gid & 1023u) * y);
    half4 a1 = a0 + half(1.0h), a2 = a0 + half(2.0h), a3 = a0 + half(3.0h);
    half4 b0 = a0 + half(4.0h), b1 = a0 + half(5.0h), b2 = a0 + half(6.0h), b3 = a0 + half(7.0h);
    for (uint i = 0; i < iters; ++i) {
        a0 = a0 * x + y; a1 = a1 * x + y; a2 = a2 * x + y; a3 = a3 * x + y;
        b0 = b0 * x + y; b1 = b1 * x + y; b2 = b2 * x + y; b3 = b3 * x + y;
    }
    out[gid] = a0 + a1 + a2 + a3 + b0 + b1 + b2 + b3;
}

// ---------------------------------------------------------------- GEMM（参考实现）
// 64x64 线程块 / 每线程 4x4 / BK=16，纯 FMA 路径，用来测“不用矩阵指令能跑多快”。
template <typename T>
inline void gemm_tile(device const T* A, device const T* B, device T* C, uint N,
                      uint bm, uint bn, threadgroup T* As, threadgroup T* Bs, uint tid) {
    const uint arow = tid / 4u, ak = (tid % 4u) * 4u;
    const uint bk = tid / 16u, bn0 = (tid % 16u) * 4u;
    const uint m0 = (tid / 16u) * 4u, n0 = (tid % 16u) * 4u;
    float4 c0 = float4(0.0f), c1 = float4(0.0f), c2 = float4(0.0f), c3 = float4(0.0f);
    for (uint k0 = 0; k0 < N; k0 += 16u) {
        for (uint i = 0; i < 4u; ++i) As[(ak + i) * 72u + arow] = A[(bm + arow) * N + k0 + ak + i];
        for (uint i = 0; i < 4u; ++i) Bs[bk * 72u + bn0 + i] = B[(k0 + bk) * N + bn + bn0 + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16u; ++k) {
            float4 a = float4(As[k * 72u + m0], As[k * 72u + m0 + 1],
                              As[k * 72u + m0 + 2], As[k * 72u + m0 + 3]);
            float4 b = float4(Bs[k * 72u + n0], Bs[k * 72u + n0 + 1],
                              Bs[k * 72u + n0 + 2], Bs[k * 72u + n0 + 3]);
            c0 = fma(float4(a.x), b, c0);
            c1 = fma(float4(a.y), b, c1);
            c2 = fma(float4(a.z), b, c2);
            c3 = fma(float4(a.w), b, c3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    C[(bm + m0 + 0u) * N + bn + n0 + 0u] = T(c0.x);
    C[(bm + m0 + 0u) * N + bn + n0 + 1u] = T(c0.y);
    C[(bm + m0 + 0u) * N + bn + n0 + 2u] = T(c0.z);
    C[(bm + m0 + 0u) * N + bn + n0 + 3u] = T(c0.w);
    C[(bm + m0 + 1u) * N + bn + n0 + 0u] = T(c1.x);
    C[(bm + m0 + 1u) * N + bn + n0 + 1u] = T(c1.y);
    C[(bm + m0 + 1u) * N + bn + n0 + 2u] = T(c1.z);
    C[(bm + m0 + 1u) * N + bn + n0 + 3u] = T(c1.w);
    C[(bm + m0 + 2u) * N + bn + n0 + 0u] = T(c2.x);
    C[(bm + m0 + 2u) * N + bn + n0 + 1u] = T(c2.y);
    C[(bm + m0 + 2u) * N + bn + n0 + 2u] = T(c2.z);
    C[(bm + m0 + 2u) * N + bn + n0 + 3u] = T(c2.w);
    C[(bm + m0 + 3u) * N + bn + n0 + 0u] = T(c3.x);
    C[(bm + m0 + 3u) * N + bn + n0 + 1u] = T(c3.y);
    C[(bm + m0 + 3u) * N + bn + n0 + 2u] = T(c3.z);
    C[(bm + m0 + 3u) * N + bn + n0 + 3u] = T(c3.w);
}

kernel void gemm_f32(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                     device float* C [[buffer(2)]], constant uint& N [[buffer(3)]],
                     uint2 tgid [[threadgroup_position_in_grid]],
                     uint2 tpos [[thread_position_in_threadgroup]]) {
    threadgroup float As[16 * 72];
    threadgroup float Bs[16 * 72];
    gemm_tile<float>(A, B, C, N, tgid.y * 64u, tgid.x * 64u, As, Bs, tpos.y * 16u + tpos.x);
}

kernel void gemm_f16(device const half* A [[buffer(0)]], device const half* B [[buffer(1)]],
                     device half* C [[buffer(2)]], constant uint& N [[buffer(3)]],
                     uint2 tgid [[threadgroup_position_in_grid]],
                     uint2 tpos [[thread_position_in_threadgroup]]) {
    threadgroup half As[16 * 72];
    threadgroup half Bs[16 * 72];
    gemm_tile<half>(A, B, C, N, tgid.y * 64u, tgid.x * 64u, As, Bs, tpos.y * 16u + tpos.x);
}

// ---------------------------------------------------------------- INT8 整数 ALU
// 结构与 peak_f32 完全一致，只把 FMA 换成 int32 乘加（IMAD），
// 用来回答「不用 tensor 通路时，整型 ALU 的真实吞吐是多少」。
kernel void peak_i8(device uint4* out [[buffer(0)]],
                    constant uint& iters [[buffer(1)]],
                    device const uint* seed [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    uint k1 = seed[0], k2 = seed[1];
    uint4 a0 = uint4(gid + 1u), a1 = uint4(gid + 2u), a2 = uint4(gid + 3u), a3 = uint4(gid + 5u);
    uint4 b0 = uint4(gid * 3u + 1u), b1 = uint4(gid * 5u + 1u),
          b2 = uint4(gid * 7u + 1u), b3 = uint4(gid * 11u + 1u);
    for (uint i = 0; i < iters; ++i) {
        a0 = a0 * k1 + k2; a1 = a1 * k1 + k2; a2 = a2 * k1 + k2; a3 = a3 * k1 + k2;
        b0 = b0 * k2 + k1; b1 = b1 * k2 + k1; b2 = b2 * k2 + k1; b3 = b3 * k2 + k1;
    }
    out[gid] = a0 + a1 + a2 + a3 + b0 + b1 + b2 + b3;
}

// ---------------------------------------------------------------- INT8 GEMM
// 量化推理在 Metal 上的真实路径：int8 输入、int32 累加。没有任何 tensor 指令可用。
template <typename T, typename ACC>
inline void gemm_tile_int(device const T* A, device const T* B, device ACC* C, uint N,
                          uint bm, uint bn, threadgroup T* As, threadgroup T* Bs, uint tid) {
    const uint arow = tid / 4u, ak = (tid % 4u) * 4u;
    const uint bk = tid / 16u, bn0 = (tid % 16u) * 4u;
    const uint m0 = (tid / 16u) * 4u, n0 = (tid % 16u) * 4u;
    int4 c0 = int4(0), c1 = int4(0), c2 = int4(0), c3 = int4(0);
    for (uint k0 = 0; k0 < N; k0 += 16u) {
        for (uint i = 0; i < 4u; ++i) As[(ak + i) * 72u + arow] = A[(bm + arow) * N + k0 + ak + i];
        for (uint i = 0; i < 4u; ++i) Bs[bk * 72u + bn0 + i] = B[(k0 + bk) * N + bn + bn0 + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16u; ++k) {
            int4 a = int4(As[k * 72u + m0], As[k * 72u + m0 + 1],
                          As[k * 72u + m0 + 2], As[k * 72u + m0 + 3]);
            int4 b = int4(Bs[k * 72u + n0], Bs[k * 72u + n0 + 1],
                          Bs[k * 72u + n0 + 2], Bs[k * 72u + n0 + 3]);
            c0 += a.x * b;
            c1 += a.y * b;
            c2 += a.z * b;
            c3 += a.w * b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    C[(bm + m0 + 0u) * N + bn + n0 + 0u] = ACC(c0.x);
    C[(bm + m0 + 0u) * N + bn + n0 + 1u] = ACC(c0.y);
    C[(bm + m0 + 0u) * N + bn + n0 + 2u] = ACC(c0.z);
    C[(bm + m0 + 0u) * N + bn + n0 + 3u] = ACC(c0.w);
    C[(bm + m0 + 1u) * N + bn + n0 + 0u] = ACC(c1.x);
    C[(bm + m0 + 1u) * N + bn + n0 + 1u] = ACC(c1.y);
    C[(bm + m0 + 1u) * N + bn + n0 + 2u] = ACC(c1.z);
    C[(bm + m0 + 1u) * N + bn + n0 + 3u] = ACC(c1.w);
    C[(bm + m0 + 2u) * N + bn + n0 + 0u] = ACC(c2.x);
    C[(bm + m0 + 2u) * N + bn + n0 + 1u] = ACC(c2.y);
    C[(bm + m0 + 2u) * N + bn + n0 + 2u] = ACC(c2.z);
    C[(bm + m0 + 2u) * N + bn + n0 + 3u] = ACC(c2.w);
    C[(bm + m0 + 3u) * N + bn + n0 + 0u] = ACC(c3.x);
    C[(bm + m0 + 3u) * N + bn + n0 + 1u] = ACC(c3.y);
    C[(bm + m0 + 3u) * N + bn + n0 + 2u] = ACC(c3.z);
    C[(bm + m0 + 3u) * N + bn + n0 + 3u] = ACC(c3.w);
}

kernel void gemm_i8(device const char* A [[buffer(0)]], device const char* B [[buffer(1)]],
                    device int* C [[buffer(2)]], constant uint& N [[buffer(3)]],
                    uint2 tgid [[threadgroup_position_in_grid]],
                    uint2 tpos [[thread_position_in_threadgroup]]) {
    threadgroup char As[16 * 72];
    threadgroup char Bs[16 * 72];
    gemm_tile_int<char, int>(A, B, C, N, tgid.y * 64u, tgid.x * 64u, As, Bs, tpos.y * 16u + tpos.x);
}

// ---------------------------------------------------------------- INT4 GEMM
// 每字节打包 2 个 4-bit 值（低 nibble = 偶数 k），解包后 int32 累加。
// 与 int8 走同一条 ALU 通路，多出来的成本就是解包本身——这是量化的真实代价。
inline int s4(uint byte, uint hi) {
    return hi ? ((int)(byte << 24) >> 28) : ((int)(byte << 28) >> 28);
}

kernel void gemm_i4(device const uchar* A [[buffer(0)]], device const uchar* B [[buffer(1)]],
                    device int* C [[buffer(2)]], constant uint& N [[buffer(3)]],
                    uint2 tgid [[threadgroup_position_in_grid]],
                    uint2 tpos [[thread_position_in_threadgroup]]) {
    threadgroup char As[16 * 72];
    threadgroup char Bs[16 * 72];
    const uint packedN = N / 2u;
    const uint tid = tpos.y * 16u + tpos.x;
    const uint arow = tid / 4u, ak = (tid % 4u) * 4u;
    const uint bk = tid / 16u, bn0 = (tid % 16u) * 4u;
    const uint m0 = (tid / 16u) * 4u, n0 = (tid % 16u) * 4u;
    const uint bm = tgid.y * 64u, bn = tgid.x * 64u;

    int4 c0 = int4(0), c1 = int4(0), c2 = int4(0), c3 = int4(0);
    for (uint k0 = 0; k0 < N; k0 += 16u) {
        const uint ka = k0 + ak;
        device const uchar* aP = A + (bm + arow) * packedN + ka / 2u;
        for (uint i = 0; i < 4u; ++i)
            As[(ak + i) * 72u + arow] = char(s4(aP[i / 2u], (ka + i) & 1u));

        const uint kb = k0 + bk;
        device const uchar* bP = B + kb * packedN + (bn + bn0) / 2u;
        for (uint i = 0; i < 4u; ++i)
            Bs[bk * 72u + bn0 + i] = char(s4(bP[i / 2u], (bn + bn0 + i) & 1u));

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16u; ++k) {
            int4 a = int4(As[k * 72u + m0], As[k * 72u + m0 + 1],
                          As[k * 72u + m0 + 2], As[k * 72u + m0 + 3]);
            int4 b = int4(Bs[k * 72u + n0], Bs[k * 72u + n0 + 1],
                          Bs[k * 72u + n0 + 2], Bs[k * 72u + n0 + 3]);
            c0 += a.x * b;
            c1 += a.y * b;
            c2 += a.z * b;
            c3 += a.w * b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    C[(bm + m0 + 0u) * N + bn + n0 + 0u] = c0.x;
    C[(bm + m0 + 0u) * N + bn + n0 + 1u] = c0.y;
    C[(bm + m0 + 0u) * N + bn + n0 + 2u] = c0.z;
    C[(bm + m0 + 0u) * N + bn + n0 + 3u] = c0.w;
    C[(bm + m0 + 1u) * N + bn + n0 + 0u] = c1.x;
    C[(bm + m0 + 1u) * N + bn + n0 + 1u] = c1.y;
    C[(bm + m0 + 1u) * N + bn + n0 + 2u] = c1.z;
    C[(bm + m0 + 1u) * N + bn + n0 + 3u] = c1.w;
    C[(bm + m0 + 2u) * N + bn + n0 + 0u] = c2.x;
    C[(bm + m0 + 2u) * N + bn + n0 + 1u] = c2.y;
    C[(bm + m0 + 2u) * N + bn + n0 + 2u] = c2.z;
    C[(bm + m0 + 2u) * N + bn + n0 + 3u] = c2.w;
    C[(bm + m0 + 3u) * N + bn + n0 + 0u] = c3.x;
    C[(bm + m0 + 3u) * N + bn + n0 + 1u] = c3.y;
    C[(bm + m0 + 3u) * N + bn + n0 + 2u] = c3.z;
    C[(bm + m0 + 3u) * N + bn + n0 + 3u] = c3.w;
}

// ---------------------------------------------------------------- 带宽
kernel void bw_read(device const float4* in [[buffer(0)]],
                    device float4* out [[buffer(1)]],
                    constant uint& n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]],
                    uint stride [[threads_per_grid]]) {
    float4 acc = float4(0.0f);
    for (uint i = gid; i < n; i += stride) acc += in[i];
    if (acc.x == 12345.678f) out[0] = acc;
}

// 填充伪随机位模式：避免零页/内存压缩把带宽测虚高
inline uint4 hash4(uint i) {
    uint h = i * 2654435761u + 0x9E3779B9u;
    h ^= h >> 15; h *= 2246822519u;
    uint h2 = h * 3266489917u; h2 ^= h2 >> 13;
    uint h3 = h ^ 0x85EBCA6Bu; h3 *= 0xC2B2AE35u; h3 ^= h3 >> 16;
    uint h4 = h2 + 0x27D4EB2Fu; h4 ^= h4 >> 15;
    return uint4(h, h2, h3, h4);
}

kernel void bw_fill(device float4* out [[buffer(1)]],
                    constant uint& n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]],
                    uint stride [[threads_per_grid]]) {
    for (uint i = gid; i < n; i += stride) out[i] = as_type<float4>(hash4(i));
}

kernel void bw_write(device float4* out [[buffer(1)]],
                     constant uint& n [[buffer(2)]],
                     uint gid [[thread_position_in_grid]],
                     uint stride [[threads_per_grid]]) {
    for (uint i = gid; i < n; i += stride) out[i] = as_type<float4>(hash4(i));
}
"""
