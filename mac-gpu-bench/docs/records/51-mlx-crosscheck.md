# 交叉验证：oMLX 真实模型 vs 本机 roofline（2026-11-20）

> 状态：**record**（一次摸底结果，用于校准 `[4] 推荐数据` 里的效率系数）。
> 复现：见本文「复现方法」。

## 结论先行

用 oMLX 0.7.2 加载本地 4-bit 模型实测，和本项目 `macgpu` 的 roofline 预测对比：

| 模型（4bit） | 文本权重 | 参数量 | prefill 实测 | 折算算力 | 占本机峰值 | decode 实测 | 折算带宽 | 占实测读带宽 |
|---|---|---|---|---|---|---|---|---|
| MiniCPM5-2B-MLX（2.7B dense） | 1.42 GB | 2.68 B | **321.3 tok/s** | 1.72 TFLOPS | **78%** | **36.46 tok/s** | 51.8 GB/s | **92%** |
| Qwen3.5-9B-MLX-4bit（9.5B dense） | 5.04 GB | 9.50 B | **95.6 tok/s** | 1.82 TFLOPS | **83%** | **12.30 tok/s** | 62.0 GB/s | **111%** |
| gemma-4-E4B-it-4bit（MoE，激活约 4B） | 4.81 GB | 8 B（激活 4B） | 275.6 tok/s | —（MoE 难折算） | — | 18.76 tok/s | — | — |

本机基准：**prefill 峰值 2.20 TFLOPS、decode 读带宽 56.1 GB/s**（同一轮 `macgpu run`）。

**匹配程度**：

- **prefill**：MLX 达到本机峰值的 **78% ~ 83%**，说明 2.2 TFLOPS 是真实可达的上限，不是纸面数字。
- **decode**：折算带宽是实测读带宽的 **92% ~ 111%**，误差 ±10% 以内；Qwen 略高于 100%，那是因为基准的读带宽是在 load≈2.5 下测的（空载时读到过 59.8 GB/s），且 MLX 的顺序流式读比我们的网格步长读更接近硬件极限。
- 结论：**roofline 公式可以直接用来估 tok/s**，`prefill = 算力 ÷ 参数量`、`decode = 带宽 ÷ 每 token 字节数`，误差 ±20% 以内。

## 校准出来的效率系数

| 场景 | prefill 算力效率 | decode 带宽效率 |
|---|---|---|
| MLX / MPS 这类成熟实现（本次实测） | **0.78 ~ 0.83** | **0.92 ~ 1.0** |
| 本项目手写 GEMM / 普通内核（见 `[2]`） | 0.30 ~ 0.60 | 0.8 ~ 0.9（若访存不规整会更低） |

也就是说 `[4] 推荐数据` 里写的 `eff 0.3~0.6` 是**手写内核**的区间；如果你跑的是 MLX / oMLX / LM Studio，
应该取 **0.8**，否则会把 prefill 低估一半。

## 复现方法

```bash
# 1) 起服务（HOME 指向可写目录，避免沙箱挡住 ~/.omlx 的日志与设置）
cd ~/Desktop/my_project/ai_project && mkdir -p .omlxhome
HOME="$PWD/.omlxhome" ~/.omlx/bin/omlx serve --port 8899 \
  --model-dir ~/.omlx/models --memory-guard aggressive --no-cache

# 2) 打 3 轮：1500 token 提示 + max_tokens=64，读 usage 里的
#    prompt_tokens_per_second / generation_tokens_per_second
python3 /tmp/bench_models.py Qwen3.5-9B-MLX-4bit MiniCPM5-2B-MLX
```

关键点：

- `temperature=0`、`--no-cache`（否则 prefix cache 会让 prefill 看起来快好几倍）。
- 每次先发一个 `max_tokens=8` 的请求预热，把模型加载时间从测量里排除。
- 指标直接取响应 `usage` 里的 oMLX 自报值（`prompt_tokens_per_second`、`generation_tokens_per_second`），与客户端墙钟一致。
- 参数量按 `文本权重字节 ÷ 0.53` 估算（4-bit affine + group 64 的 scale/zero 开销），并**扣掉 vision 塔**——vision 权重在文本推理时不读。

## 踩到的坑

| 现象 | 原因 | 处理 |
|---|---|---|
| `PermissionError: ~/.omlx/logs/server.log` | 沙箱只允许写工作区 | `HOME=$PWD/.omlxhome` 重定向状态目录 |
| `HTTP 507 Insufficient Storage` | 16 GB 机器 + 其他进程占用，`memory-guard safe` 只给 1.67 GB 余量 | `--memory-guard aggressive` |
| `HTTP 400 preflight rejected` | 1697 token 的提示在 safe 档下 prefill 峰值超限 | 同上，或缩短提示 |
| 首次请求 TTFT 15 s | 含模型加载（`model_load_duration 5.37 s`） | 先发预热请求 |
