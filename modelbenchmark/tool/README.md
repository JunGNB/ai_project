# tool：工程外部工具

本目录只放**不属于测试运行时**的辅助工具，不随 Python 包安装、不被 `modelbench` 导入：

- `download/`：数据集下载/校验的独立入口（`./modelbench download` 会调用它）；
  用于在无 oMLX、只准备数据的机器上下载 LongBench-v2 等大数据集。
- 第三方二进制/脚本（如需要）也放在这里，并在本文件说明来源与版本。

测试运行时、领域逻辑、文档生成等代码不要放这里：
- 运行时源码 → `src/modelbench/`
- 文档/门禁/任务板 → `harness/`
- 启动入口 → 根目录 `modelbench`

## 目录分类

```
tool/
├── download/              # 既有：数据集下载/校验入口（`./modelbench download` 调用）
└── dataset/               # 测评样本生成（稳定配置 + 确定性生成器，保留）
    ├── config.py          # ★ 稳定配置：数据源/许可、配额、token 目标、seed、构造规则、判题器
    ├── ds_common.py       # 公共库：hf-mirror 下载/校验、oMLX tokenizer、按 token 切分
    ├── tokenize_worker.py # 离线 tokenizer worker（build_* 调用）
    ├── build_perf.py      # 生成 performance/testset.jsonl + manifest（配额取自 config）
    ├── build_perf_subset.py # 派生 performance_10（240 条严格子集）+ manifest_10
    ├── build_acc.py       # 生成 accuracy/testset.jsonl + manifest（配额取自 config）
    └── verify.py          # 只读校验：结构/manifest/逐文件 SHA-256（gate dataset_integrity 调用）
```

约定：

- **保留**：`config.py` 与生成器决定"取哪些样本"，固定 seed + 固定配额，重复生成结果一致；全量生成会断言总量（performance 480、accuracy 6,589），配额漂移直接报错。
- **不保留一次性工具**：例如原始数据下载、临时探查脚本。`dataset/raw/` 已落盘，来源/SHA-256 以两份 testset `manifest.json` 的 `provenance` 为准；换机或重下时临时写脚本即可，不放进仓库。

## 常用命令

```bash
PY=.venv/bin/python
$PY tool/dataset/build_perf.py            # 性能：480 条（24 cells，无 warmup 数据）
$PY tool/dataset/build_perf.py --only knowledge_qa math_logic   # 只重建指定场景
$PY tool/dataset/build_acc.py             # 精度：6,589 条（250×6 + BFCL 889 + 42×100）
$PY tool/dataset/build_perf_subset.py     # 快速性能子集：240 条
$PY tool/dataset/build_acc.py --only coding general_agent        # 只重建指定场景
$PY tool/dataset/verify.py                # 只读校验（gate 的 dataset_integrity 检查）
MB_TOKEN_MODEL=Qwen3.5-9B-MLX-4bit $PY tool/dataset/build_perf.py   # 换参考 tokenizer
```

约定：逐样本 `tokens` 为实测输入 token（chat template，`/v1/tokenize`），不是分档值；性能 `bucket` 仅为选样标签。`dataset/.cache/tokenize.json` 为计数缓存，不提交；大文件见 `dataset/.gitignore`。
