# 2026-11-19 判题依赖与资产内置（消除静默 0 分）

> **状态**：records（非规范、非证据）｜ **日期**：2026-11-19 ｜
> **关联 run**：`20261119T020415Z-06f2__accuracy__minicpm-v-4.7-1b`（2,000 条，宏平均 0.4737）
> **一句话结论**：该轮 `instruction_following` 全场景 0 分、`long_context` 中文 100 条 0 分，
> 原因是**判题 pip 依赖与磁盘资产都没装/没抓**，不是样本选择，也不是测试脚本写错；修复方式是把
> 小依赖（官方 IFEval checker 96 KB + 英文 punkt 668 KB）**内置进包**、把 pip 依赖提为**核心
> 依赖**，并让 preflight 同时检查磁盘资产。

## 1. 现象（旧 run 证据）

| 场景 | 分数 | `judge_error` | 直接原因（`records.jsonl`） |
|---|---:|---:|---|
| instruction_following | 0.0 | 250/250 | `RuntimeError: IFEval checker not found at dataset/raw/ifeval` |
| long_context | 0.2014（en 0.3357 / zh 0.0） | 100/250 | `ModuleNotFoundError: No module named 'jieba'`（74 条 `f1_zh` + 26 条 `rouge_zh`） |
| general_agent | 0.576 | 0 | 2 条 `format_failure`（`unparseable_output`，模型输出问题） |

- 全 run `judge_error=350`（17.5%），全部集中在上面两个场景；`n_selected=n_measured=2000`，
  样本量、testset SHA、协议（cold/`temperature=0`/`repeats=1`）都正常。
- 该 run 的 `run.json.outcome.warnings` 只列了 6 个 pip 模块（`absl`/`nltk`/`langdetect`/
  `immutabledict`/`jieba`/`rouge`），**没有提 `dataset/raw/ifeval` 缺失**——因为旧
  `missing_dependencies()` 只用 `importlib.util.find_spec` 探测模块，看不到磁盘资产。
- 影响：宏平均被压低约 0.08；该 bundle 的这两个场景分数不可引用，修复后应重跑。

## 2. 根因

1. 判题依赖放在 `.[eval]` 可选 extra 里，而 `./modelbench setup` 默认只装 `.[dev]`；新机器/
   新环境按 README 走完安装仍然缺 6 个包。
2. 官方 IFEval checker 代码需要 `tool/dataset/accuracy/fetch.py` 才会落到 `dataset/raw/ifeval`；
   没跑过 fetch 的 checkout 直接 0 分。
3. checker 还需要 NLTK punkt 数据（官方全语种 `punkt.zip` + `punkt_tab.zip` 约 18 MB），
   而 `fetch.py` 的上游是 `raw.githubusercontent.com`，在部分网络（如本次 macOS 环境）**超时**，
   于是"按文档跑了 fetch"也可能拿不到。
4. 运行期告警只探测模块，资产缺失不在告警里，于是故障表现为"场景 0 分"而不是"环境没装好"。

## 3. 修复

- **pip 依赖提为核心依赖**：`nltk`/`langdetect`/`absl-py`/`immutabledict`/`jieba`/`rouge` 移入
  `project.dependencies`；`eval` 保留为空 extra 以兼容旧命令。
- **官方 checker 内置**：`src/modelbench/domain/judges/_vendor/instruction_following_eval/`
  （6 文件、96 KB、Apache-2.0，`__init__.py` 使其成为真包）。`checker_root()` 优先
  `dataset/raw/ifeval`（可固定版本），否则用内置副本；并要求 `evaluation_lib.py` 真实存在，
  避免半成品目录蒙混过关。
- **英文 punkt 内置**：checker 的句数约束只加载 `tokenizers/punkt/english.pickle`，词切分走
  `RegexpTokenizer(r"\w+")`。因此只内置英文子集
  （`_vendor/nltk_data/`，`english.pickle` 423 KB + `punkt_tab/english/` 244 KB），
  `_ensure_nltk_data()` 依次尝试 `dataset/raw/nltk_data` → 内置子集 → NLTK 默认路径。
  **安装后完全离线可判分**，不再依赖 GitHub 可达性。
- **preflight 检查资产**：新增 `judges.missing_assets()`；accuracy run 的启动告警与
  `validate` 预检同时报告缺失模块与缺失资产，warning 里给出修复命令；
  `judge_dependencies` 增加 `missing_assets` 字段（writer 原样持久化，报告同步展示）。
- **避免再静默**：`fetch.py` 仍保留（用于重建测试集 / 固定版本覆盖），但不再是判题前置条件。

## 4. 验证

| 项 | 结果 |
|---|---|
| `pytest test/test_judges.py -q` | 40 passed（含新增：内置 checker 存在、缺资产可被探测、契约入参） |
| `tool/system/validate_judges.py` | `instruction_following 250/250 checker_resolves bad=0`；`long_context`/`search_agent` gold_f1=1.0 bad=0（中文判题已可用）；coding 段因本机缺 `pyarrow` 未跑（数据集 extra，与本次改动无关） |
| 真实 testset 抽样判题 | ifeval 40 条 `judge_error=0`；`f1_zh`/`rouge_zh` 100 条 gold 自评 1.0、失败 0 |
| `./modelbench validate --offline` | `validation passed`，无判题缺依赖/缺资产告警 |
| wheel | 0.66 MB，包含 12 个 `_vendor` 条目（checker + punkt）；解包后 `PYTHONPATH` 指向 wheel 目录即可离线判分（`number_sentences` 用例得 1.0） |

## 5. 复现与后续

```bash
# 复现旧故障（在缺依赖/缺资产的环境）
.venv/bin/python -c "from modelbench.domain.judges import missing_assets; print(missing_assets(['ifeval']))"

# 现在：普通安装即可
./modelbench setup
.venv/bin/python tool/system/validate_judges.py     # 需 datasets extra 才能跑 coding 段
./modelbench run --method accuracy --models minicpm-v-4.7-1b
```

- 旧 bundle `20261119T020415Z-06f2` 的两个 0 分场景需**重跑**后再引用；同一 testset 下
  `instruction_following` 预期回到 ~0.7 量级、`long_context` zh 不再恒为 0。
- 升级注意：已装过的 `.venv` 重跑 `./modelbench setup`（或 `pip install -e .`）即获得核心依赖；
  内置资产随包更新，无需再跑 fetch。
