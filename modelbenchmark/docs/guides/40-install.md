# 安装与依赖（macOS / Windows）

面向**新机器**：把仓库 clone 下来后，按本文建好 Python 环境并完成自检。
本文只讲安装、依赖与验证；使用方法见 [41-user-guide.md](41-user-guide.md)，
数据集见 [dataset/README.md](../../dataset/README.md)，文档全貌见 [README.md](../README.md)。

> 约定：`<repo>` 指**仓库根目录**（clone 下来的目录）。本文用
> `/path/to/modelbenchmark`（macOS/Linux）和 `D:\path\to\modelbenchmark`
> （Windows）作为占位路径，请替换成你的实际路径。
> **路径含空格或中文时必须加引号**，例如 `cd "/path/to/我的 项目/modelbenchmark"`；
> 若 `pip`/`venv` 在中文或含空格路径下报错，把仓库放到纯 ASCII、无空格的路径重建 `.venv`。

## 1. 支持矩阵

| 平台 | 启动器 | 环境安装 / 离线工具链 | live benchmark |
|---|---|---|---|
| macOS（Apple Silicon） | `./modelbench setup` 等 | 支持 | 支持（本机 oMLX 桌面应用） |
| Windows 10/11（x64） | `.\modelbench.cmd setup` 等；PowerShell 还可用 `.\modelbench.ps1` | 支持 | 需**可达的 oMLX 实例**（oMLX 目前是 macOS 桌面应用，Windows 上不本机运行服务） |
| Linux | `./modelbench setup` 等 | 支持 | 需可达的 oMLX 实例 |

要点：

- 本仓库**没有** `[project.scripts]` console script，唯一入口是仓库根目录的启动器：
  macOS/Linux 用 `./modelbench`，Windows 用 `.\modelbench.cmd`（或 `.\modelbench.ps1`）。
  不要期待 `modelbench` 成为一个全局命令。
- `.venv/` 只是启动器创建的项目内虚拟环境，**不可跨目录/跨机器复制**
  （`pyvenv.cfg`、shebang 和解释器链接都是绝对路径）。
- 仓库不随 clone 分发数据产物与运行结果（`dataset/raw/`、`dataset/.cache/`、
  `dataset/test/**/testset.jsonl`、`output/`），到新机器后需按
  [dataset/README.md](../../dataset/README.md) 与第 10 节准备。

## 2. 前置条件

| 项 | 要求 | 说明 |
|---|---|---|
| Python | `>=3.10`，**推荐 3.12** | 来自 `pyproject.toml` 的 `requires-python`；`setup` 默认按 3.12 建 venv |
| git | 任意较新版本 | clone 仓库、拉取更新 |
| uv（可选） | 任意较新版本 | 已安装则 `setup` 用 uv 快速创建 `.venv`；依赖安装统一由 venv 内的 pip 经 `tool/bootstrap.py` 完成 |
| make（可选，仅 macOS/Linux） | 任意版本 | 只是开发者快捷封装，**不是必需**；Windows 无需 make（等价命令见第 9 节） |
| oMLX（仅 live 需要） | 可达的 oMLX 服务 | OpenAI 兼容接口 + admin API；默认 `http://127.0.0.1:8000` |

确认 Python（在终端里执行）：

```bash
# macOS / Linux
python3.12 --version
```

```bat
REM Windows：py 启动器
py -3.12 --version
```

若没有 3.12：macOS 用 `brew install python@3.12`，Windows 用 python.org 安装包并勾选
“Add python.exe to PATH”。`setup` 会在 `python3.12` → `python3` → `python` 中依次查找，
并用 `tool/bootstrap.py` 校验版本（**下限 3.10**）。显式指定解释器：

```bash
# macOS / Linux：用已有解释器（示例 3.11）
PYTHON=python3.11 ./modelbench setup

# 换 Python 版本（同时用于 uv 建 venv）
PYTHON_VERSION=3.11 ./modelbench setup
```

```bat
REM Windows：指定解释器（也可先装好 3.12 让 py -3.12 可用）
set MB_PYTHON=C:\path\to\python.exe
.\modelbench.cmd setup
```

> Windows 上 `setup` 依次尝试 `MB_PYTHON` → `py -3.12` → `python`。请确保其中一个
> 能输出 3.10+；否则先在 python.org 安装 3.12。PowerShell 里指定解释器用
> `$env:MB_PYTHON = "C:\path\to\python.exe"`。

## 3. 安装路径 A：一键 setup（推荐）

### macOS / Linux

```bash
cd /path/to/modelbenchmark
./modelbench setup
./modelbench validate --offline
```

### Windows（cmd）

```bat
cd /d D:\path\to\modelbenchmark
.\modelbench.cmd setup
.\modelbench.cmd validate --offline
```

### Windows（PowerShell）

```powershell
cd D:\path\to\modelbenchmark
.\modelbench.ps1 setup
.\modelbench.ps1 validate --offline
```

PowerShell 报“无法加载文件 … 未对文件进行数字签名 / 禁止运行脚本”时，为**当前窗口**
放行一次执行策略，或改用 `.cmd`：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\modelbench.ps1 setup
```

`setup` 会做四件事：

1. 选定 Python（默认 3.12，校验 ≥3.10），在 `<repo>/.venv` 建虚拟环境；macOS/Linux 有 uv
   时用 uv 创建，否则 `python -m venv`（Windows 启动器不依赖 uv，用 `py -3.12`/`python`）；
2. 通过 `tool/bootstrap.py` 在 venv 内以可编辑模式安装依赖：`pip install -e ".[dev]"`；
3. setuptools 构建临时文件由 `setup.cfg` 指向 `output/build`，源码目录保持干净；
4. 打印下一步命令（`./modelbench web`、`./modelbench run …`）。

macOS/Linux 的 `setup` 默认使用清华 PyPI 镜像；**Windows 默认走官方 PyPI**（可用
`PIP_INDEX_URL` 或 `UV_DEFAULT_INDEX` 设置镜像，三个平台的启动器都支持）。换源/代理见第 5 节。
`.venv/` 是隐藏目录（macOS Finder 用 `Cmd+Shift+.` 显示），已被 `.gitignore` 忽略，不要提交。

## 4. 安装路径 B：手动创建 venv

不使用一键 `setup`、或需要自己控制解释器/依赖时：

### macOS / Linux

```bash
cd /path/to/modelbenchmark
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

### Windows（cmd；PowerShell 把 `set X=Y` 换成 `$env:X = "Y"`）

```bat
cd /d D:\path\to\modelbenchmark
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

手动建好 `.venv` 后，启动器同样可以直接用（它自动寻找 `.venv` 内的解释器）：

```bash
./modelbench --version
```

```bat
.\modelbench.cmd --version
```

若手动环境没有 pip（少见，通常是 `--without-pip` 建的），先补装：

```bash
.venv/bin/python -m ensurepip --upgrade
```

> 等价于直接调用 CLI 模块：`.venv/bin/python -m modelbench.interfaces.cli --version`；
> Windows 为 `.venv\Scripts\python.exe -m modelbench.interfaces.cli --version`。

## 5. 镜像与代理

| 变量 | 作用 |
|---|---|
| `UV_DEFAULT_INDEX` | `setup` 的包索引环境变量（未设置 `PIP_INDEX_URL` 时生效，三个平台启动器都传给 venv 内 pip）；macOS/Linux 默认清华镜像，Windows 默认官方 PyPI |
| `PIP_INDEX_URL` | pip 的包索引地址；手动 venv 路径使用 |
| `PIP_TRUSTED_HOST` | 信任的自建/内网镜像域名（HTTPS 证书非公网 CA 时） |
| `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` | 出网代理；uv 与 pip 都遵循 |

macOS / Linux：

```bash
# 镜像不可用时换回官方源
UV_DEFAULT_INDEX=https://pypi.org/simple ./modelbench setup

# 手动 venv：给 pip 指定镜像（清华）与内网自签镜像
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  .venv/bin/pip install -e ".[dev]"
PIP_INDEX_URL=https://mirrors.example.com/simple \
  PIP_TRUSTED_HOST=mirrors.example.com \
  .venv/bin/pip install -e ".[dev]"

# 走 HTTP(S) 代理
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
```

Windows（cmd）：

```bat
set UV_DEFAULT_INDEX=https://pypi.org/simple
.\modelbench.cmd setup
set PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
set PIP_TRUSTED_HOST=mirrors.example.com
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Windows（PowerShell）：

```powershell
$env:UV_DEFAULT_INDEX = "https://pypi.org/simple"
$env:PIP_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
$env:PIP_TRUSTED_HOST = "mirrors.example.com"
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
```

> 以上变量只对**当前终端会话**生效。需要长期生效时写进 shell 配置（`~/.zshrc`、`~/.bashrc`）
> 或 Windows 的“系统属性 → 环境变量”。
> `PIP_TRUSTED_HOST` 只用于你信任的内网/自建镜像，不要对公网源关闭证书校验。

## 6. 可选依赖（extras）

| extra | 包含 | 何时需要 |
|---|---|---|
| `dev` | `pytest`、`pytest-timeout` | 跑测试/门禁；`setup` 默认安装 |
| `datasets` | `pyarrow`、`tokenizers`、`jinja2` | **仅数据集构建**（`tool/dataset`）需要；`pyarrow` 读 parquet，`tokenizers`+`jinja2` 用于无 oMLX.app 时的便携 tokenizer。日常 `validate`/`web`/`run` 不需要 |
| `eval` | `nltk`、`langdetect`、`absl-py`、`immutabledict`、`jieba`、`rouge` | **仅精度判题/自检**（`tool/system`）需要：官方 IFEval checker 依赖前四个；LongBench 中文 QA 的 ROUGE 指标需要 `jieba` 与 `rouge`。构造/校验测试集本身不需要 |

手动 venv 下追加安装：

```bash
.venv/bin/pip install -e ".[dev,datasets,eval]"
```

```bat
.venv\Scripts\python.exe -m pip install -e ".[dev,datasets,eval]"
```

用 uv 管理的环境等价写法：

```bash
uv pip install --python .venv/bin/python -e ".[dev,datasets,eval]"
```

## 7. 安装验证

按顺序执行，全部通过即安装成功：

### macOS / Linux

```bash
cd /path/to/modelbenchmark

./modelbench --version                 # 入口可用，打印版本号
./modelbench validate --offline        # 配置/测试集/数据集解析
.venv/bin/python -m pytest test/ -q    # DT 测试
./modelbench web --no-open             # http://127.0.0.1:8765，Ctrl+C 停止
```

### Windows（cmd）

```bat
cd /d D:\path\to\modelbenchmark
.\modelbench.cmd --version
.\modelbench.cmd validate --offline
.venv\Scripts\python.exe -m pytest test/ -q
.\modelbench.cmd web --no-open
```

各步骤说明：

- `--version`：只验证启动器与 `.venv` 就位。
- `validate --offline`：不探测 oMLX；新机器尚未准备数据时加 `--allow-missing`
  （`dataset/raw/` 为空属于预期，见第 10 节）。
- `pytest test/ -q`：单元 + fake-server 集成测试，不需要真实 oMLX。
- `web --no-open`：启动本地控制台（只监听 127.0.0.1、无鉴权），浏览器打开
  `http://127.0.0.1:8765`；`--port 8766` 可换端口。
- 完整门禁：`.venv/bin/python -m harness.gate`（提交前必须 PASS）。
- 在线自检（需 oMLX 已启动且 key 正确）：`./modelbench validate`。

## 8. 运行前提：oMLX

- live benchmark（`./modelbench run …`）必须有**可达的 oMLX 服务**：OpenAI 兼容接口
  负责推理，admin API 负责清缓存与模型加载状态。
- oMLX 目前是 **macOS 桌面应用**：在 Mac 上安装并启动即可；Windows/Linux 上需要把
  `server.base_url` 指向一台网络可达的 oMLX 实例。
- 服务地址与 key 的配置位置：
  - `server.base_url`、`server.api_key`：`src/modelbench/resources/config/default.yaml`；
  - API key 也可用环境变量 `OMLX_API_KEY`（优先级最高）；
  - 或自动读取 `~/.omlx/settings.json`。
- Windows 没有本机 oMLX 时，以下离线能力仍可完整使用：
  `validate --offline`、`tool/dataset` 数据集构建、`web --no-open`（浏览/查看历史）以及
  `runs`、`analyze`、`export-analysis`、`index` 等结果工具。
- 单机内存不足、oMLX 内存守卫（HTTP 507）等运行期问题见
  [41-user-guide.md](41-user-guide.md) 的故障排查章节。

## 9. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `python3.12: command not found` | 安装 Python 3.12；或用 `PYTHON=python3.11 ./modelbench setup` / `PYTHON_VERSION=3.11 ./modelbench setup`；Windows 确认 `py -3.12 --version` 可用，或用 `set MB_PYTHON=C:\path\to\python.exe` 指定解释器 |
| `SSL: CERTIFICATE_VERIFY_FAILED` / `SSLError` | 代理或内网镜像证书不被信任：设置 `PIP_TRUSTED_HOST=镜像域名`；或临时换官方源 `UV_DEFAULT_INDEX=https://pypi.org/simple`；macOS 用 python.org 安装包时先运行安装包自带的 `Install Certificates.command`；并检查系统时间 |
| 下载超时 / 镜像 5xx | 换索引（第 5 节）；确认代理环境变量 `HTTPS_PROXY` 正确；公司网络需要 `NO_PROXY` 排除内网域名 |
| 路径含空格/中文导致安装或运行报错 | 命令里给路径加引号；仍失败就把仓库 clone 到纯 ASCII、无空格路径，**删除 `.venv` 后重建**（不要复制 `.venv`） |
| PowerShell 拒绝执行 `modelbench.ps1` | `Set-ExecutionPolicy -Scope Process Bypass`（仅当前窗口），或改用 `.\modelbench.cmd …` |
| 提示 environment not ready（要求先 run setup） | `<repo>/.venv` 不存在或建在了别处；在仓库根目录重跑 `./modelbench setup`（Windows `.\modelbench.cmd setup`） |
| 没有安装 make | make 只是封装，不是必需：`make setup` 等价 `./modelbench setup`；`make test` 等价 `.venv/bin/python -m pytest test/`；`make docs` 等价 `.venv/bin/python -m harness.gate --only docs_links,docs_contract`；Windows 直接用 `.\modelbench.cmd …` |
| `validate` 报 oMLX 不可达（exit 3） | oMLX 未启动或地址不对；先启动 oMLX 桌面应用，或先跑 `./modelbench validate --offline`；Windows 需把 `server.base_url` 指向可达实例 |
| `401 / invalid key` | 设置 `OMLX_API_KEY`，或检查 `src/modelbench/resources/config/default.yaml` 的 `server.api_key` / `~/.omlx/settings.json` |
| 从别的机器/目录拷来 `.venv` 后各种报错 | `.venv` 含绝对路径，禁止迁移；删除后重建（第 11 节） |
| pip 权限错误 / 装进了系统 Python | 不要用 `sudo pip`，也不要 `pip install --user`；统一用 `.venv/bin/pip`（Windows `.venv\Scripts\python.exe -m pip`），确认目标是 `<repo>/.venv` |
| 忘记激活 venv / 不知道用哪个 python | 本项目不需要 `activate`：一律用根启动器（`./modelbench` / `.\modelbench.cmd`）或 `.venv` 内的绝对路径解释器 |
| `No module named pip` | 手动 venv 未带 pip：`.venv/bin/python -m ensurepip --upgrade`；或直接重跑 `./modelbench setup` |

## 10. 数据准备（不影响安装成功）

clone 只包含源码、配置、测试与文档；数据与结果需在目标机器自行准备：

- `dataset/raw/`（上游原始文件，只读，已被 `.gitignore` 忽略）：完整步骤见
  [dataset/README.md](../../dataset/README.md)。
  拉取/校验脚本是 `tool/dataset/performance/fetch.py` 与
  `tool/dataset/accuracy/fetch.py`（断点续传 + SHA-256），只读校验用 `--verify`。
- 测试集由构造器生成：`tool/dataset/performance/build.py`（240 条）与
  `tool/dataset/accuracy/build.py`（2,000 行隐藏精度资产）；只读校验用各自的 `--check`。
- **Windows / 无 oMLX.app 的机器也能重建测试集**：安装 `.[dev,datasets]` 后，
  两个 `fetch.py` 会下载通用参考 tokenizer（**cl100k_base**，性能/精度两侧共用
  同一资产：官方 blob + `derive_cl100k.py` 确定性派生的 `tokenizer.json`，
  共约 4.4 MB），无需 oMLX API、不依赖任何本地模型。
- **不要**用 `./modelbench download` 初始化 `dataset/raw`：它是旧数据层的兼容下载入口，
  与当前 `dataset/` 布局不一致。
- 数据未就绪时用 `./modelbench validate --offline --allow-missing` 先确认环境本身没问题。

## 11. 重建与卸载

环境损坏、换了 Python 版本，或想彻底重置依赖时，**只删除项目内 `.venv`** 再安装：

```bash
# macOS / Linux
rm -rf .venv
./modelbench setup
./modelbench validate --offline
```

```bat
REM Windows
rmdir /s /q .venv
.\modelbench.cmd setup
.\modelbench.cmd validate --offline
```

- 删除 `.venv` 不会影响 `dataset/`、`output/`、配置与文档。
- uv 缓存（`.uv-cache/`）可删，删除后下次安装会重新下载。
- 不建议手工编辑 `.venv` 内的包；需要换依赖版本就删环境重装。

## 12. 附：make 目标与启动器对照（macOS/Linux）

Windows 没有 make，直接用 `.\modelbench.cmd <子命令>` 等价命令。

| make 目标 | 等价命令 |
|---|---|
| `make setup` | `./modelbench setup` |
| `make check` / `make gate` | `.venv/bin/python -m harness.gate` |
| `make test` | `.venv/bin/python -m pytest test/` |
| `make docs` | `.venv/bin/python -m harness.gate --only docs_links,docs_contract` |
| `make docs-write` | `.venv/bin/python -m harness.gen_structure --write` |
