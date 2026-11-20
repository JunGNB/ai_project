# AGENT.md —— 在本仓库工作的约定

本文件写给在这个仓库里干活的 AI Agent（也给人看）。
目标：**改动可追溯、口径不漂移、数字可复现**。读完这一页再动手。

## 1. 这个项目是什么

`macgpu` 是 Mac（Apple Silicon）上的 GPU 算力与带宽基准，用 Swift + 运行时编译的 MSL 实现，
零第三方依赖。产出是 `output/results/<时间戳>/result.json` 与 `output/reports/REPORT.md`。

技术栈约束（不要引入新依赖）：

- 语言：Swift（`swiftc`，只用标准库 + Metal / MetalKit）。
- 脚本：Python 3（只用标准库）、Bash、Make。
- 内核：MSL 源码以字符串形式写在 `src/Kernels.swift`，运行时编译。

## 2. 三条不可协商的硬规则

1. **峰值内核必须写回整条向量。** 只写 `.x` 会让编译器删掉其余 3/4 的 FMA，
   历史上因此虚高 4 倍。写回 `float4` / `half4` / `int4` 整体，或用 `if (c.x == -1.0f) out[gid] = …` 之类的完整依赖。
2. **计数必须等于实际执行的乘加。** `macsPerIteration` 要和内核语句能逐条对上；
   新增内核必须同时在 `SelfCheck` 里加数值比对（CPU 参考复算），否则 `make test` 不算通过。
3. **同一指标不得有第二种口径。** 改公式/单位/线程数 = 改 spec 文档 + 改代码 + 升版本，
   三件事必须同一个提交里完成。

## 3. 目录职责

| 路径 | 放什么 | 能不能手改 |
|---|---|---|
| `src/` | Swift 源码、MSL 内核 | 是 |
| `tools/` | report.py / mcp_server.py / docs_contract.py | 是 |
| `tests/` | 冒烟测试、文档契约测试 | 是 |
| `docs/spec` | 测什么、公式、指标矩阵 | 是（改口径先改它） |
| `docs/design` | 架构、产物契约 | 是 |
| `docs/guides` | 使用说明 | 是 |
| `docs/process` | 验证与治理 | 是 |
| `docs/records` | 历史读数（只追加，不改旧记录） | 追加 |
| `output/` | 二进制、结果、报告 | **否，全部可再生，不进库** |

## 4. 新增一个测量项的标准流程

以「新增 bf16 峰值」为例，顺序不能变：

1. **spec 先行**：在 `docs/spec/10-methodology.md` 的指标矩阵加一行（kernel 名、MACs/iter、units）。
2. **内核**：在 `src/Kernels.swift` 加 `peak_mma_bf16`（或复用模板），保证写回整条向量。
3. **用例**：在 `src/Bench.swift` 的 `ComputeCases.all` 加 `ComputeCase`（含 `labelEn` / `methodEn`），
   并在 `makeBuffers` 里补 buffer 分支（注意 `itersIndex` 不能想当然写成 1）。
4. **自检**：在 `SelfCheck.peakCheck` 加对应 CPU 参考比对，并让 `main.swift` 的自检循环覆盖到它。
5. **文档**：`docs/design/21-output-contract.md` 若字段有变，同步更新；`docs/README.md` 无需改（文件没变）。
6. **验证**：`make test && make docs`，然后 `make run && make report`，数值与 `docs/records/` 基线同量级。

漏掉第 1 步或第 4 步，`make docs` / `make test` 会直接失败——这是有意设计的。

## 5. 常用命令

```bash
./macgpu run --repeat 3          # 完整测试
./macgpu selfcheck               # 数值自检（必须 PASS 才采信数字）
make report                      # 从最新结果生成报告
make docs                        # 文档/实现一致性
make test                        # 冒烟测试（含上面两项）
make clean                       # 只清二进制
```

## 6. 判断标准（Definition of Done）

一个改动算完成，需要同时满足：

- [ ] `make test` 通过
- [ ] `./macgpu selfcheck` 输出 `PASS`
- [ ] `make docs` 通过
- [ ] 报告里对应章节的数字与改动前同量级（除非本次就是修 bug，需在 `docs/records/` 追加说明）
- [ ] 没有新的第三方依赖、没有新增需要手动同步的「隐形口径」

## 7. 禁止事项

- **不要**把 `output/` 下的任何文件提交进库（已在 `.gitignore` 中忽略）。
- **不要**为了「数字好看」调整迭代次数、线程数、缓冲区大小而不改文档。
- **不要**用「结果看起来合理」代替 `selfcheck`。
- **不要**修改 `../modelbenchmark/`（另一个独立工程，有未提交的用户改动）。
- **不要**在文档里引用不存在的命令或路径；`make docs` 会检查旧名字残留。

## 8. 出问题时的排查顺序

1. `./macgpu selfcheck` 是否 PASS？不 PASS 时一切性能数字无意义。
2. `output/build/build.log` 有没有编译告警/错误。
3. `result.json` 的 `host.load_avg_1m` 是否 > 1.5（负载高会压低带宽）。
4. 内核是否被编译器重排：把内核单独拿出去，用「只改 store 写法」的对照实验验证。
