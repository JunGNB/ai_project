# docs/records/ — 时间性记录

> **状态**：records（非规范）｜ **位置**：`docs/records/` ｜ **索引**：本页

本目录存放**时间性、不可作为规范引用**的记录：评审结论、变更/迁移记录、发布说明、
一次性分析摘要等。规范/设计请放 `docs/` 下的规范/设计/操作/流程文档（见
[docs/README.md](../README.md) 索引）。

## 规则

1. **不是证据**：测试证据只认 `output/runs/<run_id>/` 与 `output/index.json`；
   本目录与根目录 `reports/` 都不是 run 结果。
2. **规范不在这里**：任何"必须/默认/口径"的结论应写进 `docs/` 的规范文档；
   本目录只记录"何时、为什么、谁"。
3. **可归档**：过期记录可整体移入 `docs/archive/`，不影响门禁（`docs_contract` 不扫描
   `docs/archive/`）。
4. **命名**：`YYYY-MM-DD-<topic>.md`；文件头写日期、关联任务/run_id、结论一句话。

## 现状

- 2026-11-09/10 的 P0/P1 文档评审与整改记录当前在根目录 `reports/`（`reports/README.md`
  为索引）。其中的**长期结论**（文档规范、口径、目录规划）已并入 `docs/`；`reports/`
  中的过程记录可随时归档/删除，不算证据。
- 迁移记录（schema 2.x → 3.0）见 [31-output-contract.md](../design/31-output-contract.md) §10。
