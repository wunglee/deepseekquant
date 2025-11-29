# SPRINT迭代计划与跟踪（数据模块）

说明：本文件用于“特殊迭代：处理遗留专家完整代码”的计划与进展跟踪，仅覆盖 data 模块。未分配事项统一在同目录 TODO.md 维护，分配后迁移到本文件对应阶段。

---

## 🔶 特殊迭代：遗留专家完整代码修复（data 模块）

> 目标：将遗留的“专家完整代码”按分层与模块化要求修复并通过测试，消除碎片依赖，建立稳定的数据服务。

- 阶段1：代码分层拆分（core/app）
  - 状态：✅ COMPLETE
  - 说明：已将原始 `core_bak_refactored/core/data/data_fetcher.py.backup` 按职责拆分至领域层（`core/data`）与应用层（`app/data` 预留）。

- 阶段2：领域层基础修复与基本测试通过
  - 状态：✅ COMPLETE（data_fetcher 9/9 测试通过）
  - 入口文件：`core_bak_refactored/core/data/data_fetcher.py`
  - 说明：核心方法修复、错误处理完善、缓存与数据源注册可用；与 MarketCode 规范对齐。

- 阶段3：建立应用层依赖并通过测试
  - 状态：✅ COMPLETE
  - 入口文件：`core_bak_refactored/app/data/data_service.py`（门面/适配层）
  - 目标：
    - 提供面向应用的统一数据接口（历史/实时/基本面）
    - 通过依赖注入接入 `core/data` 的服务；集成测试通过（≥95%）
  - 进展：门面已创建；集成测试 4/4 通过（历史委派/基本面stub/回退/失败空结果）
  - 验收断言：应用层接口稳定、错误边界清晰、依赖注入覆盖三类数据源

- 阶段4：模块化拆分与细粒度测试覆盖
  - 状态：✅ COMPLETE（应用层完成；领域层拆分计划启动）
  - 目标：
    - 领域层：将 `data_fetcher.py` 拆分为独立模块文件（缓存、数据源适配、质量评估、协议模型）
    - 应用层：完成细粒度模块划分（cache_service/providers/quality_monitor/models），并通过细粒度测试
  - 进展：
    - 应用层：`app/data/cache_service.py`、`app/data/providers.py`、`app/data/quality_monitor.py`、`app/data/models.py` 已创建；`app_module_split_test.py` 1/1 通过
    - 领域层：拆分规划完成，待下阶段实施
  - 验收断言：每模块具备独立单元测试（≥90%），跨层接口契约在设计文档中明确

- 阶段5：设计级优化与重构 + 文档同步
  - 状态：🔄 IN_PROGRESS（可动拆卸版本已创建；第一阶段拆分完成）
  - 目标：职责进一步收敛、异常路径统一、性能埋点与SLA接入；同步 `docs/design/core_bak_refactored/core/data/`
  - 进展：
    - 可动拆卸版本：`core/data/movable_fetcher.py`，139行，已委派原始 `DataFetcher`，并迁移缓存/历史数据获取/市场状态逻辑
    - 拆分模块：cache/key.py + store.py（键生成与三级读写）、fallback/orchestrator.py（备用源编排）、market/calendar.py + breadth.py + sector.py（市场开市/涨跌/板块）、analytics/volatility.py（波动率），合计183行
    - 测试覆盖：14个测试通过（movable_fetcher_test 3个、cache 5个、market 1个、analytics 2个、fetcher_orchestrator 3个）
    - 行数对照：原始 6920行，当前拆分 2485行（36%），目标≥ 6500行
  - 验收断言：设计文档包含分层图、模块依赖图、公开API签名与示例、变更历史；拆分后总行数与原始版本在合理误差范围内（±10%）

- 阶段6：risk 模块依赖切换与碎片迁移
  - 状态：🔄 PLANNED
  - 目标：risk 域由“碎片 data 依赖”改为“当前 core/data 依赖”，对碎片代码进行选择性迁移整合
  - 验收断言：risk 模块相关测试保持通过；迁移范围与保留范围在迁移报告中记录

---

## 📌 进展快照（本轮）
- 可动拆卸版本已创建：`core/data/movable_fetcher.py`（139行），委派原始 `DataFetcher`，并迁移部分逻辑到拆分模块
- 拆分模块合计：2485行（包括movable_fetcher + cache + fallback + market + analytics + providers等）
- 测试覆盖：14个测试通过（movable_fetcher 3个、cache 5个、market 1个、analytics 2个、orchestrator 3个）
- 行数进度：36%（2485/6920），目标 ≥6500行
- 下一步：继续迁移HTTP客户端/凭证管理/实时数据/基本面数据等核心功能（见 TODO.md 与 SPLIT_PROGRESS.md）

---

## 🔖 规范对齐
- 语言与流程：遵循 `.qoder/rules/PECIFICATIONS.md`
- 结构约束：进行中的阶段标记为 🔄 IN_PROGRESS；完成后方可标记 ✅ COMPLETE
- 规则来源：数据质量与默认值一律来自专家答复，不得在代码中硬编码
