# TODO（数据模块）

说明：本文件维护“特殊迭代：遗留专家完整代码修复”的未分配/待实施事项；一旦分配到具体阶段，迁移至同目录 SPRINT.md 对应节点。

---

## 🧭 模块导航
1. [应用层依赖建立与测试](#应用层依赖建立与测试)
2. [模块化拆分与细粒度测试覆盖](#模块化拆分与细粒度测试覆盖)
3. [设计级优化与重构 + 文档同步](#设计级优化与重构--文档同步)
4. [risk 依赖切换与碎片迁移](#risk-依赖切换与碎片迁移)

---

## 应用层依赖建立与测试（阶段3）
- [COMPLETE] 建立应用层门面/适配层（已迁移至 SPRINT 阶段3）
  - 路径：`core_bak_refactored/app/data/data_service.py`（已创建）
  - 目标：统一历史/实时/基本面数据接口，依赖注入接入 `core/data`
  - 验收：集成测试≥95%通过；错误边界清晰；接口契约文档化
- [COMPLETE] 集成测试方案（4/4 用例通过）
  - 内容：模拟成功/失败/回退链路（三类数据源：自定义 + stub）
  - 交付：`core_bak_refactored/tests/app/data/data_service_integration_test.py` + `data_service_fallback_test.py`

## 模块化拆分与细粒度测试覆盖（阶段4）
- [COMPLETE] 应用层模块划分
  - 交付：`app/data/cache_service.py`、`app/data/providers.py`、`app/data/quality_monitor.py`、`app/data/models.py`
  - 测试：`tests/app/data/app_module_split_test.py`（1/1 通过）
- [PLANNED] 领域层拆分计划与实施
  - 目标：从 `core/data/data_fetcher.py` 提取子模块（cache/providers/quality/models）
  - 验收：每子模块≥90%覆盖率；接口契约文档化（设计文档更新）

## 设计级优化与重构 + 文档同步（阶段5）
- [PENDING] 设计文档同步
  - 位置：`docs/design/core_bak_refactored/core/data/`
  - 内容：分层架构图/模块依赖图/API签名与示例/变更历史
- [PENDING] 性能埋点与SLA接入
  - 内容：为核心路径加入埋点；定义SLA阈值（专家确认）
  - 验收：p50/p95/p99采集；SLA告警策略文档化

## risk 依赖切换与碎片迁移（阶段6）
- [PENDING] 依赖切换
  - 目标：risk 域由“碎片 data 依赖”改为“当前 core/data 依赖”
  - 验收：risk 相关测试保持通过
- [PENDING] 碎片迁移策略
  - 内容：评估 `core_bak/*` 中 data 相关碎片，选择性迁移与弃用
  - 交付：迁移报告（范围/保留/弃用理由/影响评估）

---

## 规范约束（须遵循 `.qoder/rules/PECIFICATIONS.md`）
- 进行中的阶段标记为 🔄 IN_PROGRESS；完成后方可标记 ✅ COMPLETE
- 数据质量规则/阈值必须由专家答复明确，不得在代码中硬编码
- 代码变更必须同步设计文档；测试覆盖需按模块细化
