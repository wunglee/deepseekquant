# 项目分层计划（全局）

> 目标：将根路径 `core_bak` 的历史代码，先重构到 `core_bak_refactored`（保持安全、可测的过渡形态），最终融合到根路径 `core/`，与实际架构同构，形成统一、可维护的系统。

---

## 迁移总路线（三阶段）

1) Phase 1：从 `core_bak` → `core_bak_refactored`
- 拆分与重构：按架构分层（share / infrastructure / services / modules）
- 建立最小可用实现（MVP）与测试覆盖
- 逐模块完成职责边界澄清与依赖减耦

2) Phase 2：从 `core_bak_refactored` → `core/`
- 与实际架构融合：接口/协议对齐、命名规范统一
- 迁移路径与契约稳定性验证（不破坏已存在 `core/` 结构）
- 全局集成测试与跨模块联调

3) Phase 3：发布与运维
- 在融合完成后，进行全局发布演练（灰度→全量）
- 建立生产监控基线与回滚方案

---

## 分层架构同构目标（核心域：risk）

- L1 数据模型层：`risk_models.py`（数据结构、枚举、序列化与校验）
- L2 基础设施层：数学与工具、外部接口适配，例如 `infrastructure/*`, 货币转换、统计计算
- L3 共享配置层：`share/market_config.py` 等，市场配置与共享概念
- L4 服务层：`risk_metrics_service.py`（风险指标统一计算服务）
- L5 分析器层：`stress_testing.py`、`portfolio_risk.py`、`position_risk.py`
- L6 协调器层：`risk_calculator.py`（组装与协调调用）
- L7 管理器层：`risk_limits*.py`（限额与策略管理）
- L8 监控与处理层：`risk_monitor.py`、`processors/*`

说明：最终目标是在 `core/` 路径下达到以上分层的同构与职责稳定。

---

## 全局文件盘点（core_bak 当前状态）

- `core_bak/risk_manager.py`：正在按上述分层拆分至 `core_bak_refactored/core/risk/*`（进行中）
- `core_bak/signal_engine.py`：未迁移（待规划 Phase 1 拆分）
- `core_bak/execution_engine.py`：未迁移（待规划 Phase 1 拆分）
- `core_bak/portfolio_manager.py`：未迁移（待规划 Phase 1 拆分）
- `core_bak/data_fetcher.py`：未迁移（待规划 Phase 1 拆分）
- `core_bak/bayesian_optimizer.py`：未迁移（待规划 Phase 1 拆分）
- `core_bak/main.py`：未迁移（待规划 Phase 1 拆分）

结论：整体尚处于第一步，仅风险域开始拆分，远未到可发布状态。

---

## 当前阶段目标与任务（Risk 域）

- 目标：将 `core_bak/risk_manager.py` 完整拆分到 `core_bak_refactored/core/risk/*`，并补齐对应层的边界测试与依赖模拟。

- 拆分映射（进行中）：
  - `risk_models.py`（L1）→ 定义与校验补齐（待评审）
  - `infrastructure/*`（L2）→ 统计计算与货币转换（已部分实现）
  - `share/market_config.py`（L3）→ 市场配置与校验（已完成）
  - `risk_metrics_service.py`（L4）→ 服务层计算（已完成）
  - `stress_testing.py`、`portfolio_risk.py`、`position_risk.py`（L5）→ 压力/组合/持仓分析器（部分完成）
  - `risk_calculator.py`（L6）→ 协调器（基础实现，待评审）
  - `risk_limits*.py`（L7）→ 管理器（已实现，待验证）
  - `risk_monitor.py`、`processors/*`（L8）→ 监控与处理（待评审）

- 验证策略：
  - 单模块单元测试 + 跨层联调测试
  - 临时依赖使用可控模拟（stub/mock），在 Phase 2 替换为真实实现

---

## 里程碑定义（面向全局迁移）

- M1：Risk 域 Phase 1 拆分完成（含测试）
- M2：Risk 域 Phase 2 融合到 `core/` 完成（含集成测试）
- M3：Signal/Exec/Portfolio/Data 域完成 Phase 1 拆分
- M4：上述域融合到 `core/` 完成（含集成测试）
- M5：全局集成测试通过，进入发布演练

---

## 发布策略（与迁移耦合）

- 当前状态：暂停发布（未完成全局融合至 `core/`）
- 解锁条件：完成 Risk 域 M2 + 至少一个其他域的 M2，并通过全局集成测试

---

## 下一步（近期 1-2 周)

1) 完成 Risk 域 Phase 1 的剩余拆分与评审（`risk_models.py`、`risk_calculator.py`、`position_risk.py`、`portfolio_risk.py`）
2) 准备 Risk 域 Phase 2 融合（接口稳定性检查、命名与契约统一）
3) 启动 Signal/Exec/Portfolio/Data 的 Phase 1 拆分规划与骨架搭建

## 模块TODO覆盖清单（与TODO_INDEX.md对齐）

- 已存在模块级TODO：
  - infrastructure/TODO.md
  - core/risk/TODO.md
  - core/share/TODO.md
  - core/signal/TODO.md
  - core/strategy/TODO.md
- 待创建模块级TODO：
  - core/exec/TODO.md
  - core/portfolio/TODO.md
  - core/data/TODO.md
  - core/optimization/TODO.md

说明：Share（L3）与Infrastructure（L2）的补充、新增与重构遵循“即时实现规则”，优先提供最小可用实现并配套测试，不长期搁置为TODO。
# 项目分层计划（全局）

> 目标：将根路径 `core_bak` 的历史代码，先重构到 `core_bak_refactored`（保持安全、可测的过渡形态），最终融合到根路径 `core/`，与实际架构同构，形成统一、可维护的系统。

---

## 迁移总路线（三阶段）

1) Phase 1：从 `core_bak` → `core_bak_refactored`
- 拆分与重构：按架构分层（share / infrastructure / services / modules）
- 建立最小可用实现（MVP）与测试覆盖
- 逐模块完成职责边界澄清与依赖减耦

2) Phase 2：从 `core_bak_refactored` → `core/`
- 与实际架构融合：接口/协议对齐、命名规范统一
- 迁移路径与契约稳定性验证（不破坏已存在 `core/` 结构）
- 全局集成测试与跨模块联调

3) Phase 3：发布与运维
- 在融合完成后，进行全局发布演练（灰度→全量）
- 建立生产监控基线与回滚方案

---

## 分层架构同构目标（核心域：risk）

- L1 数据模型层：`risk_models.py`（数据结构、枚举、序列化与校验）
- L2 基础设施层：数学与工具、外部接口适配，例如 `infrastructure/*`, 货币转换、统计计算
- L3 共享配置层：`share/market_config.py` 等，市场配置与共享概念
- L4 服务层：`risk_metrics_service.py`（风险指标统一计算服务）
- L5 分析器层：`stress_testing.py`、`portfolio_risk.py`、`position_risk.py`
- L6 协调器层：`risk_calculator.py`（组装与协调调用）
- L7 管理器层：`risk_limits*.py`（限额与策略管理）
- L8 监控与处理层：`risk_monitor.py`、`processors/*`

说明：最终目标是在 `core/` 路径下达到以上分层的同构与职责稳定。

---

## 全局文件盘点（core_bak 当前状态）

- `core_bak/risk_manager.py`：正在按上述分层拆分至 `core_bak_refactored/core/risk/*`（进行中）
- `core_bak/signal_engine.py`：未迁移（待规划 Phase 1 拆分）
- `core_bak/execution_engine.py`：未迁移（待规划 Phase 1 拆分）
- `core_bak/portfolio_manager.py`：未迁移（待规划 Phase 1 拆分）
- `core_bak/data_fetcher.py`：未迁移（待规划 Phase 1 拆分）
- `core_bak/bayesian_optimizer.py`：未迁移（待规划 Phase 1 拆分）
- `core_bak/main.py`：未迁移（待规划 Phase 1 拆分）

结论：整体尚处于第一步，仅风险域开始拆分，远未到可发布状态。

---

## 当前阶段目标与任务（Risk 域）

- 目标：将 `core_bak/risk_manager.py` 完整拆分到 `core_bak_refactored/core/risk/*`，并补齐对应层的边界测试与依赖模拟。

- 拆分映射（进行中）：
  - `risk_models.py`（L1）→ 定义与校验补齐（待评审）
  - `infrastructure/*`（L2）→ 统计计算与货币转换（已部分实现）
  - `share/market_config.py`（L3）→ 市场配置与校验（已完成）
  - `risk_metrics_service.py`（L4）→ 服务层计算（已完成）
  - `stress_testing.py`、`portfolio_risk.py`、`position_risk.py`（L5）→ 压力/组合/持仓分析器（部分完成）
  - `risk_calculator.py`（L6）→ 协调器（基础实现，待评审）
  - `risk_limits*.py`（L7）→ 管理器（已实现，待验证）
  - `risk_monitor.py`、`processors/*`（L8）→ 监控与处理（待评审）

- 验证策略：
  - 单模块单元测试 + 跨层联调测试
  - 临时依赖使用可控模拟（stub/mock），在 Phase 2 替换为真实实现

---

## 里程碑定义（面向全局迁移）

- M1：Risk 域 Phase 1 拆分完成（含测试）
- M2：Risk 域 Phase 2 融合到 `core/` 完成（含集成测试）
- M3：Signal/Exec/Portfolio/Data 域完成 Phase 1 拆分
- M4：上述域融合到 `core/` 完成（含集成测试）
- M5：全局集成测试通过，进入发布演练

---

## 发布策略（与迁移耦合）

- 当前状态：暂停发布（未完成全局融合至 `core/`）
- 解锁条件：完成 Risk 域 M2 + 至少一个其他域的 M2，并通过全局集成测试

---

## 下一步（近期 1-2 周）

1) 完成 Risk 域 Phase 1 的剩余拆分与评审（`risk_models.py`、`risk_calculator.py`、`position_risk.py`、`portfolio_risk.py`）
2) 准备 Risk 域 Phase 2 融合（接口稳定性检查、命名与契约统一）
3) 启动 Signal/Exec/Portfolio/Data 的 Phase 1 拆分规划与骨架搭建
