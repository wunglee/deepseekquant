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

## 里程碑定义（面向全局迁移）

- M1：Risk 域 Phase 1 拆分完成（含测试）
- M2：Risk 域 Phase 2 融合到 `core/` 完成（含集成测试）
- M3：Signal/Exec/Portfolio/Data 域完成 Phase 1 拆分
- M4：上述域融合到 `core/` 完成（含集成测试）
- M5：全局集成测试通过，进入发布演练

## 当前进展（细节进入SPRINT文件中查看）
### Phase 1：
docs/process/core_bak_refactored/core/risk/SPRINT.md（已完成）
docs/process/core_bak_refactored/core/data/SPRINT.md（进行中）
## 当前状态
### core_bak 当前状态
- `core_bak/risk_manager.py`：已迁移
- `core_bak/data_fetcher.py`：已迁移
- `core_bak/signal_engine.py`：未迁移
- `core_bak/execution_engine.py`：未迁移
- `core_bak/portfolio_manager.py`：未迁移
- `core_bak/bayesian_optimizer.py`：未迁移
- `core_bak/main.py`：未迁移
### core_bak_refactored/core 当前状态
- core_bak_refactored/core/backtest ❌未迁移
- core_bak_refactored/core/data：🟢进行中
- core_bak_refactored/core/exec：❌未迁移
- core_bak_refactored/core/monitoring：❌未迁移
- core_bak_refactored/core/optimization：🟢进行中
- core_bak_refactored/core/portfolio：❌未迁移
- core_bak_refactored/core/risk：✅已完成
- core_bak_refactored/core/share：🟢进行中
- core_bak_refactored/core/signal：❌未迁移
- core_bak_refactored/core/strategy：❌未迁移
**状态说明**：
- ✅ **已完成**：可以依赖，接口稳定，有测试覆盖
- 🟢 **进行中**：部分可用，但接口可能变化，需谨慎依赖
- ❌ **未迁移**：不可依赖，代码可能破碎或不存在，必须Mock隔离

## 发布策略（与迁移耦合）

- 当前状态：暂停发布（未完成全局融合至 `core/`）
- 解锁条件：完成 Risk 域 M2 + 至少一个其他域的 M2，并通过全局集成测试

---
