# DEPRECATED: backtest_framework.py

## ⚠️ 废弃说明

**状态**: ❌ 已废弃  
**废弃日期**: 2025-11-25  
**原因**: 违反模块职责边界，混合了多个模块的功能

---

## 📋 原文件内容分布

原 `backtest_framework.py` 的功能已被重构并拆分到以下位置：

| 原组件 | 新位置 | 状态 |
|--------|--------|------|
| `StressTestValidator` | `core/risk/stress_test_validator.py` | ✅ 已重构 |
| `HistoricalDataProvider` | `core/data/_fragments/historical_data_provider.py` | ⏸️ 功能碎片 |
| `MockHistoricalDataProvider` | `core/data/_fragments/historical_data_provider.py` | ⏸️ 功能碎片 |
| `SyntheticPortfolio` | `core/portfolio/_fragments/synthetic_portfolio_builder.py` | ⏸️ 功能碎片 |
| `SyntheticPortfolioBuilder` | `core/portfolio/_fragments/synthetic_portfolio_builder.py` | ⏸️ 功能碎片 |
| `EventWindowBacktester` | `core/backtest/_fragments/event_window_backtester.py` | ⏸️ 功能碎片 |
| `BacktestReporter` | `core/backtest/_fragments/event_window_backtester.py` | ⏸️ 功能碎片 |

---

## 🔄 迁移指南

### 如果您之前使用：

```python
# ❌ 旧代码（已废弃）
from core.risk.backtest_framework import (
    EventWindowBacktester,
    SyntheticPortfolioBuilder,
    MockHistoricalDataProvider
)
```

### 请改为：

```python
# ✅ 新代码（风险模块部分）
from core.risk.stress_test_validator import (
    StressTestValidator,
    MockHistoricalDataSource,  # 临时Mock
    MockPortfolioBuilder       # 临时Mock
)

# ⏸️ 功能碎片（临时位置，等待目标模块完成）
# 注意：这些导入将来会改变，当对应模块开发完成后
from core.data._fragments.historical_data_provider import (
    HistoricalDataProvider,
    MockHistoricalDataProvider
)
from core.portfolio._fragments.synthetic_portfolio_builder import (
    SyntheticPortfolio,
    SyntheticPortfolioBuilder
)
from core.backtest._fragments.event_window_backtester import (
    EventWindowBacktester,
    BacktestReporter
)
```

---

## 📊 架构改进

### 原架构问题：

```
core/risk/backtest_framework.py (混合职责)
├── 数据获取 ← 属于 core/data
├── 组合构造 ← 属于 core/portfolio
├── 回测引擎 ← 属于 core/backtest
└── 风险验证 ← 属于 core/risk ✅ 唯一正确的
```

### 新架构：

```
core/risk/stress_test_validator.py (单一职责 ✅)
└── 压力测试验证 (仅风险模块职责)
    ├── 依赖: core/data (数据获取)
    ├── 依赖: core/portfolio (组合构造)
    └── 依赖: core/backtest (回测引擎)
```

---

## 🎯 设计原则

重构遵循以下原则：

1. **单一职责原则** (SRP)
   - 每个模块只负责自己的业务领域
   - risk模块不应包含数据获取或回测引擎逻辑

2. **依赖倒置原则** (DIP)
   - 通过Protocol接口定义依赖
   - 使用依赖注入降低耦合

3. **开闭原则** (OCP)
   - 功能碎片设计允许未来扩展
   - 不破坏现有代码

---

## 📝 相关文档

- [功能碎片整合指南](../core/_FRAGMENT_INTEGRATION_GUIDE.md)
- [风险模块设计文档](../../../docs/design/core_bak_refactored/core/risk/模块设计文档.md)
- [StressTestValidator实现](./stress_test_validator.py)

---

## ❓ 常见问题

### Q1: 为什么要拆分？
A: 原文件混合了4个模块的职责，违反了设计文档的模块边界定义，导致：
- 代码耦合严重
- 测试困难
- 维护成本高
- 模块职责混乱

### Q2: 功能碎片是什么？
A: 功能碎片是暂时存放在目标模块 `_fragments/` 目录中的代码，等待该模块开发完成后整合。这是一种渐进式重构策略。

### Q3: 何时删除原文件？
A: 当所有功能碎片都整合到目标模块后，原文件将被完全删除。当前保留是为了：
- 代码历史追溯
- 迁移参考
- 防止意外依赖

### Q4: 测试受影响吗？
A: 旧测试 `test_backtest_framework.py` 已废弃，新测试为 `test_stress_test_validator.py`。所有测试通过率100% (9/9)。

---

**最后更新**: 2025-11-25  
**状态**: 废弃完成，功能已迁移
