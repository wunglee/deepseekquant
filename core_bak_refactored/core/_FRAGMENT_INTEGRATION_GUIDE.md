# 功能碎片整合指南

## 📋 概述

本文档记录了从 `core/risk/backtest_framework.py` 拆分出的功能碎片，这些碎片因跨越模块职责边界而被暂时存放在各目标模块的 `_fragments/` 目录中，等待目标模块完成开发后整合。

**拆分原因**：根据 `docs/design/core_bak_refactored/core/risk/模块设计文档.md` 的职责定义，原 `backtest_framework.py` 混合了多个模块的职责，违反了单一职责原则和模块边界。

---

## 🔍 职责边界分析

| 原组件 | 原位置 | 职责归属 | 理由 |
|--------|--------|---------|------|
| `HistoricalDataProvider` | ❌ core/risk | ✅ **core/data** | 数据获取职责 |
| `MockHistoricalDataProvider` | ❌ core/risk | ✅ **core/data** | 数据模拟职责 |
| `SyntheticPortfolio` | ❌ core/risk | ✅ **core/portfolio** | 组合数据模型 |
| `SyntheticPortfolioBuilder` | ❌ core/risk | ✅ **core/portfolio** | 组合构造职责 |
| `EventWindowBacktester` | ❌ core/risk | ✅ **core/backtest** | 回测引擎职责 |
| `BacktestReporter` | ❌ core/risk | ✅ **core/backtest** | 回测报告职责 |
| `StressTestValidator` | ✅ core/risk | ✅ **core/risk** | 压力测试验证（正确归属） |

---

## 📁 功能碎片清单

### 1. 数据模块碎片

**文件**: `core/data/_fragments/historical_data_provider.py`

**包含组件**:
- `HistoricalDataProvider` (Protocol) - 历史数据接口
- `MockHistoricalDataProvider` (Class) - 模拟数据实现
- `RealHistoricalDataProvider` (Placeholder) - 真实数据实现（待完成）

**依赖关系**:
- 被依赖：`core/risk/stress_test_validator.py` (StressTestValidator)
- 被依赖：`core/backtest/_fragments/event_window_backtester.py` (EventWindowBacktester)

**整合检查清单**:
```markdown
□ 接口标准化
  □ 确认 HistoricalDataProvider 协议符合 core/data 模块设计
  □ 添加更多方法（get_stock_prices, get_option_data 等）

□ 真实数据源集成
  □ 实现 YahooFinanceAdapter
  □ 实现 JoinQuantAdapter
  □ 实现 WindAdapter（可选）

□ Mock 数据优化
  □ 迁移到 core/data/mocks/
  □ 支持更多事件场景
  □ 改进模拟数据质量

□ 数据缓存机制
  □ 实现本地缓存（文件系统）
  □ 实现内存缓存（Redis可选）
  □ 缓存过期策略

□ 调用者更新
  □ 更新 core/risk/stress_test_validator.py 的导入路径
  □ 更新测试用例
  □ 更新文档示例
```

---

### 2. 组合模块碎片

**文件**: `core/portfolio/_fragments/synthetic_portfolio_builder.py`

**包含组件**:
- `SyntheticPortfolio` (Dataclass) - 合成组合数据模型
- `SyntheticPortfolioBuilder` (Class) - 组合构造器

**提供方法**:
- `build_csi300_equal_weight()` - 沪深300等权重组合
- `build_sector_rotation()` - 行业轮动组合
- `build_ah_hybrid()` - A+H混合组合
- `build_by_type()` - 工厂方法

**依赖关系**:
- 被依赖：`core/risk/stress_test_validator.py` (StressTestValidator)
- 被依赖：`core/backtest/_fragments/event_window_backtester.py` (EventWindowBacktester)

**整合检查清单**:
```markdown
□ 与现有Portfolio类整合
  □ 确认与 core/portfolio/portfolio.py 的 Portfolio 类关系
  □ 决定是合并、继承还是保持独立

□ 扩展组合类型
  □ 市值加权组合
  □ 风险平价组合
  □ 动量组合
  □ 价值组合

□ 权重生成策略
  □ 等权重策略
  □ 市值加权策略
  □ 优化权重策略（MVO/Black-Litterman）
  □ 风险平价策略

□ 组合约束支持
  □ 最大/最小权重约束
  □ 行业/国家集中度约束
  □ 杠杆约束

□ 调用者更新
  □ 更新 core/risk/stress_test_validator.py 的导入路径
  □ 更新 core/backtest 模块的调用
  □ 更新文档和示例
```

---

### 3. 回测模块碎片

**文件**: `core/backtest/_fragments/event_window_backtester.py`

**包含组件**:
- `BacktestEvent` (Dataclass) - 回测事件定义
- `BacktestResult` (Dataclass) - 回测结果数据模型
- `EventWindowBacktester` (Class) - 事件窗口回测引擎
- `BacktestReporter` (Class) - 回测报告生成器

**提供方法**:
- `run_backtest()` - 执行回测
- `_calculate_actual_loss()` - 计算实际损失
- `_calculate_predicted_loss()` - 计算预测损失
- `generate_summary()` - 生成统计摘要
- `print_summary()` - 打印报告

**依赖关系**:
- 依赖：`core/data/_fragments/historical_data_provider.py` (HistoricalDataProvider)
- 依赖：`core/portfolio/_fragments/synthetic_portfolio_builder.py` (SyntheticPortfolio)
- 被依赖：`core/risk/stress_test_validator.py` (StressTestValidator)
- 现有实现：`core/backtest/backtest_engine.py` (BacktestEngine)

**整合检查清单**:
```markdown
□ 与BacktestEngine整合
  □ 确认 EventWindowBacktester 与 BacktestEngine 的关系
  □ 决定是合并、继承还是策略模式

□ 扩展回测方法
  □ 时间序列回测（传统方法）
  □ 事件窗口回测（Event Study）
  □ 蒙特卡洛回测
  □ 滚动窗口回测（Walk-Forward）

□ 回测指标增强
  □ 与 BacktestMetrics 类整合
  □ 添加风险调整收益指标
  □ 添加交易成本分析

□ 报告生成增强
  □ HTML报告
  □ PDF报告
  □ 可视化图表（权益曲线、回撤曲线）

□ 调用者更新
  □ 更新 core/risk/stress_test_validator.py 的导入路径
  □ 更新示例脚本
  □ 更新文档
```

---

## 🔧 当前状态：风险模块精简版

**保留在 core/risk 的组件**:

### `stress_test_validator.py` (新文件)

**职责**：
- 压力测试场景的历史有效性验证
- 场景参数准确性评估
- 损失预测误差统计

**设计亮点**：
- ✅ **依赖注入**：通过构造函数注入外部依赖（data_source, portfolio_builder）
- ✅ **接口抽象**：通过Protocol定义依赖接口，降低耦合
- ✅ **单一职责**：仅负责风险验证，不涉及数据获取和组合构造

**关键方法**：
- `validate_scenario()` - 验证单个场景
- `validate_all_scenarios()` - 批量验证
- `generate_validation_report()` - 生成验证报告

**临时Mock依赖**：
- `MockHistoricalDataSource` - 临时数据源（待data模块替换）
- `MockPortfolioBuilder` - 临时组合构造器（待portfolio模块替换）

---

## 🔄 整合流程

### Phase 1: 数据模块整合（优先级：P0）

1. **core/data 模块开发**
   - [ ] 设计数据模块架构
   - [ ] 实现 `HistoricalDataProvider` 接口
   - [ ] 集成 Yahoo Finance Adapter
   - [ ] 集成 JoinQuant Adapter
   - [ ] 实现数据缓存机制

2. **碎片整合**
   - [ ] 迁移 `historical_data_provider.py` 到 `core/data/providers/`
   - [ ] 迁移 `MockHistoricalDataProvider` 到 `core/data/mocks/`
   - [ ] 更新 `StressTestValidator` 的导入路径

3. **测试验证**
   - [ ] 运行 `test_stress_test_validator.py`
   - [ ] 验证真实数据与Mock数据对比

### Phase 2: 组合模块整合（优先级：P1）

1. **core/portfolio 模块重构**
   - [ ] 分析现有 `Portfolio` 类设计
   - [ ] 决定 `SyntheticPortfolio` 整合方式
   - [ ] 扩展组合类型和策略

2. **碎片整合**
   - [ ] 迁移 `synthetic_portfolio_builder.py` 到 `core/portfolio/builders/`
   - [ ] 更新 `StressTestValidator` 的导入路径

3. **测试验证**
   - [ ] 运行组合模块测试
   - [ ] 验证风险模块兼容性

### Phase 3: 回测模块整合（优先级：P1）

1. **core/backtest 模块开发**
   - [ ] 分析现有 `BacktestEngine` 设计
   - [ ] 决定 `EventWindowBacktester` 整合方式
   - [ ] 扩展回测方法

2. **碎片整合**
   - [ ] 迁移 `event_window_backtester.py` 到 `core/backtest/engines/`
   - [ ] 更新 `StressTestValidator` 的导入路径

3. **测试验证**
   - [ ] 运行回测模块测试
   - [ ] 端到端回测流程验证

---

## 📊 依赖关系图

```
StressTestValidator (core/risk)
       ↓ 依赖
    ┌──────────────┬──────────────────┐
    ↓              ↓                  ↓
HistoricalData  Portfolio      EventWindow
Provider        Builder        Backtester
(core/data)     (core/portfolio) (core/backtest)
    ↓                              ↓ 依赖
    └──────────────────────────────┤
           HistoricalDataProvider
```

---

## ⚠️ 重要提醒

### 1. 不要删除碎片文件

碎片文件是临时的功能实现，在目标模块完成前仍需使用。**仅在目标模块完成整合后才能删除**。

### 2. 导入路径管理

当前临时导入：
```python
# core/risk/stress_test_validator.py
from .stress_test_validator import MockHistoricalDataSource, MockPortfolioBuilder
```

未来正式导入（目标）：
```python
# core/risk/stress_test_validator.py
from core.data.providers import HistoricalDataProvider
from core.data.mocks import MockHistoricalDataProvider
from core.portfolio.builders import SyntheticPortfolioBuilder
```

### 3. 测试隔离

碎片文件有独立的测试：
- `test_stress_test_validator.py` - 测试风险模块验证器
- 碎片整合后需新增集成测试验证模块间协作

### 4. 文档同步

每次整合碎片后，必须同步更新：
- [ ] 模块设计文档.md
- [ ] 接口设计文档.md
- [ ] 本整合指南（标记已完成项）

---

## 📝 变更历史

| 日期 | 版本 | 变更内容 | 作者 |
|------|------|---------|------|
| 2025-11-25 | v1.0 | 创建功能碎片整合指南，拆分backtest_framework.py | AI Assistant |

---

## 📚 参考文档

- [风险模块设计文档](../docs/design/core_bak_refactored/core/risk/模块设计文档.md)
- [风险模块接口设计文档](../docs/design/core_bak_refactored/core/risk/接口设计文档.md)
- [回测引擎实现](../core/backtest/backtest_engine.py)
- [专家咨询记录](../docs/consultation.md)
