# DeepSeekQuant TODO 文档导航

> **组织原则**：TODO按架构分散到各模块目录，与代码同构  
> **更新日期**：2024-11-09

---

## 📍 TODO 文档位置索引

### 🔴 系统级TODO（根目录）
**文件**：`/core_bak_refactored/TODO.md`  
**内容**：跨模块任务、未确定位置的任务、架构级决策

**包含**：
- 端到端集成测试
- 多处理器协同测试
- 系统性能优化（缓存、增量计算）
- 告警通知系统
- 架构决策待定

---

### 🏗️ 基础设施层TODO
**文件**：`/core_bak_refactored/infrastructure/TODO.md`  
**内容**：纯数学/统计计算相关改进

**包含**：
- ✅ `risk_metrics.py` - StatisticalCalculator（生产就绪）
- ✅ `timeseries_calculator.py` - TimeSeriesCalculator（已完成）
- ⚠️ `cache_manager.py` - 缓存管理器（待创建）
- 📉 `technical_indicators.py` - 板块参数配置

---

### 🎯 核心层TODO

#### 信号引擎模块
**文件**：`/core_bak_refactored/core/signal/TODO.md`  
**内容**：技术指标、信号生成、信号评估

**包含**：
- ✅ `indicator_service.py` - IndicatorService（已完成）
- 🔴 `signal_generator.py` - SignalGenerator（高优先级，未开始）
- 🟡 `signal_evaluator.py` - SignalEvaluator（中优先级，未规划）

---

#### 风险管理模块
**文件**：`/core_bak_refactored/core/risk/TODO.md`  
**内容**：风险指标、风险监控、压力测试

**包含**：
- 🟡 `risk_metrics_service.py` - RiskMetricsService（待评审）
  - A股涨跌停场景处理
  - Sortino比率业务映射
  - 监管报告配置
- ✅ `portfolio_risk.py` - PortfolioRiskAnalyzer（已完成）
  - 风险归因可视化（P2）
- ✅ `stress_testing.py` - StressTester（已完成）
  - 历史极端事件回测（P1）
  - 蒙特卡洛模拟（P1）
- ✅ `risk_monitor.py` - RiskMonitor（已完成）
  - 风险仪表板（P2）

---

#### 策略模块
**文件**：`/core_bak_refactored/core/strategy/TODO.md`  
**内容**：交易策略实现

**包含**：
- 🔴 `base_strategy.py` - 策略基类（未开始）
- 🔴 `trend_following.py` - 趋势跟踪策略
- 🔴 `mean_reversion.py` - 均值回归策略
- 🔴 `breakout.py` - 突破策略
- 🟡 `strategy_backtester.py` - 回测框架

---

## 📊 当前工作重点

### 正在进行
- 🟡 **风险模块业务服务层评审** - `core/risk/risk_metrics_service.py`

### 下一步计划
1. 完成风险服务层专家复审
2. 实施A股涨跌停场景处理（P1）
3. 创建信号生成器 `signal_generator.py`（高优先级）

---

## 🎯 优先级说明

- 🔴 **P0/高优先级**：阻塞性任务，必须尽快完成
- 🟡 **P1/中优先级**：重要但非紧急，按计划推进
- 🟢 **P2/低优先级**：锦上添花，资源允许时实施

---

## 📈 模块状态总览

| 模块 | 文件 | 状态 | 优先级 | 测试覆盖 |
|------|------|------|--------|---------|
| 基础设施层 | `risk_metrics.py` | ✅ 生产就绪 | - | 21/21 |
| 基础设施层 | `timeseries_calculator.py` | ✅ 已完成 | - | 24/24 |
| 信号引擎 | `indicator_service.py` | ✅ 已完成 | - | 63/63 |
| 信号引擎 | `signal_generator.py` | 🔴 未开始 | 高 | 0/0 |
| 风险管理 | `risk_metrics_service.py` | 🟡 待评审 | 高 | 49/49 |
| 风险管理 | `portfolio_risk.py` | ✅ 已完成 | - | 8/8 |
| 风险管理 | `stress_testing.py` | ✅ 已完成 | - | 9/9 |
| 风险管理 | `risk_monitor.py` | ✅ 已完成 | - | 7/7 |
| 策略模块 | `base_strategy.py` | 🔴 未开始 | 中 | 0/0 |

**总测试覆盖**：192/192 通过 ✅

---

## 🗓️ 历史变更

- **2024-11-09**: 重构TODO组织结构
  - 从单一 `TODO.md` 拆分为模块级TODO
  - 创建导航文档
  - 按架构同构原则组织

- **2024-11-08**: 原整体 `TODO.md` 创建
