# DeepSeekQuant 架构（core_bak_refactored）

目标：明确模块边界与职责、依赖倒置策略、碎片化交付方案，并保证文档与代码一致。

模块总览
- core/signal：指标与信号生成
- core/strategy：策略编排与信号消费
- core/portfolio：组合构建与优化；提供合成组合工具
- core/backtest：事件窗口回测与报告生成（准确度指标、误差Top5）
- core/risk：压力测试、风险度量、限额管理、监控与报告（不承载backtest/data具体实现）
- core/data：历史/真实数据提供者，统一接口（Protocol）供 backtest/risk 使用
- core/market_analysis：市场情绪与波动率分析
- core/exec：执行算法与订单处理
- core/optimization：参数调优（如贝叶斯）
- core/share：共享类型与小型工具（如市场配置、汇率转换）

关键原则
- 边界严格：每个模块只负责自身业务；跨模块交互通过 Protocol 接口完成。
- 碎片化交付：尚未完备的功能以“_fragments/”形式放入目标模块，待模块完善后合并。
- 依赖倒置：策略/风险等高层策略依赖接口而非具体实现，便于替换数据源与算法。
- 测试驱动：每个碎片需有对应的端到端测试，保障最小可用路径稳定。

职责与当前状态
- risk：
  - 负责场景库与压力模拟（顺序、并发、反馈回路）、风险度量、限额与监控。
  - 仅定义依赖接口（如 HistoricalDataProvider Protocol），不包含数据/回测的具体实现。
  - 已清理越界职责：删除 currency_converter.py 与 international_config.py，相关能力统一由 [core/share/exchange_rates.py](symbol:exchange_rates.py) 与 [core/share/market_config.py](symbol:market_config.py) 提供。
  - 修正不合规导入：移除对根目录 common 的尝试导入，统一使用模块内默认配置。
- backtest：
  - 事件窗口回测引擎与报告生成，摘要包含准确度与误差Top5。
  - [EventWindowBacktester](symbol:event_window_backtester.py) 与 [BacktestReporter](symbol:event_window_backtester.py) 驻留在 core/backtest/_fragments/。
- data：
  - 历史数据提供者（Mock/Real），实现 [HistoricalDataProvider](symbol:historical_data_provider.py) 接口。
  - 事件期价格生成采用确定性日漂移，严格匹配预期跌幅；成交量非负夹紧。
- portfolio：
  - 合成组合与构造器迁移至 [core/portfolio/_fragments/synthetic_portfolio.py](symbol:synthetic_portfolio.py)，供回测使用。

依赖接口
- HistoricalDataProvider（Protocol）：由 risk/backtest 消费，具体实现位于 data 模块。
- BacktestRunner（接口角色）：由 backtest 模块提供实现（EventWindowBacktester）。

碎片策略与标记
- 统一在目标模块的“_fragments/”目录落地，并在头部注释标记“来源模块与合并意图”。
- 模块完整后，合并碎片至稳定位置并移除兼容重导出。

导入与约束
- 严格限制在 core_bak_refactored/ 范围内与标准/第三方库；禁止从根 core/、infrastructure/、common.py 导入。
- 已修复：
  - [factor_model.py](symbol:factor_model.py) 移除对 common.RISK_MODEL_CONFIG 的导入，使用本模块默认参数（含 cache_ttl_seconds）。
  - [portfolio_risk.py](symbol:portfolio_risk.py) 不再尝试导入 common，统一使用本地默认 risk_model_config。

质量与一致性
- 回测集成质量：准确度（误差≤20%）阈值≥90%，并发相关性矩阵支持对称方向回退与波动率微调；实际损失计算具备NaN与价格边界检查。
- 数据生成：事件窗口价格严格匹配预期跌幅；成交量非负夹紧，确保可重复性。
- 风险模块：风险度量与限额校验保持模块职责纯净；国际化增强与市场机制检测逻辑不依赖越界配置。

演进路线
- 完成 risk/backtest/data 的接口化对接后，逐步移除 risk/backtest_framework.py 的兼容重导出，仅保留协议层。
- 引入真实数据提供者后，重新校准回测质量阈值，并扩展预测函数以纳入更多压力参数（如波动率冲击）。
- 碎片合并：在目标模块成熟后，将 _fragments 内容合并为稳定实现，并更新对应测试。

模块依赖关系架构图
```mermaid
graph TD
    Risk --> Data
    Risk --> Share
    Backtest --> Data
    Backtest --> Portfolio
    Portfolio --> Share
    Strategy --> Signal
    Exec --> Portfolio
    Optimization --> Strategy
    Data --> Share
 core_bak_refactored 架构设计文档

## 📅 更新日志

### 2025-12-02: 职责归位与模块重构（彻底清理版）

**核心变更**：
1. **data 模块职责清理**：
   - ✅ **彻底删除**所有向后兼容代码（无委托方法遗留）
   - ✅ 删除兼容导入文件：`core/data/analytics/sentiment.py`, `performance.py`
   - ✅ 从 `data_utils.py` 删除 87行委托方法，保留纯净的基础处理工具
   - ✅ 修正导入路径：`market_config` → `market.market_config`, `market_enums` → `market.market_enums`

2. **新增模块**：
   - `core/backtest/event_analysis.py` (177行) - 事件驱动分析
     * `EventConfig` 数据类
     * `EventAnalyzer` 静态工具类：4个方法（safe_get_event_data, calculate_actual_return, calculate_prediction_error, validate_event_data）
   - `core/share/data_analysis_utils.py` (141行) - 共享数据分析工具
     * `DataAnalysisUtils` 静态工具类：3个方法（align_time_series, ensure_series, validate_dataframe）
   - `core/market_analysis/sentiment_analyzer.py` (355行) - 市场情绪评估
     * 3个函数：assess_market_sentiment, assess_liquidity_conditions, determine_volatility_regime
   - `infrastructure/monitoring/performance_monitor.py` (255行) - 性能监控
     * `PerformanceMonitor` 类
     * `create_performance_report` 函数

3. **测试同构化**：
   - ✅ 创建 `tests/units/core/backtest/event_analysis_test.py` (172行, 9个测试)
   - ✅ 更新所有引用文件的导入路径：
     * `tests/fixtures/core/backtest/backtest_fixtures.py`
     * `tests/units/core/data/data_utils_test.py`
     * `tests/units/core/data/providers/historical_data_provider_test.py`
     * `core/risk/risk_preprocessor.py`
     * `core/share/performance_stats.py`
   - ✅ 修复测试日志断言（logger名称：`DeepSeekQuant.EventAnalysis`）
   - ✅ 修复导入错误：`share.market_config` → `share.market.market_config`, `share.market_enums` → `share.market.market_enums`

4. **职责划分**（严格遵守单一职责原则）：
   - **data 模块**：数据源基本处理、转换、验证（OHLC转收益率、时间序列处理）
   - **backtest 模块**：事件驱动回测分析能力（事件窗口数据提取、误差计算）
   - **market_analysis 模块**：市场情绪与波动率分析（VIX评估、流动性条件、波动率区间）
   - **share 模块**：跨模块共享的高级数据分析工具（时间序列对齐、数据验证）
   - **infrastructure/monitoring**：通用性能监控基础设施（请求追踪、性能报告）

5. **代码统计**：
   - 删除：155行（87行委托方法 + 68行兼容导入文件）
   - 新增：928行（职责归位后的新代码）
   - 净变化：+773行（职责清晰，无冗余）

6. **测试验证**（全通过）：
   - 单元测试：21+ 测试（data_utils_test, risk_preprocessor_test, event_analysis_test 等）
   - E2E 集成测试：6/6 通过
   - 集成测试：6/6 通过（industry_parameter_validation）
   - ✅ 无向后兼容遗留，代码纯净
   - ✅ 测试与代码完全同构

**架构影响**：
- ✅ data 模块现在专注于数据源处理，不包含高级业务逻辑
- ✅ backtest/market_analysis/share 模块承载相应的业务能力
- ✅ 依赖方向正确：高级模块依赖基础模块，符合分层架构
- ✅ 测试与代码同构，符合规范要求
- ✅ 所有导入路径正确，无遗留引用

**清理验证**：
- ✅ 零向后兼容遗留（grep验证：0个 `@deprecated` 标记）
- ✅ 零委托方法（data_utils.py 仅保留基础工具）
- ✅ 零兼容导入文件（analytics/sentiment.py, performance.py 已删除）
- ✅ 所有测试文件导入路径已更新

---

### 2025-12-02: 代码冗余清理与性能监控架构优化

**优化内容**：
1. **删除冗余聚合器**：移除 `core/share/aggregation_manager.py`（325行），保留 `core/data/aggregation/aggregator.py`
2. **性能监控架构改造**：`PerformanceMonitor` 改造为底层请求追踪器，集成到 `PerformanceStatsManager`
3. **删除冗余缓存管理器**：
   - 移除 `core/share/cache_manager.py`（154行）
   - 移除 `core/risk/cache_manager.py`（449行）
   - 统一使用 `infrastructure/cache/CacheManager`（三层缓存架构）
4. **枚举按业务内聚拆分**：
   - 删除 `core/share/enums.py`（通用枚举文件）
   - 创建 `core/data/enums.py`（数据模块专属枚举，7个枚举类）
   - 创建 `core/monitoring/enums.py`（监控模块专属枚举，2个枚举类）
   - 删除 `core/monitoring/alert_manager.py` 中的重复枚举定义
   - 更新 5 个文件的导入路径
5. **验证模块按业务内聚归位**：
   - 将 `infrastructure/validation/validator.py`（282行）迁移到 `core/data/validation/`
   - 原因：`validator.py` 是 MarketData 业务验证逻辑（OHLC关系、价格合理性），属于业务层
   - 保留 `infrastructure/type_validators.py`（457行）：通用技术验证（类型、长度、数值范围）
   - 删除 `infrastructure/validation/` 目录
   - 更新 `infrastructure/__init__.py` 和 `core/data/__init__.py` 的导入路径
6. **业务模块按业务内聚归位**：
   - 迁移 `infrastructure/portfolio_optimizers.py`（133行）到 `core/portfolio/optimizers/`
   - 迁移 `infrastructure/execution_algos.py`（182行）到 `core/exec/algorithms/`
   - 迁移 `infrastructure/acquisition_functions.py`（190行）和 `gaussian_process.py`（149行）到 `core/optimization/bayesian/`
   - 原因：这些模块是业务逻辑，非通用技术基础设施
   - 更新导入路径：`portfolio_processor.py`, `execution_strategy.py`
   - 迁移测试文件到对应业务模块目录

**架构变更**：
- **聚合层统一**：`DataAggregator` 作为唯一聚合器（386行，11个方法）
- **缓存管理统一**：`infrastructure/cache/CacheManager` 作为唯一缓存管理器（三层缓存：L1内存+L2 LRU+L3 Redis）
  - 删除业务层重复实现：core/share、core/risk 的 CacheManager
  - 所有业务模块统一从 infrastructure.cache 导入
- **枚举内聚管理**：按业务模块归属，数据枚举归 `core/data/`，监控枚举归 `core/monitoring/`
- **性能监控分层**：
  - 底层：`PerformanceMonitor`（请求级别追踪）
  - 业务层：`PerformanceStatsManager`（业务指标 + 请求追踪）

**扫描数据**（基于规范要求 - 25个模式完整扫描）：

| 编号 | 模式 | 出现次数 | 说明 |
|------|------|----------|------|
| 1 | except Exception as e: | 450 | 异常处理模式 |
| 2 | if len( | 174 | 数据长度检查 |
| 3 | .get(' | 1226 | 配置提取模式 |
| 4 | return float( | 86 | float转换 |
| 5 | np.percentile( | 19 | 百分位数计算 |
| 6 | isinstance( | 220 | 类型检查 |
| 7 | logger. | 1284 | 日志调用 |
| 8 | try: | 624 | try语句起始 |
| 9 | sum( | 210 | sum聚合 |
| 10 | abs( | 162 | 绝对值计算 |
| 11 | float(abs( | 12 | 组合转换 |
| 12 | np.clip( | 24 | 数值裁剪 |
| 13 | .values() | 63 | 字典值遍历 |
| 14 | is None | 191 | None检查 |
| 15 | len() == 0 | 58 | 空列表检查 |
| 16 | max/min( | 242 | 最值计算 |
| 17 | pd.Series( | 115 | Series构造 |
| 18 | np.array( | 89 | 数组构造 |
| 19 | f" | 1608 | f-string格式化 |
| 20 | getattr( | 61 | 属性获取 |
| 21 | hasattr( | 113 | 属性检查 |
| 22 | enumerate( | 35 | 枚举迭代 |
| 23 | zip( | 16 | 并行迭代 |
| 24 | list comprehension | 155 | 列表推导式 |
| 25 | dict comprehension | 59 | 字典推导式 |

**统计汇总**：
- Python文件总数：307个
- 总扫描模式数：25个
- 高频模式（>500次）：f-string(1608), logger(1284), .get(1226), try(624)
- 潜在优化点：异常处理统一（450次），配置提取标准化（1226次）

**优化成果**：
- 删除重复代码：**928行**
  - aggregation_manager.py: 325行
  - core/share/cache_manager.py: 154行
  - core/risk/cache_manager.py: 449行
- 按业务内聚拆分与归位：
  - 枚举拆分：data_enums.py → core/data/enums.py, alert_enums.py → core/monitoring/enums.py
  - 验证归位：infrastructure/validation/validator.py → core/data/validation/（MarketData业务验证）
  - 业务模块归位：
    * portfolio_optimizers.py → core/portfolio/optimizers/（组合优化业务）
    * execution_algos.py → core/exec/algorithms/（执行算法业务）
    * acquisition_functions.py + gaussian_process.py → core/optimization/bayesian/（贝叶斯优化业务）
- 架构更清晰：缓存、聚合、枚举、验证、业务算法统一管理，遵循分层原则
- 向后兼容：默认模式不启用请求追踪
- 测试验证：
  - core/share: 81/81 passed
  - infrastructure/cache: 105/105 passed
  - core/data/validation: 8/8 passed
  - core/portfolio/optimizers: 3/3 passed
  - core/exec/algorithms: 3/3 passed
  - core/optimization/bayesian: 5/5 passed
- 新增测试：7个测试文件（enums_test, config_manager_test, models_test, performance_stats_test, data/enums_test, monitoring/enums_test, data/validation_test）

**详细报告**：见下文"性能监控架构"章节

---

**依赖说明**：
- **Share（业务基础）**：纯配置与工具模块，不依赖任何业务模块；提供市场配置、汇率转换等共享能力。
- **Data**：依赖 Share 获取市场配置（如交易日历、市场代码映射）。
- **Portfolio**：依赖 Share 获取汇率转换与市场参数。
- **Risk**：依赖 Share 获取市场配置与汇率适配器。
- **Backtest**：依赖 Data（数据提供者）和 Portfolio（合成组合构建器）。
- **Strategy/Exec/Optimization**：高层业务模块，依赖下层服务。

**架构原则**：
- Share 为最底层，仅依赖标准库与第三方库，不依赖任何业务模块。
- Data 作为数据抽象层，依赖 Share 的市场配置。
- 业务模块（Risk/Backtest/Portfolio）通过 Protocol 接口依赖 Data，实现依赖倒置。

---

## 性能监控架构（2025-12-02 改造）

### 架构层次

```
业务层：PerformanceStatsManager（core/share/performance_stats.py）
  ↓ 集成
底层：PerformanceMonitor（infrastructure/monitoring/performance_monitor.py）
```

### 职责划分

**底层追踪器：PerformanceMonitor**
- 职责：细粒度跟踪单个API请求性能
- 指标：requests_total, cache_hits, avg_response_time, source_usage, error_counts
- 使用场景：需要详细的请求级别性能分析时

**业务层管理器：PerformanceStatsManager**
- 职责：业务层性能统计 + 可选集成底层追踪器
- 指标：throughput, success_rate, reliability, stability_score, anomalies_detected
- 集成方式：通过 `enable_request_tracking=True` 启用底层追踪

### 使用模式

**模式1：默认模式**
```python
manager = PerformanceStatsManager()  # 不启用请求追踪
manager.increment_counter('data_points_processed', 100)
summary = manager.get_summary()  # 仅包含业务层指标
```

**模式2：启用请求追踪**
```python
manager = PerformanceStatsManager(enable_request_tracking=True)
manager.record_request('AAPL', True, 0.5, 'yahoo')  # 委托给底层追踪器
summary = manager.get_summary()  # 包含业务层 + 请求级别指标
```

### 数据聚合层统一

**唯一聚合器：DataAggregator**（core/data/aggregation/aggregator.py）
- 方法数：11个
- 核心功能：
  - `aggregate_ohlcv()` - OHLCV数据时间聚合
  - `calculate_rolling_metrics()` - 滚动窗口指标
  - `aggregate_by_symbol()` - 按股票代码聚合
  - `calculate_vwap()` - 成交量加权平均价
  - `merge_data_sources()` - 数据源合并

**已删除**：`core/share/aggregation_manager.py`（96%重复代码）

### 优化验证

- ✅ 删除冗余：325行重复代码
- ✅ 测试通过：19/19（DataAggregator）
- ✅ 向后兼容：默认模式无破坏性变更
- ✅ 架构清晰：性能监控分层，聚合器统一

