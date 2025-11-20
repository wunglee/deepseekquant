# 风险管理模块 TODO

> **层级**：Core Layer - Risk Management  
> **路径**：`core/risk/`  
> **职责**：风险指标计算、风险监控、压力测试
> **生产状态**：模块生产就绪（专家复审通过）；全局发布取决于完成 Phase 2（core_bak_refactored → core 融合）与全局集成测试

---

## 🎉 最新进展：生产就绪确认

**专家复审结果** (2024-11-14):
- **评分**: 96/100分
- **结论**: ✅ **批准立即发布生产版本**
- **P1修正**: 6项全部正确落地（100%完成）
- **P1.5边界测试**: ✅ 已补充，18/18通过
- **测试通过率**: 123/128 (96.1%)
- **复审文档**: `docs/answer.md`

**关键改进已落地**:
1. ✅ 动态相关性调整（基于市场波动率）
2. ✅ 美股合规日志增强（event_id + ISO8601）
3. ✅ 数据质量自动化处理（C/D级告警）
4. ✅ 场景相关性矩阵微调（货币危机0.45、A股0.3）
5. ✅ 新增SG市场配置
6. ✅ JP市场调整为严格模式（专家建议）
7. ✅ **P1.5边界条件测试补充**（9个新测试，含性能测试）

**发布说明**：发布策略由全局计划统一安排；在完成 Phase 2（融合至 core）与全局集成测试通过前，本模块不单独推动发布

---

## 📋 风险模块全面评审计划（自底向上）

### 评审阶段与文件映射

#### **阶段1：数据模型层** ⚠️ 当前阶段
- **文件**: `risk_models.py` (278行)
- **状态**: ❌ 未评审（P0最高优先级）
- **依赖**: 无（最底层，被所有模块依赖）
- **评审文档**: `docs/ask.md`（第3轮咨询，已创建）
- **重点**: 枚举完整性、数据结构合理性、序列化健壮性

#### **阶段2：基础设施层** ✅ 已完成
- **文件**: `infrastructure/risk_metrics.py`
- **状态**: ✅ 专家Stage 1已评审
- **依赖**: 无

#### **阶段3：业务服务层** ✅ 已完成
- **文件**: `risk_metrics_service.py`
- **状态**: ✅ 专家Stage 2 + P0修复完成
- **依赖**: StatisticalCalculator, risk_models
- **测试**: 17/17业务层 + 11/11国际化测试通过

#### **阶段4：国际化配置层** ✅ 已完成
- **文件**: `international_config.py`, `international_enhancements.py`, `market_detectors.py`
- **状态**: ✅ 国际化支持完成
- **依赖**: RiskMetricsService
- **测试**: 11/11测试通过

#### **阶段5：专业分析器层（并列）** ⚠️ 部分待评审

**5A. 组合风险分析器** ⚠️ 待评审
- **文件**: `portfolio_risk.py` (342行)
- **状态**: P1-1功能增强完成，❌ 未经专家评审
- **依赖**: RiskMetricsService, StatisticalCalculator
- **测试**: 8/8测试通过
- **评审重点**: 
  - 组合收益计算方法（对数vs简单收益）
  - Barra因子风险归因准确性
  - 边际风险贡献计算验证

**5B. 持仓风险分析器** ⚠️ 待评审
- **文件**: `position_risk.py` (223行)
- **状态**: 基础实现完成，❌ 未评审
- **依赖**: StatisticalCalculator
- **评审重点**: 
  - 流动性风险模型（参与率冲击模型）
  - 清算时间估算准确性
  - 价格冲击模型参数验证

**5C. 压力测试器** ✅ 已完成
- **文件**: `stress_testing.py` (733行)
- **状态**: ✅ P1-2内置场景库完成（基于专家指导）
- **依赖**: RiskMetricsService, risk_models
- **测试**: 9/9测试通过
- **内容**: 9种历史场景 + 场景相关性矩阵

**5D. 风险计算协调器** ⚠️ 待评审
- **文件**: `risk_calculator.py` (160行)
- **状态**: 基础实现，❌ 未评审
- **依赖**: RiskMetricsService
- **评审重点**: 
  - 协调逻辑正确性
  - 与RiskMetricsService的职责边界

#### **阶段6：管理器层** ✅ 已完成
- **文件**: `risk_limits.py` (560行) + `risk_limits_enhanced.py` (1202行)
- **状态**: ✅ P1-3智能化完成 + 专家2轮咨询修正
- **依赖**: 无直接依赖分析器（独立检查限额）
- **测试**: 36/36测试通过（100%）
- **内容**: 智能阈值、优先级处理、市场差异化

#### **阶段7：监控层** ⚠️ 待评审
- **文件**: `risk_monitor.py` (268行)
- **状态**: 基础实现完成，❌ 未评审
- **依赖**: risk_models
- **测试**: 7/7测试通过
- **评审重点**: 
  - 实时监控机制合理性
  - 告警触发逻辑准确性
  - 监控线程安全性

#### **阶段8：协调层** ⚠️ 待评审
- **文件**: `risk_processor.py` (180行)
- **状态**: 基础实现，❌ 需要集成测试
- **依赖**: 所有上述组件
- **测试范围**: 
  - 所有组件协同工作
  - 端到端场景验证
  - 性能压力测试
  - 错误处理和降级

---

## 🎯 当前任务：阶段1 - 数据模型层评审

### 执行计划

**步骤1**: ✅ 创建专家咨询文档
- 文件：`docs/ask.md`（第3轮咨询）
- 状态：已创建（411行，7个核心问题）
- 时间：2024-11-12

**步骤2**: ⏳ 等待专家反馈
- 文件：`docs/answer.md`
- 等待专家回复7个问题

**步骤3**: 📋 待执行 - 根据专家指导修正
- 修改：`risk_models.py`
- 可能涉及：
  - 调整枚举定义（RiskLevel/RiskType/RiskMetric）
  - 优化数据类字段
  - 增强容错机制
  - 补充文档注释

**步骤4**: ✅ 已完成 - 补充测试
- 文件：`risk_models_test.py`
- 覆盖：RiskLevel映射、legacy兼容、LimitBreach序列化/旧字段映射、Recommendation优先级校验与序列化、TimeHorizon属性

**步骤5**: 📋 待执行 - 更新consultation.md
- 追加第3轮咨询问答到 `docs/consultation.md`

---

## 🗓️ 后续阶段优先级

### P0优先级（阶段1完成后立即执行）
1. **阶段5A：组合风险分析器评审** - 有复杂Barra模型
2. **阶段5B：持仓风险分析器评审** - 流动性风险核心

### P1优先级
3. **阶段5D：风险计算协调器评审** - 明确职责边界
4. **阶段7：监控层评审** - 功能相对独立

### P2优先级
5. **阶段8：总协调器集成测试** - 整体验证

---

## 🟢 risk_metrics_service.py - RiskMetricsService

**状态**：✅ 国际化支持完成，生产就绪  
**最后更新**：2024-11-09  
**测试覆盖**：17/17 通过（业务服务层）+ 11/11 通过（国际化支持）+ 207/207 通过（全项目）  
**依赖**：✅ infrastructure/risk_metrics.py 已完成

### 已完成
- [x] 基础风险指标（VaR, CVaR, Sharpe, Sortino）
- [x] Beta、Alpha计算
- [x] 最大回撤
- [x] **P0修复：CVaR参数法保守估计**（替代1.1倍系数）
- [x] **P0修复：分层置信度配置**（95%/99%，满足监管要求）
- [x] **P0修复：动态无风险利率**（支持实时市场数据）
- [x] **P0修复：符号约定统一**（VaR/CVaR明确返回正数表示损失）
- [x] **P0修复：A股涨跌停检测机制**（4种板块类型 + 日志警告 + 4个专项测试）
- [x] **🌍 国际化支持**：CN/US/HK/JP/EU市场配置
- [x] **🌍 市场机制检测**：涨跌停、熔断、LULD、VCM
- [x] **🌍 增强版夏普比率**：市场特定风险溢价、异常调整
- [x] **🌍 跨市场风险对比**：多市场风险分析
- [x] 17个业务服务层测试 + 11个国际化测试
- [x] **阶段3：国际化补充检查** - RiskCalculator集成MarketConfigManager，专家评审87/100，批准发布

### 待办事项（业务层改进储备）

> **📝P0修复总结（2024-11-09）**
> - ✅ CVaR保守估计策略：使用参数法CVaR替代1.1倍系数
> - ✅ 默认置信度分层：支持95%/99%，区分日常监控和监管报告
> - ✅ 无风险利率动态化：支持动态配置和参数传入
> - ✅ 符号约定统一：VaR/CVaR返回正数表示损失，文档清晰
> - ✅ A股涨跌停处理：检测机制 + 日志警告 + 4个专项测试
> 
> **📄复审文档**：`/consultation/expert_review_stage2_business_service.md`  
> **测试结果**：17/17 通过（业务服务层）+ 196/196 通过（全项目）

> **🌍国际化补充检查（2024-11-12）**
> - ✅ RiskCalculator集成MarketConfigManager
> - ✅ 配置验证和自动补全机制
> - ✅ 所有日志添加market_type信息
> - ✅ 修复蒙特卡洛随机种子硬编码问题
> 
> **👨‍🔬专家评审结果**：✅ 批准发布（87/100，达到生产就绪标准）  
> **测试结果**：10/10 通过

---

#### ⚠️ P1: 货币单位一致性检查（专家建议 - 下阶段补充）

**问题说明**：  
投资组合包含多币种资产时，VaR计算可能因货币单位不一致而产生错误。

**修复方案**：
```python
def _validate_currency_consistency(self, portfolio_state, market_data):
    """验证投资组合货币单位一致性"""
    currencies = set()
    for symbol, allocation in portfolio_state.allocations.items():
        currency = market_data['prices'][symbol].get('currency', 'UNKNOWN')
        currencies.add(currency)
    
    if len(currencies) > 1:
        logger.warning(f"多币种投资组合检测: {currencies}")
        return False
    return True
```

**待办**：
- [ ] 在RiskCalculator中添加货币一致性验证
- [ ] 支持多币种警告和日志记录
- [ ] 添加单元测试

---

#### 💡 P2: 实时市场数据集成（专家建议 - 未来版本）

**目标**：  
集成实时市场数据源（Bloomberg/Wind）获取实时无风险利率。

**实现示例**：
```python
def _fetch_live_risk_free_rate(self, market_type: str) -> float:
    """从Bloombeg/Wind获取实时无风险利率"""
    # 实现实时数据接口
    pass
```

**待办**：
- [ ] 调研数据源API（Bloomberg/Wind）
- [ ] 设计数据接口层
- [ ] 实现实时数据缓存机制
- [ ] 添加数据有效性验证

---

#### ⚠️ P1: 配置热更新支持（专家建议 - 近期实施）

**目标**：  
为`MarketConfigManager`和`RiskCalculator`增加运行时安全热更新能力。

**实施要点**：
- [ ] 设计配置版本与变更检测（仅内部使用，不暴露版本控制功能）
- [ ] 新增接口：`update_market_config(new_config: Dict)`（原子性更新）
- [ ] 增加回滚机制：更新失败自动恢复旧配置
- [ ] 增加订阅通知：相关服务在热更新后重新初始化
- [ ] 添加并发安全保障（锁/副本交换）
- [ ] 补充单元测试与集成测试

---

#### ✅ P1.5: 货币一致性增强与市场特定策略（专家建议 - 已完成）

**目标**：完善阶段1的货币一致性检查，按市场定制分级策略，提升美股场景支持。

**已完成**：
- [x] 增加风险参数货币检查：`_check_risk_parameters_currency(data)`
- [x] 更精细的警告分类：`_classify_currency_warnings(warnings)`（info/warning/error）
- [x] 市场特定严重性：`_get_market_specific_severity(warning, market_type)`（US/CN差异）
- [x] 按市场类型的默认严格模式：`_get_default_strict_mode(market_type)` （US/HK/SG/JP严格）
- [x] 数据源质量评估：`_assess_data_source_quality(prices)`（currency覆盖度评级）
- [x] 美股合规日志：`_us_compliance_logging(currency_warnings)`（SEC/FINRA合规事件）
- [x] 记录跨模块依赖：汇率转换服务，见共享业务模块（core/share/exchange_rates.py）
- [x] 实现适配器注入与调用：`attach_exchange_rate_adapter` + `_unify_currency_for_portfolio`
- [x] 新增联调测试：`currency_adapter_integration_test.py`，14 tests passed
- [x] 共享模块迁移：MarketConfigManager迁移到core/share/market_config.py
- [x] 全量回归测试：105/110 passed（无新增回归）
- [x] **P1修正落地**：动态相关性、美股合规日志增强、数据质量处理、SG市场、JP严格模式

**评审文档**：`docs/ask.md` + `docs/answer.md`（专家95分，批准发布）  
**完成时间**：2024-11-14

---

## 📅 P2优化计划（基于专家answer.md建议）

### 高优先级（发布后1-2周）

#### 🔴 P2-1: 性能基准测试
**紧急度**: 🔴 高  
**目标**: 建立性能基准与容量规划  
**时间**: 发布后1周

**实施计划**:
```python
# 性能测试套件
@pytest.mark.performance
def test_currency_check_large_portfolio():
    """1000个标的货币检查性能"""
    large_portfolio = generate_portfolio(1000)
    start = time.time()
    result = calculator._runtime_currency_check(large_portfolio)
    elapsed = time.time() - start
    assert elapsed < 0.5  # 目标500ms
    
@pytest.mark.performance  
def test_stress_testing_parallel():
    """9场景并行压力测试"""
    # 并行化测试，目标<100ms
    pass
```

**待办**:
- [ ] 设计性能测试场景（100/1000/10000标的）
- [ ] 建立性能基准线
- [ ] 集成到CI/CD流程
- [ ] 生产环境容量规划

---

#### 🟡 P2-2: 边界条件测试补充
**紧急度**: 🟡 中  
**目标**: 提高鲁棒性  
**时间**: 发布后2周

**测试场景**:
- [ ] 极端汇率波动（1:1000）
- [ ] 零成交量标的
- [ ] 负价格场景
- [ ] API超时/失败场景
- [ ] 内存溢出场景

---

### 中优先级（发布后3-4周）

#### 🟡 P2-3: 配置热重载
**紧急度**: 🟡 中  
**目标**: 运维便利  
**时间**: 发布后3周

**实现方案**:
```python
class MarketConfigManager:
    def update_config(self, new_config: Dict, validate: bool = True) -> bool:
        """运行时安全热更新配置"""
        # 1. 验证新配置
        if validate and not self._validate_config(new_config):
            return False
        
        # 2. 备份旧配置
        old_config = copy.deepcopy(self.market_registry)
        
        # 3. 原子性更新
        try:
            self.market_registry.update(new_config)
            self._notify_subscribers()  # 通知相关服务
            return True
        except Exception as e:
            # 4. 回滚
            self.market_registry = old_config
            logger.error(f"配置更新失败，已回滚: {e}")
            return False
```

**待办**:
- [ ] 设计配置版本管理
- [ ] 实现原子性更新
- [ ] 增加订阅通知机制
- [ ] 并发安全保障

---

### 低优先级（发布后4-8周）

#### 🟢 P2-4: 货币检查结果缓存
**紧急度**: 🟢 低  
**目标**: 优化体验  
**时间**: 发布后4周

```python
from functools import lru_cache
import hashlib

class CachedCurrencyChecker:
    def __init__(self, ttl_seconds=300):  # 5分钟缓存
        self._cache = TTLCache(maxsize=1000, ttl=ttl_seconds)
    
    def check_currency_consistency(self, market_data):
        cache_key = self._generate_cache_key(market_data)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = self._do_check(market_data)
        self._cache[cache_key] = result
        return result
```

---

#### 🟢 P2-5: 异步检查
**紧急度**: 🟢 低  
**目标**: 非核心路径优化  
**时间**: 发布后6周

---

#### 🟢 P2-6: 语义化版本控制
**紧急度**: 🟢 低  
**目标**: 长期维护  
**时间**: 发布后8周

```python
class MarketConfigManager:
    def __init__(self):
        self.api_version = "1.0.0"  # 语义化版本
        self.compatibility = {"1.0.0": ["1.1.0", "1.2.0"]}  # 兼容性矩阵
```

---

#### 🟢 P2-7: 场景相关性矩阵缓存
**紧急度**: 🟢 低  
**目标**: 性能优化  
**时间**: 发布后4周

---

#### P1: Sortino比率baseline参数业务映射
**问题**：基础设施层的baseline参数需要业务层映射  
**解决方案**：
```python
class RiskMetricsService:
    def calculate_sortino_ratio(self, returns, risk_free_rate=None, 
                               minimum_acceptable_return=None,
                               annualization_factor=252):
        """
        计算Sortino比率
        
        Args:
            risk_free_rate: 无风险利率（用于超额收益计算）
            minimum_acceptable_return: 最小可接受收益率（MAR）
        """
        # 确定baseline
        if minimum_acceptable_return is not None:
            baseline = minimum_acceptable_return
        elif risk_free_rate is not None:
            baseline = risk_free_rate
        else:
            baseline = 0.0  # 绝对损失风险
        
        # 调用基础设施层
        downside_std = self.calculator.calculate_downside_deviation(
            returns, baseline=baseline
        )
        
        excess_return = np.mean(returns) - baseline
        
        # 年化处理
        annualized_return = excess_return * annualization_factor
        annualized_downside_std = downside_std * np.sqrt(annualization_factor)
        
        return annualized_return / annualized_downside_std if annualized_downside_std != 0 else 0.0
```

**待办**：
- [ ] 实现 `calculate_sortino_ratio` 增强版
- [ ] 从配置中获取 `risk_free_rate`
- [ ] 支持策略级 `minimum_acceptable_return` 配置
- [ ] 添加单元测试

#### P2: 监管报告一致性配置
**待办**：
- [ ] 创建 `config/risk_config.py`
- [ ] 配置默认收益率计算方法
- [ ] 配置监管报告参数映射

---

## 🟢 portfolio_risk.py - PortfolioRiskAnalyzer

**状态**：✅ 已完成  
**测试覆盖**：8/8 通过

### 已完成
- [x] 组合风险分析
- [x] 因子风险归因
- [x] 持仓集中度风险
- [x] 8个测试用例

### 待办事项

#### P2: 风险归因可视化
**目标**：绘制风险分解图表（饼图、瀑布图）

```python
class RiskVisualizer:
    """风险可视化器"""
    
    def plot_factor_attribution(self, attribution_result):
        """绘制因子风险分解饼图"""
        # 市场风险、行业风险、风格风险、特质风险
        pass
    
    def plot_position_contribution_heatmap(self, contributions):
        """持仓风险贡献热力图"""
        # 各持仓对组合风险的贡献
        pass
    
    def plot_risk_trend(self, risk_history):
        """风险指标时间序列"""
        # VaR、波动率、最大回撤时间序列图
        pass
```

**待办**：
- [ ] 选择可视化库（matplotlib/plotly）
- [ ] 实现因子归因饼图
- [ ] 实现持仓贡献热力图
- [ ] 实现风险趋势图

---

## 🟢 stress_testing.py - StressTester

**状态**：✅ 已完成  
**测试覆盖**：9/9 通过

### 已完成
- [x] 6种基础压力场景
- [x] 场景分析框架
- [x] 9个测试用例

### 待办事项

#### P1: 历史极端事件回测
**目标**：使用历史真实数据复现极端事件

```python
class AdvancedStressTester(StressTester):
    """高级压力测试器"""
    
    HISTORICAL_EVENTS = {
        '2008_financial_crisis': {
            'start_date': '2008-09-15',
            'end_date': '2009-03-09',
            'market_drop': -0.40,  # 市场下跌40%
        },
        '2015_china_stock_crash': {
            'start_date': '2015-06-12',
            'end_date': '2015-08-26',
            'market_drop': -0.43,  # A股暴跌43%
        },
        '2020_covid_crash': {
            'start_date': '2020-02-20',
            'end_date': '2020-03-23',
            'market_drop': -0.34,  # 疫情引发暴跌
        }
    }
    
    def run_historical_scenario(self, event_name: str, portfolio):
        """历史极端事件回测"""
        event = self.HISTORICAL_EVENTS[event_name]
        # 获取历史数据
        # 重放市场行情
        # 计算组合损益
        pass
```

**待办**：
- [ ] 收集历史极端事件数据
- [ ] 实现历史场景回测框架
- [ ] 生成压力测试报告

#### P1: 蒙特卡洛模拟
**目标**：多情景组合概率分析

```python
def run_monte_carlo_simulation(self, portfolio, num_simulations=10000):
    """
    蒙特卡洛压力测试
    
    Returns:
        {
            'var_95': -0.05,  # 95%置信度VaR
            'cvar_95': -0.08,  # 95%置信度CVaR
            'worst_case': -0.15,  # 最坏情况
            'distribution': np.array([...])  # 损益分布
        }
    """
    pass
```

**待办**：
- [ ] 实现蒙特卡洛模拟引擎
- [ ] 参数估计（波动率、相关性）
- [ ] 结果可视化

#### P2: 自定义情景编辑器
**待办**：
- [ ] 设计情景配置格式（YAML/JSON）
- [ ] 实现情景加载器
- [ ] GUI情景编辑器（可选）

---

## 🟢 risk_monitor.py - RiskMonitor

**状态**：✅ 已完成  
**测试覆盖**：7/7 通过

### 已完成
- [x] 实时风险监控
- [x] 风险限额管理
- [x] 告警机制
- [x] 7个测试用例

### 待办事项

#### P2: 风险指标时间序列可视化
```python
class RiskMonitorDashboard:
    """风险监控仪表板"""
    
    def plot_real_time_metrics(self):
        """实时风险指标图表"""
        # VaR、CVaR、波动率实时曲线
        pass
```

**待办**：
- [ ] 实现实时数据采集
- [ ] 实现时间序列图表
- [ ] 集成到监控面板

---

## 🗓️ 实施顺序建议

### 短期（1-2周）
1. 🟡 RiskMetricsService 专家复审
2. 🟡 实现A股涨跌停场景处理
3. 🟡 实现Sortino比率增强版

### 中期（1个月）
4. 🟢 历史极端事件回测
5. 🟢 蒙特卡洛模拟

### 长期（3个月）
6. 🔵 风险归因可视化
7. 🔵 风险监控仪表板

---

## 🗓️ 历史变更

- **2024-11-09**: 创建风险模块独立TODO
- **2024-11-09**: risk_metrics_service.py 待进入评审阶段
- **2024-11-09**: 新增业务层改进储备清单
            'cvar_95': -0.08,  # 95%置信度CVaR
            'worst_case': -0.15,  # 最坏情况
            'distribution': np.array([...])  # 损益分布
        }
    """
    pass
```

**待办**：
- [ ] 实现蒙特卡洛模拟引擎
- [ ] 参数估计（波动率、相关性）
- [ ] 结果可视化

#### P2: 自定义情景编辑器
**待办**：
- [ ] 设计情景配置格式（YAML/JSON）
- [ ] 实现情景加载器
- [ ] GUI情景编辑器（可选）

---

## 🟢 risk_monitor.py - RiskMonitor

**状态**：✅ 已完成  
**测试覆盖**：7/7 通过

### 已完成
- [x] 实时风险监控
- [x] 风险限额管理
- [x] 告警机制
- [x] 7个测试用例

### 待办事项

#### P2: 风险指标时间序列可视化
```python
class RiskMonitorDashboard:
    """风险监控仪表板"""
    
    def plot_real_time_metrics(self):
        """实时风险指标图表"""
        # VaR、CVaR、波动率实时曲线
        pass
```

**待办**：
- [ ] 实现实时数据采集
- [ ] 实现时间序列图表
- [ ] 集成到监控面板

---

## 🗓️ 实施顺序建议

### 短期（1-2周）
1. 🟡 RiskMetricsService 专家复审
2. 🟡 实现A股涨跌停场景处理
3. 🟡 实现Sortino比率增强版

### 中期（1个月）
4. 🟢 历史极端事件回测
5. 🟢 蒙特卡洛模拟

### 长期（3个月）
6. 🔵 风险归因可视化
7. 🔵 风险监控仪表板

---

## 🗓️ 历史变更

- **2024-11-09**: 创建风险模块独立TODO
- **2024-11-09**: risk_metrics_service.py 待进入评审阶段
- **2024-11-09**: 新增业务层改进储备清单
- **2024-11-09**: 新增业务层改进储备清单
