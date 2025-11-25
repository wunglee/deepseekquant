# 风险管理模块 TODO（结构化清单）

> **层级**：Core Layer - Risk Management  
> **路径**：`core/risk/`  
> **职责**：风险指标计算、风险监控、压力测试
> **生产状态**：模块生产就绪（专家复审通过）；全局发布取决于完成 Phase 2（core_bak_refactored → core 融合）与全局集成测试

---

## 🧭 模块导航

1. [未分配事项（业务目标）](#未分配事项)  
2. [待办事项（技术储备）](#待办事项)  
3. [已完成事项（迁移至代码注释）](#已完成事项)

---

## 未分配事项

说明：仅维护未分配到具体迭代的业务目标或特性级待办，不落到具体代码文件维度。被分配后，迁移到本模块 SPRINT.md 相应迭代节点。

条目字段：标题、背景/价值、验收标准、优先级、关联领域、依赖/前置、实施方案（专家实施要点/代码建议）、相关文件、迁移状态。

### 持仓风险分析器历史验证与生产化（5B-5）
- 标题：历史压力场景回测与跨市场一致性校准
  - 背景/价值：提升模型在极端市场条件下的准确性与可靠性；确保跨市场风险指标的可比性
  - 验收标准：
    - 历史回测准确性：压力场景误差≤15%（2008金融危机、2015股灾等5个场景）
    - 跨市场一致性：汇率调整+流动性校准后，跨市场风险可比性≥85%
    - 行业差异化：金融/科技/周期股冲击系数差异≥10%
  - 优先级：P2（生产化增强）
  - 关联领域：持仓风险分析
  - 依赖/前置：
    - 历史压力场景数据（2008金融危机、2015股灾、2020疫情、2011欧债、2013钱荒）
    - 跨市场汇率数据与流动性参数
    - 行业分类数据（金融/科技/周期/防御）
  - 实施方案：
    - 接入历史压力场景数据（需咨询专家数据来源与口径）
    - 实现calibrate_cross_market_consistency()：基于USD基准的跨市场校准
    - 添加行业特性参数：金融股流动性溢价、科技股波动性调整、周期股系统性风险系数
    - 生产环境数据验证：与实际交易数据对比，建立持续校准机制
  - 相关文件：`core_bak_refactored/core/risk/position_risk.py`
  - 迁移状态：待分配（需专家确认数据来源与业务规则）

### 风险识别与响应
- 标题：动态性能阈值优化
  - 背景/价值：按组合规模调整SLA阈值，提高性能稳定性
  - 验收标准：<=100→5000ms，<=500→10000ms，>500→30000ms；采集p50/p95/p99
  - 优先级：P1
  - 关联领域：识别与响应
  - 依赖/前置：性能采集与统计
  - 实现方案：建立SLA策略（按组合规模分级），采集计算耗时样本并统计分位数，策略与阈值外部化到配置，不绑定具体文件；迭代分配后由SPRINT节点落地。
  - 相关文件：`core_bak_refactored/core/risk/portfolio_risk.py`
  - 迁移状态：未分配

### 审计与追溯
- 标题：数据质量监控增强
  - 背景/价值：提升数据完整性与一致性，降低异常影响
  - 验收标准：DATA_QUALITY_WARNING触发率≤3%，误报≤1%；关键字段完整性100%
  - 优先级：P1
  - 关联领域：审计与追溯
  - 依赖/前置：data_quality_policy 配置
  - 实现方案：定义数据质量策略（完整性/一致性规则集），构建自动标注与校验流程，合规标识与策略外部化；迭代分配后在门面入口挂载校验。
  - 相关文件：`core_bak_refactored/core/risk/portfolio_risk.py`
  - 迁移状态：未分配

- 标题：货币单位一致性检查
  - 背景/价值：避免多币种导致VaR与风险指标偏差
  - 验收标准：多币种检测与告警覆盖率100%；一致性验证通过率100%
  - 优先级：P1
  - 关联领域：审计与追溯
  - 依赖/前置：汇率适配器、币种字段覆盖
  - 实现方案：统一货币校验策略（组合维度），通过适配器进行汇率标准化与一致性检查，产生业务日志与告警；落地入口由SPRINT分配。
  - 相关文件：`core_bak_refactored/core/risk/risk_calculator.py`, `core_bak_refactored/core/share/exchange_rates.py`
  - 迁移状态：未分配

### 触发可靠性与成本
- 标题：配置管理界面（可视化）
  - 背景/价值：降低运维成本；支持阈值热更新与审计
  - 验收标准：参数热更新成功率100%；变更审计完整性100%
  - 优先级：P1
  - 关联领域：触发可靠性与成本
  - 依赖/前置：MarketConfigManager 接口
  - 实现方案：提供配置策略的UI/接口层，变更走审计流水与回滚策略，与运行时配置更新解耦；界面落地与后端接口由SPRINT分配。
  - 相关文件：`core_bak_refactored/core/share/market_config.py`
  - 迁移状态：未分配

- 标题：配置热更新支持
  - 背景/价值：运行时安全更新参数，支持回滚
  - 验收标准：原子性更新与回滚；并发安全；覆盖单元/集成测试
  - 优先级：P1
  - 关联领域：触发可靠性与成本
  - 依赖/前置：版本检测与通知机制
  - 实现方案：引入版本化配置与原子更新接口，失败自动回滚，并发安全通过锁/副本交换保证；与各模块入口解耦，分配后在SPRINT节点挂接。
  - 迁移状态：未分配

### 市场差异化合规
- 标题：集中度风险监控
  - 背景/价值：HHI阈值预警，完善CONCENTRATION_RISK标识
  - 验收标准：HHI≥阈值触发率与误报率达标；合规标识逻辑一致
  - 优先级：P1
  - 关联领域：市场差异化合规
  - 依赖/前置：组合权重数据质量
  - 实现方案：制定集中度阈值策略与预警规则，接入组合权重数据进行计算与标识触发，策略与阈值外部化；分配后在入口挂接。
  - 相关文件：`core_bak_refactored/core/risk/portfolio_risk.py`
  - 迁移状态：未分配

- 标题：实时市场数据集成
  - 背景/价值：动态无风险利率与关键参数更新
  - 验收标准：数据有效性验证；缓存机制；接口健壮性
  - 优先级：P2
  - 关联领域：市场差异化合规
  - 依赖/前置：数据源API与接口层
  - 实现方案：抽象实时数据接口层，提供数据校验与缓存策略，参数更新通过配置化策略应用；具体数据源由SPRINT迭代分配。
  - 相关文件：`core_bak_refactored/core/risk/risk_metrics_service.py`, `core_bak_refactored/core/share/market_config.py`
  - 迁移状态：未分配

---

## 待办事项

### 待办事项（技术债务 - 第5轮遗留）

#### ⚠️ P2: 压力测试代码优化重构（技术债务）

**背景**：  
第5轮Phase 1和Phase 2完成后，未执行规范要求的代码走查与优化重构。

**技术问题清单**：
- 重复代码：`_simulate_currency_crisis`和`_simulate_geopolitical_risk`中的恢复期计算逻辑重复
- 日志级别不当：使用`logger.debug`替代`logger.info`，生产环境可能缺少关键信息
- 缺少输入验证：未检查`total_exposure`为零的边界情况
- 数值稳定性：恢复期计算未限制极端值范围

**实施要点**：
- [ ] 提取公共方法：`_calculate_recovery_opportunity_cost(direct_loss, recovery_months)`
- [ ] 添加输入验证：检查`total_exposure == 0`并返回0.0
- [ ] 升级日志级别：关键计算步骤使用`logger.info`替代`debug`
- [ ] 数值稳定性：限制`recovery_months`在0-120月范围，避免指数爆炸
- [ ] 异常处理增强：添加`exc_info=True`到错误日志

相关文件：`core_bak_refactored/core/risk/stress_testing.py`  
状态评估：🔲 未实施（优先级P2，Phase 3完成后处理）

#### ⚠️ P2: 设计文档同步（技术债务）

**背景**：  
根据规范第741-839行要求，代码变更后必须同步更新设计文档。第5轮新增了2个场景、2个方法、2个数据类，但未同步文档。

**需要同步的变更**：
1. **模块设计文档**：
   - 新增场景说明（1997亚洲金融危机、2022俄乌冲突）
   - 更新变更历史（v1.x, 2025-11-25）
   
2. **接口设计文档**：
   - 新增数据类API：`StressTestResult`、`CombinedStressTestResult`
   - 新增方法文档：`_simulate_currency_crisis()`、`_simulate_geopolitical_risk()`
   - 更新使用示例（展示新数据类用法）

**实施要点**：
- [ ] 更新`docs/design/core/risk/模块设计文档.md`
- [ ] 更新`docs/design/core/risk/接口设计文档.md`
- [ ] 添加版本历史条目
- [ ] 标记变更章节（如：`[v1.1 新增]`）

相关文件：  
- `docs/design/core/risk/模块设计文档.md`
- `docs/design/core/risk/接口设计文档.md`

状态评估：🔲 未实施（优先级P2，Phase 3完成后处理）

---

### 待办事项（业务层改进储备）

> **📝P0修复总结（2024-11-09）**
> - ✅ CVaR保守估计策略：使用参数法CVaR替代1.1倍系数
> - ✅ 默认置信度分层：支持95%/99%，区分日常监控和监管报告
> - ✅ 无风险利率动态化：支持动态配置和参数传入
> - ✅ 符号约定统一：VaR/CVaR返回正数表示损失，文档清晰
> - ✅ A股涨跌停处理：检测机制 + 日志警告 + 4个专项测试

---

#### ⚠️ P1: 货币单位一致性检查（专家建议 - 下迭代补充）

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

相关文件：`core_bak_refactored/core/risk/risk_calculator.py`, `core_bak_refactored/core/share/exchange_rates.py`

子项（下一层级）：
**待办**：
- [ ] 在RiskCalculator中添加货币一致性验证
- [ ] 支持多币种警告和日志记录
- [ ] 添加单元测试

状态评估：✅ 已部分实现，在 [risk_calculator.py](file:///Users/wangli/Library/Mobile%20Documents/com~apple~CloudDocs/历史项目/projects/deepseekquant/core_bak_refactored/core/risk/risk_calculator.py) 中已实现货币一致性检查相关功能，包括 [_runtime_currency_check](file:///Users/wangli/Library/Mobile%20Documents/com~apple~CloudDocs/历史项目/projects/deepseekquant/core_bak_refactored/core/risk/risk_calculator.py#L184-L215)、[_handle_currency_warnings](file:///Users/wangli/Library/Mobile%20Documents/com~apple~CloudDocs/历史项目/projects/deepseekquant/core_bak_refactored/core/risk/risk_calculator.py#L217-L232) 等方法。需要补充单元测试。

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

状态评估：🔲 未实现，仍处于待办状态。

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

状态评估：🔲 未实现，仍处于待办状态。

---

#### ✅ P1.5: 货币一致性增强与市场特定策略（专家建议 - 已完成）

**评审文档**：`docs/ask.md` + `docs/answer.md`（专家95分，批准发布）  
**完成时间**：2024-11-14

状态评估：✅ 已完成，在 [risk_calculator.py](file:///Users/wangli/Library/Mobile%20Documents/com~apple~CloudDocs/历史项目/projects/deepseekquant/core_bak_refactored/core/risk/risk_calculator.py) 中已实现完整的货币一致性检查功能。

---

## 风险计算优化实施（2024-11-21）
**待办**:
- [ ] P2-1: 多级缓存架构（L1+L2 Redis+L3磁盘，目标命中率>85%）
- [ ] P2-2: 智能预热系统（基于模式的预测预热）
- [ ] P2-3: 业务感知失效规则（市场波动率、重大事件检测）
- [ ] P2-4: 风险指标体系完善（Sortino/Omega/Calmar比率）
- [ ] P2-5: 多市场因子适配（动态适配器）

**性能目标调整**:
| 指标 | 当前目标 | 专家建议 |
|------|----------|----------|
| 缓存命中率 | >70% | **>85%** |
| 并行加速比 | 2-3x | **3-5x** |
| 因子模型延迟 | <50ms | **<20ms** |

**关键决策**:
1. 因子模型集成时机：立即开始渐进式集成（PHASE_2: PCA + 部分真实数据）
2. 增量计算与因子模型结合：强烈建议结合

状态评估：🔲 未实现，仍处于待办状态。

---

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

相关文件：`core_bak_refactored/core/share/market_config.py`, `core_bak_refactored/core/risk/risk_calculator.py`

子项（下一层级）：
**待办**:
- [ ] 设计配置版本管理
- [ ] 实现原子性更新
- [ ] 增加订阅通知机制
- [ ] 并发安全保障

状态评估：🔲 未实现，仍处于待办状态。

---

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

**待办**:
- [ ] 设计缓存键生成策略
- [ ] 实现缓存装饰器
- [ ] 添加缓存失效机制
- [ ] 补充性能测试

状态评估：🔲 未实现，仍处于待办状态。

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

相关文件：`core_bak_refactored/core/risk/risk_metrics_service.py`, `core_bak_refactored/infrastructure/risk_metrics.py`

子项（下一层级）：
**待办**：
- [ ] 实现 `calculate_sortino_ratio` 增强版
- [ ] 从配置中获取 `risk_free_rate`
- [ ] 支持策略级 `minimum_acceptable_return` 配置
- [ ] 添加单元测试

状态评估：🔲 未实现，仍处于待办状态。

---

#### P2: 监管报告一致性配置
**待办**：
- [ ] 创建 `config/risk_config.py`
- [ ] 配置默认收益率计算方法
- [ ] 配置监管报告参数映射

状态评估：🔲 未实现，仍处于待办状态。

---

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

相关文件：`core_bak_refactored/core/risk/portfolio_risk.py`, `core_bak_refactored/core/risk/position_risk.py`

子项（下一层级）：
**待办**：
- [ ] 选择可视化库（matplotlib/plotly）
- [ ] 实现因子归因饼图
- [ ] 实现持仓贡献热力图
- [ ] 实现风险趋势图

状态评估：🔲 未实现，仍处于待办状态。

---

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

相关文件：`core_bak_refactored/core/risk/stress_testing.py`

子项（下一层级）：
**待办**：
- [ ] 收集历史极端事件数据
- [ ] 实现历史场景回测框架
- [ ] 生成压力测试报告

状态评估：✅ 已部分实现，压力测试器已完成并在 [stress_testing.py](file:///Users/wangli/Library/Mobile%20Documents/com~apple~CloudDocs/历史项目/projects/deepseekquant/core_bak_refactored/core/risk/stress_testing.py) 中实现。

---

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

相关文件：`core_bak_refactored/core/risk/risk_calculator.py`, `core_bak_refactored/core/risk/risk_metrics_service.py`

子项（下一层级）：
**待办**：
- [ ] 实现蒙特卡洛模拟引擎
- [ ] 参数估计（波动率、相关性）
- [ ] 结果可视化

状态评估：✅ 已部分实现，在 [risk_calculator.py](file:///Users/wangli/Library/Mobile%20Documents/com~apple~CloudDocs/历史项目/projects/deepseekquant/core_bak_refactored/core/risk/risk_calculator.py) 中已实现 [calculate_var_monte_carlo](file:///Users/wangli/Library/Mobile%20Documents/com~apple~CloudDocs/历史项目/projects/deepseekquant/core_bak_refactored/core/risk/risk_calculator.py#L133-L177) 方法。

---

## 🟢 risk_monitor.py - RiskMonitor
#### P2: 风险指标时间序列可视化
相关文件：`core_bak_refactored/core/risk/risk_monitor.py`
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

状态评估：🔲 未实现，仍处于待办状态。

---

## 已完成事项

> **说明**：以下事项已完成，详细信息已迁移至代码文件注释中。

- [x] risk_metrics_service.py - RiskMetricsService（国际化支持、P0修复等）
- [x] risk_calculator.py - RiskCalculator（货币一致性检查增强等）
- [x] portfolio_risk.py - PortfolioRiskAnalyzer（P2.1审计字段增强等）

状态评估：
✅ risk_metrics_service.py - 已完成，包含国际化支持和P0修复
✅ risk_calculator.py - 已完成，包含货币一致性检查增强
✅ portfolio_risk.py - 已完成，包含P2.1审计字段增强