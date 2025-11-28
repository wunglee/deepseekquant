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
  - 实施方案：建立SLA策略（按组合规模分级），采集计算耗时样本并统计分位数，策略与阈值外部化到配置，不绑定具体文件；迭代分配后由SPRINT节点落地。
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

- 标题：动态货币严格模式（基于组合特征）
  - 背景/价值：当前严格模式仅按市场静态配置，无法识别跨境组合、多币种组合，存在低估风险或过度限制的可能。
  - 验收标准：
    - 能根据组合的多币种占比、跨境敞口、监管叠加规则，动态决定是否启用严格模式；
    - 严格模式触发准确率>95%，误报率<5%；
    - 与第2轮专家答复中 `_determine_dynamic_strict_mode` 设计保持一致。
  - 优先级：✅ 已完成（Phase 1，2025-11-27）
  - 关联领域：审计与追溯 / 货币一致性检查
  - 实施成果（第4轮验收）：
    - ✅ 第3轮专家已确认阈值/权重结构（multi_currency_ratio_threshold=0.30、cross_border_exposure_threshold市场差异化、component_weights、comprehensive_trigger_score=0.65）
    - ✅ 实施完成：`_determine_dynamic_strict_mode` 方法及4个辅助方法（`_calculate_multi_currency_score`、`_calculate_cross_border_score`、`_calculate_regulatory_overlay_score`、`_calculate_comprehensive_score`）
    - ✅ 测试覆盖：15/15通过（含JP/SG/CN/EU市场阈值、边界条件0.64 vs 0.65）
    - ✅ cross_border_exposure 数据来源确认：由Portfolio模块提供（字段缺失时返回None）
    - ✅ regulatory_overlay_rules 接口占位：可选维度，无数据时返回None，不影响其他维度综合评分
  - 迁移状态：✅ 已完成（2025-11-27）

- 标题：美股合规阻断与人工审批机制
  - 背景/价值：当前 `_us_compliance_logging` 仅记录合规事件，不阻断交易或触发审批，难以满足高等级合规场景。
  - 验收标准：
    - 当估值影响>2% 或存在监管货币违规时，能够正确触发“阻断”；
    - MEDIUM 级别事件触发人工审批比例<10%；
    - 阈值与处理逻辑与第2轮专家答复中的 block_thresholds 建议保持一致。
  - 优先级：P1（Phase 2）
  - 关联领域：触发可靠性与成本 / 合规审计
  - 依赖/前置：
    - 与 exec 模块的风险门控设计对齐（下单前统一入口）；
    - 专家确认阻断/审批场景边界。
  - 实施方案：
    - 在 risk 域保留合规评分与事件封装，不直接做交易阻断；
    - 在 exec 域的风险门控逻辑中，根据合规事件字段决定是否阻断或转人工审批。

- 标题：多维度数据质量评估模型（DataQualityAssessment）
  - 背景/价值：当前数据质量评估仅基于货币字段覆盖率，难以全面反映数据完整性、准确性、及时性等维度，对风险评估支撑有限。
  - 验收标准：
    - 根据 completeness/accuracy/consistency/timeliness/reliability 五个维度给出 0–100 综合评分；
    - 数据质量评级准确率>90%，与风险误差相关系数>0.8（参考第2轮专家答复指标）。
  - 优先级：P2（Phase 3）
  - 关联领域：审计与追溯 / 数据质量监控
  - 实施成果（第4轮Phase 1）：
    - ✅ 第3轮专家已确认 base_weights/grade_thresholds/usage_scenarios
    - ✅ 第一阶段完成：仅实现 completeness 维度（基于 currency_coverage），reporting_only不影响计算
    - ✅ 实施方法：`_assess_data_quality_multi`、`_convert_score_to_grade`、`_calculate_currency_coverage`（共享辅助方法）
    - ✅ 测试覆盖：B/C/D级数据质量测试通过（75≤B<90、60≤C<75、D<60）
    - ✅ 分级阈值：A≥90、B≥75、C≥60、D<60
  - 依赖/前置（Phase 3后续维度）：
    - 第4轮专家建议：accuracy（价格离群值检测）、consistency（时间序列连续性）、timeliness（数据延迟统计）、reliability（历史质量记录）
    - 市场差异化配置：是否需要按市场设置不同权重和阈值（如US更关注accuracy、HK更关注completeness）
    - 与 data 模块的实际数据可用性对齐
  - 实施方案（后续阶段）：
    - 设计 `DataQualityAssessment` 类，提供 `comprehensive_quality_score(market_data)` 接口
    - 使用第2轮/第4轮 answer 中示例的权重作为初始实现，后续通过 ask 确认后固化
  - 迁移状态：🔄 Phase 1完成（completeness），Phase 3规划中（其他维度）

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
  - 迁移状态：未实现

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

## 已完成事项

> **说明**：以下事项已完成，详细信息已迁移至代码文件注释中。

- [x] risk_metrics_service.py - RiskMetricsService（国际化支持、P0修复等）
- [x] risk_calculator.py - RiskCalculator（货币一致性检查增强等）
- [x] portfolio_risk.py - PortfolioRiskAnalyzer（P2.1审计字段增强等）
- [x] calculate_var_monte_carlo迁移 - 将Monte Carlo VaR计算从RiskCalculator迁移至RiskMetricsService

状态评估：
✅ risk_metrics_service.py - 已完成，包含国际化支持和P0修复
✅ risk_calculator.py - 已完成，包含货币一致性检查增强
✅ portfolio_risk.py - 已完成，包含P2.1审计字段增强
✅ calculate_var_monte_carlo迁移 - 已完成，符合专家要求
