# 第5轮咨询 — 准入准备

## 📋 Phase边界声明（必须）
- 当前阶段：风险模块协调器与前置检查，本轮进行专家确认；不讨论生产发布与跨域融合。
- 系统范围：仅限`core_bak_refactored`；不修改根目录`core/`模块。

## 背景说明
- 本轮为文档与架构对齐咨询轮：代码仅做小步辅助性调整（如缓存/风险模块内部职责澄清与校验增强），不改变对外行为；现有单元测试保持通过（15/15）。

#### 📚 依赖上下文与设计文档清单（必须）
- 设计文档：
  - `docs/design/core_bak_refactored/core/risk/模块设计文档.md`
  - `docs/design/core_bak_refactored/core/risk/接口设计文档.md`
  - `docs/design/core_bak_refactored/ARCHITECTURE.md`
- 共享配置与基础设施：
  - `core_bak_refactored/core/share/market_config.py`
  - `core_bak_refactored/infrastructure/cache_service.py`

## 🧩 代码评审

### 上一轮答复摘要与本轮改进（必须)
- 摘要回顾（上一轮 `docs/answer.md`）：
  - 动态严格模式：配置驱动、分市场阈值、综合触发分数；缺失配置/数据返回None。
  - 多维数据质量：Phase 1仅completeness，分级阈值A/B/C/D；report-only不影响计算。
  - 高优先级改进建议：
    - 4.1.1：在 RiskCalculator 初始化时验证必要配置项（仅记录告警，不阻断）。
    - 4.1.2：支持市场差异化配置的读取（market_specific 回退默认）。
- 代码清单汇总：
  - 核心实现：`core_bak_refactored/core/risk/risk_calculator.py`
  - 单元测试：`core_bak_refactored/tests/units/core/risk/risk_calculator_test.py`
  - 配置与依赖：`core_bak_refactored/core/share/market_config.py`、`core_bak_refactored/infrastructure/cache_service.py`
- 本轮改进映射：
  - 已新增 `RiskCalculator._validate_required_fields()` 并在初始化后调用；对动态严格模式与数据质量评估的关键配置项进行告警校验。
  - 已新增 `RiskCalculator._get_market_specific_config(config_key, default_config)`；当存在 `market_specific` 时读取当前市场配置，否则回退默认。
  - 不改变既有指标计算与行为；单元测试 15/15 通过。
- 改进考虑：
  - 严格遵循“配置驱动、无默认值”，监管维度可选；缺失字段不替代估算，只记录告警。
  - 校验仅提示问题，不阻断初始化；市场差异化读取保持向后兼容（默认回退）。

### 业务视角的代码实现评审要点
- 触发可靠性：综合评分的维度选择与加权完全由配置驱动；缺失维度不替代估算，避免误触发。
- 市场差异化：跨境阈值按市场读取；测试覆盖体现差异化生效。
- 数据质量治理：Phase 1仅报告不干预计算；分级标准清晰可验证。
- 合规路径：US合规日志保留；监管维度可选占位，待规则与数据就绪后再参与评分。
- 契约一致性：输入/输出/异常路径与业务口径一致；协调器不承载算法实现，仅委托与前置检查。

### ✅ 本轮改进验收清单（专家确认）
- 监管：规则Schema、评分公式、阻断阈值、动作映射、审计字段、exec集成点。
- 跨境：数据来源与计算标准；缺失策略。
- 数据质量：Phase 3维度、各市场权重/阈值、各等级动作策略。
- 性能与审计：目标与留存；必需字段。

### 本轮改进（清单与关键评审）（必须)
- 文件：`core_bak_refactored/core/risk/risk_calculator.py`
  - 关键方法：
    - `_determine_dynamic_strict_mode(data)`：读取`dynamic_currency_strict_mode`配置（enabled/weights/thresholds），按可用维度计算综合评分；当配置或数据缺失时返回None（不覆盖静态）。
    - `_calculate_multi_currency_score(allocations, market_data, cfg)`：基于`base_currency`计算非基准货币权重占比。
    - `_calculate_cross_border_score(portfolio, cfg)`：读取`portfolio.cross_border_exposure`，字段缺失返回None。
    - `_calculate_regulatory_overlay_score(allocations, cfg)`：占位实现；无规则/数据时返回None；保持监管维度可选。
    - `_calculate_comprehensive_score(scores, cfg)`：仅对存在的维度按`component_weights`加权；权重与维度键名严格一致。
    - `_assess_data_quality_multi(market_data, dq_cfg)`：Phase 1仅计算`completeness=currency_coverage×100`；`usage_scenarios.report_only=True`，不影响指标。
    - `_convert_score_to_grade(score, thresholds)`：A≥90、B≥75、C≥60、D<60。
    - `_calculate_currency_coverage(prices)`：返回(总标的数、有currency的标的数、覆盖率)。
  - 字段/返回/异常与业务口径映射：
    - 输入字段：`market_data.prices[*].currency`、`portfolio.allocations[*].weight`、`portfolio.cross_border_exposure`、配置项（阈值/权重/触发分）。
    - 返回约定：布尔决策/评分浮点/数据质量字典；异常路径不抛未声明异常，记录日志后返回安全值（如None或{}）。
    - 业务口径：严格遵循“配置驱动、无默认值”；监管维度可选；缺失字段导致维度不参与评分而非替代估算。
- 文件：`core_bak_refactored/tests/units/core/risk/risk_calculator_test.py`
  - 关键评审点：
    - 分市场阈值验证（JP/SG/CN/EU）；
    - 边界值（0.64不触发、0.66触发）；
    - 数据质量分级（B/C/D）与阈值边界；
    - 缺失字段容错（cross_border_exposure缺失返回None）。

### 🧩 架构变更与影响（如果有）
- 协调器保持不实现算法；动态严格决策与数据质量评估定位为“前置检查/报告”。
- 监管维度在有规则/数据时参与评分；无则返回None；综合评分按可用维度加权。
- `cross_border_exposure`由Portfolio模块提供；Risk不设默认值，不在缺失时自行估算。

## 本轮业务问题（下一轮需解决，非本轮验收内容）

### 领域知识
- 请您确认监管维度的规则Schema与评分公式：权重结构、优先级因子与聚合；并按市场（US/HK/CN/JP/SG/EU）提供示例口径。
- 请您确认Portfolio为`cross_border_exposure`唯一数据源；启用动态严格模式时该字段为必需；请说明计算口径（地域/货币/综合）与一致性示例。
- 请您确认数据质量Phase 3维度定义与数据来源：accuracy/consistency/timeliness/reliability 的指标说明。
- 请您确认审计留存口径与最小必需字段清单（内部≥90天、监管≥3年）。

### 优化机会
- 请您确认各市场的数据质量权重与阈值的差异化优化（如US更关注accuracy、HK更关注completeness）。
- 请您确认阻断阈值与动作映射的优化方向，降低误触发并提升一致性。
- 请您确认性能目标达成路径（100标的×63日窗口P95≤500ms、极端P99≤1500ms）。

### 实施路径
- 请您确认监管维度在Phase 2是否“必选”；如需“必选”，请提供最小可行规则集与数据源。
- 请您确认exec域集成点：下单门控的调用位置与必需字段（风险分、监管标记、原因、时间戳）。
- 请您确认缺失策略：当关键字段缺失时动态严格模式返回None（不覆盖静态）；不发明默认值。

## 🔗 相关文件（参考）
- `core_bak_refactored/core/risk/risk_calculator.py`
- `core_bak_refactored/tests/units/core/risk/risk_calculator_test.py`

## 📝 说明（必须）
- 重要：请尽可能详尽和充分，不要遗漏和简化，谢谢！
