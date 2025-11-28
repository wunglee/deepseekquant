# 第6轮咨询 — Phase 1 验收与下一轮业务口径澄清

## 📋 Phase边界声明

- ✅ 当前阶段为 **5D 风险计算协调器 Phase 1 验收评审**，范围界定如下：
  - **系统边界**：仅限 `core_bak_refactored` 临时系统；不涉及生产部署与核心系统融合。
  - **Phase 1 范围**：货币一致性检查（静态+动态严格模式）、数据质量评估（completeness 维度）、US 合规日志、智能缓存失效上下文集成、分市场阈值与回退链路一致性。
  - **本轮状态**：Phase 1 代码已实施；15/15 单元测试通过；业务行为"仅报告、不阻断"（usage_scenarios.reporting_only=True, affect_calculation=False）。
  - **Phase 2/3 规划**：合规阻断门控（exec 域）、多维度数据质量评估（accuracy/consistency/timeliness/reliability）已转入 TODO，不纳入本轮 Phase 1 验收。

---

## 背景说明

- 本轮为 **5D 风险计算协调器 Phase 1** 的业务评审，聚焦业务口径与合规路径澄清；本轮已完成货币检查、数据质量 completeness 维度、US 合规日志与智能缓存上下文集成的实现，保持"仅报告、不影响计算"原则；15/15 单元测试全部通过。
- 本轮评审目标：请您确认 Phase 1 实现的业务正确性与完整性；并为下一轮（Phase 2/3 或其他业务增强）提供业务口径与路径指导。

### 📚 依赖上下文与设计文档清单

**核心设计文档**：
- `docs/design/core_bak_refactored/core/risk/模块设计文档.md`（v1.3，2025-11-27 更新）
- `docs/design/core_bak_refactored/core/risk/接口设计文档.md`（v1.3，2025-11-27 更新）
- `docs/process/core_bak_refactored/core/risk/SPRINT.md`（5D 迭代目标与进展）

**基础配置与共享模块**：
- `core_bak_refactored/core/share/market_config.py`：6 市场配置管理（US/HK/CN/JP/EU/SG）
- `core_bak_refactored/core/share/exchange_rates.py`：货币转换与汇率适配器
- `core_bak_refactored/infrastructure/cache_service.py`：智能缓存失效管理器

---

## 🧩 代码评审

### 上一轮答复摘要与本轮改进

#### 上一轮专家答复关键点回顾

**第 5 轮答复（合并版 `docs/answer.md`）的核心确认**：
1. **触发可靠性**：综合评分完全由配置驱动；缺失维度不参与计算；无配置时返回 None 保持静态模式。
2. **市场差异化**：US/HK/JP/SG/CN/EU 各市场阈值与权重已确认；US 跨境敞口 25%、HK 40%、JP 20%、SG 35%、CN 50%、EU 30%。
3. **数据质量治理 Phase 1**：仅 completeness 维度（基于 currency_coverage），reporting_only 不影响计算；A≥90、B≥75、C≥60、D<60。
4. **合规路径（US）**：SEC/FINRA 结构化合规事件记录；当前 automated_action='LOG_ONLY'（不阻断）。
5. **协调器职责边界**：仅委托与前置检查，不实现算法；配置驱动、无默认值。

#### 本轮代码清单汇总

**核心实现文件（本轮改进）**：
- `core_bak_refactored/core/risk/risk_calculator.py`
  - 新增方法：`_validate_required_fields()`、`_get_market_specific_config()`、`_runtime_currency_check()`、`_determine_dynamic_strict_mode()`、`_assess_data_quality_multi()`、`_us_compliance_logging()`
  - 集成智能缓存失效上下文：`calculate_all_metrics()` 中调用 `get_smart_invalidation_manager().check_and_invalidate(context)`

**测试文件（本轮验证）**：
- `core_bak_refactored/tests/units/core/risk/risk_calculator_test.py`
  - 15 个单元测试用例（US/HK/JP/SG/CN/EU 市场阈值、边界条件 0.64 vs 0.65、B/C/D 级数据质量）

**配置与基础设施依赖**：
- `core_bak_refactored/core/share/market_config.py`：6 市场基准货币与阈值
- `core_bak_refactored/infrastructure/cache_service.py`：智能缓存失效管理器

#### 本轮改进映射（对应上一轮答复）

| 上一轮答复条目 | 本轮改进实施 | 改进考虑 |
|--------------|------------|---------|
| **配置项验证**（4.1.1） | ✅ `_validate_required_fields()`：动态严格模式与数据质量评估的关键配置项验证（仅告警，不阻断） | 配置驱动；缺失时告警而非发明默认值 |
| **市场差异化配置读取**（4.1.2） | ✅ `_get_market_specific_config()`：`market_specific → fallback default` 模式 | 保证市场差异化配置的读取行为稳定；无配置时回退到默认 |
| **触发可靠性 — 配置驱动**（实现符合性 1） | ✅ `_determine_dynamic_strict_mode()` 与四个辅助方法：缺失维度返回 None；无配置不覆盖静态模式 | 配置驱动；不发明默认值；维度缺失时综合评分返回 None |
| **市场差异化 — 跨市场阈值**（实现符合性 2） | ✅ 测试覆盖 US/HK/JP/SG/CN/EU 市场阈值差异化（15 个用例） | 6 市场阈值生效；边界条件（0.64 vs 0.65）精确验证 |
| **数据质量治理 Phase 1**（实现符合性 3） | ✅ `_assess_data_quality_multi()`：completeness 维度（currency_coverage × 100），reporting_only=True, affect_calculation=False | Phase 1 仅报告；A/B/C/D 分级；不影响风险计算 |
| **US 合规日志**（实现符合性 4） | ✅ `_us_compliance_logging()`：SEC/FINRA 结构化事件记录；automated_action='LOG_ONLY' | 仅 US 市场触发；不阻断交易；结构化日志 |
| **协调器职责边界**（实现符合性 5） | ✅ `calculate_all_metrics()` 委托模式：前置检查 → 委托 `RiskMetricsService` | 协调器仅委托与前置检查；不实现算法 |

---

### 业务视角的代码实现评审要点

#### 1. 货币一致性检查与动态严格模式

**业务契约**：
- **静态严格模式**：US/HK/SG/JP 默认严格（基于监管要求）；CN/EU 默认非严格（市场特征）。
- **动态严格模式**：根据组合特征（多币种占比、跨境敞口）动态决策是否启用严格模式；无配置或数据不足时返回 None，保持静态模式。

**业务口径一致性**：
- `_runtime_currency_check()`：检测多币种、货币缺失、基准货币不一致、组合货币不一致；仅日志告警，不阻断。
- `_determine_dynamic_strict_mode()`：
  - 多币种评分 = 非基准货币权重比例；
  - 跨境敞口评分 = portfolio['cross_border_exposure']（由 Portfolio 模块提供）；
  - 监管叠加评分 = 当前返回 None（规则/数据缺失时不计算）；
  - 综合评分 = 加权平均（仅当配置要求的维度都有分数时计算）；
  - 决策：综合评分 ≥ comprehensive_trigger_score 时启用严格模式。

**触发可靠性**：
- **边界条件验证**：0.64 < 0.65（不触发）、0.66 ≥ 0.65（触发），确保阈值精确生效。
- **跨市场阈值差异化**：US 25%、HK 40%、JP 20%、SG 35%、CN 50%、EU 30%；测试覆盖 6 市场。

#### 2. 数据质量评估（Phase 1：completeness）

**业务契约**：
- **Phase 1 范围**：仅实现 completeness 维度（基于 currency_coverage）；其他维度（accuracy/consistency/timeliness/reliability）留待 Phase 3。
- **usage_scenarios**：reporting_only=True, affect_calculation=False（仅报告，不影响风险计算）。

**业务口径一致性**：
- `_assess_data_quality_multi()`：
  - 维度得分：completeness = currency_coverage × 100（0-100）；
  - 综合评分：当前 Phase 1 仅有 completeness 一个维度，overall_score = completeness；
  - 等级映射：A≥90、B≥75、C≥60、D<60（可配置阈值）。

**分级准确性**：
- **测试覆盖**：A 级（100%）、B 级（80%）、C 级（60%）、D 级（20%）；4 个等级均验证通过。

#### 3. US 合规日志与审计留存

**业务契约**：
- **触发条件**：仅 US 市场 + 存在货币警告时触发。
- **事件结构**：event_id（UUID）、event_type（CURRENCY_INCONSISTENCY）、message、timestamp（ISO8601）、market、severity（MEDIUM/HIGH）、automated_action（LOG_ONLY）。

**合规路径**：
- 当前 Phase 1：仅结构化日志记录，不阻断交易（automated_action='LOG_ONLY'）。
- Phase 2 规划：根据 severity 与 block_thresholds 触发阻断或人工审批（exec 域门控，非 risk 域职责）。

#### 4. 智能缓存失效上下文集成

**业务契约**：
- **触发条件**：波动率超阈值、熔断机制触发、极端相关性崩溃、涨跌停比例超阈值、重大市场事件。
- **上下文字段**：time_window、param_version、market_data_updated、volatility、market_type、portfolio_size、data_quality_rating、volatility_tier（NORMAL/MEDIUM/HIGH/EXTREME）、market_status、circuit_breaker_triggered、extreme_correlation_breakdown、limit_hit_ratio、major_market_event、trigger_score、affected_symbols_count。

**触发可靠性**：
- **评分机制**：trigger_score = (returns_std / threshold) + 事件权重加权（circuit_breaker/extreme_correlation/limit_hits/major_event）。
- **影响范围估算**：affected_symbols_count = 组合标的数量（根据 portfolio_state）。

---

### ✅ 本轮改进验收清单（专家确认）

请您逐条确认以下 Phase 1 实现的业务正确性：

#### A. 货币一致性与动态严格模式
- [ ] **静态严格模式口径**：US/HK/SG/JP 默认严格、CN/EU 默认非严格，是否符合监管要求与市场特征？
- [ ] **跨市场阈值差异化**：US 25%、HK 40%、JP 20%、SG 35%、CN 50%、EU 30%，是否符合各市场跨境投资监管边界？
- [ ] **综合评分触发阈值**：0.65（默认）是否为合理的严格模式启用边界？
- [ ] **cross_border_exposure 数据来源**：由 Portfolio 模块提供；缺失时返回 None 不参与综合评分，是否符合业务口径？
- [ ] **regulatory_overlay_rules 可选性**：监管叠加规则当前为可选维度（无数据时返回 None），是否符合 Phase 1 范围定位？

#### B. 数据质量评估（Phase 1：completeness）
- [ ] **Phase 1 范围界定**：仅 completeness 维度（currency_coverage），reporting_only 不影响计算，是否符合 Phase 1 目标？
- [ ] **分级阈值口径**：A≥90、B≥75、C≥60、D<60，是否符合数据质量监控标准？
- [ ] **usage_scenarios 定位**：reporting_only=True, affect_calculation=False，是否为合理的 Phase 1 → Phase 2 过渡策略？

#### C. US 合规日志与审计留存
- [ ] **触发范围**：仅 US 市场 + 货币警告，是否符合 SEC/FINRA 合规要求？
- [ ] **事件 severity 分级**：多币种=MEDIUM、货币不一致=HIGH，是否符合合规风险等级定义？
- [ ] **automated_action 口径**：当前仅 LOG_ONLY（不阻断），是否为合理的 Phase 1 策略？Phase 2 需在何种条件下启用阻断？

#### D. 智能缓存失效与触发可靠性
- [ ] **波动率触发阈值**：默认 0.05（可配置），是否为合理的市场波动敏感度边界？
- [ ] **涨跌停比例阈值**：默认 0.3（可配置），是否符合 A 股/港股等市场的极端行情判断标准？
- [ ] **事件权重配置**：circuit_breaker、extreme_correlation、limit_hits、major_event 各自权重，是否需要按市场差异化？

---

### 本轮改进（清单与关键评审）

#### 文件 1：`core_bak_refactored/core/risk/risk_calculator.py`

**关键方法 1**：`_validate_required_fields()`

**业务口径**：
- **职责**：验证动态严格模式与数据质量评估的关键配置项（component_weights、comprehensive_trigger_score、base_weights、grade_thresholds）。
- **行为**：仅记录告警，不阻断初始化；配置驱动，不发明默认值。

**方法签名**：
```python
def _validate_required_fields(self) -> None
```

**输入**：无（读取 `self.config`）  
**输出**：无（仅日志告警）  
**异常**：不抛出异常（仅 logger.warning）

**业务评审点**：
- 配置项验证是否覆盖关键字段？
- 告警级别（warning）是否符合配置缺失的业务影响？

---

**关键方法 2**：`_get_market_specific_config(config_key: str, default_config: Dict[str, Any])`

**业务口径**：
- **职责**：获取市场特定配置，支持 `market_specific[market_type] → default` 回退链路。
- **行为**：优先读取 market_specific；缺失时回退到 default_config；异常时也回退（不阻断）。

**方法签名**：
```python
def _get_market_specific_config(self, config_key: str, default_config: Dict[str, Any]) -> Dict[str, Any]
```

**输入**：
- `config_key`：配置键名（例：'data_quality_assessment'）
- `default_config`：默认配置字典

**输出**：市场特定配置字典（Dict[str, Any]）  
**异常**：不抛出异常（异常时返回 default_config）

**业务评审点**：
- 回退链路是否符合市场差异化配置的容错要求？
- 默认配置的优先级是否低于市场特定配置？

---

**关键方法 3**：`_runtime_currency_check(data: Dict[str, Any])`

**业务口径**：
- **职责**：运行时货币一致性检查（仅日志，不阻断）。
- **检测项**：多币种、货币字段缺失、基准货币不在检测货币中、组合货币≠基准货币。

**方法签名**：
```python
def _runtime_currency_check(self, data: Dict[str, Any]) -> List[str]
```

**输入**：`data`（包含 market_data/portfolio）  
**输出**：警告列表（List[str]）  
**异常**：不抛出异常

**业务评审点**：
- 检测项是否覆盖主要货币不一致场景？
- 警告列表是否足够支撑后续分级处理（info/warning/error）？

---

**关键方法 4**：`_determine_dynamic_strict_mode(data: Dict[str, Any])`

**业务口径**：
- **职责**：动态严格模式决策器（仅按配置阈值判断；无配置则不覆盖静态模式）。
- **评分逻辑**：
  - 多币种评分 = 非基准货币权重比例；
  - 跨境敞口评分 = portfolio['cross_border_exposure']（Portfolio 模块提供）；
  - 监管叠加评分 = 当前返回 None（规则/数据缺失）；
  - 综合评分 = 加权平均（仅当配置要求的维度都有分数时计算）。
- **决策规则**：综合评分 ≥ comprehensive_trigger_score 时返回 True；否则返回 False；无配置/数据不足时返回 None。

**方法签名**：
```python
def _determine_dynamic_strict_mode(self, data: Dict[str, Any]) -> Optional[bool]
```

**输入**：`data`（包含 portfolio/market_data）  
**输出**：True（启用严格）、False（禁用严格）、None（保持静态模式）  
**异常**：不抛出异常（异常时返回 None）

**业务评审点**：
- 综合评分的加权逻辑是否符合多维度风险叠加的业务规则？
- 维度缺失时返回 None（不参与综合评分）是否为合理的容错策略？
- cross_border_exposure 由 Portfolio 模块提供的口径是否明确？

---

**关键方法 5**：`_assess_data_quality_multi(market_data: Dict[str, Any], dq_cfg: Dict[str, Any])`

**业务口径**：
- **职责**：多维度数据质量评估（Phase 1 仅 completeness 维度）。
- **评分逻辑**：
  - completeness = currency_coverage × 100（0-100）；
  - overall_score = completeness（Phase 1 仅一个维度）；
  - quality_grade = 根据 grade_thresholds 映射（A≥90、B≥75、C≥60、D<60）。
- **usage_scenarios**：reporting_only=True, affect_calculation=False（不影响风险计算）。

**方法签名**：
```python
def _assess_data_quality_multi(self, market_data: Dict[str, Any], dq_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]
```

**输入**：
- `market_data`：市场数据（包含 prices）
- `dq_cfg`：数据质量评估配置（enabled、base_weights、grade_thresholds）

**输出**：
- 成功：`{'overall_score': float, 'dimension_scores': {...}, 'quality_grade': str}`
- 失败：None（配置未启用或不完整）

**异常**：不抛出异常（异常时返回 None）

**业务评审点**：
- Phase 1 仅 completeness 维度是否足够支撑"仅报告"的业务目标？
- A/B/C/D 分级阈值是否符合数据质量监控标准？
- Phase 3 补充 accuracy/consistency/timeliness/reliability 时，是否需要重新校准权重与阈值？

---

**关键方法 6**：`_us_compliance_logging(currency_warnings: List[str], data_quality: Optional[Dict[str, Any]])`

**业务口径**：
- **职责**：US 市场合规日志记录（SEC/FINRA）。
- **触发条件**：market_type='US' + 存在货币警告。
- **事件结构**：event_id（UUID）、event_type（CURRENCY_INCONSISTENCY）、message、timestamp（ISO8601）、market、severity（MEDIUM/HIGH）、automated_action（LOG_ONLY）。

**方法签名**：
```python
def _us_compliance_logging(self, currency_warnings: List[str], data_quality: Optional[Dict[str, Any]] = None) -> None
```

**输入**：
- `currency_warnings`：货币警告列表
- `data_quality`：数据质量评估结果（可选）

**输出**：无（仅日志）  
**异常**：不抛出异常

**业务评审点**：
- severity 分级（多币种=MEDIUM、货币不一致=HIGH）是否符合 SEC/FINRA 合规风险定义？
- automated_action='LOG_ONLY'（不阻断）是否为 Phase 1 的合理策略？
- Phase 2 在何种条件下应启用阻断或人工审批？

---

#### 文件 2：`core_bak_refactored/tests/units/core/risk/risk_calculator_test.py`

**测试覆盖关键点**：
1. **动态严格模式触发**：US 市场多币种+跨境敞口 → 综合评分 0.7 ≥ 0.65 → 触发严格模式（test_dynamic_strict_enabled_triggers）。
2. **跨市场阈值差异化**：HK 40%、JP 20%、SG 35%、CN 50%、EU 30%（6 个测试用例）。
3. **边界条件精确验证**：0.64 < 0.65（不触发）、0.66 ≥ 0.65（触发）（test_dynamic_strict_boundary_64/65）。
4. **数据质量分级**：A 级（100%）、B 级（80%）、C 级（60%）、D 级（20%）（4 个测试用例）。
5. **配置缺失回退**：dynamic_currency_strict_mode.enabled=False → 返回 None（test_dynamic_strict_disabled_returns_none）。

**业务验收断言**：
- 15/15 单元测试全部通过，覆盖率 100%。
- 跨市场阈值、边界条件、数据质量分级均验证通过。

---

### 🧩 架构变更与影响

**本轮无重大架构变更**，仅在现有协调器基础上增强：
- **职责边界保持**：RiskCalculator 仍为纯协调器（委托给 RiskMetricsService）。
- **配置驱动原则**：无默认值；配置缺失时返回 None 或回退到静态模式。
- **向后兼容**：新增方法均为内部方法（`_` 前缀），不影响公开 API。

---

## 本轮业务问题（下一轮需解决，非本轮验收内容）

### 领域知识

- **请您确认 Phase 3 数据质量维度定义与权重**：
  - accuracy（价格离群值检测）：数据来源与计算方法？各市场权重差异（US 更关注 accuracy，CN 更关注 completeness）？
  - consistency（时间序列连续性）：检测口径与容忍度？
  - timeliness（数据延迟统计）：阈值定义（实时/T+1/T+N）？
  - reliability（历史质量记录）：评分依据与衰减机制？

- **请您确认监管维度 regulatory_overlay_rules 的业务 Schema**：
  - 各市场监管规则的优先级与权重分配（US：SEC/FINRA 规则、HK：SFC 规则、CN：证监会规则）？
  - 规则评分方法（continuous/binary）与 violation_penalty 定义？
  - 监管维度在 Phase 2 是否"必选"，还是可选维度？

- **请您确认审计留存口径与字段最小集**：
  - 内部审计（≥90 天）与监管审计（≥3 年）的字段差异？
  - 必须留存的数据质量字段（data_quality_grade、currency_coverage）是否足够？
  - 争议追溯（≥1 年）需要保留哪些中间结果？

### 优化机会

- **请您确认各市场数据质量权重与阈值的业务价值提升点**：
  - 是否需要根据市场特征动态调整权重（US 更关注 accuracy、HK 更关注 completeness、CN 更关注 consistency）？
  - A/B/C/D 分级阈值是否需要按市场差异化（US：A≥95、CN：A≥85）？
  - 动态权重调整机制（根据市场波动、监管事件）的业务触发条件？

- **请您确认阻断阈值与动作映射的业务价值优化方向**：
  - 数据质量 C/D 级时，是否需要触发"限制自动交易"或"人工审批"？
  - US 合规事件 severity=HIGH 时，是否需要在 exec 域触发阻断？
  - 不同风险等级（LOW/MEDIUM/HIGH/CRITICAL）对应的动作映射（LOG_ONLY/WARN/REQUIRE_MANUAL_REVIEW/BLOCK_AUTO_TRADING/HALT_TRADING）？

- **请您确认性能目标达成路径**：
  - 组合规模（≤100 标的）重算 P95 时延≤500ms，是否为合理的 SLA 边界？
  - 智能缓存失效触发频率（波动率>阈值、熔断、涨跌停）是否会导致缓存命中率下降？
  - 是否需要按组合规模分级 SLA（≤100→500ms、≤500→1000ms、>500→3000ms）？

### 实施路径

- **请您确认 exec 域集成点与风控门控设计**：
  - risk 域输出的合规事件字段（event_id、severity、automated_action）是否足够支撑 exec 域门控决策？
  - exec 域门控应在"下单前"还是"风险计算后"统一入口？
  - 阻断/审批场景边界（估值影响>2%、监管货币违规、数据质量 D 级）是否需要业务确认？

- **请您确认缺失策略的业务处理路径**：
  - 当 cross_border_exposure 缺失时，动态严格模式返回 None（保持静态模式），是否为合理的降级策略？
  - 当 regulatory_overlay_rules 缺失时，监管维度不参与综合评分，是否符合业务口径？
  - 当数据质量配置未启用时，仅使用货币覆盖度评级（A/B/C/D），是否足够支撑审计要求？

---

## 🔗 相关文件

**核心实现文件（本轮改进）**：
- `core_bak_refactored/core/risk/risk_calculator.py`

**测试文件（本轮验证）**：
- `core_bak_refactored/tests/units/core/risk/risk_calculator_test.py`

**基础配置与共享模块**：
- `core_bak_refactored/core/share/market_config.py`
- `core_bak_refactored/core/share/exchange_rates.py`
- `core_bak_refactored/infrastructure/cache_service.py`

**设计文档**：
- `docs/design/core_bak_refactored/core/risk/模块设计文档.md`（v1.3）
- `docs/design/core_bak_refactored/core/risk/接口设计文档.md`（v1.3）

---

## 📝 说明（必须）

- 重要：请尽可能详尽和充分，不要遗漏和简化，谢谢！
