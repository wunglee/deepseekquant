# 第3轮咨询 - 迁移准入标准与阶段边界声明（5B-5 / 临时系统）

## 🔒 阶段边界与称谓统一（必须阅读）
- 当前工作系统：`core_bak_refactored`（临时独立系统，用于整理 `core_bak` 的中间产物）。
- 架构边界：所有开发/测试均仅限在 `core_bak_refactored/` 下进行；禁止修改根目录未来系统代码。
- 生产部署限制：在未完成迁移至根目录未来系统且未通过专家验收之前，**禁止讨论任何生产部署或生产化准备**；“阶段C：生产部署准备”不适用于当前临时系统。
- 轮次衔接：第1轮（目标口径确认）→ 第2轮（阶段A+B完成与职责归位）→ 第3轮（当前，仅咨询迁移准入标准与边界）。
- 称谓统一：以下问题均以“您”为称谓。

---

## 📁 相关文件清单（本次更新涉及）
### 核心实现文件（本次修改）
- 无（本轮仅生成/提交 `docs/ask.md`）

### 测试文件（本次验证）
- 无（本轮不改动测试）

### 配置与文档
- `docs/ask.md`（第3轮咨询，迁移准入标准）

---

## 📌 上一轮核心结论（简要提炼）
- 阶段A：5事件端到端流程验证完成；跨市场一致性≥0.85、数据质量评分≥90%达到框架验证。
- 阶段B：GICS一级行业参数统计验证完成（样本量≥1000、t检验p<0.05、参数范围与经济合理性成立）。
- 架构优化：业务逻辑职责归位（risk/data/tests），测试模块纯委托，无业务实现。
- 真实数据依赖：Yahoo Finance存在限流与中国指数不可用问题，现采用自动降级至Mock（依赖缺失处理已实现）。
- 临时系统边界：当前在 `core_bak_refactored`，不讨论生产部署，仅咨询迁移准入标准。

---

## 📁 上一轮修改的代码清单与需评审的关键部分（含详细解释）
（目的：迁移前对已完成实现进行业务口径层面的评审与确认）

- 组件一：`core_bak_refactored/core/risk/cross_market_calibrator.py`
  - 关键方法与业务口径映射：
    - `normalize_to_usd(value, source_currency, event_window_data)`：USD统一计量；使用事件窗口期**日均中间价**（避免极端值）。
    - `apply_liquidity_adjustment(raw_risk_metric, market_id, days_required)`：流动性调整因子（US/EU=0.95，CN/HK=0.90，JP/SG=0.85）；A股T+1与港股LULD的机制处理是否符合您第1轮口径。

- 组件二：`core_bak_refactored/core/backtest/_fragments/uat_validator.py`
  - 关键方法与业务口径映射：
    - `validate_weighted_average_error(errors_by_event, event_type_mapping)`：事件类型差异化阈值，但总体加权平均误差≤15%。
    - `validate_triple_indicator_system(predictions, actuals)`：三级指标体系（MAPE≤15% + 方向准确率≥90% + 尾部误差≤25%且占比≤20%）。

- 组件三：`core_bak_refactored/core/risk/stress_testing.py`
  - 关键方法与业务口径映射：
    - `IndustryParameterAnalyzer.analyze_and_validate(samples)`：GICS一级行业统计与t检验；`compute_industry_parameters` 与 `compute_t_tests` 的统计口径与您给定范围一致性。
    - `IndustryParameterAnalyzer.generate_test_samples(n_samples, seed)`：行业样本生成的参数范围与经济合理性（金融>防御）。

- 组件四：`core_bak_refactored/core/data/_fragments/data_utils.py`
  - 关键方法与业务口径映射：
    - `safe_get_event_data(provider, event, window_days, baseline_days)`：事件窗口数据的安全获取与返回结构兼容（dict/df）；错误静默策略是否可接受。
    - `calculate_actual_return(event_window_df)` / `calculate_return(data, price_column)`：收益率计算口径与字段一致性（close）。

- 测试与工具层：
  - `core_bak_refactored/tests/common/test_assertions.py`：通用断言工具（质量评分、误差阈值、统计显著性、性能断言）。
  - `core_bak_refactored/tests/core/backtest/test_fixtures.py`：纯委托架构（tests只编排，调用业务模块方法；不承载业务逻辑）。

---

## 🧩 背景说明（本轮仅咨询迁移准入标准）
- 目标：在**不讨论生产部署**的前提下，明确从 `core_bak_refactored` 迁移到根目录未来系统的**准入标准**与**边界口径**。
- 约束：保持现有接口契约稳定、职责划分清晰、文档与测试齐备；真实数据源与降级策略的接受口径需要您确认。

---

## 🏗️ 我们的架构组织（职责边界澄清）
- risk：压力测试与行业参数统计（不承载data/backtest具体实现）；对接校准与UAT验证口径。
- data：数据提供与常用处理工具（安全取数、收益计算、数据校验）。
- tests：测试编排与断言（纯委托至业务模块），不包含业务逻辑实现。

---

## ❓ 核心问题（请您逐项给出口径与阈值）
1) 迁移准入标准（`core_bak_refactored` → 根目录未来系统）
- 代码与测试：最低通过率（例如≥98%？）、测试覆盖维度是否已足够（端到端/行业参数/跨市场一致性）。
- 架构与文档：需同步的设计文档清单（模块/接口/架构）与更新粒度；接口契约稳定性判定标准（签名/返回/异常）。
- 版本与审计：迁移版本标注方式（tag/semver）与审计字段（report_id/timestamp/metadata）。

2) 迁移路径与流程
- 策略：按模块分批（risk → data → backtest）或一次性整体迁移？
- 依赖：迁移前需标记为“稳定”的依赖（数据源/汇率/市场配置）。
- 记录：是否需要在 `docs/consultation.md` 与各模块 `SPRINT.md` 追加“迁移确认记录”。

3) 数据源与降级策略的接受口径
- 真实数据源：是否必须先接入 JoinQuant/Wind/Tushare 等备源再迁移？
- 降级策略：对 Yahoo 限流与中国指数不可用的 Mock 降级，是否可作为迁移前的临时接受口径？是否需新增交叉验证？

4) UAT评审版报告（迁移前可生成的“非生产模板”）
- 是否需要生成迁移评审版UAT报告？最低内容清单（历史回测摘要、跨市场一致性、行业参数统计、数据质量、性能指标）。

5) 行业参数与跨市场校准的最终阈值确认
- 行业差异阈值：坚持≥10%，或接受≥6%（考虑随机性）的过渡口径？是否建议扩大样本量或细分行业？
- 跨市场一致性：相关性≥0.85是否需增加市场权重或分层阈值？

---

## 🔍 业务视角的代码实现评审要点（方法/字段/返回结构与业务口径一致性）
- 方法签名与参数语义：`normalize_to_usd`、`apply_liquidity_adjustment`、`validate_weighted_average_error`、`validate_triple_indicator_system`、`analyze_and_validate`、`safe_get_event_data`、`calculate_actual_return` 等与业务口径的一致性核验。
- 返回结构与异常处理：事件窗口数据（df/dict）兼容；错误静默与降级行为是否需要注明在评审版报告中。
- 字段与计量口径：价格列 `close`、汇率选取（日均中间价）、参与率与折扣因子、机制修正系数（LULD/T+1）。

---

## 📝 评审请求（请您明确答复）
- 请对上述“迁移准入标准”“路径与流程”“数据源与降级口径”“UAT评审版报告”“阈值确认”等逐项给出业务口径与验收阈值。
- 请确认组件与方法的业务一致性（签名/参数/返回/异常），并指出需调整或补充之处。
- 我们将在收到您的答复后，严格按口径在 `core_bak_refactored/` 内完成对齐性修正与迁移准备，随后进入迁移评审。

---

**重要：请尽可能详尽和充分，不要遗漏和简化，谢谢！**
