# 第12轮咨询 - 风险域阶段1 模块业务评审（提问）

**模块**: 风险管理 (core/risk/)
**阶段**: Phase 1（从 `core_bak/risk_manager.py` 拆分到 `core_bak_refactored/core/risk/*`）
**提交时间**: 2025-11-12
**评审类型**: 模块业务合理性与优化机会评审（只针对本轮）

---

## 📁 相关文件清单（本次更新涉及）
- core_bak_refactored/core/risk/portfolio_risk.py
- core_bak_refactored/core/risk/position_risk.py
- core_bak_refactored/core/risk/risk_monitor.py
- core_bak_refactored/core/risk/risk_models.py

---

## 评审目标（简明）
- 检查业务规则与逻辑合理性
- 识别量化领域可优化机会（参数校准、边界条件、异常场景、市场差异化）
- 提出测试覆盖增强建议

---

## 评审问题（独立本轮，不参考历史）

### portfolio_risk.py
1) 组合收益与风险归因：缺失数据与停牌场景下，是否建议进行权重重标化与稳健协方差（如Ledoit-Wolf）以降低漂移？
2) 风险呈现口径：VaR/CVaR在业务报表中是否保留符号而在风控计算中使用绝对值更合理？请给出一致性建议。

### position_risk.py
1) 单仓VaR模型：在跳跃风险与厚尾分布（学生t）场景中，是否需要峰度修正或分布替换以贴合真实交易风险？
2) 参与率冲击：`impact = α * participation_rate^β` 的参数是否应按市场/板块校准并设置上限截断以避免过拟合？

### risk_monitor.py
1) 告警分级：是否建议引入指标权重矩阵（VaR、MDD、TE等）与市场差异化阈值（US/HK/JP/SG）以提升分级敏感度？
2) 稳定性保障：在高并发告警场景下，是否需要防抖/节流、熔断与降级策略以避免告警风暴？

### risk_models.py
1) 语义与监管一致性：`RiskLevel` 与 `ImpactLevel` 的区分是否满足监管与审计口径？是否需要数值范围/单位校验器？
2) 容错与审计：`from_dict` 的容错策略是否需要统一日志等级与审计字段（如source、confidence），以增强生产稳健性？

---

**重要：请尽可能详尽和充分，不要遗漏和简化，谢谢！**
