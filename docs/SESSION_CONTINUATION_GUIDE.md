# 会话续作指南（Continuation Context Guide）

> 目的：让一个全新会话在不丢失上下文的情况下，无误地继续当前任务，完整还原专家提供的完整版能力。
> 更新时间：2025-11-27
> 项目根路径：/Users/wangli/Library/Mobile Documents/com~apple~CloudDocs/历史项目/projects/deepseekquant

---

## 1. 当前任务目标与方向
- 目标：完全还原并以专家提供的完整版为主，实现数据模块的生产级能力；将专家“碎片”中真实的增量能力谨慎保留并系统性整合。
- 架构方向：
  - 采用方案B（激进清理）：删除来源不明或重复的实现，以专家完整版为主干。
  - 保留3个专家碎片（增量能力）：`data_quality_enhancer.py`、`historical_data_provider.py`、`yahoo_finance_provider.py`。
  - 领域层与应用层分离：Data模块仅保留领域能力；应用层（Dashboard/API）迁移到 `core_bak_refactored/app/`。

---

## 2. 已完成的关键动作（可复用的稳定基线）
- 激进清理（方案B）
  - 已删除15个来源不明文件（完整备份于 `core_bak_refactored/core/data_backup_20251127/`）。
  - 已复制专家完整版 `core_bak/data_fetcher.py` 到 `core_bak_refactored/core/data/data_fetcher.py`（约8652行，338KB）。
- 修复类名冲突：在 `data_fetcher.py` 中将第1版 `DataQualityMonitor` 重命名为 `DataQualityMonitorBasic`，保留第2版为 `DataQualityMonitor`（增强版，推荐使用）。
- 修复语法错误：`data_fetcher.py` 第2696、2705行 f-string 引号冲突，已改为局部变量注入字符串。
- 修复碎片依赖：
  - `data_quality_enhancer.py` 内联定义 `DataQualityReport`，不再依赖已删除的 `quality_types.py`。
  - `historical_data_provider.py` 内联定义 `DataQualityReport` 与临时 `DataSource` 枚举；为区域优先级 `REGIONAL_PRIORITY` 提供默认字典（替代已删除的 `market_enums`）。
  - `yahoo_finance_provider.py` 内联定义 `DataQualityReport`。
- 完善导出：`core_bak_refactored/core/data/__init__.py` 已导出专家完整版与碎片的核心类，便于直接使用。
- 安装并验证依赖：
  - 已安装：ta-lib、redis、websockets、cryptography、tenacity、cachetools、beautifulsoup4（均在 `deepseekquant-dev` 环境）。
  - 已验证导入：`DataFetcher`、`DataQualityMonitor`、`DataQualityEnhancer` 正常导入。

---

## 3. 现状总览（文件与能力）
- 领域层（核心）：`core_bak_refactored/core/data/`
  - `data_fetcher.py`（专家完整版）
    - `DataSourceType`、`DataFrequency`、`MarketData`、`DataFetcher`、`DataValidator`、`DataQualityMonitorBasic`、`DataQualityMonitor`、`DataQualityMonitorFactory`、`DeepSeekQuantSystem`、以及（应迁移到应用层的）`DataQualityDashboard`、`DataQualityAPIService`。
  - `data_quality_enhancer.py`（第6轮专家碎片）：质量驱动的多源智能切换（独特增量）。
  - `historical_data_provider.py`（第2轮专家 + Phase 5B-5）：Protocol接口、区域化优先级、事件窗口与停牌处理（显著增量）。
  - `yahoo_finance_provider.py`（第6轮专家碎片）：指数代码映射、标准化输出、Mock回退（中高价值增量）。
- 应用层（我们已存在的实现）：`core_bak_refactored/app/data_quality/`（`dashboard.py`、`api_service.py` 等）。
- 重要说明：`data_migration_analysis.md` 的“100%完成”目录结构已过时（方案B后，该结构不再真实），请以本指南与 `core_bak_refactored/core/data/` 的实际文件为准。

---

## 4. 关键结论（碎片 vs 完整版，对比与保留策略）
- 专家完整版已具备：基础多源切换（失败驱动）、全面的数据获取与验证框架、缓存与性能监控。
- 专家碎片的独特增量：
  - `DataQualityEnhancer`：质量评分驱动的多源智能切换（质量阈值、源间质量对比、IQR异常检测）。
  - `HistoricalDataProvider`：Protocol接口、区域化源优先级、Phase 5B-5事件窗口与停牌处理、真实事件驱动的Mock数据生成。
  - `YahooFinanceDataProvider`：指数代码映射、统一输出格式、Mock回退。
- 决策：3个碎片全部保留为独立模块，不与完整版冲突；后续择机将增量融入（或组合使用）。

---

## 5. 环境与验证（新会话快速起步）
- 激活环境（示例）：
  - conda/mamba/venv 任选；当前使用的是 `deepseekquant-dev`（若新会话无此环境，请新建并安装依赖）。
- 安装依赖（新会话请执行）：
  - `pip install ta-lib redis websockets cryptography tenacity cachetools beautifulsoup4 yfinance requests aiohttp` 
  - 可选：`pip install pandas numpy`（若未安装）。
- 快速验证（必须通过）：
  - 进入项目根：`cd /Users/wangli/.../deepseekquant`
  - 运行：
    - `python -c "from core_bak_refactored.core.data import DataFetcher, DataQualityMonitor, DataQualityEnhancer; print('OK')"`
  - 预期输出：`OK`
- 重要提示：JoinQuant/Wind/Tushare 等商业/闭源API暂未集成；对应在 `historical_data_provider.py` 中以 stub 方式占位。请勿在无凭据的情况下尝试安装或调用。

---

## 6. 下一步任务（按优先级执行）
- P1：处理应用层重复（统一到专家实现）
  1) 从 `data_fetcher.py` 提取 `DataQualityDashboard` 与 `DataQualityAPIService` 到 `core_bak_refactored/app/data_quality/`。
  2) 用专家版本替换/合并我们现有的 `dashboard.py` 与 `api_service.py`，保留专家实现为主；对我们新增的接口保留注释：`TODO：待专家评审`。
  3) 保持分层：应用层仅依赖领域层；删除 data 模块中的应用层残留。
- P2：整合碎片的增量能力（不破坏专家主干）
  1) 在 `DataFetcher` 的 fallback 机制周边，设计与 `DataQualityEnhancer` 的质量阈值/质量对比策略的组合接口（先组合使用，后再考虑融合）。
  2) 在历史数据路径，优先尝试 `RealHistoricalDataProvider` 的区域化优先级与事件窗口增强；若与现有流程冲突，保持 Provider 独立使用（通过组合调用）。
  3) 为 Yahoo 路径增加指数代码映射与统一格式的预处理钩子（先在 Provider 内做，必要时在 `DataFetcher` 的 Yahoo 分支挂接）。
- P3：测试补齐与示例
  1) 基于专家完整版能力，编写新的集成测试（重点：多源、验证、质量监控）。
  2) 为碎片的增量能力编写场景化测试（质量阈值切换、区域化优先级、指数映射与格式统一）。
  3) 提供“组合使用”示例脚本，演示 DataFetcher + Enhancer + Provider 的最佳实践。

---

## 7. 关键约束与注意事项
- 不恢复已删除的15个文件（已确认与专家完整版/碎片存在重复或来源不明）。
- 所有新增/合并点统一使用注释标识：`TODO：专家碎片整合 - 待评审`。
- 以最小改动为原则：优先组合使用（composition），避免“大手术式”重构，待稳定后再融合。
- 保持分层：领域层（core）与应用层（app）严格解耦；应用层不可回引入领域实现到 data_fetcher.py。
- 依赖管理：仅安装必要依赖；涉及商业API（JoinQuant/Wind/Tushare）保留 stub，不做自动安装。

---

## 8. 快速入口（新会话如何继续）
- 打开本指南后，先运行“环境与验证”的检查命令，确认导入一切正常。
- 按“下一步任务（P1）”开始提取专家的 Dashboard/API 到 app 层；完成后删除 data 层中的应用层类。
- 然后按“P2”组合使用碎片的增量能力，避免破坏专家主干逻辑。
- 最后按“P3”补齐测试与示例。

---

## 9. 参考与其他文档（已生成）
- `docs/process/core_bak_refactored/core/data/FRAGMENT_INTEGRATION_ANALYSIS.md`：碎片整合分析
- `docs/process/core_bak_refactored/core/data/INTEGRATION_STRATEGY.md`：整合策略
- `docs/process/core_bak_refactored/core/data/COMPLETION_REPORT.md`：完成报告（方案B执行总结）
- `docs/process/core_bak_refactored/core/data/EXPERT_FRAGMENTS_VS_COMPLETE.md`：碎片 vs 完整版对比
- `docs/process/core_bak_refactored/core/data/RESTORATION_ROADMAP.md`：完整版能力还原路线图

---

## 10. 复盘与状态声明
- 当前状态：专家完整版已导入可用；3个碎片已修复依赖并保留；导出已完善；环境验证通过。
- 需注意：`data_migration_analysis.md` 的“100%完成”结构为历史快照，已与现状不一致；请以本指南为准继续任务。

---

## 11. 附：极简健康检查命令
```
cd "/Users/wangli/Library/Mobile Documents/com~apple~CloudDocs/历史项目/projects/deepseekquant"
python -c "from core_bak_refactored.core.data import DataFetcher, DataQualityMonitor, DataQualityEnhancer; print('OK')"
```
如输出 `OK`，表示导入与依赖正常，可继续执行P1任务。
