# data/_fragments 迁移计划

## 📌 迁移目标

将 `core_bak_refactored/core/data/_fragments` 中的代码整合到 `core_bak_refactored/core/data` 的恰当位置，使得 risk 模块继续依赖 data 模块并通过测试。

## 📊 现状评估

### 测试状态
- ✅ risk 模块测试：210/210 通过
- ✅ _fragments 测试：51/51 通过
- ✅ risk 模块当前**未直接依赖** _fragments（通过 grep 验证）

### _fragments 目录结构
```
core_bak_refactored/core/data/_fragments/
├── __init__.py                      # 说明文档
├── data_quality_checker.py          # 590行 - 数据质量多维度检查器
├── data_quality_enhancer.py         # 169行 - 多源数据智能切换
├── data_quality_monitor.py          # 33行 - 配置驱动的质量监控（简化版）
├── data_quality_policy.py           # 20行 - 数据质量策略配置
├── data_utils.py                    # 225行 - 数据处理工具函数
├── historical_data_provider.py      # 48.5KB - 历史数据提供者
├── policy_config_loader.py          # 21行 - 策略配置加载器
├── quality_types.py                 # 43行 - 质量报告数据类型
├── tushare_provider.py              # 11.8KB - Tushare数据源
└── yahoo_finance_provider.py        # 18.1KB - Yahoo Finance数据源
```

### core/data 已有模块结构
```
core_bak_refactored/core/data/
├── quality/                         # 质量相关（仅1个文件）
│   └── metrics.py                   # 9行 - 聚合读取质量指标
├── providers/                       # 数据提供者（8个文件）
│   ├── yahoo.py, alphavantage.py, finnhub.py, ...
├── cache/                           # 缓存管理（7个文件）
├── analytics/                       # 分析功能（5个文件）
├── market/                          # 市场数据（5个文件）
├── validation/                      # 数据验证（2个文件）
├── data_fetcher.py                  # 273.6KB - 主数据获取器（包含大量质量监控代码）
└── ... (其他20+目录)
```

## 🎯 迁移策略

### 原则
1. **消除重复**：优先使用 `core/data` 已有的相同意图的逻辑
2. **技术优先**：`_fragments` 技术上更优时，整合到 `core/data`（仅限技术手段，非业务逻辑）
3. **业务保守**：业务逻辑只用 `core/data` 的，不做处理（只有业务专家才知道哪种更优）
4. **职责单一**：迁移到恰当的模块内，确保职责单一，消除冗余
5. **测试保证**：迁移后 risk 模块测试继续通过

### 迁移映射

#### 1️⃣ 质量检查模块（优先迁移）

| _fragments 文件 | 目标位置 | 迁移策略 | 理由 |
|----------------|----------|---------|------|
| `quality_types.py` | `quality/types.py` | ✅ **直接迁移** | 数据类型定义，无冲突 |
| `data_quality_policy.py` | `quality/policy.py` | ✅ **直接迁移** | 策略配置，无冲突 |
| `policy_config_loader.py` | `quality/policy_loader.py` | ✅ **直接迁移** | 配置加载器，无冲突 |
| `data_quality_checker.py` | `quality/checker.py` | ⚠️ **整合迁移** | 与 `data_fetcher.py` 中的质量检查有重复，需整合 |
| `data_quality_monitor.py` | `quality/monitor.py` | ⚠️ **整合迁移** | 与 `data_fetcher.py` 中的 `DataQualityMonitor` 类重复，需整合 |
| `data_quality_enhancer.py` | `quality/enhancer.py` | ✅ **直接迁移** | 多源切换增强功能，独立特性 |

**整合策略 - `data_quality_checker.py`**：
- `data_fetcher.py` 中有大量质量检查代码（完整性/一致性/连续性/合理性）
- `_fragments/data_quality_checker.py` 更模块化、更清晰，职责单一
- **迁移方案**：
  1. 将 `_fragments/data_quality_checker.py` 迁移到 `quality/checker.py`
  2. `data_fetcher.py` 中的质量检查方法委派给 `quality/checker.py`
  3. 保留 `data_fetcher.py` 的质量监控循环和报告生成（这是业务流程层面的组织）

**整合策略 - `data_quality_monitor.py`**：
- `data_fetcher.py` 中有完整的 `DataQualityMonitor` 类（4000+行，包含监控循环、警报、性能统计）
- `_fragments/data_quality_monitor.py` 是简化版（33行，配置驱动）
- **迁移方案**：
  1. 保留 `data_fetcher.py` 中的完整版 `DataQualityMonitor`（专家提供的第2版实现）
  2. 将 `_fragments/data_quality_monitor.py` 的配置驱动特性整合到完整版
  3. 不创建独立的 `quality/monitor.py`（避免冗余）

#### 2️⃣ 数据提供者模块

| _fragments 文件 | 目标位置 | 迁移策略 | 理由 |
|----------------|----------|---------|------|
| `historical_data_provider.py` | `providers/historical.py` | ⚠️ **整合迁移** | 与 `core/data/historical_data_provider.py` 重复，需整合 |
| `tushare_provider.py` | `providers/tushare.py` | ✅ **直接迁移** | `providers/` 中没有 tushare，直接添加 |
| `yahoo_finance_provider.py` | `providers/yahoo.py` | ⚠️ **比对整合** | `providers/yahoo.py` 已存在，需比对功能 |

**整合策略 - `historical_data_provider.py`**：
- `core/data/historical_data_provider.py` (49.1KB) 已存在
- `_fragments/historical_data_provider.py` (48.5KB) 功能相似
- **迁移方案**：
  1. 比对两者差异（备用源切换、交叉验证、市场优先级）
  2. 将 `_fragments` 中的优势特性整合到 `core/data/historical_data_provider.py`
  3. 不创建 `providers/historical.py`（避免冗余）

**整合策略 - `yahoo_finance_provider.py`**：
- `providers/yahoo.py` (4.2KB) 已存在但较简单
- `_fragments/yahoo_finance_provider.py` (18.1KB) 功能更丰富
- **迁移方案**：
  1. 比对功能差异
  2. 如果 `_fragments` 版本技术上更优，替换 `providers/yahoo.py`
  3. 如果是业务逻辑差异，保留 `providers/yahoo.py`

#### 3️⃣ 工具模块

| _fragments 文件 | 目标位置 | 迁移策略 | 理由 |
|----------------|----------|---------|------|
| `data_utils.py` | `transformation/utils.py` | ⚠️ **比对整合** | 检查是否与现有工具函数重复 |

**整合策略 - `data_utils.py`**：
- 包含通用数据处理函数（收益率计算、数据验证、事件数据获取）
- **迁移方案**：
  1. 检查 `transformation/` 和其他模块是否已有类似功能
  2. 如果无重复，创建 `transformation/utils.py` 或 `utils.py`
  3. 如果有重复，整合到已有模块

## 📋 迁移步骤

### Phase 1: 准备阶段（当前）
- [x] 评估 `_fragments` 内容与 `core/data` 重复情况
- [x] 确认 risk 模块测试状态
- [x] 制定迁移计划

### Phase 2: 质量模块迁移（优先）
1. 创建 `quality/` 子模块结构：
   - [ ] `quality/types.py` - 从 `_fragments/quality_types.py` 迁移
   - [ ] `quality/policy.py` - 从 `_fragments/data_quality_policy.py` 迁移
   - [ ] `quality/policy_loader.py` - 从 `_fragments/policy_config_loader.py` 迁移
   - [ ] `quality/checker.py` - 从 `_fragments/data_quality_checker.py` 迁移
   - [ ] `quality/enhancer.py` - 从 `_fragments/data_quality_enhancer.py` 迁移

2. 整合质量检查到 `data_fetcher.py`：
   - [ ] 修改 `data_fetcher.py` 的质量检查方法，委派给 `quality/checker.py`
   - [ ] 保留监控循环和报告生成逻辑在 `data_fetcher.py`

3. 更新测试：
   - [ ] 迁移 `tests/units/core/data/_fragments/data_quality_*.py` 到 `tests/units/core/data/quality/`
   - [ ] 验证测试通过

### Phase 3: 数据提供者迁移
1. 整合历史数据提供者：
   - [ ] 比对 `_fragments/historical_data_provider.py` 与 `core/data/historical_data_provider.py`
   - [ ] 整合优势特性到 `core/data/historical_data_provider.py`
   
2. 迁移 Tushare 提供者：
   - [ ] `providers/tushare.py` - 从 `_fragments/tushare_provider.py` 迁移
   - [ ] 迁移测试到 `tests/units/core/data/providers/tushare_test.py`

3. 整合 Yahoo Finance 提供者：
   - [ ] 比对 `_fragments/yahoo_finance_provider.py` 与 `providers/yahoo.py`
   - [ ] 如果 `_fragments` 更优，替换 `providers/yahoo.py`
   - [ ] 更新测试

### Phase 4: 工具模块迁移
1. 整合数据工具：
   - [ ] 比对 `_fragments/data_utils.py` 与现有工具函数
   - [ ] 决定迁移到 `transformation/utils.py` 或新位置
   - [ ] 迁移测试

### Phase 5: 清理与验证
1. 验证迁移：
   - [ ] 运行 `core/data` 所有测试，确保通过
   - [ ] 运行 `core/risk` 所有测试，确保通过
   - [ ] 确认没有导入 `_fragments` 的引用

2. 删除 `_fragments`：
   - [ ] 删除 `core_bak_refactored/core/data/_fragments/` 目录
   - [ ] 删除 `core_bak_refactored/tests/units/core/data/_fragments/` 目录

3. 更新文档：
   - [ ] 更新 `SPRINT.md`，标记迁移完成
   - [ ] 更新架构文档，反映新的模块组织

## ⚠️ 注意事项

1. **导入路径更新**：迁移后需更新所有导入语句
2. **测试同步迁移**：每个文件迁移后立即迁移对应测试
3. **小步迭代**：每完成一个模块迁移后运行测试
4. **保留历史**：迁移前确保 `_fragments` 代码有备份

## 📈 进度追踪

- [ ] Phase 1: 准备阶段
- [ ] Phase 2: 质量模块迁移
- [ ] Phase 3: 数据提供者迁移
- [ ] Phase 4: 工具模块迁移
- [ ] Phase 5: 清理与验证

---

**版本**: v1.0  
**创建时间**: 2025-11-29  
**状态**: 规划中
