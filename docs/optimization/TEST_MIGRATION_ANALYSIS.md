# 测试迁移分析文档

## 概述

本文档分析 `historical_data_provider_test.py` 中需要修复或废弃的测试用例。这些测试失败是因为使用了已废弃的旧架构 API。

**架构变更日期**: 2025-12-06  
**变更类型**: 从 `primary_source/backup_sources` 架构迁移到 `market_sources` 架构

---

## 测试失败统计

| 类别 | 失败数 | 说明 |
|------|--------|------|
| **使用废弃参数** | 6 个 | 使用了 `primary_source` 参数 |
| **调用废弃方法** | 4 个 | 调用了 `_get_recommended_source_for_region()` 方法 |
| **断言错误码不匹配** | 1 个 | 错误码命名变化 |
| **合计** | **11 个** | - |

---

## 详细分析

### 类别 1: 使用废弃参数 `primary_source` (6个测试)

#### 问题描述

**旧架构**:
```python
# ❌ 已废弃
RealHistoricalDataProvider(
    primary_source='akshare',
    backup_sources=['yahoo', 'tushare']
)
```

**新架构**:
```python
# ✅ 正确方式
RealHistoricalDataProvider(
    market_sources={
        MarketCode.CN.value: DataSource.AKSHARE.value,
        MarketCode.HK.value: DataSource.YAHOO.value,
        MarketCode.US.value: DataSource.YAHOO.value,
        # ...
    }
)
```

#### 受影响的测试

##### 1. `test_real_provider_health_check_skips_unavailable` (行 96-125)

**当前代码**:
```python
def test_real_provider_health_check_skips_unavailable(self):
    provider = RealHistoricalDataProvider(
        primary_source='akshare',  # ❌ 废弃参数
        enable_cross_validation=False
    )
    # ...
```

**失败信息**:
```
TypeError: RealHistoricalDataProvider.__init__() got an unexpected keyword argument 'primary_source'
```

**建议方案**:
- **选项 A (推荐)**: 重写为使用 `market_sources`
- **选项 B**: 标记为 `@pytest.mark.skip(reason="旧架构已废弃")`
- **选项 C**: 删除此测试（如果新架构不再需要此功能）

**重写示例**:
```python
def test_real_provider_health_check_skips_unavailable(self):
    """测试健康检查：跳过不可用数据源"""
    provider = RealHistoricalDataProvider(
        market_sources={
            MarketCode.CN.value: DataSource.AKSHARE.value
        },
        enable_cross_validation=False
    )
    
    # 新架构：数据源配置在 market_sources 中
    result = provider.get_index_prices('000300.SH', '2015-06-01', '2015-06-15')
    assert not result.empty, "AKShare应该能成功获取A股数据"
```

---

##### 2. `test_regional_priority_cn_market` (行 127-146)

**当前代码**:
```python
def test_regional_priority_cn_market(self):
    provider = RealHistoricalDataProvider(
        primary_source='akshare'  # ❌ 废弃参数
    )
    
    recommended = provider._get_recommended_source_for_region('000300.SH')  # ❌ 废弃方法
    assert recommended == 'akshare'
```

**失败信息**:
```
TypeError: RealHistoricalDataProvider.__init__() got an unexpected keyword argument 'primary_source'
```

**建议方案**:
- **删除或重写**: 新架构不再有"推荐数据源"概念，而是直接配置每个市场的数据源
- 新架构测试重点应该是：验证市场正确映射到配置的数据源

**重写示例**:
```python
def test_cn_market_uses_configured_source(self):
    """测试A股市场使用配置的数据源"""
    provider = RealHistoricalDataProvider(
        market_sources={
            MarketCode.CN.value: DataSource.AKSHARE.value
        }
    )
    
    # 新架构：验证从 market_sources 配置中正确获取数据源
    market_code = provider._extract_market_code('000300.SH')
    assert market_code == MarketCode.CN
    
    provider_id = provider.market_sources.get(market_code.value)
    assert provider_id == DataSource.AKSHARE.value
```

---

##### 3. `test_regional_priority_us_market` (行 148-160)

**当前代码**:
```python
def test_regional_priority_us_market(self):
    provider = RealHistoricalDataProvider(
        primary_source='akshare'  # ❌ 废弃参数
    )
    
    recommended = provider._get_recommended_source_for_region('SPX.US')  # ❌ 废弃方法
    assert recommended == 'akshare'
```

**失败信息**: 同上

**重写示例**:
```python
def test_us_market_uses_configured_source(self):
    """测试美股市场使用配置的数据源"""
    provider = RealHistoricalDataProvider(
        market_sources={
            MarketCode.US.value: DataSource.YAHOO.value
        }
    )
    
    market_code = provider._extract_market_code('^GSPC')
    assert market_code == MarketCode.US
    
    provider_id = provider.market_sources.get(market_code.value)
    assert provider_id == DataSource.YAHOO.value
```

---

##### 4. `test_regional_priority_hk_market` (行 162-173)

**当前代码**:
```python
def test_regional_priority_hk_market(self):
    provider = RealHistoricalDataProvider(
        primary_source='akshare'  # ❌ 废弃参数
    )
    
    recommended = provider._get_recommended_source_for_region('0700.HK')  # ❌ 废弃方法
    assert recommended == 'akshare'
```

**失败信息**: 同上

**重写示例**:
```python
def test_hk_market_uses_configured_source(self):
    """测试港股市场使用配置的数据源"""
    provider = RealHistoricalDataProvider(
        market_sources={
            MarketCode.HK.value: DataSource.YAHOO.value
        }
    )
    
    market_code = provider._extract_market_code('HSI')
    assert market_code == MarketCode.HK
    
    provider_id = provider.market_sources.get(market_code.value)
    assert provider_id == DataSource.YAHOO.value
```

---

##### 5. `test_all_sources_fail_logs_health_summary` (行 175-195)

**当前代码**:
```python
def test_all_sources_fail_logs_health_summary(self, log_capture):
    provider = RealHistoricalDataProvider(
        primary_source='akshare'  # ❌ 废弃参数
    )
    # ...
```

**失败信息**: 同上

**重写示例**:
```python
def test_source_failure_raises_runtime_error(self):
    """测试数据源失败时抛出RuntimeError"""
    provider = RealHistoricalDataProvider(
        market_sources={
            MarketCode.CN.value: DataSource.AKSHARE.value
        }
    )
    
    # Mock数据源使其失败
    if DataSource.AKSHARE.value in provider._adapters:
        adapter = provider._adapters[DataSource.AKSHARE.value]
        adapter.get_index_prices = Mock(side_effect=RuntimeError("数据源不可用"))
    
    # 新架构：数据源失败时抛出RuntimeError
    with pytest.raises(RuntimeError, match="数据源"):
        provider.get_index_prices('000300.SH', '2015-01-01', '2015-01-10')
```

---

##### 6. `test_tushare_stub_in_cn_priority` (行 210-221)

**当前代码**:
```python
def test_tushare_stub_in_cn_priority(self):
    provider = RealHistoricalDataProvider(
        primary_source='akshare'  # ❌ 废弃参数
    )
    
    recommended = provider._get_recommended_source_for_region('000001.SH')  # ❌ 废弃方法
    assert recommended == 'akshare'
```

**失败信息**: 同上

**建议**: 与测试 #2 重复，可以删除或合并

---

### 类别 2: 调用废弃方法 `_get_recommended_source_for_region()` (4个测试)

#### 问题描述

**旧架构**:
```python
# ❌ 已删除的方法
recommended = provider._get_recommended_source_for_region(symbol)
```

**新架构**:
```python
# ✅ 直接从配置获取
market_code = provider._extract_market_code(symbol)
provider_id = provider.market_sources.get(market_code.value)
```

#### 受影响的测试

##### 7. `test_all_markets_covered` (行 223-244)

**当前代码**:
```python
def test_all_markets_covered(self):
    provider = RealHistoricalDataProvider()
    
    test_symbols = {
        MarketCode.CN: '000300.SH',
        MarketCode.US: 'SPX.US',
        MarketCode.HK: '0700.HK',
    }
    
    for market, symbol in test_symbols.items():
        recommended = provider._get_recommended_source_for_region(symbol)  # ❌ 废弃方法
        assert recommended is not None
```

**失败信息**:
```
AttributeError: 'RealHistoricalDataProvider' object has no attribute '_get_recommended_source_for_region'
```

**重写示例**:
```python
def test_all_markets_have_configured_sources(self):
    """测试所有市场都有配置的数据源"""
    provider = RealHistoricalDataProvider()
    
    test_symbols = {
        MarketCode.CN: '000300.SH',
        MarketCode.US: '^GSPC',
        MarketCode.HK: 'HSI',
    }
    
    for market, symbol in test_symbols.items():
        market_code = provider._extract_market_code(symbol)
        provider_id = provider.market_sources.get(market_code.value)
        assert provider_id is not None, f"市场 {market.value} 应该有配置的数据源"
```

---

##### 8. `test_jp_market_priority` (行 246-254)
##### 9. `test_eu_market_priority` (行 256-263)
##### 10. `test_sg_market_priority` (行 265-272)

**当前代码**:
```python
def test_jp_market_priority(self):
    provider = RealHistoricalDataProvider()
    recommended = provider._get_recommended_source_for_region('9984.T')  # ❌ 废弃方法
```

**失败信息**: 同上

**建议方案**:
- **选项 A**: 重写为测试市场代码识别功能
- **选项 B**: 删除（如果这些市场当前未配置数据源）

**重写示例**:
```python
def test_jp_market_code_extraction(self):
    """测试日本市场代码识别"""
    provider = RealHistoricalDataProvider()
    market_code = provider._extract_market_code('9984.T')
    assert market_code == MarketCode.JP

def test_eu_market_code_extraction(self):
    """测试欧洲市场代码识别"""
    provider = RealHistoricalDataProvider()
    market_code = provider._extract_market_code('BMW.DE')
    assert market_code == MarketCode.EU

def test_sg_market_code_extraction(self):
    """测试新加坡市场代码识别"""
    provider = RealHistoricalDataProvider()
    market_code = provider._extract_market_code('STI.SI')
    assert market_code == MarketCode.SG
```

---

### 类别 3: 断言错误码不匹配 (1个测试)

##### 11. `test_insufficient_sources_handling` (行 278-290)

**当前代码**:
```python
def test_insufficient_sources_handling(self):
    provider = RealHistoricalDataProvider()
    
    report = provider.cross_validate_sources(
        '000300.SH',
        '2015-06-01',
        '2015-06-15',
        sources=[DataSource.MOCK.value]
    )
    
    assert report['passed'] is True
    assert report['reason'] == 'insufficient_sources'  # ❌ 错误码已变更
```

**失败信息**:
```
AssertionError: assert 'insufficient_sources_specified' == 'insufficient_sources'
```

**修复方案**: 简单修改断言

**修复示例**:
```python
def test_insufficient_sources_handling(self):
    """测试数据源不足时的处理"""
    provider = RealHistoricalDataProvider()
    
    report = provider.cross_validate_sources(
        '000300.SH',
        '2015-06-01',
        '2015-06-15',
        sources=[DataSource.MOCK.value]
    )
    
    assert report['passed'] is True
    assert report['reason'] == 'insufficient_sources_specified'  # ✅ 修正错误码
```

---

## 迁移优先级建议

### 高优先级 (必须修复)

1. **错误码断言修复** - 测试 #11
   - 工作量：5分钟
   - 影响：低
   - 建议：立即修复

### 中优先级 (建议重写)

2. **市场数据源配置测试** - 测试 #2, #3, #4
   - 工作量：30分钟
   - 影响：中
   - 建议：重写为测试新架构的市场配置功能

3. **市场代码识别测试** - 测试 #8, #9, #10
   - 工作量：20分钟
   - 影响：低
   - 建议：重写为测试 `_extract_market_code()` 方法

### 低优先级 (考虑删除)

4. **健康检查测试** - 测试 #1, #5
   - 工作量：60分钟
   - 影响：中
   - 建议：评估新架构是否仍需要此功能，否则删除

5. **重复测试** - 测试 #6, #7
   - 工作量：0分钟
   - 影响：无
   - 建议：直接删除重复测试

---

## 总结

### 工作量估算

| 任务 | 测试数量 | 预估工作量 |
|------|----------|------------|
| **简单修复** (错误码) | 1 | 5分钟 |
| **重写测试** (市场配置) | 3 | 30分钟 |
| **重写测试** (市场识别) | 3 | 20分钟 |
| **评估删除** (健康检查) | 2 | 30分钟 |
| **直接删除** (重复测试) | 2 | 5分钟 |
| **总计** | **11** | **~90分钟** |

### 建议执行步骤

1. ✅ **第一步**: 修复测试 #11（错误码断言）- 5分钟
2. ✅ **第二步**: 重写测试 #2, #3, #4（市场配置）- 30分钟
3. ✅ **第三步**: 重写测试 #8, #9, #10（市场识别）- 20分钟
4. ⚠️ **第四步**: 评估测试 #1, #5 是否需要保留 - 30分钟
5. ✅ **第五步**: 删除重复测试 #6, #7 - 5分钟

### 注意事项

- ⚠️ **不要直接删除所有失败测试**：部分测试验证的功能在新架构中仍然存在，只是API变化了
- ✅ **优先重写而非删除**：新架构的核心功能（市场配置、市场识别）仍需要测试覆盖
- 📝 **文档同步**：修改测试后，同步更新测试文档中的架构说明

---

**文档生成时间**: 2025-12-09  
**架构版本**: market_sources (新) vs primary_source/backup_sources (旧)  
**状态**: 待决策
