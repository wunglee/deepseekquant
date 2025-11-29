# 测试代码 MarketCode 使用规范

## 核心原则

测试代码中的市场类型配置**保留字符串形式**，由被测代码内部通过 `MarketCode.parse()` 自动转换为枚举。

## 原因

1. **测试数据真实性**: 配置文件（YAML/JSON）中使用字符串（如 `'CN'`, `'US'`），测试应模拟真实场景
2. **向后兼容**: 现有大量测试无需修改
3. **集中转换**: 转换逻辑集中在业务代码中，测试无需关心枚举细节

## 示例

### ✅ 推荐（测试代码保留字符串）
```python
def test_cn_market_risk():
    config = {
        'market_type': 'CN',  # 字符串，将被 parse 为 MarketCode.CN
        'market_configs': {
            'CN': {  # 配置键也是字符串
                'trading_days': 245
            }
        }
    }
    calculator = RiskCalculator(config)
    # calculator.market_type 现在是 MarketCode.CN 枚举
    assert calculator.market_type == MarketCode.CN
```

### ❌ 不推荐（测试代码直接使用枚举）
```python
from core_bak_refactored.core.share.market_enums import MarketCode

def test_cn_market_risk():
    config = {
        'market_type': MarketCode.CN,  # ❌ 不符合真实配置格式
        'market_configs': {
            str(MarketCode.CN): {...}  # ❌ 过于复杂
        }
    }
```

## 特殊情况：需要枚举断言

当测试需要验证市场类型时，使用枚举进行断言：

```python
def test_market_type_parsing():
    from core_bak_refactored.core.share.market_enums import MarketCode
    
    config = {'market_type': 'JP'}  # 字符串输入
    analyzer = PositionRiskAnalyzer(config)
    
    # 断言：内部已转换为枚举
    assert analyzer.market_type == MarketCode.JP
    assert isinstance(analyzer.market_type, MarketCode)
```

## 配置迁移说明

### YAML 配置文件
**现状**:
```yaml
risk:
  market_type: "CN"
  market_configs:
    CN:
      trading_days: 245
    US:
      trading_days: 252
```

**无需修改**: 
- 字符串 `"CN"` / `"US"` 等保持原样
- 代码中通过 `MarketCode.parse('CN')` 自动转换为 `MarketCode.CN`

### JSON 配置文件
```json
{
  "market_type": "JP",
  "market_configs": {
    "JP": {
      "trading_days": 245
    }
  }
}
```

**无需修改**: parse 方法支持所有有效字符串（'CN', 'US', 'HK', 'JP', 'EU', 'SG'）

## Symbol 后缀识别补充说明

`_detect_market_type` 方法在无 metadata 时使用 symbol 后缀启发式：

### 已支持
- `.US` → `MarketCode.US`
- `.HK` / `.HKG` → `MarketCode.HK`
- `.SH` / `.SZ` / `.CN` → `MarketCode.CN`

### 暂未支持（回退 UNKNOWN）
- `.JP` → 日本（需补充规则）
- `.EU` / `.DE` / `.FR` 等 → 欧洲（需补充规则）
- `.SG` → 新加坡（需补充规则）

### 推荐做法
在 MarketData 的 metadata 中显式提供 `market_type`:
```python
data = MarketData(
    symbol='7203.T',  # Toyota (东京证券交易所)
    metadata={'market_type': 'JP'},  # 显式指定，避免启发式失败
    # ... other fields
)
```

或在配置中强制指定：
```python
config = {
    'market_type': 'JP',  # 全局设定
    # ... other config
}
```

## 总结

| 场景 | 使用方式 | 示例 |
|------|---------|------|
| 测试配置 | 字符串 | `'market_type': 'CN'` |
| 配置文件（YAML/JSON） | 字符串 | `market_type: "US"` |
| 业务代码内部 | 枚举 | `self.market_type = MarketCode.parse(...)` |
| 测试断言 | 枚举 | `assert x.market_type == MarketCode.CN` |
| 配置键访问 | 字符串 | `config['market_configs'][str(market_type)]` |

---

**更新时间**: 2025-11-29  
**适用版本**: core_bak_refactored v2+
