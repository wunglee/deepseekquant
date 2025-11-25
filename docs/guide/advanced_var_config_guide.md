# 高级VaR策略配置指南

> **版本**: v1.0  
> **最后更新**: 2024-11-14  
> **适用模块**: `PositionRiskAnalyzer`

---

## 📋 概述

`PositionRiskAnalyzer` 支持多种高级VaR计算方法，适用于厚尾分布、极端风险和压力期场景。通过配置化启用，可在生产环境灵活切换策略。

---

## 🔧 配置参数

### 1. 启用高级VaR

在风险配置中设置以下参数：

```python
config = {
    # 启用高级VaR策略（默认False，使用简单历史分位方法）
    'advanced_var_enabled': True,
    
    # 选择VaR方法（默认'evt'）
    'position_var_method': 'evt',  # 可选: 'normal', 't_distribution', 'evt', 'historical_simulation'
    
    # VaR置信水平（默认0.99）
    'var_confidence_level': 0.99
}

analyzer = PositionRiskAnalyzer(config)
```

---

## 📊 支持的VaR方法

### 1. `normal` - 正态分布法

**适用场景**：收益率接近正态分布的资产  
**优点**：计算快速，理论基础清晰  
**缺点**：低估尾部风险

**配置示例**：
```python
config = {
    'advanced_var_enabled': True,
    'position_var_method': 'normal',
    'var_confidence_level': 0.95
}
```

---

### 2. `t_distribution` - 学生t分布法

**适用场景**：收益率呈现厚尾特征（峰度>3）  
**优点**：更好拟合厚尾分布，捕捉极端波动  
**缺点**：需要足够数据估计自由度

**配置示例**：
```python
config = {
    'advanced_var_enabled': True,
    'position_var_method': 't_distribution',
    'var_confidence_level': 0.99
}
```

**建议**：至少100个数据点以获得稳定的分布参数估计

---

### 3. `evt` - 极值理论（POT方法） **[推荐]**

**适用场景**：关注极端尾部风险（99%+置信水平）  
**优点**：专门针对极端事件建模，理论严谨  
**缺点**：数据不足时回退到历史分位方法

**配置示例**：
```python
config = {
    'advanced_var_enabled': True,
    'position_var_method': 'evt',
    'var_confidence_level': 0.99
}
```

**技术细节**：
- 使用广义帕累托分布（GPD）拟合超过90%分位数的尾部数据
- 阈值选择：90%分位数（可在代码中调整）
- 最少超额样本：10个（否则回退）

---

### 4. `historical_simulation` - 历史模拟+压力VaR

**适用场景**：历史数据充足，关注压力期表现  
**优点**：非参数方法，包含压力期VaR增强  
**缺点**：依赖历史数据质量

**配置示例**：
```python
config = {
    'advanced_var_enabled': True,
    'position_var_method': 'historical_simulation',
    'var_confidence_level': 0.95
}
```

**输出**：
- `var_hs`：标准历史模拟VaR
- `var_stress`：压力期VaR（取历史最差窗口）

---

## ⚙️ 自动特性

### 跳跃风险修正

所有方法自动叠加跳跃风险调整因子（基于峰度）：

```python
# 跳跃修正公式（内置）
jump_adjustment = max(0.0, min(0.10, (kurtosis - 3.0) * 0.01))
final_var = base_var * (1 + jump_adjustment)
```

**影响**：峰度越高，VaR修正幅度越大（最高+10%）

---

### 数据不足回退

当数据点少于50个时，自动回退到简单历史分位方法（95%置信水平）：

```python
if len(returns) < 50:
    # 自动回退
    return {'var_simple': calculate_single_position_var(symbol, returns, 0.95)}
```

---

## 🧪 测试验证

### 边界测试

已包含以下测试用例覆盖：

1. ✅ 高级VaR在`analyze_position`中自动调用（`test_advanced_var_enabled_in_analyze_position`）
2. ✅ EVT方法正确性（`test_advanced_var_method_evt`）
3. ✅ 历史模拟+压力VaR（`test_advanced_var_method_historical_simulation`）
4. ✅ 数据不足回退逻辑（`test_advanced_var_insufficient_data_fallback`）

### 性能测试

- **EVT方法**：200个数据点，耗时 < 10ms
- **学生t分布**：200个数据点，耗时 < 5ms
- **历史模拟**：200个数据点，耗时 < 3ms

---

## 📖 使用示例

### 完整示例：生产环境配置

```python
from core_bak_refactored.core.risk.position_risk import PositionRiskAnalyzer

# 生产环境配置
production_config = {
    # 高级VaR启用
    'advanced_var_enabled': True,
    'position_var_method': 'evt',  # 极值理论，适合极端风险
    'var_confidence_level': 0.99,  # 99%置信水平（监管要求）
}

analyzer = PositionRiskAnalyzer(production_config)

# 分析持仓
symbol = 'AAPL'
position = get_position(symbol)  # 获取持仓对象
market_data = get_market_data()  # 获取市场数据

result = analyzer.analyze_position(symbol, position, market_data)
print(f"Position VaR: {result['position_var']:.2f}")
```

### 策略切换示例

```python
# 日常监控：快速正态分布法
daily_config = {
    'advanced_var_enabled': True,
    'position_var_method': 'normal',
    'var_confidence_level': 0.95
}

# 月度报告：严格EVT方法
monthly_config = {
    'advanced_var_enabled': True,
    'position_var_method': 'evt',
    'var_confidence_level': 0.99
}

# 压力测试：历史模拟+压力VaR
stress_config = {
    'advanced_var_enabled': True,
    'position_var_method': 'historical_simulation',
    'var_confidence_level': 0.99
}
```

---

## ⚠️ 注意事项

### 1. 数据质量要求

- **最少数据点**：50个（否则自动回退）
- **推荐数据点**：
  - 正态/历史模拟：≥100
  - 学生t分布：≥100
  - EVT方法：≥200

### 2. 置信水平选择

- **日常监控**：95%
- **风险限额**：99%
- **监管报告**：99%（巴塞尔协议要求）

### 3. 方法选择建议

| 资产特征 | 推荐方法 | 理由 |
|---------|---------|------|
| 正态分布 | `normal` | 计算快，理论成熟 |
| 厚尾分布 | `t_distribution` | 更好拟合峰度 |
| 极端风险关注 | `evt` | 专门针对尾部 |
| 压力测试 | `historical_simulation` | 包含压力VaR |

---

## 🔗 相关文档

- **架构设计**: `docs/design/ARCHITECTURE.md`
- **工作规范**: `.qoder/rules/PECIFICATIONS.md`
- **测试用例**: `core_bak_refactored/tests/core/risk/position_risk_test.py`

---

**维护者**: DeepSeekQuant 开发团队  
**最后审核**: 2024-11-14  
**状态**: ✅ 生效中
