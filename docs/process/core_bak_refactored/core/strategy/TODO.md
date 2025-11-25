# 策略模块 TODO

> **层级**：Core Layer - Strategy  
> **路径**：`core/strategy/`  
> **职责**：交易策略实现、策略模式封装

> **状态**：❌ 未开始  
> **优先级**：中  
> **依赖**：signal_generator.py 待完成

---

## 📁 目录结构规划

```
core/strategy/
├── __init__.py
├── base_strategy.py          # 策略基类
├── trend_following.py         # 趋势跟踪策略
├── mean_reversion.py          # 均值回归策略
├── breakout.py                # 突破策略
└── TODO.md                    # 本文件
```

---

## 🔴 高优先级任务

### 1. 创建策略基类
**文件**：`base_strategy.py`

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class TradingSignal:
    """交易信号数据类"""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # 0.0-1.0
    timestamp: datetime
    reason: Dict[str, Any]  # 信号生成原因

class TradingStrategy(ABC):
    """策略基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    def generate_signals(self, market_data, indicators) -> List[TradingSignal]:
        """
        生成交易信号
        
        Args:
            market_data: 市场数据
            indicators: 技术指标字典
        
        Returns:
            交易信号列表
        """
        pass
    
    @abstractmethod
    def validate_signal(self, signal: TradingSignal) -> bool:
        """验证信号有效性"""
        pass
    
    def get_position_size(self, signal: TradingSignal, 
                         portfolio_value: float) -> float:
        """计算仓位大小（可被子类重写）"""
        # 默认固定百分比
        return portfolio_value * self.config.get('position_size_pct', 0.1)
```

**待办**：
- [ ] 实现策略基类
- [ ] 定义信号数据结构
- [ ] 实现基础仓位管理
- [ ] 编写单元测试

---

### 2. 实现趋势跟踪策略
**文件**：`trend_following.py`

```python
class TrendFollowingStrategy(TradingStrategy):
    """趋势跟踪策略"""
    
    def generate_signals(self, market_data, indicators):
        """
        趋势跟踪信号生成
        
        买入条件：
        - MACD金叉 (macd > macd_signal)
        - ADX强趋势 (adx > 25)
        - 价格突破布林带中轨 (close > bb_middle)
        
        卖出条件：
        - MACD死叉 (macd < macd_signal)
        - ADX弱趋势 (adx < 20)
        - 价格跌破布林带中轨 (close < bb_middle)
        """
        signals = []
        
        # 买入逻辑
        if (indicators['macd'] > indicators['macd_signal'] and
            indicators['adx'] > 25 and
            market_data['close'] > indicators['bb_middle']):
            
            signal = TradingSignal(
                symbol=market_data['symbol'],
                action='BUY',
                strength=self._calculate_signal_strength(indicators),
                timestamp=market_data['timestamp'],
                reason={
                    'macd_crossover': True,
                    'adx': indicators['adx'],
                    'price_vs_bb': 'above_middle'
                }
            )
            signals.append(signal)
        
        # 卖出逻辑
        elif (indicators['macd'] < indicators['macd_signal'] and
              indicators['adx'] < 20):
            # ...
            pass
        
        return signals
    
    def _calculate_signal_strength(self, indicators):
        """计算信号强度（0.0-1.0）"""
        strength = 0.0
        
        # ADX贡献（0-0.4）
        strength += min(indicators['adx'] / 50, 0.4)
        
        # MACD差值贡献（0-0.3）
        macd_diff = abs(indicators['macd'] - indicators['macd_signal'])
        strength += min(macd_diff * 10, 0.3)
        
        # 其他指标...
        
        return min(strength, 1.0)
```

**待办**：
- [ ] 实现趋势跟踪策略
- [ ] 实现信号强度计算
- [ ] 参数可配置化
- [ ] 回测验证

---

### 3. 实现均值回归策略
**文件**：`mean_reversion.py`

```python
class MeanReversionStrategy(TradingStrategy):
    """均值回归策略"""
    
    def generate_signals(self, market_data, indicators):
        """
        均值回归信号生成
        
        买入条件：
        - RSI超卖 (rsi < 30)
        - 价格低于布林带下轨 (close < bb_lower)
        - CCI超卖 (cci < -100)
        
        卖出条件：
        - RSI超买 (rsi > 70)
        - 价格高于布林带上轨 (close > bb_upper)
        - CCI超买 (cci > 100)
        """
        pass
```

**待办**：
- [ ] 实现均值回归策略
- [ ] 超买超卖判断逻辑
- [ ] 回测验证

---

### 4. 实现突破策略
**文件**：`breakout.py`

```python
class BreakoutStrategy(TradingStrategy):
    """突破策略"""
    
    def generate_signals(self, market_data, indicators):
        """
        突破信号生成
        
        买入条件：
        - 价格突破布林带上轨
        - 成交量放大（volume > 1.5 * avg_volume）
        - ADX确认趋势 (adx > 20)
        """
        pass
```

**待办**：
- [ ] 实现突破策略
- [ ] 成交量确认逻辑
- [ ] 假突破过滤

---

## 🟡 中优先级任务

### 5. 策略回测框架
**文件**：`strategy_backtester.py`

```python
class StrategyBacktester:
    """策略回测器"""
    
    def backtest(self, strategy: TradingStrategy, 
                historical_data, 
                initial_capital: float = 100000):
        """
        回测策略表现
        
        Returns:
            {
                'total_return': 0.25,  # 总收益率
                'sharpe_ratio': 1.5,   # 夏普比率
                'max_drawdown': -0.15, # 最大回撤
                'win_rate': 0.60,      # 胜率
                'trades': [...]        # 交易记录
            }
        """
        pass
```

**待办**：
- [ ] 实现回测引擎
- [ ] 计算策略绩效指标
- [ ] 生成回测报告

---

## 🗓️ 实施顺序

### 阶段1：基础框架（2周）
1. 创建 `base_strategy.py`
2. 定义信号数据结构
3. 实现策略基类

### 阶段2：策略实现（3周）
4. 实现趋势跟踪策略
5. 实现均值回归策略
6. 实现突破策略

### 阶段3：回测验证（2周）
7. 实现回测框架
8. 历史数据验证
9. 性能优化

---

## 🗓️ 历史变更

- **2024-11-09**: 创建策略模块TODO（占位）
