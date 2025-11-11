# 信号引擎模块 TODO

> **层级**：Core Layer - Signal Engine  
> **路径**：`core/signal/`  
> **职责**：技术指标计算、信号生成、信号评估

---

## 🟢 indicator_service.py - IndicatorService

**状态**：✅ 已完成（专家复审通过）  
**最后更新**：2024-11-08  
**测试覆盖**：63/63 通过

### 已完成
- [x] 参数系统优化（市场+时间周期）
- [x] 新增VWAP、CCI指标
- [x] 分层批量计算
- [x] 63个测试用例

### 待办事项
- [ ] **P2**: 板块参数配置（主板/创业板/科创板）
  ```python
  indicator = TechnicalIndicators(
      market='CN', 
      timeframe='daily',
      sector='gem'  # 新增参数
  )
  ```

---

## 🔴 signal_generator.py - SignalGenerator

**状态**：❌ 未开始  
**优先级**：高  
**依赖**：✅ indicator_service.py 已完成

### 目标
封装多指标组合逻辑，生成交易信号

### 待办事项

#### 1. 创建信号生成器基础框架
- [ ] **P0**: 创建 `signal_generator.py` 文件
  ```python
  class SignalGenerator:
      """信号生成器 - 封装指标组合逻辑"""
      
      def __init__(self, indicator_service: IndicatorService):
          self.indicator_service = indicator_service
          self.weights = self._default_weights()
      
      def _default_weights(self):
          return {
              'trend': 0.4,      # 趋势指标权重
              'momentum': 0.3,   # 动量指标权重  
              'volume': 0.2,     # 成交量权重
              'volatility': 0.1   # 波动率权重
          }
  ```

#### 2. 实现多指标共振逻辑
- [ ] **P0**: 实现 `multi_indicator_signal()` 方法
  ```python
  def multi_indicator_signal(self, data, weights=None):
      """
      多指标加权信号生成
      
      Returns:
          {
              'signal': 1/-1/0,  # 买入/卖出/中性
              'strength': 0.0-1.0,  # 信号强度
              'components': {...}  # 各指标贡献
          }
      """
      pass
  ```

#### 3. 实现评分系统
- [ ] **P0**: 趋势强度评分
  ```python
  def _calculate_trend_score(self, indicators):
      """
      趋势强度评分（0-100）
      组合: MACD金叉/死叉 + ADX趋势强度 + +DI/-DI方向
      """
      score = 0
      
      # MACD信号（40分）
      if indicators['macd'] > indicators['macd_signal']:
          score += 40
      
      # ADX趋势强度（30分）
      if indicators['adx'] > 25:
          score += 30
      
      # 方向指标（30分）
      if indicators['plus_di'] > indicators['minus_di']:
          score += 30
      
      return score
  ```

- [ ] **P0**: 动量得分计算
  ```python
  def _calculate_momentum_score(self, indicators):
      """
      动量得分 (RSI + KDJ)
      """
      score = 0
      
      # RSI信号（50分）
      rsi = indicators['rsi']
      if 30 < rsi < 70:  # 正常区间
          score += 25
      if rsi > 50:  # 多头
          score += 25
      
      # KDJ信号（50分）
      k, d = indicators['kdj_k'], indicators['kdj_d']
      if k > d:  # 金叉
          score += 50
      
      return score
  ```

#### 4. 测试
- [ ] **P0**: 单元测试
  - 测试单指标信号生成
  - 测试多指标加权组合
  - 测试边界条件（数据不足、NaN处理）

- [ ] **P1**: 历史数据回测验证
  - 使用真实历史数据验证信号质量
  - 统计信号胜率
  - 分析假信号过滤效果

---

## 🟡 signal_evaluator.py - SignalEvaluator

**状态**：❌ 未规划  
**优先级**：中  
**依赖**：signal_generator.py 待完成

### 目标
信号质量评估和回测分析

### 待办事项
- [ ] **P1**: 创建信号评估框架
  ```python
  class SignalEvaluator:
      """信号质量评估器"""
      
      def evaluate_signal_quality(self, signals, actual_returns):
          """
          评估信号质量
          
          Returns:
              {
                  'accuracy': 0.65,  # 准确率
                  'precision': 0.72,  # 精确率
                  'recall': 0.58,  # 召回率
                  'sharpe': 1.5,  # 信号夏普比率
              }
          """
          pass
  ```

- [ ] **P1**: 信号胜率统计
  - 买入信号胜率
  - 卖出信号胜率
  - 不同市场环境下的表现

- [ ] **P2**: 信号效果回测
  - 基于历史数据回测信号表现
  - 生成回测报告

---

## 🗓️ 实施顺序建议

### 阶段1：信号生成（当前优先级）
1. ✅ 完成 IndicatorService（已完成）
2. 🔴 创建 SignalGenerator
3. 🔴 实现多指标共振逻辑
4. 🔴 趋势/动量评分系统
5. 🔴 信号生成测试

### 阶段2：信号评估
1. 🟡 创建 SignalEvaluator
2. 🟡 实现信号质量评估
3. 🟡 历史数据回测验证

---

## 🗓️ 历史变更

- **2024-11-09**: 创建信号模块独立TODO
- **2024-11-08**: indicator_service.py 专家复审通过
