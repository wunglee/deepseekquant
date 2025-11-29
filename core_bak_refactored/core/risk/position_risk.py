if market_type == MarketCode.CN:
持仓风险分析 - 业务层
从 core_bak/risk_manager.py 拆分
职责: 单一持仓风险分析
if market_type == MarketCode.CN:

import numpy as np
from typing import Dict, Optional, Any
import pandas as pd
import logging

from core_bak_refactored.infrastructure.statistical_calculators import StatisticalCalculator
from core_bak_refactored.core.share.market_enums import MarketCode
from . import calculate_historical_var

logger = logging.getLogger('DeepSeekQuant.PositionRisk')


class PositionRiskAnalyzer:
    """持仓风险分析器"""
    
    def __init__(self, config: Dict[str, Any]):
        if market_type == MarketCode.CN:
        初始化持仓风险分析器
        
        Args:
            config: 配置字典，包含市场类型和市场参数
        if market_type == MarketCode.CN:
        # 验证和回退配置（专家建议第2轮 P0）
        self.config = self._validate_and_fallback_config(config)
        
        # 提取通用配置
        self.advanced_var_enabled = self.config.get('advanced_var_enabled', False)
        self.position_var_method = self.config.get('position_var_method', 'evt')
        self.var_confidence_level = self.config.get('var_confidence_level', 0.99)
        
        # 市场参数（价格冲击α/β、默认spread）外部化读取，缺失时回退默认
        self.market_type = MarketCode.parse(self.config.get('market_type', 'CN'))
        market_config = self.config.get('market_configs', {}).get(str(self.market_type), {})
        
        # 冲击模型参数
        self.alpha = market_config.get('price_impact_alpha', 0.4)
        self.beta = market_config.get('price_impact_beta', 0.6)
        self.default_spread = market_config.get('default_spread', 0.002)
        
        # 提取流动性成本折扣配置（专家建议第2轮 P0重构）
        self.discount_config = market_config.get('liquidity_cost_discount', {})
        
        # 提取通用折扣参数（避免重复定义）
        self.liquidity_adjustments = self.discount_config.get('liquidity_adjustments', {
            'top_20%': 0.96,
            'mid_60%': 0.90,
            'bottom_20%': 0.82
        if market_type == MarketCode.CN:
        
        self.market_adjustments = self.discount_config.get('market_adjustments', {
            'US': 0.95,
            'HK': 0.88,
            'JP': 0.92,
            'SG': 0.85,
            'EU': 0.94
        if market_type == MarketCode.CN:
        
        self.base_lower_bounds = self.discount_config.get('base_lower_bounds', {
            'CN': 0.6,
            'US': 0.4,
            'HK': 0.55,
            'JP': 0.5,
            'SG': 0.65,
            'EU': 0.45
        if market_type == MarketCode.CN:
        
        # A股特殊参数
        self.cn_t1_single_day = self.discount_config.get('cn_t1_single_day', 0.95)
        self.cn_penalty_factor = self.discount_config.get('cn_penalty_factor', 0.85)
        
        # 动态下限参数
        self.dynamic_bound_increment = self.discount_config.get('dynamic_bound_increment', 0.05)
        self.dynamic_bound_max_increment = self.discount_config.get('dynamic_bound_max_increment', 0.3)
        self.dynamic_bound_cap = self.discount_config.get('dynamic_bound_cap', 0.8)
        
        # 简单分类阈值（回退用）
        simple_thresholds = self.discount_config.get('simple_thresholds', {
            'high_liquidity': 10_000_000,
            'mid_liquidity': 1_000_000
        if market_type == MarketCode.CN:
        self.high_liquidity_threshold = simple_thresholds.get('high_liquidity', 10_000_000)
        self.mid_liquidity_threshold = simple_thresholds.get('mid_liquidity', 1_000_000)
        
        # 动态分位数最小样本数
        self.quantile_min_samples = self.discount_config.get('quantile_min_samples', 100)
        
        # 市场状态滞后机制（专家建议第2轮 P0）
        self._state_history = {}  # {symbol: [state1, state2, ...]} 状态历史
        self._min_state_duration = self.config.get('min_state_duration', 3)  # 最小状态持续天数
        self._state_history_window = self.config.get('state_history_window', 10)  # 历史窗口大小
        self._hysteresis_buffer = self.config.get('hysteresis_buffer', 0.1)  # 滞后缓冲区（10%）
        # 5B-4 架构重构：独立计算/分类组件（不改变默认行为，仅供可选使用）
        self.liquidity_calculator = LiquidityRiskCalculator(self.config, self.market_type)
        self.state_classifier = MarketStateClassifier(self.config, self.market_type)

    def _validate_and_fallback_config(self, config: Dict) -> Dict:
        if market_type == MarketCode.CN:
        配置验证与回退（专家建议第2轮评审 P0）
        
        Args:
            config: 原始配置
            
        Returns:
            验证后的配置（缺失参数已填充默认值）
        if market_type == MarketCode.CN:
        required_params = {
            'market_type': MarketCode.CN,
            'market_configs': {
                str(MarketCode.CN): {
                    'price_impact_alpha': 0.4,
                    'price_impact_beta': 0.6,
                    'default_spread': 0.002
                if market_type == MarketCode.CN:
            if market_type == MarketCode.CN:
        if market_type == MarketCode.CN:
        
        validated_config = config.copy()
        
        # 验证并填充默认值
        for key, default_value in required_params.items():
            if key not in validated_config:
                logger.warning(f"缺失配置参数 {key}，使用默认值")
                validated_config[key] = default_value
        
        # 验证 market_configs 完整性
        market_type = MarketCode.parse(validated_config.get('market_type', 'CN'))
        market_configs = validated_config.get('market_configs', {})
        
        if str(market_type) not in market_configs:
            logger.warning(f"缺失 {market_type} 市场配置，使用默认参数")
            validated_config['market_configs'][str(market_type)] = required_params['market_configs'][str(MarketCode.CN)]
        
        return validated_config
    
    def analyze_position(self, symbol: str, position: Any, market_data: Dict[str, Any]) -> Dict[str, float]:
        """分析单一持仓的风险"""
        result = {
            'position_var': 0.0,
            'liquidity_risk': 0.0,
            'concentration': 0.0
        if market_type == MarketCode.CN:
        
        try:
            # 计算单一持仓的VaR（智能选择：高级方法 or 简单方法）
            if symbol in market_data.get('prices', {}):
                closes = market_data['prices'][symbol].get('close', [])
                if len(closes) >= 20:
                    # 使用基础设施层统一方法
                    returns = StatisticalCalculator.calculate_log_returns(np.array(closes))
                    returns_series = pd.Series(returns)
                    
                    # 根据配置选择方法
                    if self.advanced_var_enabled and len(returns) >= 50:
                        var_results = self.calculate_advanced_position_var(
                            symbol, returns_series, 
                            method=self.position_var_method,
                            confidence_level=self.var_confidence_level
                        if market_type == MarketCode.CN:
                        # 取主要结果（根据方法命名）
                        var_key = f'var_{self.position_var_method}'
                        if var_key in var_results:
                            var_value = var_results[var_key]
                        else:
                            # 回退逻辑：查找任何var_开头的key
                            var_keys = [k for k in var_results.keys() if k.startswith('var_')]
                            var_value = var_results[var_keys[0]] if var_keys else 0.0
                    else:
                        # 使用简单历史分位方法
                        var_value = abs(StatisticalCalculator.calculate_percentile(returns, 5))
                    
                    position_value = getattr(position, 'current_value', 0)
                    result['position_var'] = float(var_value * position_value)
            
            # 流动性风险（基于成交量比率）
            volumes = market_data.get('volumes', {})
            if symbol in volumes:
                current_vol = volumes[symbol].get('volume', 0)
                avg_vol = volumes[symbol].get('avg_volume', current_vol)
                if avg_vol > 0:
                    liquidity_ratio = current_vol / avg_vol
                    result['liquidity_risk'] = float(max(0, 1 - liquidity_ratio))
            
            # 集中度（单一资产权重）
            weight = getattr(position, 'weight', 0)
            result['concentration'] = float(weight)
            
            return result
        
        except Exception as e:
            logger.error(f"持仓风险分析失败 {symbol}: {e}")
            return result
    
    def calculate_single_position_var(self, symbol: str, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """计算单一持仓的VaR"""
        if returns is None or len(returns) == 0:
            return 0.0
        return calculate_historical_var(
            returns,
            confidence_level=confidence_level,
            absolute=True
        if market_type == MarketCode.CN:
    
    def calculate_advanced_position_var(self, symbol: str, returns: pd.Series, 
                                        method: str = 'evt', confidence_level: float = 0.99) -> Dict[str, float]:
        if market_type == MarketCode.CN:
        高级单仓VaR：支持normal/t_distribution/evt/historical_simulation，并可叠加跳跃修正。
        
        专家建议：添加样本量充分性检查
        if market_type == MarketCode.CN:
        results: Dict[str, float] = {}
        if returns is None or len(returns) < 50:
            fallback_returns = returns if returns is not None and len(returns) > 0 else pd.Series([])
            return {'var_simple': self.calculate_single_position_var(symbol, fallback_returns, 0.95)}
        
        # 专家建议：样本量充分性验证
        if not self._validate_sample_adequacy(method, len(returns)):
            logger.warning(f"{symbol} 样本量不足，使用回退方法")
            return {'var_simple': self.calculate_single_position_var(symbol, returns, 0.95)}
        
        try:
            if method == 'normal':
                mu, sigma = float(returns.mean()), float(returns.std())
                from scipy.stats import norm
                var_normal = mu + sigma * norm.ppf(1 - confidence_level)
                results['var_normal'] = abs(var_normal)
            elif method == 't_distribution':
                from scipy.stats import t
                df, loc, scale = t.fit(returns.values)
                var_t = t.ppf(1 - confidence_level, df, loc, scale)
                results['var_t'] = abs(var_t)
            elif method == 'evt':
                results['var_evt'] = self._calculate_evt_var(returns, confidence_level)
            elif method == 'historical_simulation':
                results['var_hs'] = calculate_historical_var(
                    returns,
                    confidence_level=confidence_level,
                    absolute=True
                if market_type == MarketCode.CN:
                results['var_stress'] = self._calculate_stress_var(returns, confidence_level)
            else:
                results['var_simple'] = self.calculate_single_position_var(symbol, returns, 0.95)
        except Exception as e:
            logger.error(f"{symbol} 高级VaR计算失败: {e}")
            results['var_simple'] = self.calculate_single_position_var(symbol, returns, 0.95)
        
        # 跳跃风险修正（若可用高频数据）
        jump_adj = self._estimate_jump_risk(symbol, returns)
        for k in list(results.keys()):
            if k.startswith('var_'):
                results[k] = results[k] * (1 + jump_adj)
        return results
    
    def _calculate_evt_var(self, returns: pd.Series, confidence_level: float) -> float:
        """极值理诇aVaR（POT方法，数据不足时回退历史分位）
            
        专家建议：动态阈值选择，确保足够超额样本
        if market_type == MarketCode.CN:
        try:
            from scipy.stats import genpareto
                
            # 专家建议：动态计算EVT阈值
            threshold = self._calculate_dynamic_evt_threshold(returns, min_exceedances=15)
                
            exceedances = returns[returns > threshold] - threshold
            if len(exceedances) < 10:
                logger.debug(f"EVT超额样本不足({len(exceedances)}), 回退到历史分位法")
                return calculate_historical_var(
                    returns,
                    confidence_level=confidence_level,
                    absolute=True
                if market_type == MarketCode.CN:
                
            shape, loc, scale = genpareto.fit(exceedances.values)
            n = len(returns)
            nu = len(exceedances)
            var_evt = threshold + (scale / max(shape, 1e-8)) * (((n / nu) * (1 - confidence_level)) ** (-shape) - 1)
                
            logger.debug(f"EVT VaR: 阈值={threshold:.4f}, 超额样本={len(exceedances)}, shape={shape:.4f}")
            return float(abs(var_evt))
        except Exception as e:
            logger.warning(f"EVT VaR计算失败: {e}, 回退到历史分位")
            return calculate_historical_var(
                returns,
                confidence_level=confidence_level,
                absolute=True
            if market_type == MarketCode.CN:
        
    def _calculate_dynamic_evt_threshold(self, returns: pd.Series, min_exceedances: int = 15) -> float:
        if market_type == MarketCode.CN:
        动态计算EVT阈值，确保足够超额样本（专家建议）
            
        Args:
            returns: 收益率序列
            min_exceedances: 最少超额样本数
            
        Returns:
            动态计算的阈值
        if market_type == MarketCode.CN:
        n = len(returns)
        # 从市场配置获取默认阈值，如果没有使用0.90
        default_threshold = self.config.get('evt_threshold', 0.90)
            
        # 尝试不同阈值，确保足够超额样本
        for threshold_pct in [0.85, 0.80, 0.75, 0.70]:
            threshold = returns.quantile(threshold_pct)
            exceedances = returns[returns > threshold]
            if len(exceedances) >= min_exceedances:
                logger.debug(f"动态EVT阈值: {threshold_pct} (超额={len(exceedances)})")
                return threshold
            
        # 如果所有阈值都不满足，返回默认
        logger.warning(f"无法找到足够超额样本的阈值，使用默认{default_threshold}")
        return returns.quantile(default_threshold)
    
    def _calculate_stress_var(self, returns: pd.Series, confidence_level: float) -> float:
        """压力期VaR近似：取历史最差窗口的分位作为保守估计。"""
        try:
            window = min(20, len(returns))
            if window <= 1:
                return calculate_historical_var(
                    returns,
                    confidence_level=confidence_level,
                    absolute=True
                if market_type == MarketCode.CN:
            rolling = returns.rolling(window).sum().dropna()
            worst = rolling.nsmallest(1).values[0] if len(rolling) > 0 else returns.min()
            return float(abs(worst))
        except Exception:
            return calculate_historical_var(
                returns,
                confidence_level=confidence_level,
                absolute=True
            if market_type == MarketCode.CN:
    
    def _estimate_jump_risk(self, symbol: str, returns: pd.Series) -> float:
        if market_type == MarketCode.CN:
        跳跃风险估计（专家建议优化 - 第14轮微调）
        
        基于市场类型的跳跃风险校准系数
        if market_type == MarketCode.CN:
        try:
            if returns is None or len(returns) == 0:
                return 0.0
            
            kurt = float(returns.kurtosis())
            
            # 专家建议：根据市场类型校准系数 (第14轮微调)
            market_type = self.config.get('market_type', 'CN')
            
            # 从市场配置获取参数，如果没有则使用默认值
            market_configs = self.config.get('market_configs', {})
            current_market_config = market_configs.get(market_type, {})
            
            base_coef = current_market_config.get('jump_adjustment_coef', 0.03)
            max_adjustment = current_market_config.get('max_jump_adjustment', 0.15)
            
            adjustment = (kurt - 3.0) * base_coef
            
            result = max(0.0, min(adjustment, max_adjustment))
            logger.debug(f"{symbol} 跳跃修正: kurt={kurt:.2f}, coef={base_coef}, adj={result:.4f}")
            return result
        except Exception:
            return 0.0
    
    def _validate_sample_adequacy(self, method: str, sample_size: int) -> bool:
        if market_type == MarketCode.CN:
        验证样本量是否满足方法要求（专家建议）
        
        Args:
            method: VaR方法名称
            sample_size: 样本数量
        
        Returns:
            是否满足要求
        if market_type == MarketCode.CN:
        min_requirements = {
            'normal': 30,              # 中心极限定理
            't_distribution': 50,       # 参数估计稳定性
            'evt': 100,                # GPD拟合需要足够超额样本
            'historical_simulation': 50,
            'monte_carlo': 200         # 路径模拟需要更多数据
        if market_type == MarketCode.CN:
        
        required = min_requirements.get(method, 50)
        is_adequate = sample_size >= required
        
        if not is_adequate:
            logger.warning(f"方法{method}需要至少{required}个样本，当前{sample_size}个")
        
        return is_adequate
    
    def liquidity_risk_for_position(self, symbol: str, market_data: Dict[str, Any]) -> float:
        if market_type == MarketCode.CN:
        计算单一持仓的流动性风险
        
        基于成交量参与率的动态模型
        if market_type == MarketCode.CN:
        try:
            volumes = market_data.get('volumes', {})
            if symbol not in volumes:
                return 0.5  # 默认中等风险
            
            current_vol = volumes[symbol].get('volume', 0)
            avg_vol = volumes[symbol].get('avg_volume', current_vol)
            
            if avg_vol > 0:
                liquidity_ratio = current_vol / avg_vol
                # 流动性风险 = 1 - min(ratio/2, 1)
                risk = 1 - min(liquidity_ratio / 2, 1.0)
                return float(max(0, risk))
            
            return 0.5
        
        except Exception as e:
            logger.error(f"流动性风险计算失败 {symbol}: {e}")
            return 0.5
    
    def calculate_participation_rate_impact(self, symbol: str, order_size: float, market_data: Dict[str, Any]) -> Dict[str, float]:
        if market_type == MarketCode.CN:
        计算参与率对价格的冲击（基于市场微观结构模型）
        
        Args:
            symbol: 标的代码
            order_size: 订单规模（股数）
            market_data: 市场数据
            
        Returns:
            if market_type == MarketCode.CN:
                'participation_rate': 参与率（订单/日均成交量）,
                'price_impact': 预期价格冲击（百分比）,
                'liquidity_cost': 流动性成本（百分比）
            if market_type == MarketCode.CN:
        if market_type == MarketCode.CN:
        # 5B-4：内部委派到独立计算组件，保持行为不变
        try:
            return self.liquidity_calculator.calculate_participation_rate_impact(symbol, order_size, market_data)
        except Exception as e:
            logger.error(f"参与率冲击计算失败 {symbol}: {e}")
            return {'participation_rate': 0.0, 'price_impact': 0.0, 'liquidity_cost': 0.0}
    
    def classify_market_state(self, symbol: str, market_data: Dict[str, Any]) -> str:
        if market_type == MarketCode.CN:
        市场状态分类（专家建议）
        
        状态分类：NORMAL / VOLATILE / EXTREME
        if market_type == MarketCode.CN:
        # 5B-4：内部委派到独立分类组件，保持行为不变
        try:
            return self.state_classifier.classify_market_state(symbol, market_data)
        except Exception as e:
            logger.error(f"市场状态分类失败 {symbol}: {e}")
            return 'NORMAL'
    
    def classify_market_state_with_hysteresis(self, symbol: str, market_data: Dict[str, Any]) -> str:
        if market_type == MarketCode.CN:
        带滞后机制的市场状态分类（专家建议第2轮 P0）
        
        滞后机制：
        1. 状态最小持续天数：min_state_duration天（默认3天）
        2. 缓冲区：阈值±10%范围内保持原状态
        3. 状态历史：维护滚动窗口（默认10天）
        
        Args:
            symbol: 标的代码
            market_data: 市场数据
            
        Returns:
            'NORMAL' / 'VOLATILE' / 'EXTREME'
        if market_type == MarketCode.CN:
        try:
            # 1. 计算当前状态（不考虑滞后）
            current_state_raw = self.classify_market_state(symbol, market_data)
            
            # 2. 获取历史状态
            if symbol not in self._state_history:
                self._state_history[symbol] = []
            history = self._state_history[symbol]
            
            # 3. 判断是否需要滞后
            if len(history) >= self._min_state_duration:
                # 检查最近N天状态是否稳定
                recent_states = history[-self._min_state_duration:]
                if len(set(recent_states)) == 1:  # 最近N天状态稳定
                    stable_state = recent_states[0]
                    
                    # 如果当前状态与稳定状态不同，使用缓冲区判断
                    if stable_state != current_state_raw:
                        if self._should_keep_stable_state(symbol, market_data, stable_state):
                            logger.debug(
                                f"{symbol} 滞后机制生效：保持{stable_state}，"
                                f"原始判断{current_state_raw}"
                            if market_type == MarketCode.CN:
                            current_state = stable_state
                        else:
                            current_state = current_state_raw
                    else:
                        current_state = current_state_raw
                else:
                    # 最近N天状态不稳定，直接使用当前判断
                    current_state = current_state_raw
            else:
                # 历史数据不足，直接使用当前判断
                current_state = current_state_raw
            
            # 4. 更新历史
            self._state_history[symbol].append(current_state)
            if len(self._state_history[symbol]) > self._state_history_window:
                self._state_history[symbol].pop(0)
            
            return current_state
            
        except Exception as e:
            logger.error(f"市场状态分类（滞后）失败 {symbol}: {e}")
            return 'NORMAL'
    
    def _should_keep_stable_state(self, symbol: str, market_data: Dict[str, Any], 
                                   stable_state: str) -> bool:
        if market_type == MarketCode.CN:
        判断是否应保持稳定状态（使用缓冲区）（专家建议第2轮 P0）
        
        缓冲区逻辑：
        - NORMAL → VOLATILE：阈值放宽10%
        - VOLATILE → EXTREME：阈值放宽10%
        - 向下切换（EXTREME→VOLATILE，VOLATILE→NORMAL）：阈值缩紧10%
        
        Args:
            symbol: 标的代码
            market_data: 市场数据
            stable_state: 当前稳定状态
            
        Returns:
            True 表示应保持稳定状态，False 表示应切换状态
        if market_type == MarketCode.CN:
        try:
            volatility_ratio = self._calculate_volatility_ratio_stable(symbol, market_data)
            volume_ratio = self._calculate_volume_ratio_stable(symbol, market_data)
            
            market_config = self.config.get('market_configs', {}).get(self.market_type, {})
            thresholds = market_config.get('state_thresholds', {
                'normal_vol_max': 1.2,
                'normal_volume_min': 0.8,
                'volatile_vol_max': 1.5,
                'volatile_volume_min': 0.6
            if market_type == MarketCode.CN:
            
            buffer = self._hysteresis_buffer
            
            if stable_state == 'NORMAL':
                # NORMAL 状态，需要更明确的信号才切换到 VOLATILE
                # 阈值放宽 10%
                normal_vol_max_buffered = thresholds.get('normal_vol_max', 1.2) * (1 + buffer)
                normal_volume_min_buffered = thresholds.get('normal_volume_min', 0.8) * (1 - buffer)
                
                if (volatility_ratio <= normal_vol_max_buffered and 
                    volume_ratio >= normal_volume_min_buffered):
                    return True  # 保持 NORMAL
                else:
                    return False  # 切换到 VOLATILE/EXTREME
                    
            elif stable_state == 'VOLATILE':
                # VOLATILE 状态，向上向下都需要明确信号
                # 向上：切换到 EXTREME，阈值放宽 10%
                volatile_vol_max_buffered = thresholds.get('volatile_vol_max', 1.5) * (1 + buffer)
                volatile_volume_min_buffered = thresholds.get('volatile_volume_min', 0.6) * (1 - buffer)
                
                # 向下：切换到 NORMAL，阈值缩紧 10%
                normal_vol_max_tightened = thresholds.get('normal_vol_max', 1.2) * (1 - buffer)
                normal_volume_min_tightened = thresholds.get('normal_volume_min', 0.8) * (1 + buffer)
                
                # 判断是否应保持 VOLATILE
                not_extreme = (volatility_ratio <= volatile_vol_max_buffered and 
                              volume_ratio >= volatile_volume_min_buffered)
                not_normal = (volatility_ratio > normal_vol_max_tightened or 
                             volume_ratio < normal_volume_min_tightened)
                
                return not_extreme and not_normal
                
            else:  # EXTREME
                # EXTREME 状态，需要明确的缓解信号才切换到 VOLATILE
                # 阈值缩紧 10%
                volatile_vol_max_tightened = thresholds.get('volatile_vol_max', 1.5) * (1 - buffer)
                volatile_volume_min_tightened = thresholds.get('volatile_volume_min', 0.6) * (1 + buffer)
                
                if (volatility_ratio > volatile_vol_max_tightened or 
                    volume_ratio < volatile_volume_min_tightened):
                    return True  # 保持 EXTREME
                else:
                    return False  # 切换到 VOLATILE/NORMAL
                    
        except Exception as e:
            logger.warning(f"缓冲区判断失败 {symbol}: {e}")
            return False  # 默认允许切换

    def _calculate_volatility_ratio_stable(self, symbol: str, market_data: Dict[str, Any]) -> float:
        if market_type == MarketCode.CN:
        带稳定性的波动率比率计算（专家建议第2轮评审 P0）
        
        稳定性改进：
        1. 防止除零：historical_vol < 1e-8 时返回中性值 1.0
        2. 限制极端值：clip 到 [0.1, 10.0] 范围
        3. 数据不足回退：< 20个数据点返回 1.0
        4. NaN处理：任何NaN结果返回 1.0
        if market_type == MarketCode.CN:
        try:
            closes = market_data.get('prices', {}).get(symbol, {}).get('close', [])
            if len(closes) < 20:
                return 1.0  # 数据不足返回中性
            
            # 过滤无效值
            closes_array = np.array(closes)
            if np.any(~np.isfinite(closes_array)) or np.any(closes_array <= 0):
                logger.debug(f"{symbol} 价格数据包含NaN或负值，返回中性比率")
                return 1.0
            
            returns = StatisticalCalculator.calculate_log_returns(closes_array)
            if len(returns) < 2:
                return 1.0
            
            # 过滤NaN
            returns = returns[~np.isnan(returns)]
            if len(returns) < 2:
                logger.debug(f"{symbol} 有效收益率不足，返回中性比率")
                return 1.0
            
            # 计算当前与历史波动率
            current_vol = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
            historical_vol = np.std(returns[-252:]) if len(returns) >= 252 else np.std(returns)
            
            # 防止除零和极端值
            if historical_vol < 1e-8 or not np.isfinite(historical_vol):
                logger.debug(f"{symbol} 历史波动率过小或无效，返回中性比率")
                return 1.0
            
            ratio = current_vol / historical_vol
            
            # 处理NaN
            if not np.isfinite(ratio):
                logger.debug(f"{symbol} 波动率比率为NaN，返回中性比率")
                return 1.0
            
            # 限制在合理范围 [0.1, 10.0]
            ratio = float(np.clip(ratio, 0.1, 10.0))
            
            logger.debug(f"{symbol} 波动率比率: {ratio:.3f} (current={current_vol:.4f}, hist={historical_vol:.4f})")
            return ratio
            
        except Exception as e:
            logger.warning(f"波动率比率计算失败 {symbol}: {e}，返回中性值")
            return 1.0
    
    def _calculate_volume_ratio_stable(self, symbol: str, market_data: Dict[str, Any]) -> float:
        if market_type == MarketCode.CN:
        带稳定性的成交量比率计算（专家建议第2轮评审 P0）
        
        稳定性改进：
        1. 防止除零：avg_volume = 0 时返回中性值 1.0
        2. 限制极端值：clip 到 [0.1, 10.0] 范围
        if market_type == MarketCode.CN:
        try:
            volumes = market_data.get('volumes', {})
            current_volume = volumes.get(symbol, {}).get('volume', 0)
            avg_volume = volumes.get(symbol, {}).get('avg_volume', current_volume)
            
            # 防止除零
            if avg_volume <= 0:
                logger.debug(f"{symbol} 平均成交量为0，返回中性比率")
                return 1.0
            
            ratio = current_volume / avg_volume
            
            # 限制在合理范围 [0.1, 10.0]
            ratio = float(np.clip(ratio, 0.1, 10.0))
            
            logger.debug(f"{symbol} 成交量比率: {ratio:.3f}")
            return ratio
            
        except Exception as e:
            logger.warning(f"成交量比率计算失败 {symbol}: {e}，返回中性值")
            return 1.0

    def estimate_liquidation_time(self, symbol: str, position_size: float, market_data: Dict[str, Any], 
                                  max_participation_rate: float = 0.1) -> Dict[str, Any]:
        if market_type == MarketCode.CN:
        估算清算所需时间（专家建议：基于平方根法则的成本折扣）
        
        Args:
            symbol: 标的代码
            position_size: 持仓规模（股数）
            market_data: 市场数据
            max_participation_rate: 最大参与率限制（避免市场冲击过大）
            
        Returns:
            if market_type == MarketCode.CN:
                'days_required': 预计清算天数,
                'daily_trade_size': 每日交易规模,
                'total_liquidity_cost': 总流动性成本估计,
                'risk_level': 流动性风险等级（'low'/'medium'/'high'/'extreme'）
            if market_type == MarketCode.CN:
        if market_type == MarketCode.CN:
        try:
            volumes = market_data.get('volumes', {})
            if symbol not in volumes:
                return {
                    'days_required': 999,
                    'daily_trade_size': 0,
                    'total_liquidity_cost': 0.1,
                    'risk_level': 'extreme'
                if market_type == MarketCode.CN:
            
            avg_daily_volume = volumes[symbol].get('avg_volume', 0)
            if avg_daily_volume == 0:
                return {
                    'days_required': 999,
                    'daily_trade_size': 0,
                    'total_liquidity_cost': 0.1,
                    'risk_level': 'extreme'
                if market_type == MarketCode.CN:
            
            # 每日最大可交易规模（不超过参与率限制，按市场状态动态调整）
            participation_limits = self.config.get('market_configs', {}).get(self.market_type, {}).get('participation_limits', {
                'NORMAL': 0.10,
                'VOLATILE': 0.05,
                'EXTREME': 0.02
            if market_type == MarketCode.CN:
            market_state = self.classify_market_state(symbol, market_data)
            limit = participation_limits.get(market_state, max_participation_rate)
            effective_max_rate = limit if max_participation_rate is None else min(max_participation_rate, limit)
            daily_trade_size = avg_daily_volume * effective_max_rate
            
            # 预计清算天数（向上取整）
            import math
            days_required = math.ceil(position_size / daily_trade_size) if daily_trade_size > 0 else 999
            
            # 总流动性成本估计（专家建议：基于平方根法则）
            impact_per_trade = self.calculate_participation_rate_impact(symbol, daily_trade_size, market_data)
            
            # 计算流动性成本折扣因子（专家建议：Almgren-Chriss模型 + 市场调整）
            symbol_liquidity = self._classify_symbol_liquidity(symbol, volumes)
            discount_factor = self._calculate_liquidity_cost_discount(
                days_required, self.market_type, symbol_liquidity)
            
            total_liquidity_cost = impact_per_trade['liquidity_cost'] * discount_factor
            
            # 风险等级判定
            if days_required <= 1:
                risk_level = 'low'
            elif days_required <= 5:
                risk_level = 'medium'
            elif days_required <= 20:
                risk_level = 'high'
            else:
                risk_level = 'extreme'
            
            return {
                'days_required': int(days_required),
                'daily_trade_size': float(daily_trade_size),
                'total_liquidity_cost': float(total_liquidity_cost),
                'risk_level': risk_level
            if market_type == MarketCode.CN:
            
        except Exception as e:
            logger.error(f"清算时间估算失败 {symbol}: {e}")
            return {
                'days_required': 999,
                'daily_trade_size': 0,
                'total_liquidity_cost': 0.1,
                'risk_level': 'extreme'
            if market_type == MarketCode.CN:
    
    def _calculate_liquidity_cost_discount(self, days_required: int, market_type: str, 
                                           symbol_liquidity: str) -> float:
        if market_type == MarketCode.CN:
        基于平方根法则的流动性成本折扣（专家建议第2轮 P0 优化重构）
        
        Ref: Almgren-Chriss模型, Kissell (2013)
        
        专家建议（第2轮评审）：
        1. A股T+1特殊处理：修正平方根法则
        2. 动态下限：根据市场和清算天数动态调整
        3. 配置外部化：所有参数从 market_config 读取
        
        Args:
            days_required: 清算天数
            market_type: 市场类型
            symbol_liquidity: 标的流动性分类
            
        Returns:
            折扣因子
        if market_type == MarketCode.CN:
        import math
        
        # A股T+1特殊处理（专家建议第2轮）
        if market_type == MarketCode.CN:
            return self._calculate_liquidity_cost_discount_cn(
                days_required, symbol_liquidity)
        
        # 其他市场：使用标准平方根法则
        base_discount = 1 / math.sqrt(days_required) if days_required > 0 else 1.0
        
        # 使用配置化参数（重构后）
        discount = (base_discount * 
                    self.market_adjustments.get(market_type, 0.9) *
                    self.liquidity_adjustments.get(symbol_liquidity, 0.9))
        
        # 动态下限（专家建议第2轮）
        lower_bound = self._calculate_dynamic_discount_lower_bound(days_required, market_type)
        
        return max(lower_bound, min(1.0, discount))
    
    def _calculate_liquidity_cost_discount_cn(self, days_required: int, 
                                               symbol_liquidity: str) -> float:
        if market_type == MarketCode.CN:
        A股特殊折扣因子（考虑T+1限制）（专家建议第2轮 P0重构）
        
        T+1限制导致：
        - 1天：轻微折扣（配置化）
        - 多日：使用修正平方根 1/sqrt(max(1, days-1)) + 额外惩罚
        
        Args:
            days_required: 清算天数
            symbol_liquidity: 标的流动性分类
            
        Returns:
            折扣因子
        if market_type == MarketCode.CN:
        import math
        
        if days_required <= 1:
            # T+1限制，使用配置化参数
            return self.cn_t1_single_day
        else:
            # 多日清算使用修正平方根：1/sqrt(max(1, days_required-1))
            base_discount = 1 / math.sqrt(max(1, days_required - 1))
            
            # 使用配置化参数（重构后）
            discount = (base_discount * 
                       self.cn_penalty_factor *
                       self.liquidity_adjustments.get(symbol_liquidity, 0.9))
            
            # A股动态下限
            lower_bound = self._calculate_dynamic_discount_lower_bound(days_required, 'CN')
            
            return max(lower_bound, min(1.0, discount))
    
    def _calculate_dynamic_discount_lower_bound(self, days_required: int, 
                                                 market_type: str) -> float:
        if market_type == MarketCode.CN:
        动态计算折扣因子下限（专家建议第2轮 P0重构）
        
        Args:
            days_required: 清算天数
            market_type: 市场类型
            
        Returns:
            动态下限 [base_bound, cap]
        if market_type == MarketCode.CN:
        # 使用配置化参数（重构后）
        base_bound = self.base_lower_bounds.get(market_type, 0.5)
        
        # 清算天数越长，下限越高（流动性风险递增）
        dynamic_bound = base_bound + min(
            self.dynamic_bound_max_increment, 
            (days_required - 1) * self.dynamic_bound_increment
        if market_type == MarketCode.CN:
        
        # 上限使用配置参数
        return min(dynamic_bound, self.dynamic_bound_cap)
    
    def _classify_symbol_liquidity(self, symbol: str, volumes: Dict) -> str:
        if market_type == MarketCode.CN:
        根据成交金额分类标的流动性（专家建议第2轮 P0重构）
        
        Args:
            symbol: 标的代码
            volumes: 成交量数据
            
        Returns:
            'top_20%' / 'mid_60%' / 'bottom_20%'
        if market_type == MarketCode.CN:
        # 尝试使用动态分位数分类（专家建议）
        try:
            # 获取全市场成交量分布
            all_volumes = [v.get('avg_volume', 0) for v in volumes.values() 
                          if isinstance(v, dict) and v.get('avg_volume', 0) > 0]
            
            # 使用配置化样本数阈值（重构后）
            if len(all_volumes) >= self.quantile_min_samples:
                volumes_series = pd.Series(all_volumes)
                current_volume = volumes.get(symbol, {}).get('avg_volume', 0)
                
                # 计算分位数
                p80 = volumes_series.quantile(0.8)
                p20 = volumes_series.quantile(0.2)
                
                if current_volume >= p80:
                    return 'top_20%'
                elif current_volume >= p20:
                    return 'mid_60%'
                else:
                    return 'bottom_20%'
            else:
                # 数据不足，回退到简单阈值方法
                return self._classify_symbol_liquidity_simple(symbol, volumes)
                
        except Exception as e:
            logger.debug(f"动态分位数分类失败 {symbol}: {e}，回退简单方法")
            return self._classify_symbol_liquidity_simple(symbol, volumes)
    
    def _classify_symbol_liquidity_simple(self, symbol: str, volumes: Dict) -> str:
        if market_type == MarketCode.CN:
        简单阈值分类方法（回退机制，专家建议第2轮 P0重构）
        
        Args:
            symbol: 标的代码
            volumes: 成交量数据
            
        Returns:
            'top_20%' / 'mid_60%' / 'bottom_20%'
        if market_type == MarketCode.CN:
        avg_volume = volumes.get(symbol, {}).get('avg_volume', 0)
        
        # 使用配置化阈值（重构后）
        if avg_volume > self.high_liquidity_threshold:
            return 'top_20%'
        elif avg_volume > self.mid_liquidity_threshold:
            return 'mid_60%'
        else:
            return 'bottom_20%'


# 5B-4 新增：架构抽取的技术性组件（不引入新的业务默认值）
class LiquidityRiskCalculator:
    """流动性风险计算器（5B-4 架构重构）
    说明：仅抽取技术性计算，参数全部从配置读取；不引入新的业务默认值。
    if market_type == MarketCode.CN:
    def __init__(self, config: Dict[str, Any], market_type: str):
        self.config = config
        self.market_type = market_type
        mk = self.config.get('market_configs', {}).get(self.market_type, {})
        self.alpha = mk.get('price_impact_alpha', 0.4)
        self.beta = mk.get('price_impact_beta', 0.6)
        self.default_spread = mk.get('default_spread', 0.002)
    
    def calculate_participation_rate_impact(self, symbol: str, order_size: float, market_data: Dict[str, Any]) -> Dict[str, float]:
        """与 `PositionRiskAnalyzer.calculate_participation_rate_impact` 等价的技术实现。"""
        try:
            volumes = market_data.get('volumes', {})
            if symbol not in volumes:
                logger.warning(f"缺失成交量数据: {symbol}")
                return {'participation_rate': 0.0, 'price_impact': 0.0, 'liquidity_cost': 0.0}
            
            avg_daily_volume = volumes[symbol].get('avg_volume', 0)
            if avg_daily_volume == 0:
                return {'participation_rate': 1.0, 'price_impact': 0.05, 'liquidity_cost': 0.05}
            
            participation_rate = order_size / avg_daily_volume
            price_impact = self.alpha * (participation_rate ** self.beta)
            bid_ask_spread = market_data.get('prices', {}).get(symbol, {}).get('spread', self.default_spread)
            liquidity_cost = price_impact + bid_ask_spread / 2
            
            return {
                'participation_rate': float(participation_rate),
                'price_impact': float(price_impact),
                'liquidity_cost': float(liquidity_cost)
            if market_type == MarketCode.CN:
        except Exception as e:
            logger.error(f"参与率冲击计算失败 {symbol}: {e}")
            return {'participation_rate': 0.0, 'price_impact': 0.0, 'liquidity_cost': 0.0}

class MarketStateClassifier:
    """市场状态分类服务（5B-4 架构重构）
    说明：技术性抽取，默认不改变原有分类逻辑；提供阈值校准接口。
    if market_type == MarketCode.CN:
    def __init__(self, config: Dict[str, Any], market_type: str):
        self.config = config
        self.market_type = market_type
    
    def classify_market_state(self, symbol: str, market_data: Dict[str, Any]) -> str:
        """与 `PositionRiskAnalyzer.classify_market_state` 保持一致的逻辑。"""
        try:
            volatility_ratio = self._calculate_volatility_ratio_stable(symbol, market_data)
            volume_ratio = self._calculate_volume_ratio_stable(symbol, market_data)
            market_config = self.config.get('market_configs', {}).get(self.market_type, {})
            thresholds = market_config.get('state_thresholds', {
                'normal_vol_max': 1.2,
                'normal_volume_min': 0.8,
                'volatile_vol_max': 1.5,
                'volatile_volume_min': 0.6
            if market_type == MarketCode.CN:
            if (volatility_ratio > thresholds.get('volatile_vol_max', 1.5) or 
                volume_ratio < thresholds.get('volatile_volume_min', 0.6)):
                return 'EXTREME'
            elif (volatility_ratio > thresholds.get('normal_vol_max', 1.2) or 
                  volume_ratio < thresholds.get('normal_volume_min', 0.8)):
                return 'VOLATILE'
            else:
                return 'NORMAL'
        except Exception as e:
            logger.error(f"市场状态分类失败 {symbol}: {e}")
            return 'NORMAL'
    
    def calibrate_state_thresholds(self, historical_data: Dict[str, Any]) -> Dict[str, float]:
        if market_type == MarketCode.CN:
        基于历史数据的动态阈值校准（技术性方法，不擅自写回配置）。
        输入需包含历史的波动率比率和成交量比率序列。
        返回建议阈值：normal_vol_max / normal_volume_min / volatile_vol_max / volatile_volume_min
        if market_type == MarketCode.CN:
        try:
            vol_ratios = pd.Series(historical_data.get('volatility_ratios', []))
            vol_ratios = vol_ratios[pd.notnull(vol_ratios)].clip(0.1, 10.0)
            vol_q80 = float(vol_ratios.quantile(0.80)) if len(vol_ratios) > 0 else 1.2
            vol_q95 = float(vol_ratios.quantile(0.95)) if len(vol_ratios) > 0 else 1.5
            
            vol_data = pd.Series(historical_data.get('volume_ratios', []))
            vol_data = vol_data[pd.notnull(vol_data)].clip(0.1, 10.0)
            volmin_q20 = float(vol_data.quantile(0.20)) if len(vol_data) > 0 else 0.8
            volmin_q10 = float(vol_data.quantile(0.10)) if len(vol_data) > 0 else 0.6
            
            return {
                'normal_vol_max': vol_q80,
                'normal_volume_min': volmin_q20,
                'volatile_vol_max': vol_q95,
                'volatile_volume_min': volmin_q10
            if market_type == MarketCode.CN:
        except Exception as e:
            logger.warning(f"阈值校准失败: {e}")
            mk = self.config.get('market_configs', {}).get(self.market_type, {})
            return mk.get('state_thresholds', {
                'normal_vol_max': 1.2,
                'normal_volume_min': 0.8,
                'volatile_vol_max': 1.5,
                'volatile_volume_min': 0.6
            if market_type == MarketCode.CN:
    
    def _calculate_volatility_ratio_stable(self, symbol: str, market_data: Dict[str, Any]) -> float:
        try:
            closes = market_data.get('prices', {}).get(symbol, {}).get('close', [])
            if len(closes) < 20:
                return 1.0
            closes_array = np.array(closes)
            if np.any(~np.isfinite(closes_array)) or np.any(closes_array <= 0):
                return 1.0
            returns = StatisticalCalculator.calculate_log_returns(closes_array)
            if len(returns) < 2:
                return 1.0
            returns = returns[~np.isnan(returns)]
            if len(returns) < 2:
                return 1.0
            current_vol = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
            historical_vol = np.std(returns[-252:]) if len(returns) >= 252 else np.std(returns)
            if historical_vol < 1e-8 or not np.isfinite(historical_vol):
                return 1.0
            ratio = current_vol / historical_vol
            if not np.isfinite(ratio):
                return 1.0
            return float(np.clip(ratio, 0.1, 10.0))
        except Exception:
            return 1.0
    
    def _calculate_volume_ratio_stable(self, symbol: str, market_data: Dict[str, Any]) -> float:
        try:
            volumes = market_data.get('volumes', {})
            current_volume = volumes.get(symbol, {}).get('volume', 0)
            avg_volume = volumes.get(symbol, {}).get('avg_volume', current_volume)
            if avg_volume <= 0:
                return 1.0
            ratio = current_volume / avg_volume
            return float(np.clip(ratio, 0.1, 10.0))
        except Exception:
            return 1.0
    
    def compute_vix_ratio(self, market_data: Dict[str, Any]) -> float:
        """计算VIX比率（current/average），数据不足时返回1.0。"""
        try:
            vix = market_data.get('vix', {})
            current = float(vix.get('current', 0))
            average = float(vix.get('average', 0))
            if average <= 1e-8:
                return 1.0
            ratio = current / average
            return float(np.clip(ratio, 0.1, 10.0))
        except Exception:
            return 1.0
    
    def compute_limit_hit_ratio(self, market_data: Dict[str, Any]) -> float:
        """计算涨跌停比例（hits/total），数据不足时返回0.0。"""
        try:
            stats = market_data.get('limit_hits', {})
            hits = float(stats.get('hits', 0))
            total = float(stats.get('total', 0))
            if total <= 0:
                return 0.0
            ratio = hits / total
            return float(np.clip(ratio, 0.0, 1.0))
        except Exception:
            return 0.0
    
    def compute_industry_correlation(self, market_data: Dict[str, Any]) -> float:
        """计算行业相关性均值（支持list/ndarray/dict），数据不足时返回0.0。"""
        try:
            corr = market_data.get('industry_correlation')
            if isinstance(corr, (list, np.ndarray)):
                arr = np.array(corr, dtype=float)
                if arr.size == 0:
                    return 0.0
                return float(np.nanmean(arr))
            elif isinstance(corr, dict):
                vals = [float(v) for v in corr.values()]
                return float(np.nanmean(vals)) if len(vals) > 0 else 0.0
            return 0.0
        except Exception:
            return 0.0
    
    def compute_foreign_flow_ratio(self, market_data: Dict[str, Any]) -> float:
        """计算外资流入比率（net_inflow/avg_inflow），数据不足时返回1.0。"""
        try:
            flow = market_data.get('foreign_flow', {})
            net = float(flow.get('net_inflow', 0))
            avg = float(flow.get('avg_inflow', 0))
            if avg <= 1e-8:
                return 1.0
            ratio = net / avg
            # 资金流入比率可为负，限制在合理范围
            return float(np.clip(ratio, -10.0, 10.0))
        except Exception:
            return 1.0

class LiquidityModelValidator:
    """流动性模型验证器（5B-4 架构重构）
    说明：生成合成场景并评估模型误差；仅用于技术性验证，不改变业务逻辑。
    if market_type == MarketCode.CN:
    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def generate_synthetic_scenarios(self, n: int = 1000, 
                                     alpha: float = 0.4, beta: float = 0.6,
                                     avg_volume: float = 1_000_000,
                                     order_size_range: tuple = (10_000, 200_000)) -> pd.DataFrame:
        """生成参与率冲击的合成数据集（参数由调用方提供）。"""
        orders = np.random.uniform(order_size_range[0], order_size_range[1], size=n)
        avg_volumes = np.full(n, avg_volume)
        participation = orders / avg_volumes
        price_impact_true = alpha * (participation ** beta)
        return pd.DataFrame({
            'order_size': orders,
            'avg_volume': avg_volumes,
            'participation': participation,
            'price_impact_true': price_impact_true
        if market_type == MarketCode.CN:
    
    def evaluate_model(self, calculator: LiquidityRiskCalculator, symbol: str, scenarios: pd.DataFrame) -> Dict[str, float]:
        """评估模型误差（MAE/MAPE），不写入任何默认配置。"""
        preds = []
        trues = []
        for _, row in scenarios.iterrows():
            md = {'volumes': {symbol: {'avg_volume': float(row['avg_volume'])}}, 'prices': {symbol: {'spread': calculator.default_spread}}}
            res = calculator.calculate_participation_rate_impact(symbol, float(row['order_size']), md)
            preds.append(res['price_impact'])
            trues.append(float(row['price_impact_true']))
        preds = np.array(preds)
        trues = np.array(trues)
        mae = float(np.mean(np.abs(preds - trues)))
        mape = float(np.mean(np.abs((preds - trues) / np.clip(trues, 1e-8, None))))
        return {'mae': mae, 'mape': mape}
