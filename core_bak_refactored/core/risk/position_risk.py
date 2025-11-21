"""
持仓风险分析 - 业务层
从 core_bak/risk_manager.py 拆分
职责: 单一持仓风险分析
"""

import numpy as np
from typing import Dict, Optional, Any
import pandas as pd
import logging
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from core_bak_refactored.infrastructure.risk_metrics import StatisticalCalculator

logger = logging.getLogger('DeepSeekQuant.PositionRisk')


class PositionRiskAnalyzer:
    """持仓风险分析器"""
    
    def __init__(self, config: Dict):
        self.config = config
        # 高级VaR配置：支持配置化启用与方法选择
        self.advanced_var_enabled = config.get('advanced_var_enabled', False)
        self.position_var_method = config.get('position_var_method', 'evt')  # 默认EVT方法
        # 支持方法: 'normal', 't_distribution', 'evt', 'historical_simulation'
        self.var_confidence_level = config.get('var_confidence_level', 0.99)
    
    def analyze_position(self, symbol: str, position: Any, market_data: Dict[str, Any]) -> Dict[str, float]:
        """分析单一持仓的风险"""
        result = {
            'position_var': 0.0,
            'liquidity_risk': 0.0,
            'concentration': 0.0
        }
        
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
                        )
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
                        var_5pct = np.percentile(returns, 5)
                        var_value = abs(var_5pct)
                    
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
        var = np.percentile(returns.values, (1 - confidence_level) * 100)
        return float(abs(var))
    
    def calculate_advanced_position_var(self, symbol: str, returns: pd.Series, 
                                        method: str = 'evt', confidence_level: float = 0.99) -> Dict[str, float]:
        """
        高级单仓VaR：支持normal/t_distribution/evt/historical_simulation，并可叠加跳跃修正。
        
        专家建议：添加样本量充分性检查
        """
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
                var_hs = np.percentile(returns.values, (1 - confidence_level) * 100)
                results['var_hs'] = abs(var_hs)
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
        """极值理讻aVaR（POT方法，数据不足时回退历史分位）
            
        专家建议：动态阈值选择，确保足够超额样本
        """
        try:
            from scipy.stats import genpareto
                
            # 专家建议：动态计算EVT阈值
            threshold = self._calculate_dynamic_evt_threshold(returns, min_exceedances=15)
                
            exceedances = returns[returns > threshold] - threshold
            if len(exceedances) < 10:
                logger.debug(f"EVT超额样本不足({len(exceedances)}), 回退到历史分位法")
                return abs(np.percentile(returns.values, (1 - confidence_level) * 100))
                
            shape, loc, scale = genpareto.fit(exceedances.values)
            n = len(returns)
            nu = len(exceedances)
            var_evt = threshold + (scale / max(shape, 1e-8)) * (((n / nu) * (1 - confidence_level)) ** (-shape) - 1)
                
            logger.debug(f"EVT VaR: 阈值={threshold:.4f}, 超额样本={len(exceedances)}, shape={shape:.4f}")
            return float(abs(var_evt))
        except Exception as e:
            logger.warning(f"EVT VaR计算失败: {e}, 回退到历史分位")
            return float(abs(np.percentile(returns.values, (1 - confidence_level) * 100)))
        
    def _calculate_dynamic_evt_threshold(self, returns: pd.Series, min_exceedances: int = 15) -> float:
        """
        动态计算EVT阈值，确保足够超额样本（专家建议）
            
        Args:
            returns: 收益率序列
            min_exceedances: 最少超额样本数
            
        Returns:
            动态计算的阈值
        """
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
                return float(abs(np.percentile(returns.values, (1 - confidence_level) * 100)))
            rolling = returns.rolling(window).sum().dropna()
            worst = rolling.nsmallest(1).values[0] if len(rolling) > 0 else returns.min()
            return float(abs(worst))
        except Exception:
            return float(abs(np.percentile(returns.values, (1 - confidence_level) * 100)))
    
    def _estimate_jump_risk(self, symbol: str, returns: pd.Series) -> float:
        """
        跳跃风险估计（专家建议优化 - 第14轮微调）
        
        基于市场类型的跳跃风险校准系数
        """
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
        """
        验证样本量是否满足方法要求（专家建议）
        
        Args:
            method: VaR方法名称
            sample_size: 样本数量
        
        Returns:
            是否满足要求
        """
        min_requirements = {
            'normal': 30,              # 中心极限定理
            't_distribution': 50,       # 参数估计稳定性
            'evt': 100,                # GPD拟合需要足够超额样本
            'historical_simulation': 50,
            'monte_carlo': 200         # 路径模拟需要更多数据
        }
        
        required = min_requirements.get(method, 50)
        is_adequate = sample_size >= required
        
        if not is_adequate:
            logger.warning(f"方法{method}需要至少{required}个样本，当前{sample_size}个")
        
        return is_adequate
    
    def liquidity_risk_for_position(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """
        计算单一持仓的流动性风险
        
        基于成交量参与率的动态模型
        """
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
        """
        计算参与率对价格的冲击（基于市场微观结构模型）
        
        Args:
            symbol: 标的代码
            order_size: 订单规模（股数）
            market_data: 市场数据
            
        Returns:
            {
                'participation_rate': 参与率（订单/日均成交量）,
                'price_impact': 预期价格冲击（百分比）,
                'liquidity_cost': 流动性成本（百分比）
            }
        """
        try:
            volumes = market_data.get('volumes', {})
            if symbol not in volumes:
                logger.warning(f"缺失成交量数据: {symbol}")
                return {'participation_rate': 0.0, 'price_impact': 0.0, 'liquidity_cost': 0.0}
            
            avg_daily_volume = volumes[symbol].get('avg_volume', 0)
            if avg_daily_volume == 0:
                return {'participation_rate': 1.0, 'price_impact': 0.05, 'liquidity_cost': 0.05}
            
            # 计算参与率
            participation_rate = order_size / avg_daily_volume
            
            # 价格冲击模型：impact = α * (participation_rate)^β
            # α: 市场冲击系数（A股市场约0.3-0.5）
            # β: 非线性指数（通常0.5-0.7）
            alpha = 0.4  # 可配置
            beta = 0.6
            price_impact = alpha * (participation_rate ** beta)
            
            # 流动性成本 = 价格冲击 + 买卖价差
            bid_ask_spread = market_data.get('prices', {}).get(symbol, {}).get('spread', 0.002)  # 默认0.2%
            liquidity_cost = price_impact + bid_ask_spread / 2  # 单边成本
            
            return {
                'participation_rate': float(participation_rate),
                'price_impact': float(price_impact),
                'liquidity_cost': float(liquidity_cost)
            }
            
        except Exception as e:
            logger.error(f"参与率冲击计算失败 {symbol}: {e}")
            return {'participation_rate': 0.0, 'price_impact': 0.0, 'liquidity_cost': 0.0}
    
    def estimate_liquidation_time(self, symbol: str, position_size: float, market_data: Dict[str, Any], 
                                  max_participation_rate: float = 0.1) -> Dict[str, Any]:
        """
        估算清算所需时间
        
        Args:
            symbol: 标的代码
            position_size: 持仓规模（股数）
            market_data: 市场数据
            max_participation_rate: 最大参与率限制（避免市场冲击过大）
            
        Returns:
            {
                'days_required': 预计清算天数,
                'daily_trade_size': 每日交易规模,
                'total_liquidity_cost': 总流动性成本估计,
                'risk_level': 流动性风险等级（'low'/'medium'/'high'/'extreme'）
            }
        """
        try:
            volumes = market_data.get('volumes', {})
            if symbol not in volumes:
                return {
                    'days_required': 999,
                    'daily_trade_size': 0,
                    'total_liquidity_cost': 0.1,
                    'risk_level': 'extreme'
                }
            
            avg_daily_volume = volumes[symbol].get('avg_volume', 0)
            if avg_daily_volume == 0:
                return {
                    'days_required': 999,
                    'daily_trade_size': 0,
                    'total_liquidity_cost': 0.1,
                    'risk_level': 'extreme'
                }
            
            # 每日最大可交易规模（不超过参与率限制）
            daily_trade_size = avg_daily_volume * max_participation_rate
            
            # 预计清算天数（向上取整）
            import math
            days_required = math.ceil(position_size / daily_trade_size) if daily_trade_size > 0 else 999
            
            # 总流动性成本估计（考虑多日累积冲击）
            impact_per_trade = self.calculate_participation_rate_impact(symbol, daily_trade_size, market_data)
            total_liquidity_cost = impact_per_trade['liquidity_cost'] * days_required * 0.8  # 0.8折扣因子（分批降低冲击）
            
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
            }
            
        except Exception as e:
            logger.error(f"清算时间估算失败 {symbol}: {e}")
            return {
                'days_required': 999,
                'daily_trade_size': 0,
                'total_liquidity_cost': 0.1,
                'risk_level': 'extreme'
            }

