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
from infrastructure.risk_metrics import StatisticalCalculator

logger = logging.getLogger('DeepSeekQuant.PositionRisk')


class PositionRiskAnalyzer:
    """持仓风险分析器"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def analyze_position(self, symbol: str, position: Any, market_data: Dict[str, Any]) -> Dict[str, float]:
        """分析单一持仓的风险"""
        result = {
            'position_var': 0.0,
            'liquidity_risk': 0.0,
            'concentration': 0.0
        }
        
        try:
            # 计算单一持仓的VaR（简化）
            if symbol in market_data.get('prices', {}):
                closes = market_data['prices'][symbol].get('close', [])
                if len(closes) >= 20:
                    # 使用基础设施层统一方法
                    returns = StatisticalCalculator.calculate_log_returns(np.array(closes))
                    var_5pct = np.percentile(returns, 5)
                    position_value = getattr(position, 'current_value', 0)
                    result['position_var'] = float(abs(var_5pct * position_value))
            
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

