"""
市场情绪评估模块

职责：
- 评估市场情绪（综合VIX、Put/Call比率、涨跌家数比）
- 计算恐惧贪婪指数
- 分析看涨看跌比率
- 识别情绪极端状态
- 评估市场流动性状况
- 确定波动率状态

来源：从 core/data/analytics/sentiment.py 迁移而来（属于高级市场分析，非数据源处理）
"""
from typing import Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


async def assess_market_sentiment(fetcher: Any) -> Dict[str, Any]:
    """
    评估市场情绪
    
    综合多个指标评估市场情绪，包括：
    - VIX指数
    - Put/Call比率
    - 涨跌家数比
    - 新高新低数
    
    Args:
        fetcher: DataFetcher实例
    
    Returns:
        市场情绪评估结果，包含：
        - sentiment_score: 情绪评分(0-1, 0=极度悲观, 1=极度乐观)
        - bullish_bearish_ratio: 看涨看跌比率
        - fear_greed_index: 恐惧贪婪指数(0-100)
        - put_call_ratio: Put/Call比率
        - market_outlook: 市场展望(bearish/neutral_bearish/neutral/neutral_bullish/bullish)
        - sentiment_extremes: 是否处于情绪极端状态
        - contrarian_indicator: 是否触发逆向指标
    
    Example:
        >>> sentiment = await assess_market_sentiment(fetcher)
        >>> # {'sentiment_score': 0.6, 'market_outlook': 'neutral_bullish', ...}
    """
    try:
        # 简化实现：实际生产中应该调用真实的API获取这些指标
        # 这里使用模拟数据作为示例
        
        # 1. 获取VIX指数（恐慌指数）
        vix_value = await _get_vix_value(fetcher)
        
        # 2. 获取Put/Call比率
        put_call_ratio = await _get_put_call_ratio(fetcher)
        
        # 3. 获取涨跌家数比
        from core_bak_refactored.core.share.market.breadth import get_advance_decline
        advance_decline = await get_advance_decline(fetcher)
        ad_ratio = advance_decline.get('advance_decline_ratio', 1.0)
        
        # 4. 计算综合情绪评分
        # VIX评分：VIX越低，市场越乐观
        if vix_value < 12:
            vix_score = 0.9  # 极度乐观
        elif vix_value < 20:
            vix_score = 0.7  # 乐观
        elif vix_value < 30:
            vix_score = 0.5  # 中性
        elif vix_value < 40:
            vix_score = 0.3  # 悲观
        else:
            vix_score = 0.1  # 极度悲观
        
        # Put/Call比率评分：比率越低，市场越乐观
        if put_call_ratio < 0.7:
            pc_score = 0.9
        elif put_call_ratio < 0.9:
            pc_score = 0.7
        elif put_call_ratio < 1.1:
            pc_score = 0.5
        elif put_call_ratio < 1.3:
            pc_score = 0.3
        else:
            pc_score = 0.1
        
        # 涨跌比评分
        if ad_ratio > 2.0:
            ad_score = 0.9
        elif ad_ratio > 1.5:
            ad_score = 0.7
        elif ad_ratio > 0.8:
            ad_score = 0.5
        elif ad_ratio > 0.5:
            ad_score = 0.3
        else:
            ad_score = 0.1
        
        # 综合评分（加权平均）
        sentiment_score = (vix_score * 0.4 + pc_score * 0.3 + ad_score * 0.3)
        
        # 计算恐惧贪婪指数（0-100）
        fear_greed_index = int(sentiment_score * 100)
        
        # 看涨看跌比率
        bullish_bearish_ratio = 1 / put_call_ratio if put_call_ratio > 0 else float('inf')
        
        # 市场展望
        if sentiment_score >= 0.8:
            market_outlook = 'bullish'
        elif sentiment_score >= 0.6:
            market_outlook = 'neutral_bullish'
        elif sentiment_score >= 0.4:
            market_outlook = 'neutral'
        elif sentiment_score >= 0.2:
            market_outlook = 'neutral_bearish'
        else:
            market_outlook = 'bearish'
        
        # 情绪极端状态
        sentiment_extremes = (sentiment_score >= 0.9 or sentiment_score <= 0.1)
        
        # 逆向指标（极端情绪时触发）
        contrarian_indicator = sentiment_extremes
        
        result = {
            'sentiment_score': sentiment_score,
            'bullish_bearish_ratio': bullish_bearish_ratio,
            'fear_greed_index': fear_greed_index,
            'put_call_ratio': put_call_ratio,
            'market_outlook': market_outlook,
            'sentiment_extremes': sentiment_extremes,
            'contrarian_indicator': contrarian_indicator,
            'vix_value': vix_value,
            'advance_decline_ratio': ad_ratio,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(
            f"市场情绪评估完成: 评分={sentiment_score:.2f}, "
            f"展望={market_outlook}, 恐惧贪婪指数={fear_greed_index}"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"市场情绪评估失败: {e}")
        return {
            'sentiment_score': 0.5,
            'bullish_bearish_ratio': 1.0,
            'fear_greed_index': 50,
            'put_call_ratio': 1.0,
            'market_outlook': 'neutral',
            'sentiment_extremes': False,
            'contrarian_indicator': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


async def _get_vix_value(fetcher: Any) -> float:
    """
    获取VIX指数值
    
    Args:
        fetcher: DataFetcher实例
    
    Returns:
        VIX值
    """
    try:
        # 获取VIX实时数据
        vix_data = await fetcher.get_historical_data(
            ['^VIX'],
            period='1d',
            interval='1d',
            data_type='ohlcv',
            adjustments=False
        )
        
        if '^VIX' in vix_data and vix_data['^VIX']:
            latest = vix_data['^VIX'][-1]
            if hasattr(latest, 'close'):
                return latest.close
            elif isinstance(latest, dict):
                return latest.get('close', 20.0)
        
        # 默认值
        return 20.0
        
    except Exception as e:
        logger.warning(f"获取VIX值失败: {e}")
        return 20.0  # 返回历史平均值


async def _get_put_call_ratio(fetcher: Any) -> float:
    """
    获取Put/Call比率
    
    Args:
        fetcher: DataFetcher实例
    
    Returns:
        Put/Call比率
    """
    try:
        # 简化实现：实际应该从期权数据API获取
        # 这里返回一个合理的默认值
        return 0.8
        
    except Exception as e:
        logger.warning(f"获取Put/Call比率失败: {e}")
        return 1.0  # 中性值


def assess_liquidity_conditions() -> Dict[str, Any]:
    """
    评估市场流动性状况
    
    基于多个流动性指标的综合评估：
    - 买卖价差
    - 市场深度
    - 执行质量
    - 成交量集中度
    - 市场冲击成本
    
    Returns:
        流动性评估结果
    
    Example:
        >>> liquidity = assess_liquidity_conditions()
        >>> # {'liquidity_score': 0.8, 'liquidity_risk': 'low', ...}
    """
    try:
        # 简化实现：实际生产中应该基于真实的流动性数据
        result = {
            'liquidity_score': 0.8,  # 0-1之间的分数
            'bid_ask_spread': 'normal',  # tight/normal/wide
            'market_depth': 'good',  # excellent/good/moderate/poor
            'execution_quality': 'high',  # high/medium/low
            'liquidity_risk': 'low',  # low/moderate/high/extreme
            'volume_concentration': 'moderate',  # low/moderate/high
            'market_impact_cost': 'low',  # low/moderate/high
            'trading_activity': 'normal',  # low/normal/high
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(
            f"流动性评估完成: 评分={result['liquidity_score']:.2f}, "
            f"风险={result['liquidity_risk']}"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"流动性评估失败: {e}")
        return {
            'liquidity_score': 0.5,
            'bid_ask_spread': 'normal',
            'market_depth': 'moderate',
            'execution_quality': 'medium',
            'liquidity_risk': 'moderate',
            'volume_concentration': 'moderate',
            'market_impact_cost': 'moderate',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def determine_volatility_regime(vix_value: float = 20.0) -> Dict[str, Any]:
    """
    确定波动率状态
    
    基于VIX和其他波动率指标的综合判断，识别当前的波动率状态：
    - low: 低波动率环境（VIX < 15）
    - normal: 正常波动率（15 <= VIX < 25）
    - high: 高波动率（25 <= VIX < 40）
    - extreme: 极端波动率（VIX >= 40）
    
    Args:
        vix_value: VIX指数值
    
    Returns:
        波动率状态评估结果
    
    Example:
        >>> regime = determine_volatility_regime(vix_value=18.5)
        >>> # {'regime': 'normal', 'vix_level': 'moderate', ...}
    """
    try:
        # 确定波动率级别
        if vix_value < 12:
            regime = 'low'
            vix_level = 'very_low'
        elif vix_value < 20:
            regime = 'normal'
            vix_level = 'moderate'
        elif vix_value < 30:
            regime = 'high'
            vix_level = 'elevated'
        elif vix_value < 40:
            regime = 'high'
            vix_level = 'high'
        else:
            regime = 'extreme'
            vix_level = 'extreme'
        
        # 波动率聚集性（简化判断）
        volatility_clustering = (regime in ['high', 'extreme'])
        
        # 状态置信度
        if regime in ['low', 'extreme']:
            regime_confidence = 0.9  # 极端状态较明确
        else:
            regime_confidence = 0.7  # 中间状态不太确定
        
        # 预期持续时间
        if regime == 'extreme':
            expected_duration = 'short_term'  # 极端状态通常短暂
        elif regime == 'high':
            expected_duration = 'medium_term'
        else:
            expected_duration = 'long_term'  # 低波动可能持续较久
        
        result = {
            'regime': regime,
            'vix_level': vix_level,
            'vix_value': vix_value,
            'volatility_clustering': volatility_clustering,
            'regime_confidence': regime_confidence,
            'expected_duration': expected_duration,
            'risk_adjustment_factor': vix_value / 20.0,  # 相对于正常水平的调整因子
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(
            f"波动率状态确定: {regime}, VIX={vix_value:.2f}, "
            f"置信度={regime_confidence:.2%}"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"波动率状态确定失败: {e}")
        return {
            'regime': 'normal',
            'vix_level': 'moderate',
            'vix_value': 20.0,
            'volatility_clustering': False,
            'regime_confidence': 0.5,
            'expected_duration': 'unknown',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
