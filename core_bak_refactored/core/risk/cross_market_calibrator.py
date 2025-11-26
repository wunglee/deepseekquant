"""
跨市场一致性校准器 - Phase 5B-5
基于专家answer.md第1轮第2节指导实现

职责：
- USD统一计量转换
- 分市场流动性调整因子
- A股T+1和港股LULD特殊处理
- 跨市场风险指标一致性验证（相关性≥0.85）

设计原则：
- 数据标准化：所有风险指标统一为USD计量
- 市场特性处理：针对不同市场机制进行调整
- 一致性验证：确保跨市场风险可比性
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass, field

logger = logging.getLogger('DeepSeekQuant.CrossMarketCalibrator')


@dataclass
class MarketConfig:
    """市场配置（专家answer.md第1轮2.1节）"""
    market_id: str
    liquidity_tier: str  # 'high', 'medium', 'low'
    participation_rate_limit: float  # 参与率上限
    discount_factor: float  # 折扣因子
    special_mechanisms: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossMarketConsistencyResult:
    """跨市场一致性验证结果"""
    correlation: float  # 相关性
    consistency_score: float  # 一致性评分 (0-1)
    passed: bool  # 是否通过（≥0.85）
    market_pairs: List[Tuple[str, str]]
    details: Dict[str, Any] = field(default_factory=dict)


class CrossMarketCalibrator:
    """
    跨市场一致性校准器（Phase 5B-5）
    
    基于专家answer.md第1轮2.1节和2.2节指导：
    
    **2.1 USD统一计量标准**
    - 基准货币：所有风险指标统一换算为USD计量
    - 汇率基准：使用事件窗口期内的日均中间价
    - 流动性调整因子：
      * 高流动性市场(US/EU)：参与率上限10%, 折扣因子0.95
      * 中等流动性市场(CN/HK)：参与率上限5%, 折扣因子0.90
      * 低流动性市场(JP/SG)：参与率上限2%, 折扣因子0.85
    
    **2.2 市场机制特殊处理**
    - A股T+1机制：单日清算折扣因子0.95，多日清算使用1/sqrt(max(1, days-1))修正
    - 港股LULD机制：波动性调整系数1.2
    - 一致性目标：同一风险事件下，不同市场风险指标的相关性≥0.85
    """
    
    # 市场配置字典（专家answer.md第1轮2.1节）
    MARKET_CONFIGS = {
        'US': MarketConfig(
            market_id='US',
            liquidity_tier='high',
            participation_rate_limit=0.10,
            discount_factor=0.95,
            special_mechanisms={}
        ),
        'EU': MarketConfig(
            market_id='EU',
            liquidity_tier='high',
            participation_rate_limit=0.10,
            discount_factor=0.95,
            special_mechanisms={}
        ),
        'CN': MarketConfig(
            market_id='CN',
            liquidity_tier='medium',
            participation_rate_limit=0.05,
            discount_factor=0.90,
            special_mechanisms={
                't1_restriction': True,
                't1_single_day_discount': 0.95
            }
        ),
        'HK': MarketConfig(
            market_id='HK',
            liquidity_tier='medium',
            participation_rate_limit=0.05,
            discount_factor=0.90,
            special_mechanisms={
                'luld_mechanism': True,
                'volatility_adjustment': 1.2
            }
        ),
        'JP': MarketConfig(
            market_id='JP',
            liquidity_tier='low',
            participation_rate_limit=0.02,
            discount_factor=0.85,
            special_mechanisms={}
        ),
        'SG': MarketConfig(
            market_id='SG',
            liquidity_tier='low',
            participation_rate_limit=0.02,
            discount_factor=0.85,
            special_mechanisms={}
        )
    }
    
    # 汇率缓存（日均中间价）
    _exchange_rate_cache: Dict[Tuple[str, str], float] = {}
    
    def __init__(self, base_currency: str = 'USD'):
        """
        初始化跨市场校准器
        
        Args:
            base_currency: 基准货币（默认USD）
        """
        self.base_currency = base_currency
        self._consistency_history = {}  # 一致性验证历史记录
    
    def normalize_to_usd(self, 
                         value: float, 
                         source_currency: str,
                         event_window_data: Optional[pd.DataFrame] = None) -> float:
        """
        将金额标准化为USD（专家answer.md第1轮2.1节）
        
        Args:
            value: 原始金额
            source_currency: 源货币（CNY/HKD/JPY/EUR等）
            event_window_data: 事件窗口数据（用于计算日均中间价）
        
        Returns:
            USD金额
        """
        if source_currency == self.base_currency:
            return value
        
        # 获取汇率（日均中间价）
        exchange_rate = self._get_average_exchange_rate(
            source_currency, 
            self.base_currency,
            event_window_data
        )
        
        usd_value = value * exchange_rate
        logger.debug(f"USD标准化: {value} {source_currency} = {usd_value} {self.base_currency} (rate={exchange_rate})")
        
        return usd_value
    
    def _get_average_exchange_rate(self,
                                    from_currency: str,
                                    to_currency: str,
                                    event_window_data: Optional[pd.DataFrame] = None) -> float:
        """
        获取日均中间价汇率（专家answer.md第1轮2.1节）
        
        Args:
            from_currency: 源货币
            to_currency: 目标货币
            event_window_data: 事件窗口数据（包含汇率序列）
        
        Returns:
            日均汇率
        """
        # 检查缓存
        cache_key = (from_currency, to_currency)
        if cache_key in self._exchange_rate_cache:
            return self._exchange_rate_cache[cache_key]
        
        # 如果提供了事件窗口数据，计算实际日均汇率
        if event_window_data is not None and 'exchange_rate' in event_window_data.columns:
            avg_rate = event_window_data['exchange_rate'].mean()
            self._exchange_rate_cache[cache_key] = avg_rate
            return avg_rate
        
        # 否则使用静态汇率表（近似值，实际应从数据源获取）
        static_rates = {
            ('CNY', 'USD'): 0.14,    # 1 CNY ≈ 0.14 USD
            ('HKD', 'USD'): 0.128,   # 1 HKD ≈ 0.128 USD
            ('JPY', 'USD'): 0.0067,  # 1 JPY ≈ 0.0067 USD
            ('EUR', 'USD'): 1.08,    # 1 EUR ≈ 1.08 USD
            ('SGD', 'USD'): 0.74,    # 1 SGD ≈ 0.74 USD
        }
        
        rate = static_rates.get(cache_key, 1.0)
        logger.debug(f"使用静态汇率: {from_currency}/{to_currency} = {rate}")
        
        self._exchange_rate_cache[cache_key] = rate
        return rate
    
    def apply_liquidity_adjustment(self, 
                                   raw_risk_metric: float,
                                   market_id: str,
                                   days_required: int = 1) -> float:
        """
        应用流动性调整因子（专家answer.md第1轮2.1节和2.2节）
        
        Args:
            raw_risk_metric: 原始风险指标
            market_id: 市场代码
            days_required: 清算所需天数
        
        Returns:
            调整后的风险指标
        """
        market_config = self.MARKET_CONFIGS.get(market_id)
        if market_config is None:
            logger.warning(f"未知市场 {market_id}，使用默认调整因子0.90")
            discount_factor = 0.90
        else:
            discount_factor = market_config.discount_factor
        
        # A股T+1特殊处理（专家answer.md第1轮2.2节）
        if market_id == 'CN' and 't1_restriction' in (market_config.special_mechanisms if market_config else {}):
            if days_required == 1:
                # 单日清算：使用T+1单日折扣
                discount_factor = market_config.special_mechanisms.get('t1_single_day_discount', 0.95)
            else:
                # 多日清算：使用修正公式 1/sqrt(max(1, days-1))
                multi_day_discount = 1.0 / np.sqrt(max(1, days_required - 1))
                discount_factor = min(discount_factor, multi_day_discount)
                logger.debug(f"A股T+1多日清算: days={days_required}, discount={discount_factor:.4f}")
        
        # 港股LULD波动性调整（专家answer.md第1轮2.2节）
        # 注意：此处仅处理清算时间折扣，波动性调整由calculate_volatility_adjustment处理
        
        adjusted_metric = raw_risk_metric * discount_factor
        logger.debug(f"流动性调整: {raw_risk_metric:.4f} -> {adjusted_metric:.4f} (market={market_id}, discount={discount_factor:.4f})")
        
        return adjusted_metric
    
    def calculate_volatility_adjustment(self, 
                                        raw_volatility: float,
                                        market_id: str) -> float:
        """
        计算波动性调整（专家answer.md第1轮2.2节：港股LULD）
        
        Args:
            raw_volatility: 原始波动率
            market_id: 市场代码
        
        Returns:
            调整后的波动率
        """
        market_config = self.MARKET_CONFIGS.get(market_id)
        
        # 港股LULD机制：波动性调整系数1.2
        if market_id == 'HK' and market_config and 'luld_mechanism' in market_config.special_mechanisms:
            volatility_adjustment = market_config.special_mechanisms.get('volatility_adjustment', 1.2)
            adjusted_volatility = raw_volatility * volatility_adjustment
            logger.debug(f"港股LULD波动性调整: {raw_volatility:.4f} -> {adjusted_volatility:.4f} (factor={volatility_adjustment})")
            return adjusted_volatility
        
        return raw_volatility
    
    def validate_cross_market_consistency(self,
                                         market_risk_metrics: Dict[str, pd.Series],
                                         min_correlation: float = 0.85) -> CrossMarketConsistencyResult:
        """
        验证跨市场一致性（专家answer.md第1轮2.2节：相关性≥0.85）
        
        Args:
            market_risk_metrics: {market_id: risk_metric_series}，已标准化为USD
            min_correlation: 最小相关性阈值（默认0.85）
        
        Returns:
            CrossMarketConsistencyResult
        """
        if len(market_risk_metrics) < 2:
            logger.warning("市场数量不足2个，无法验证跨市场一致性")
            return CrossMarketConsistencyResult(
                correlation=0.0,
                consistency_score=0.0,
                passed=False,
                market_pairs=[],
                details={'error': 'insufficient_markets'}
            )
        
        # 计算两两市场间的相关性
        market_ids = list(market_risk_metrics.keys())
        correlations = []
        market_pairs = []
        
        for i in range(len(market_ids)):
            for j in range(i + 1, len(market_ids)):
                market_a = market_ids[i]
                market_b = market_ids[j]
                
                series_a = market_risk_metrics[market_a]
                series_b = market_risk_metrics[market_b]
                
                # 对齐时间序列（取交集）
                common_index = series_a.index.intersection(series_b.index)
                if len(common_index) < 10:
                    logger.warning(f"市场 {market_a} 和 {market_b} 共同数据点不足10个，跳过")
                    continue
                
                aligned_a = series_a.loc[common_index]
                aligned_b = series_b.loc[common_index]
                
                # 计算皮尔逊相关系数
                correlation = aligned_a.corr(aligned_b)
                correlations.append(correlation)
                market_pairs.append((market_a, market_b))
                
                logger.debug(f"市场 {market_a} vs {market_b}: 相关性 = {correlation:.4f}")
        
        if not correlations:
            return CrossMarketConsistencyResult(
                correlation=0.0,
                consistency_score=0.0,
                passed=False,
                market_pairs=[],
                details={'error': 'no_valid_pairs'}
            )
        
        # 计算平均相关性
        avg_correlation = np.mean(correlations)
        
        # 计算一致性评分（0-1）
        # 评分公式：(avg_correlation - min_threshold) / (1 - min_threshold)
        # 例如：avg=0.85, min=0.85 -> score=0
        #       avg=1.0, min=0.85 -> score=1
        consistency_score = max(0.0, (avg_correlation - min_correlation) / (1.0 - min_correlation))
        
        # 判断是否通过
        passed = avg_correlation >= min_correlation
        
        result = CrossMarketConsistencyResult(
            correlation=avg_correlation,
            consistency_score=consistency_score,
            passed=passed,
            market_pairs=market_pairs,
            details={
                'individual_correlations': dict(zip([f"{a}-{b}" for a, b in market_pairs], correlations)),
                'min_correlation': min(correlations) if correlations else 0.0,
                'max_correlation': max(correlations) if correlations else 0.0,
                'threshold': min_correlation
            }
        )
        
        logger.info(f"跨市场一致性验证: 平均相关性={avg_correlation:.4f}, 通过={passed}")
        
        return result
    
    def calibrate_risk_metrics(self,
                               raw_metrics: Dict[str, Dict[str, Any]],
                               event_window_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Dict[str, Any]]:
        """
        校准风险指标（USD标准化 + 流动性调整）
        
        Args:
            raw_metrics: {market_id: {'value': float, 'currency': str, 'days_required': int, ...}}
            event_window_data: {market_id: DataFrame with exchange_rate}
        
        Returns:
            校准后的风险指标（USD计量，已应用流动性调整）
        """
        calibrated_metrics = {}
        
        for market_id, metric_data in raw_metrics.items():
            try:
                # 1. USD标准化
                raw_value = metric_data.get('value', 0.0)
                currency = metric_data.get('currency', 'USD')
                
                window_data = event_window_data.get(market_id) if event_window_data else None
                usd_value = self.normalize_to_usd(raw_value, currency, window_data)
                
                # 2. 流动性调整
                days_required = metric_data.get('days_required', 1)
                adjusted_value = self.apply_liquidity_adjustment(usd_value, market_id, days_required)
                
                # 3. 波动性调整（如适用）
                raw_volatility = metric_data.get('volatility', None)
                adjusted_volatility = None
                if raw_volatility is not None:
                    adjusted_volatility = self.calculate_volatility_adjustment(raw_volatility, market_id)
                
                calibrated_metrics[market_id] = {
                    'raw_value': raw_value,
                    'raw_currency': currency,
                    'usd_value': usd_value,
                    'adjusted_value': adjusted_value,
                    'liquidity_discount': self.MARKET_CONFIGS[market_id].discount_factor if market_id in self.MARKET_CONFIGS else 0.90,
                    'days_required': days_required,
                    'raw_volatility': raw_volatility,
                    'adjusted_volatility': adjusted_volatility
                }
                
                logger.info(f"校准完成 {market_id}: {raw_value:.2f} {currency} -> {adjusted_value:.2f} USD")
                
            except Exception as e:
                logger.error(f"校准失败 {market_id}: {e}")
                calibrated_metrics[market_id] = {
                    'error': str(e),
                    'raw_value': metric_data.get('value', 0.0),
                    'adjusted_value': 0.0
                }
        
        return calibrated_metrics
