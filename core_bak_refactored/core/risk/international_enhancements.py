"""
国际化增强功能

包含：
- 增强版夏普比率（考虑市场特定风险溢价）
- 市场异常处理和调整
- 跨市场风险对比
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class InternationalEnhancements:
    """国际化增强功能混入类"""
    
    def calculate_sharpe_ratio_enhanced(
        self, 
        returns: pd.Series, 
        risk_free_rate: Optional[float] = None,
        include_market_premium: bool = True,
        adjust_for_anomalies: bool = True,
        prices: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        增强版夏普比率计算 - 支持市场特定调整
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率
            include_market_premium: 是否包含市场特定风险溢价
            adjust_for_anomalies: 是否对市场异常进行调整
            prices: 价格序列（用于异常检测）
            
        Returns:
            包含标准夏普和增强夏普的详细结果
        """
        try:
            # 基础夏普计算
            standard_sharpe = self.calculate_sharpe_ratio(returns, risk_free_rate)
            
            # 市场异常检测和调整
            adjusted_returns = returns.copy()
            adjustment_factors = {}
            anomalies = {}
            
            if adjust_for_anomalies:
                anomalies = self._detect_market_anomalies(returns, prices)
                self.anomaly_history.extend(anomalies.values())
                
                if anomalies:
                    adjustment_result = self._adjust_for_market_anomalies(returns, anomalies)
                    adjusted_returns = adjustment_result['adjusted_returns']
                    adjustment_factors['anomaly_adjustment'] = adjustment_result['adjustment_factor']
            
            # 市场特定风险溢价调整
            market_premium = 0.0
            if include_market_premium:
                market_premium = self._get_market_specific_risk_premium(returns)
                adjustment_factors['market_premium'] = market_premium
            
            # 计算增强夏普比率
            rf_rate = risk_free_rate if risk_free_rate is not None else self.get_risk_free_rate()
            daily_rf = rf_rate / self.trading_days_per_year
            
            # 调整无风险利率（考虑市场风险溢价）
            adjusted_daily_rf = daily_rf + market_premium / self.trading_days_per_year
            
            excess_returns = adjusted_returns - adjusted_daily_rf
            mean_excess = float(np.mean(excess_returns))
            volatility = self.calculate_volatility(adjusted_returns, annualize=False)
            
            if volatility == 0:
                enhanced_sharpe = 0.0
            else:
                sharpe_daily = mean_excess / volatility
                enhanced_sharpe = sharpe_daily * np.sqrt(self.trading_days_per_year)
            
            return {
                'standard_sharpe': float(standard_sharpe),
                'enhanced_sharpe': float(enhanced_sharpe),
                'adjustment_factors': adjustment_factors,
                'market_type': self.market_type,
                'anomalies_detected': len(anomalies),
                'market_premium_included': market_premium
            }
            
        except Exception as e:
            logger.error(f"增强夏普比率计算失败: {e}")
            # 回退到标准计算
            return {
                'standard_sharpe': self.calculate_sharpe_ratio(returns, risk_free_rate),
                'enhanced_sharpe': self.calculate_sharpe_ratio(returns, risk_free_rate),
                'adjustment_factors': {'error': str(e)},
                'market_type': self.market_type,
                'anomalies_detected': 0,
                'market_premium_included': 0.0
            }
    
    def _adjust_for_market_anomalies(
        self, 
        returns: pd.Series, 
        anomalies: Dict[str, Any]
    ) -> Dict[str, Any]:
        """根据市场异常调整收益率"""
        if not anomalies:
            return {
                'adjusted_returns': returns,
                'adjustment_factor': 1.0,
                'adjustment_method': 'none'
            }
        
        adjusted_returns = returns.copy()
        adjustment_factor = 1.0
        
        for anomaly_id, anomaly in anomalies.items():
            if anomaly['type'] == 'circuit_breaker' and anomaly['severity'] == 'high':
                # 熔断情况下，波动率被低估，适当放大
                adjustment_factor *= 1.15  # 15%的保守调整
                logger.info("应用熔断调整因子: 1.15")
                
            elif anomaly['type'] == 'limit_up_down':
                # 涨跌停情况下，收益率分布被截断
                # 使用缩尾处理（winsorization）
                adjusted_returns = self._winsorize_returns(adjusted_returns, 0.05)
                adjustment_factor *= 1.10  # 10%调整
                logger.info("应用涨跌停调整因子: 1.10")
        
        return {
            'adjusted_returns': adjusted_returns,
            'adjustment_factor': adjustment_factor,
            'adjustment_method': 'conservative'
        }
    
    def _winsorize_returns(self, returns: pd.Series, percentile: float = 0.05) -> pd.Series:
        """缩尾处理（Winsorization）"""
        lower_bound = returns.quantile(percentile)
        upper_bound = returns.quantile(1 - percentile)
        return returns.clip(lower=lower_bound, upper=upper_bound)
    
    def _get_market_specific_risk_premium(self, returns: pd.Series) -> float:
        """获取市场特定风险溢价"""
        base_premiums = {
            'CN': 0.015,  # A股政策风险溢价1.5%
            'US': 0.010,   # 美股流动性风险溢价1.0%
            'HK': 0.020,   # 港股新兴市场溢价2.0%
            'JP': 0.008,   # 日股成熟市场溢价0.8%
            'EU': 0.009    # 欧股成熟市场溢价0.9%
        }
        
        base_premium = base_premiums.get(self.market_type, 0.01)
        
        # 基于市场波动性动态调整
        volatility_ratio = self._calculate_volatility_ratio(returns)
        dynamic_premium = base_premium * volatility_ratio
        
        logger.debug(
            f"市场{self.market_type}风险溢价: 基础{base_premium:.3f}, "
            f"动态调整后{dynamic_premium:.3f}"
        )
        
        return dynamic_premium
    
    def _calculate_volatility_ratio(self, returns: pd.Series) -> float:
        """计算波动率比率（用于动态风险溢价）"""
        if len(returns) < 20:
            return 1.0
        
        # 计算当前波动率与历史平均的比率
        current_vol = self.calculate_volatility(returns, annualize=True)
        
        # 使用滚动窗口计算历史波动率
        if len(returns) >= 60:  # 3个月数据
            historical_vol = self.calculate_volatility(returns.iloc[:-20], annualize=True)
        else:
            historical_vol = current_vol
        
        if historical_vol == 0:
            return 1.0
        
        ratio = current_vol / historical_vol
        # 限制在合理范围内 [0.5, 2.0]
        return max(0.5, min(2.0, ratio))
    
    def calculate_cross_market_risk_comparison(
        self, 
        returns_map: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """
        跨市场风险对比分析
        
        Args:
            returns_map: 市场名称到收益率序列的映射 {'US': returns_us, 'CN': returns_cn}
            
        Returns:
            跨市场风险对比报告
        """
        comparison_report = {
            'comparison_date': pd.Timestamp.now(),
            'markets_analyzed': list(returns_map.keys()),
            'risk_metrics': {},
            'relative_risk': {}
        }
        
        # 导入必要的类（避免循环导入）
        from core.risk.risk_metrics_service import RiskMetricsService
        
        for market_name, returns in returns_map.items():
            # 为每个市场创建临时的风险服务实例
            market_config = {
                'market_type': market_name,
                'market_configs': self.market_configs
            }
            
            try:
                market_service = RiskMetricsService(market_config)
                
                # 计算综合风险指标
                metrics = market_service.calculate_all_metrics(returns)
                
                # 计算增强夏普比率
                if hasattr(market_service, 'calculate_sharpe_ratio_enhanced'):
                    enhanced_sharpe = market_service.calculate_sharpe_ratio_enhanced(returns)
                else:
                    enhanced_sharpe = {'enhanced_sharpe': metrics.get('sharpe_ratio', 0)}
                
                # 检测市场异常
                anomalies = market_service.market_detector.detect_anomalies(returns)
                
                comparison_report['risk_metrics'][market_name] = {
                    'basic_metrics': metrics,
                    'enhanced_sharpe': enhanced_sharpe,
                    'anomalies_detected': len(anomalies)
                }
                
            except Exception as e:
                logger.error(f"市场{market_name}风险分析失败: {e}")
                comparison_report['risk_metrics'][market_name] = {'error': str(e)}
        
        # 计算相对风险指标
        if len(returns_map) >= 2:
            comparison_report['relative_risk'] = self._calculate_relative_risk_measures(
                comparison_report['risk_metrics']
            )
        
        return comparison_report
    
    def _calculate_relative_risk_measures(self, risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """计算相对风险指标"""
        relative_measures = {}
        market_names = [m for m in risk_metrics.keys() if 'error' not in risk_metrics[m]]
        
        if len(market_names) < 2:
            return relative_measures
        
        # 计算波动率比率
        volatilities = {}
        for market in market_names:
            if 'basic_metrics' in risk_metrics[market]:
                volatilities[market] = risk_metrics[market]['basic_metrics'].get('volatility', 0)
        
        if len(volatilities) >= 2:
            base_market = market_names[0]
            for other_market in market_names[1:]:
                if volatilities.get(base_market, 0) > 0:
                    ratio = volatilities.get(other_market, 0) / volatilities[base_market]
                    relative_measures[f'volatility_ratio_{other_market}_to_{base_market}'] = ratio
        
        # 计算风险调整收益差异
        sharpes = {}
        for market in market_names:
            if 'enhanced_sharpe' in risk_metrics[market]:
                sharpes[market] = risk_metrics[market]['enhanced_sharpe'].get('enhanced_sharpe', 0)
        
        if len(sharpes) >= 2:
            base_market = market_names[0]
            for other_market in market_names[1:]:
                relative_measures[f'sharpe_difference_{other_market}_minus_{base_market}'] = (
                    sharpes.get(other_market, 0) - sharpes.get(base_market, 0)
                )
        
        return relative_measures
