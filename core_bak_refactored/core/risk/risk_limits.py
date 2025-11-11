"""
风险限额管理 - 业务层
从 core_bak/risk_manager.py 拆分
职责: 风险限额检查、违规处理
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

from .risk_models import RiskLimit, PositionLimit, RiskControlAction, RiskLevel, RiskType, RiskMetric
from .risk_metrics_service import RiskMetricsService

logger = logging.getLogger('DeepSeekQuant.RiskLimits')


class RiskLimitsManager:
    """风险限额管理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.risk_metrics_service = RiskMetricsService(config)
        self.risk_limits: Dict[str, RiskLimit] = {}
        self.position_limits: Dict[str, PositionLimit] = {}
        self._initialize_limits()
    
    def _initialize_limits(self):
        try:
            # 从配置加载风险限额
            limits_data = self.config.get('risk_limits', [])
            for limit_data in limits_data:
                try:
                    # 支持枚举对象或字符串映射
                    if isinstance(limit_data.get('risk_type'), dict):
                        limit_data['risk_type'] = RiskType(limit_data['risk_type']['value'])
                    elif isinstance(limit_data.get('risk_type'), str):
                        limit_data['risk_type'] = RiskType(limit_data['risk_type'])
                    
                    if isinstance(limit_data.get('metric'), dict):
                        limit_data['metric'] = RiskMetric(limit_data['metric']['value'])
                    elif isinstance(limit_data.get('metric'), str):
                        limit_data['metric'] = RiskMetric(limit_data['metric'])
                    
                    if isinstance(limit_data.get('action'), dict):
                        limit_data['action'] = RiskControlAction(limit_data['action']['value'])
                    elif isinstance(limit_data.get('action'), str):
                        limit_data['action'] = RiskControlAction(limit_data['action'])
                    
                    risk_limit = RiskLimit(**limit_data)
                    limit_key = f"{risk_limit.risk_type.value}_{risk_limit.metric.value}"
                    self.risk_limits[limit_key] = risk_limit
                except Exception as e:
                    logger.warning(f"风险限额加载失败: {e}")

            # 从配置加载头寸限额
            position_limits_data = self.config.get('position_limits', {})
            for symbol, limit_data in position_limits_data.items():
                try:
                    position_limit = PositionLimit(symbol=symbol, **limit_data)
                    self.position_limits[symbol] = position_limit
                except Exception as e:
                    logger.warning(f"头寸限额加载失败 {symbol}: {e}")

            logger.info(f"已加载 {len(self.risk_limits)} 个风险限额和 {len(self.position_limits)} 个头寸限额")

        except Exception as e:
            logger.error(f"风险限额初始化失败: {e}")
    
    def validate_input(self, portfolio_state, market_data: Dict[str, Any]) -> bool:
        """验证输入数据"""
        try:
            # 检查组合状态
            if not portfolio_state or not portfolio_state.allocations:
                logger.warning("无效的组合状态")
                return False

            # 检查市场数据
            required_market_fields = ['timestamp', 'prices', 'volumes']
            if not all(field in market_data for field in required_market_fields):
                logger.warning("市场数据不完整")
                return False

            # 检查价格数据
            for symbol, allocation in portfolio_state.allocations.items():
                if symbol not in market_data['prices']:
                    logger.warning(f"缺少价格数据: {symbol}")
                    return False

                price_data = market_data['prices'][symbol]
                if 'close' not in price_data or len(price_data['close']) < 20:
                    logger.warning(f"价格数据不足: {symbol}")
                    return False

            return True

        except Exception as e:
            logger.error(f"输入数据验证失败: {e}")
            return False
    
    def check_limits(self, portfolio_state, risk_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """检查所有风险限额"""
        limit_breaches = []

        try:
            # 检查市场风险限额
            market_risk_breaches = self._check_market_risk_limits(risk_metrics)
            limit_breaches.extend(market_risk_breaches)

            # 检查信用风险限额
            credit_risk_breaches = self._check_credit_risk_limits(portfolio_state)
            limit_breaches.extend(credit_risk_breaches)

            # 检查流动性风险限额
            liquidity_risk_breaches = self._check_liquidity_risk_limits(portfolio_state, risk_metrics)
            limit_breaches.extend(liquidity_risk_breaches)

            # 检查集中度风险限额
            concentration_breaches = self._check_concentration_limits(portfolio_state)
            limit_breaches.extend(concentration_breaches)

            # 检查杠杆风险限额
            leverage_breaches = self._check_leverage_limits(portfolio_state)
            limit_breaches.extend(leverage_breaches)

            return limit_breaches

        except Exception as e:
            logger.error(f"风险限额检查失败: {e}")
            return []
    
    def _check_market_risk_limits(self, risk_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        breaches = []

        try:
            # VaR限额检查
            var_limit = self.risk_limits.get('market_risk_value_at_risk')
            if var_limit and 'var_95' in risk_metrics:
                var_value = risk_metrics['var_95']
                if var_value > var_limit.threshold:
                    breaches.append({
                        'limit_type': 'value_at_risk',
                        'metric': 'var_95',
                        'current_value': var_value,
                        'threshold': var_limit.threshold,
                        'breach_amount': var_value - var_limit.threshold,
                        'severity': 'critical' if var_value > var_limit.threshold * 1.5 else 'high',
                        'suggested_action': '减少高风险头寸，增加对冲',
                        'time_horizon': var_limit.time_horizon
                    })

            # ES限额检查
            es_limit = self.risk_limits.get('market_risk_expected_shortfall')
            if es_limit and 'cvar_95' in risk_metrics:
                es_value = risk_metrics['cvar_95']
                if es_value > es_limit.threshold:
                    breaches.append({
                        'limit_type': 'expected_shortfall',
                        'metric': 'cvar_95',
                        'current_value': es_value,
                        'threshold': es_limit.threshold,
                        'breach_amount': es_value - es_limit.threshold,
                        'severity': 'critical' if es_value > es_limit.threshold * 1.5 else 'high',
                        'suggested_action': '加强尾部风险保护',
                        'time_horizon': es_limit.time_horizon
                    })

            return breaches

        except Exception as e:
            logger.error(f"市场风险限额检查失败: {e}")
            return []
    
    def _check_credit_risk_limits(self, portfolio_state) -> List[Dict[str, Any]]:
        breaches = []

        try:
            # 这里需要实际的信用风险数据
            # 简化实现：检查高收益债券和低评级资产

            high_yield_exposure = 0
            for allocation in portfolio_state.allocations.values():
                # 假设有信用评级信息
                credit_rating = allocation.metadata.get('credit_rating', 'investment_grade')
                if credit_rating in ['high_yield', 'junk', 'below_investment_grade']:
                    high_yield_exposure += allocation.weight

            # 检查高收益债券限额
            hy_limit = self.risk_limits.get('credit_risk_high_yield')
            if hy_limit and high_yield_exposure > hy_limit.threshold:
                breaches.append({
                    'limit_type': 'high_yield_exposure',
                    'metric': 'high_yield_weight',
                    'current_value': high_yield_exposure,
                    'threshold': hy_limit.threshold,
                    'breach_amount': high_yield_exposure - hy_limit.threshold,
                    'severity': 'medium',
                    'suggested_action': '减少高收益债券头寸',
                    'time_horizon': hy_limit.time_horizon
                })

            return breaches

        except Exception as e:
            logger.error(f"信用风险限额检查失败: {e}")
            return []
    
    def _check_liquidity_risk_limits(self, portfolio_state, risk_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """检查流动性风险限额"""
        breaches = []

        try:
            # 检查流动性风险
            liquidity_limit = self.risk_limits.get('liquidity_risk_score')
            if liquidity_limit and 'liquidity_risk' in risk_metrics:
                liquidity_risk = risk_metrics['liquidity_risk']
                if liquidity_risk > liquidity_limit.threshold:
                    breaches.append({
                        'limit_type': 'liquidity_risk',
                        'metric': 'liquidity_score',
                        'current_value': liquidity_risk,
                        'threshold': liquidity_limit.threshold,
                        'breach_amount': liquidity_risk - liquidity_limit.threshold,
                        'severity': 'high',
                        'suggested_action': '增加流动性资产，减少非流动性头寸',
                        'time_horizon': liquidity_limit.time_horizon
                    })

            return breaches

        except Exception as e:
            logger.error(f"流动性风险限额检查失败: {e}")
            return []
    
    def _check_concentration_limits(self, portfolio_state) -> List[Dict[str, Any]]:
        breaches = []

        try:
            # 检查单一资产集中度
            for symbol, allocation in portfolio_state.allocations.items():
                position_limit = self.position_limits.get(symbol)
                if position_limit and allocation.weight > position_limit.max_weight:
                    breaches.append({
                        'limit_type': 'single_asset_concentration',
                        'metric': 'asset_weight',
                        'symbol': symbol,
                        'current_value': allocation.weight,
                        'threshold': position_limit.max_weight,
                        'breach_amount': allocation.weight - position_limit.max_weight,
                        'severity': 'high',
                        'suggested_action': f'减少 {symbol} 的头寸',
                        'time_horizon': 'immediate'
                    })

            # 检查行业集中度
            sector_exposures = {}
            for allocation in portfolio_state.allocations.values():
                sector = allocation.sector
                if sector not in sector_exposures:
                    sector_exposures[sector] = 0
                sector_exposures[sector] += allocation.weight

            for sector, exposure in sector_exposures.items():
                sector_limit = self.config.get('sector_limits', {}).get(sector, 0.3)
                if exposure > sector_limit:
                    breaches.append({
                        'limit_type': 'sector_concentration',
                        'metric': 'sector_weight',
                        'sector': sector,
                        'current_value': exposure,
                        'threshold': sector_limit,
                        'breach_amount': exposure - sector_limit,
                        'severity': 'medium',
                        'suggested_action': f'减少 {sector} 行业的暴露',
                        'time_horizon': '1d'
                    })

            return breaches

        except Exception as e:
            logger.error(f"集中度限额检查失败: {e}")
            return []
    
    def _check_leverage_limits(self, portfolio_state) -> List[Dict[str, Any]]:
        breaches = []

        try:
            # 计算杠杆比率
            if portfolio_state.total_value > 0:
                leverage_ratio = portfolio_state.leveraged_value / portfolio_state.total_value

                # 检查杠杆限额
                leverage_limit = self.risk_limits.get('leverage_risk_ratio')
                if leverage_limit and leverage_ratio > leverage_limit.threshold:
                    breaches.append({
                        'limit_type': 'leverage_ratio',
                        'metric': 'leverage',
                        'current_value': leverage_ratio,
                        'threshold': leverage_limit.threshold,
                        'breach_amount': leverage_ratio - leverage_limit.threshold,
                        'severity': 'critical',
                        'suggested_action': '降低杠杆，减少借入资金',
                        'time_horizon': 'immediate'
                    })

            return breaches

        except Exception as e:
            logger.error(f"杠杆限额检查失败: {e}")
            return []
    
    def calculate_limit_utilization(self, risk_assessment) -> Dict[str, float]:
        utilization = {}

        try:
            if not risk_assessment:
                return utilization

            # 计算各种限额的使用率
            risk_metrics = {
                'value_at_risk': getattr(risk_assessment, 'value_at_risk', 0),
                'expected_shortfall': getattr(risk_assessment, 'expected_shortfall', 0),
                'liquidity_risk': getattr(risk_assessment, 'liquidity_risk', 0),
                'concentration_risk': getattr(risk_assessment, 'concentration_risk', 0),
                'leverage_risk': getattr(risk_assessment, 'leverage_risk', 0)
            }

            for limit_key, risk_limit in self.risk_limits.items():
                metric_name = risk_limit.metric.value
                if metric_name in risk_metrics:
                    current_value = risk_metrics[metric_name]
                    if risk_limit.threshold != 0:
                        utilization[limit_key] = abs(current_value / risk_limit.threshold)

            return utilization

        except Exception as e:
            logger.error(f"限额使用率计算失败: {e}")
            return {}


