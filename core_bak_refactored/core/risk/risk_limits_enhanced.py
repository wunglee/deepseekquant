"""
风险限额管理增强模块 - P1-3智能化功能
扩展RiskLimitsManager，添加：
1. 智能阈值分层系统 (0.9/1.0/1.2/1.5)
2. 基于投资组合理论的智能推荐
3. 多重违规优先级处理
4. 市场差异化限额管理
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger('DeepSeekQuant.RiskLimitsEnhanced')


# =============================================================================
# P1-3-A: 智能阈值分层系统
# =============================================================================

class ThresholdTier(Enum):
    """阈值层级枚举"""
    GREEN = 0.9      # 绿色区域：预警准备
    YELLOW = 1.0     # 黄色区域：正常限额
    ORANGE = 1.2     # 橙色区域：需要关注
    RED = 1.5        # 红色区域：需要行动


@dataclass
class ThresholdBreach:
    """阈值违规详情"""
    metric_name: str
    current_value: float
    base_threshold: float
    tier: ThresholdTier
    utilization_ratio: float  # 使用率
    severity_score: float     # 严重性评分 (0-100)
    alert_level: str          # 'info', 'warning', 'critical'
    recommended_actions: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# P1-3-D: 市场特定限额配置
# =============================================================================

MARKET_SPECIFIC_LIMITS = {
    'CN': {
        'description': 'A股市场特定限额',
        'single_stock_max_weight': 0.10,     # 单股10%限制
        'sector_max_weight': 0.30,           # 行业30%限制
        'leverage_max': 1.0,                 # 禁止融资融券（普通账户）
        'margin_account_leverage_max': 2.0,  # 融资融券账户最多2倍
        'daily_turnover_limit': 0.20,        # 日换手率20%限制
        'concentration_top10': 0.60,         # 前10大持仓不超过60%
        'st_stock_max_weight': 0.05,         # ST股票单只5%
        'limit_down_exposure_max': 0.15,     # 跌停风险敞口15%
        'regulatory_framework': 'CSRC',
    },
    'US': {
        'description': '美股市场特定限额',
        'single_stock_max_weight': 0.15,     # 单股15%限制（更宽松）
        'sector_max_weight': 0.40,           # 行业40%限制
        'leverage_max': 4.0,                 # Reg T保证金规则：最多4倍
        'day_trading_min_equity': 25000,     # 日内交易最低2.5万美元
        'pattern_day_trader_limit': 4.0,     # PDT规则：4倍杠杆
        'concentration_top10': 0.70,         # 前10大持仓不超过70%
        'otc_stock_max_weight': 0.08,        # OTC股票单只8%
        'penny_stock_max_weight': 0.05,      # 仙股单只5%
        'regulatory_framework': 'SEC/FINRA',
    },
    'HK': {
        'description': '港股市场特定限额',
        'single_stock_max_weight': 0.12,
        'sector_max_weight': 0.35,
        'leverage_max': 2.5,
        'concentration_top10': 0.65,
        'mainland_stock_max_weight': 0.10,   # 沪深港通股票单只10%
        'small_cap_max_weight': 0.08,        # 小盘股单只8%
        'regulatory_framework': 'SFC',
    }
}


class SmartThresholdChecker:
    """智能阈值检查器（P1-3-A）"""
    
    def __init__(self):
        self.threshold_tiers = {tier: tier.value for tier in ThresholdTier}
        self.breach_history: List[ThresholdBreach] = []
    
    def check_smart_threshold(self, metric_name: str, current_value: float, 
                             base_threshold: float) -> Optional[ThresholdBreach]:
        """
        智能阈值分层检查
        
        Args:
            metric_name: 指标名称
            current_value: 当前值
            base_threshold: 基础阈值
            
        Returns:
            阈值违规详情（如果有）
        """
        try:
            # 计算使用率
            utilization = abs(current_value / base_threshold) if base_threshold != 0 else 0
            
            # 确定触及的阈值层级
            tier = self._determine_threshold_tier(utilization)
            if tier is None:
                return None  # 未触及任何警戒线
            
            # 计算严重性评分 (0-100)
            severity_score = self._calculate_severity_score(utilization, tier)
            
            # 确定告警级别
            alert_level = self._determine_alert_level(tier, severity_score)
            
            # 生成推荐行动
            recommended_actions = self._generate_threshold_actions(metric_name, tier, utilization)
            
            breach = ThresholdBreach(
                metric_name=metric_name,
                current_value=current_value,
                base_threshold=base_threshold,
                tier=tier,
                utilization_ratio=utilization,
                severity_score=severity_score,
                alert_level=alert_level,
                recommended_actions=recommended_actions
            )
            
            # 记录违规历史
            self.breach_history.append(breach)
            
            return breach
            
        except Exception as e:
            logger.error(f"智能阈值检查失败: {e}")
            return None
    
    def _determine_threshold_tier(self, utilization: float) -> Optional[ThresholdTier]:
        """确定触及的阈值层级"""
        if utilization >= ThresholdTier.RED.value:
            return ThresholdTier.RED
        elif utilization >= ThresholdTier.ORANGE.value:
            return ThresholdTier.ORANGE
        elif utilization >= ThresholdTier.YELLOW.value:
            return ThresholdTier.YELLOW
        elif utilization >= ThresholdTier.GREEN.value:
            return ThresholdTier.GREEN
        return None
    
    def _calculate_severity_score(self, utilization: float, tier: ThresholdTier) -> float:
        """
        计算严重性评分
        
        评分规则：
        - 绿色区域(0.9-1.0): 10-30分
        - 黄色区域(1.0-1.2): 30-60分
        - 橙色区域(1.2-1.5): 60-85分
        - 红色区域(1.5+):   85-100分
        """
        if tier == ThresholdTier.GREEN:
            # 0.9-1.0 映射到 10-30
            return 10 + (utilization - 0.9) / 0.1 * 20
        elif tier == ThresholdTier.YELLOW:
            # 1.0-1.2 映射到 30-60
            return 30 + (utilization - 1.0) / 0.2 * 30
        elif tier == ThresholdTier.ORANGE:
            # 1.2-1.5 映射到 60-85
            return 60 + (utilization - 1.2) / 0.3 * 25
        else:  # RED
            # 1.5+ 映射到 85-100
            excess = min(utilization - 1.5, 0.5)
            return 85 + (excess / 0.5) * 15
    
    def _determine_alert_level(self, tier: ThresholdTier, severity_score: float) -> str:
        """确定告警级别"""
        if tier == ThresholdTier.RED or severity_score >= 85:
            return 'critical'
        elif tier == ThresholdTier.ORANGE or severity_score >= 60:
            return 'warning'
        else:
            return 'info'
    
    def _generate_threshold_actions(self, metric_name: str, tier: ThresholdTier, 
                                     utilization: float) -> List[str]:
        """生成推荐行动"""
        actions = []
        
        if tier == ThresholdTier.GREEN:
            actions.append(f"{metric_name}接近阈值（{utilization:.1%}），建议提前准备应对措施")
        elif tier == ThresholdTier.YELLOW:
            actions.append(f"{metric_name}已触及正常限额（{utilization:.1%}），需要密切监控")
            actions.append("建议评估当前持仓结构，制定调整计划")
        elif tier == ThresholdTier.ORANGE:
            actions.append(f"{metric_name}超出正常限额（{utilization:.1%}），需要采取行动")
            actions.append("建议在24小时内调整持仓，降低风险敞口")
            actions.append("通知风险管理团队进行评估")
        else:  # RED
            actions.append(f"{metric_name}严重超限（{utilization:.1%}），必须立即行动")
            actions.append("立即停止新增高风险头寸")
            actions.append("启动紧急平仓程序，优先处理高风险资产")
            actions.append("上报风险管理委员会")
        
        return actions


class PortfolioOptimizationAdvisor:
    """投资组合优化顾问（P1-3-B）"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.risk_free_rate = config.get('risk_free_rate', 0.03)
        self.target_sharpe = config.get('target_sharpe_ratio', 1.0)
        self.target_risk_return_ratio = config.get('target_risk_return_ratio', 2.0)
    
    def generate_recommendations(self, portfolio_state, risk_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        生成基于投资组合理论的优化建议
        
        应用理论：
        1. 均值-方差优化（Markowitz）
        2. 夏普比率最大化
        3. 最小方差组合
        4. 有效前沿分析
        """
        recommendations = []
        
        try:
            # 1. 夏普比率优化建议
            recommendations.extend(self._optimize_sharpe_ratio(risk_metrics))
            
            # 2. 最小方差组合建议
            recommendations.extend(self._optimize_minimum_variance(portfolio_state, risk_metrics))
            
            # 3. 有效前沿位置分析
            recommendations.extend(self._analyze_efficient_frontier(risk_metrics))
            
            # 4. 风险-收益均衡建议
            recommendations.extend(self._optimize_risk_return_balance(risk_metrics))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"组合优化建议生成失败: {e}")
            return []
    
    def _optimize_sharpe_ratio(self, risk_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """夏普比率优化"""
        recommendations = []
        
        try:
            current_return = risk_metrics.get('expected_return', 0)
            current_volatility = risk_metrics.get('volatility', 0.15)
            
            if current_volatility > 0:
                current_sharpe = (current_return - self.risk_free_rate) / current_volatility
            else:
                current_sharpe = 0
            
            if current_sharpe < self.target_sharpe:
                recommendations.append({
                    'type': 'sharpe_optimization',
                    'priority': 'medium',
                    'current_sharpe': current_sharpe,
                    'target_sharpe': self.target_sharpe,
                    'description': f'当前夏普比率{current_sharpe:.2f}低于目标{self.target_sharpe:.2f}',
                    'actions': [
                        '增加高夏普比率资产权重（如优质成长股、高评级债券）',
                        '减少低效资产权重（夏普比率<0.5的资产）',
                        f'目标：提升收益率{(self.target_sharpe - current_sharpe) * current_volatility * 100:.1f}个基点，'
                        f'或降低波动率{(current_volatility - (current_return - self.risk_free_rate) / self.target_sharpe) * 100:.1f}个基点'
                    ]
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"夏普比率优化失败: {e}")
            return []
    
    def _optimize_minimum_variance(self, portfolio_state, risk_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """最小方差组合优化"""
        recommendations = []
        
        try:
            current_volatility = risk_metrics.get('volatility', 0.15)
            
            # 估算理论最小波动率（基于分散化）
            n_assets = len(portfolio_state.allocations) if hasattr(portfolio_state, 'allocations') else 10
            avg_correlation = 0.3  # 假设平均相关性
            theoretical_min_vol = current_volatility * np.sqrt((1 + (n_assets - 1) * avg_correlation) / n_assets)
            
            # 如果当前波动率显著高于理论最小值
            if current_volatility > theoretical_min_vol * 1.2:
                excess_vol = current_volatility - theoretical_min_vol
                recommendations.append({
                    'type': 'minimum_variance',
                    'priority': 'medium',
                    'current_volatility': current_volatility,
                    'theoretical_min': theoretical_min_vol,
                    'excess_volatility': excess_vol,
                    'description': f'当前波动率{current_volatility:.2%}高于理论最小值{theoretical_min_vol:.2%}',
                    'actions': [
                        '增加低相关性资产配置（如债券、商品、对冲基金）',
                        '优化资产间相关性结构，降低同向波动风险',
                        f'通过分散化可降低波动率约{excess_vol * 100:.1f}个基点',
                        '考虑引入对冲工具（期权、期货）进一步降低波动'
                    ]
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"最小方差优化失败: {e}")
            return []
    
    def _analyze_efficient_frontier(self, risk_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """有效前沿分析"""
        recommendations = []
        
        try:
            current_return = risk_metrics.get('expected_return', 0)
            current_volatility = risk_metrics.get('volatility', 0.15)
            
            # 估算有效前沿的切点组合（最优夏普比率组合）
            market_return = self.config.get('market_return', 0.08)
            market_volatility = self.config.get('market_volatility', 0.18)
            market_sharpe = (market_return - self.risk_free_rate) / market_volatility
            
            # 当前组合在有效前沿上的位置
            efficient_return = self.risk_free_rate + market_sharpe * current_volatility
            
            if current_return < efficient_return * 0.9:  # 容差10%
                return_gap = efficient_return - current_return
                recommendations.append({
                    'type': 'efficient_frontier',
                    'priority': 'high',
                    'current_position': {'return': current_return, 'volatility': current_volatility},
                    'efficient_position': {'return': efficient_return, 'volatility': current_volatility},
                    'return_gap': return_gap,
                    'description': f'当前组合位于有效前沿下方，收益率缺口{return_gap:.2%}',
                    'actions': [
                        f'在相同风险水平下，可提升收益率{return_gap * 100:.1f}个基点',
                        '调整资产配置向有效前沿移动',
                        '增加收益率/风险比更优的资产',
                        '清理无效资产（低于有效前沿的资产）'
                    ]
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"有效前沿分析失败: {e}")
            return []
    
    def _optimize_risk_return_balance(self, risk_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """风险-收益均衡优化"""
        recommendations = []
        
        try:
            var_95 = abs(risk_metrics.get('var_95', 0))
            expected_return = risk_metrics.get('expected_return', 0)
            
            # 风险收益比（回报/风险比率）
            if var_95 > 0:
                risk_return_ratio = expected_return / var_95
            else:
                risk_return_ratio = 0
            
            if risk_return_ratio < self.target_risk_return_ratio:
                recommendations.append({
                    'type': 'risk_return_balance',
                    'priority': 'medium',
                    'current_ratio': risk_return_ratio,
                    'target_ratio': self.target_risk_return_ratio,
                    'description': f'风险收益比{risk_return_ratio:.2f}低于目标{self.target_risk_return_ratio:.2f}',
                    'actions': [
                        '提高收益率：增加高质量成长型资产',
                        '降低VaR：通过对冲和分散化减少尾部风险',
                        f'需提升回报{(self.target_risk_return_ratio - risk_return_ratio) * var_95 * 100:.1f}个基点，'
                        f'或降低VaR至{expected_return / self.target_risk_return_ratio:.2%}'
                    ]
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"风险收益均衡优化失败: {e}")
            return []


class BreachPrioritizer:
    """违规优先级处理器（P1-3-C）"""
    
    def prioritize_breaches(self, breaches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        多重违规智能优先级排序
        
        评分维度：
        1. 基础严重性（severity权重：30%）
        2. 违规幅度（breach_amount权重：25%）
        3. 时间紧急性（time_horizon权重：20%）
        4. 级联影响（cascading_impact权重：15%）
        5. 监管影响（regulatory_impact权重：10%）
        """
        if not breaches:
            return []
        
        try:
            # 为每个违规计算综合优先级评分
            prioritized = []
            for breach in breaches:
                priority_score = self._calculate_breach_priority(breach)
                cascading_impact = self._analyze_cascading_impact(breach, breaches)
                
                enhanced_breach = breach.copy()
                enhanced_breach['priority_score'] = priority_score
                enhanced_breach['cascading_impact'] = cascading_impact
                enhanced_breach['处理优先级'] = self._determine_priority_level(priority_score)
                
                prioritized.append(enhanced_breach)
            
            # 按优先级评分降序排序
            prioritized.sort(key=lambda x: x['priority_score'], reverse=True)
            
            # 添加处理顺序
            for idx, breach in enumerate(prioritized, 1):
                breach['处理顺序'] = idx
            
            return prioritized
            
        except Exception as e:
            logger.error(f"违规优先级排序失败: {e}")
            return breaches
    
    def _calculate_breach_priority(self, breach: Dict[str, Any]) -> float:
        """计算违规优先级评分（0-100）"""
        score = 0.0
        
        try:
            # 1. 严重性评分（30%）
            severity_map = {'low': 20, 'medium': 50, 'high': 75, 'critical': 100}
            severity = breach.get('severity', 'medium')
            severity_score = severity_map.get(severity, 50)
            score += severity_score * 0.30
            
            # 2. 违规幅度评分（25%）
            threshold = breach.get('threshold', 1.0)
            current_value = abs(breach.get('current_value', 0))
            if threshold != 0:
                breach_ratio = current_value / threshold
                breach_score = min((breach_ratio - 1.0) / 0.5 * 100, 100)
                score += breach_score * 0.25
            
            # 3. 时间紧急性评分（20%）
            time_horizon = breach.get('time_horizon', '1d')
            urgency_map = {
                'immediate': 100, '1h': 90, '4h': 75,
                '1d': 60, '1w': 40, '1m': 20
            }
            urgency_score = urgency_map.get(time_horizon, 50)
            score += urgency_score * 0.20
            
            # 4. 监管影响评分（10%）
            limit_type = breach.get('limit_type', '')
            regulatory_impact = 50
            if 'leverage' in limit_type:
                regulatory_impact = 90
            elif 'concentration' in limit_type:
                regulatory_impact = 70
            elif 'liquidity' in limit_type:
                regulatory_impact = 80
            score += regulatory_impact * 0.10
            
            return min(score, 100)
            
        except Exception as e:
            logger.error(f"优先级评分计算失败: {e}")
            return 50.0
    
    def _analyze_cascading_impact(self, breach: Dict[str, Any], all_breaches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析级联影响"""
        impact = {
            'affected_limits': [],
            'impact_score': 0.0,
            'chain_reaction': []
        }
        
        try:
            limit_type = breach.get('limit_type', '')
            
            # 定义级联关系
            cascading_rules = {
                'leverage_ratio': ['concentration', 'liquidity', 'margin_call'],
                'value_at_risk': ['margin_call', 'liquidity'],
                'liquidity_risk': ['market_impact', 'execution_cost'],
                'concentration': ['specific_risk', 'liquidity']
            }
            
            if limit_type in cascading_rules:
                potential_impacts = cascading_rules[limit_type]
                
                for other_breach in all_breaches:
                    other_type = other_breach.get('limit_type', '')
                    if any(impact_type in other_type for impact_type in potential_impacts):
                        impact['affected_limits'].append(other_type)
                        impact['chain_reaction'].append(f"{limit_type} → {other_type}")
                
                impact['impact_score'] = len(impact['affected_limits']) * 15
            
            return impact
            
        except Exception as e:
            logger.error(f"级联影响分析失败: {e}")
            return impact
    
    def _determine_priority_level(self, priority_score: float) -> str:
        """确定优先级别"""
        if priority_score >= 80:
            return 'P0-紧急'
        elif priority_score >= 60:
            return 'P1-高优先级'
        elif priority_score >= 40:
            return 'P2-中优先级'
        else:
            return 'P3-低优先级'


class MarketSpecificLimitsChecker:
    """市场差异化限额检查器（P1-3-D）"""
    
    def __init__(self, market_type: str = 'CN'):
        self.market_type = market_type
        self.market_limits = MARKET_SPECIFIC_LIMITS.get(market_type, MARKET_SPECIFIC_LIMITS['CN'])
    
    def check_market_limits(self, portfolio_state) -> List[Dict[str, Any]]:
        """检查市场特定限额"""
        breaches = []
        
        try:
            # 1. 检查市场特定的单股限额
            breaches.extend(self._check_single_stock_limits(portfolio_state))
            
            # 2. 检查市场特定的杠杆限额
            breaches.extend(self._check_leverage_limits(portfolio_state))
            
            # 3. 检查市场特定的集中度限额
            breaches.extend(self._check_concentration_limits(portfolio_state))
            
            # 4. 检查监管特定要求
            breaches.extend(self._check_regulatory_requirements(portfolio_state))
            
            logger.info(f"市场特定限额检查完成: {self.market_type}, 发现{len(breaches)}项违规")
            return breaches
            
        except Exception as e:
            logger.error(f"市场特定限额检查失败: {e}")
            return []
    
    def _check_single_stock_limits(self, portfolio_state) -> List[Dict[str, Any]]:
        """检查单股限额"""
        breaches = []
        max_weight = self.market_limits.get('single_stock_max_weight', 0.10)
        
        if hasattr(portfolio_state, 'allocations'):
            for symbol, allocation in portfolio_state.allocations.items():
                if allocation.weight > max_weight:
                    breaches.append({
                        'limit_type': f'{self.market_type}_single_stock_limit',
                        'symbol': symbol,
                        'current_value': allocation.weight,
                        'threshold': max_weight,
                        'severity': 'high',
                        'suggested_action': f'{self.market_type}市场规定单股最高{max_weight:.0%}',
                        'regulatory_framework': self.market_limits.get('regulatory_framework', '')
                    })
        
        return breaches
    
    def _check_leverage_limits(self, portfolio_state) -> List[Dict[str, Any]]:
        """检查杠杆限额"""
        breaches = []
        max_leverage = self.market_limits.get('leverage_max', 1.0)
        
        if hasattr(portfolio_state, 'total_value') and portfolio_state.total_value > 0:
            current_leverage = portfolio_state.leveraged_value / portfolio_state.total_value
            
            if current_leverage > max_leverage:
                breaches.append({
                    'limit_type': f'{self.market_type}_leverage_limit',
                    'current_value': current_leverage,
                    'threshold': max_leverage,
                    'severity': 'critical',
                    'suggested_action': f'{self.market_type}市场最高{max_leverage}倍杠杆',
                    'regulatory_framework': self.market_limits.get('regulatory_framework', '')
                })
        
        return breaches
    
    def _check_concentration_limits(self, portfolio_state) -> List[Dict[str, Any]]:
        """检查集中度限额"""
        breaches = []
        top10_limit = self.market_limits.get('concentration_top10', 0.60)
        
        if hasattr(portfolio_state, 'allocations'):
            weights = [alloc.weight for alloc in portfolio_state.allocations.values()]
            weights_sorted = sorted(weights, reverse=True)
            top10_weight = sum(weights_sorted[:10])
            
            if top10_weight > top10_limit:
                breaches.append({
                    'limit_type': f'{self.market_type}_top10_concentration',
                    'current_value': top10_weight,
                    'threshold': top10_limit,
                    'severity': 'medium',
                    'suggested_action': f'{self.market_type}市场前10持仓不超{top10_limit:.0%}'
                })
        
        return breaches
    
    def _check_regulatory_requirements(self, portfolio_state) -> List[Dict[str, Any]]:
        """检查监管特定要求"""
        if self.market_type == 'CN':
            return self._check_cn_specific_rules(portfolio_state)
        elif self.market_type == 'US':
            return self._check_us_specific_rules(portfolio_state)
        elif self.market_type == 'HK':
            return self._check_hk_specific_rules(portfolio_state)
        return []
    
    def _check_cn_specific_rules(self, portfolio_state) -> List[Dict[str, Any]]:
        """A股市场特定规则"""
        breaches = []
        st_limit = self.market_limits.get('st_stock_max_weight', 0.05)
        st_exposure = 0
        
        if hasattr(portfolio_state, 'allocations'):
            for symbol, allocation in portfolio_state.allocations.items():
                if 'ST' in symbol.upper():
                    st_exposure += allocation.weight
            
            if st_exposure > st_limit:
                breaches.append({
                    'limit_type': 'CN_st_stock_limit',
                    'current_value': st_exposure,
                    'threshold': st_limit,
                    'severity': 'high',
                    'suggested_action': f'ST股票最高{st_limit:.0%}'
                })
        
        return breaches
    
    def _check_us_specific_rules(self, portfolio_state) -> List[Dict[str, Any]]:
        """美股市场特定规则"""
        breaches = []
        
        if hasattr(portfolio_state, 'total_value'):
            min_equity = self.market_limits.get('day_trading_min_equity', 25000)
            day_trades = portfolio_state.metadata.get('day_trades_count', 0) if hasattr(portfolio_state, 'metadata') else 0
            
            if portfolio_state.total_value < min_equity and day_trades >= 4:
                breaches.append({
                    'limit_type': 'US_pdt_rule',
                    'current_value': portfolio_state.total_value,
                    'threshold': min_equity,
                    'severity': 'critical',
                    'suggested_action': 'PDT规则要求最低$25,000账户余额'
                })
        
        return breaches
    
    def _check_hk_specific_rules(self, portfolio_state) -> List[Dict[str, Any]]:
        """港股市场特定规则"""
        return []  # 简化实现
