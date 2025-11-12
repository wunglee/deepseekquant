"""
风险限额管理增强模块 - P1-3智能化功能

基于专家审核后的指导实施（docs/consultation.md）

扩展RiskLimitsManager，添加：
1. 智能阈值分层系统 (0.85/1.0/1.15/1.3) - 专家调整后
2. 基于投资组合理论的智能推荐 - 仅生成建议，不执行优化
3. 多重违规优先级处理 - 动态权重分配
4. 市场差异化限额管理 - 符合监管要求

职责边界说明：
- ✅ 风险评估、限额检查、违规检测（本模块）
- ❌ 实际组合优化执行 → core/portfolio/模块
- ❌ 市场状态判定 → core/strategy/模块
- ❌ 可视化仪表板 → apps/模块
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
    """阈值层级枚举（专家审核后的参数）"""
    GREEN = 0.85     # 绿色区域：提前预警（从0.9调整）
    YELLOW = 1.0     # 黄色区域：标准限额
    ORANGE = 1.15    # 橙色区域：更敏感（从1.2调整）
    RED = 1.3        # 红色区域：监管要求（从1.5调整）


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
# P1-3-D: 市场特定限额配置（专家审核后的参数）
# =============================================================================

MARKET_SPECIFIC_LIMITS = {
    'CN': {
        'description': 'A股市场特定限额（专家确认：第2轮咨询）',
        # 🔒 监管明确要求（需法务审核）
        'single_stock_max_weight': 0.10,      # 🔒 证监会《证券投资基金运作管理办法》第31条
        'leverage_max': 1.0,                  # 🔒 《证券法》禁止普通账户杠杆
        'margin_account_leverage_max': 2.0,   # 🔒 《融资融券业务管理办法》
        
        # ⚖️ 专家建议调整（基于风控经验）
        'daily_turnover_limit': 0.15,         # ⚖️ 专家调整：0.20→0.15（保守估计）
        'limit_down_exposure_max': 0.10,      # ⚖️ 专家调整：0.15→0.10（波动性考虑）
        
        # 🆕 专家新增建议
        '创业板_stock_max_weight': 0.08,      # 🆕 专家建议：基于创业板波动率1.8倍
        '科创板_stock_max_weight': 0.08,      # 🆕 专家建议：基于科创板波动率2.0倍
        
        # 📊 行业惯例（非强制但普遍遵守）
        'sector_max_weight': 0.30,            # 📊 行业惯例，非强制但普遍遵守
        'concentration_top10': 0.60,          # 📊 基金业协会自律规则
        'st_stock_max_weight': 0.05,          # 📊 机构内部风控要求
        
        'regulatory_framework': 'CSRC',
    },
    'US': {
        'description': '美股市场特定限额（专家确认）',
        'single_stock_max_weight': 0.15,
        'sector_max_weight': 0.40,
        'leverage_max': 4.0,
        # PDT规则完整实现（专家补充）
        'day_trading_min_equity': 25000,      # ✅ FINRA规则4210
        'pattern_day_trader_limit': 4.0,      # ✅ Reg T保证金规则
        'concentration_top10': 0.70,
        'otc_stock_max_weight': 0.05,         # ⚠️ 调整：从0.08→0.05（专家建议更保守）
        'penny_stock_max_weight': 0.03,       # ⚠️ 调整：从0.05→0.03（专家建议）
        'regulatory_framework': 'SEC/FINRA',
    },
    'HK': {
        'description': '港股市场特定限额（专家确认）',
        'single_stock_max_weight': 0.10,      # ⚠️ 调整：从0.12→0.10（符合SFC要求）
        'sector_max_weight': 0.30,            # ⚠️ 调整：从0.35→0.30（专家建议）
        'leverage_max': 2.0,                  # ⚠️ 调整：从2.5→2.0（专家建议更保守）
        'concentration_top10': 0.60,          # ⚠️ 调整：从0.65→0.60（专家建议）
        'mainland_stock_max_weight': 0.10,    # 沪深港通股票单只10%
        'small_cap_max_weight': 0.05,         # ⚠️ 调整：从0.08→0.05（专家建议）
        'regulatory_framework': 'SFC',
    }
}


class SmartThresholdChecker:
    """智能阈值检查器（P1-3-A）- 基于专家指导修正"""
    
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
        计算严重性评分（专家修正：使用指数函数而非线性）
        
        评分规则（专家调整后）：
        - 绿色区域(0.85-1.0): 10-30分，指数增长
        - 黄色区域(1.0-1.15): 30-60分
        - 橙色区域(1.15-1.3): 60-85分
        - 红色区域(1.3+):   85-100分，快速增长
        """
        if tier == ThresholdTier.GREEN:
            # 0.85-1.0 → 10-30分，指数增长(专家建议)
            base = 10
            excess = (utilization - 0.85) / 0.15
            return base + 20 * (excess ** 1.5)
        
        elif tier == ThresholdTier.YELLOW:
            # 1.0-1.15 → 30-60分
            base = 30
            excess = (utilization - 1.0) / 0.15
            return base + 30 * (excess ** 1.3)
        
        elif tier == ThresholdTier.ORANGE:
            # 1.15-1.3 → 60-85分
            base = 60
            excess = (utilization - 1.15) / 0.15
            return base + 25 * (excess ** 1.2)
        
        else:  # RED
            # 1.3+ → 85-100分，快速增长(专家建议)
            base = 85
            excess = min(utilization - 1.3, 0.3)  # 限制最大超额30%
            return base + 15 * (excess / 0.3) ** 0.8
    
    def _determine_alert_level(self, tier: ThresholdTier, severity_score: float) -> str:
        """确定告警级别（专家修正：考虑连续违规）"""
        # 检查连续违规历史（专家建议）
        recent_breaches = self._get_recent_breaches(hours=24)
        consecutive_count = len(recent_breaches)
        
        base_level = {
            ThresholdTier.GREEN: 'info',
            ThresholdTier.YELLOW: 'warning',
            ThresholdTier.ORANGE: 'warning',
            ThresholdTier.RED: 'critical'
        }[tier]
        
        # 连续违规升级逻辑（专家建议）
        if consecutive_count >= 3:
            return 'critical'
        elif consecutive_count >= 2 and base_level != 'critical':
            return 'warning'
        elif severity_score >= 90:  # 极高严重性
            return 'critical'
        else:
            return base_level
    
    def _get_recent_breaches(self, hours: int = 24) -> List[ThresholdBreach]:
        """获取最近N小时的违规记录"""
        from datetime import timedelta
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [b for b in self.breach_history if b.timestamp >= cutoff_time]
    
    def _generate_threshold_actions(self, metric_name: str, tier: ThresholdTier, 
                                     utilization: float) -> List[str]:
        """标准化的推荐行动模板（专家提供）"""
        actions = []
        excess_pct = (utilization - 1.0) * 100
        
        if tier == ThresholdTier.GREEN:
            actions.extend([
                f"{metric_name}使用率{utilization:.1%}，接近阈值边界",
                "建议：1) 监控相关指标变化 2) 准备应急方案 3) 通知风控团队关注"
            ])
            
        elif tier == ThresholdTier.YELLOW:
            actions.extend([
                f"{metric_name}已触及限额（超额{excess_pct:.1f}%）",
                "强制行动：1) 24小时内提交调整计划 2) 每日报告使用率 3) 限制新增风险敞口",
                f"目标：将使用率降至{max(1.0, utilization-0.1):.1%}以下"
            ])
            
        elif tier == ThresholdTier.ORANGE:
            actions.extend([
                f"{metric_name}严重超限（超额{excess_pct:.1f}%）",
                "立即行动：1) 12小时内开始减仓 2) 冻结相关交易权限 3) 上报风险管理委员会",
                f"目标：48小时内将使用率降至1.0以下"
            ])
            
        else:  # RED
            actions.extend([
                f"{metric_name}极度危险（超额{excess_pct:.1f}%）",
                "紧急措施：1) 立即停止所有相关交易 2) 启动强制平仓程序 3) 召开紧急风控会议",
                "目标：4小时内将使用率降至1.15以下，24小时内降至1.0以下"
            ])
        
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
    """违规优先级处理器（P1-3-C）- 基于专家指导修正"""
    
    def prioritize_breaches(self, breaches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        多重违规智能优先级排序（专家修正：动态权重分配）
        
        评分维度（专家确认）：
        1. 基础严重性（权重：30%）
        2. 违规幅度（权重：25%）
        3. 时间紧急性（权重：20%）
        4. 级联影响（权重：15%）
        5. 监管影响（权重：10%）
        """
        if not breaches:
            return []
        
        try:
            # 为每个违规计算综合优先级评分
            prioritized = []
            for breach in breaches:
                priority_score = self._calculate_breach_priority(breach)
                cascading_impact = self._analyze_cascading_impact(breach, breaches)
                
                # 级联影响额外加分（15%权重）
                priority_score += cascading_impact['impact_score'] * 0.15
                priority_score = min(priority_score, 100)
                
                enhanced_breach = breach.copy()
                enhanced_breach['priority_score'] = priority_score
                enhanced_breach['cascading_impact'] = cascading_impact
                enhanced_breach['priority_level'] = self._determine_priority_level(
                    priority_score, 
                    cascading_impact['systemic_risk_level']
                )
                
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
        """计算违规优先级评分（专家修正：动态权重分配）"""
        # 基础权重配置（专家确认）
        base_weights = {
            'severity': 0.30,           # 严重性
            'breach_amount': 0.25,      # 违规幅度
            'time_horizon': 0.20,       # 时间紧急性
            'cascading_impact': 0.15,   # 级联影响
            'regulatory_impact': 0.10   # 监管影响
        }
        
        # 根据违规类型调整权重（专家建议）
        limit_type = breach.get('limit_type', '')
        weight_adjustments = {
            'leverage': {'severity': 0.35, 'regulatory_impact': 0.15, 'time_horizon': 0.15},
            'value_at_risk': {'severity': 0.35, 'time_horizon': 0.25, 'breach_amount': 0.20},
            'liquidity': {'time_horizon': 0.30, 'breach_amount': 0.20, 'regulatory_impact': 0.15},
            'concentration': {'breach_amount': 0.30, 'regulatory_impact': 0.15, 'severity': 0.25}
        }
        
        # 应用权重调整
        weights = base_weights.copy()
        for adj_type, adjustments in weight_adjustments.items():
            if adj_type in limit_type:
                for key, value in adjustments.items():
                    if key in weights:
                        weights[key] = value
        
        # 归一化权重
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        score = 0.0
        
        try:
            # 1. 严重性评分（专家修正：7级映射）
            severity_map = {
                'catastrophic': 100,  # 灾难性
                'critical': 85,       # 严重
                'high': 70,           # 高
                'medium_high': 55,    # 中高
                'medium': 40,         # 中等
                'low': 25,            # 低
                'info': 10            # 信息性
            }
            severity = breach.get('severity', 'medium')
            severity_score = severity_map.get(severity, 40)
            score += severity_score * weights['severity']
            
            # 2. 违规幅度评分
            threshold = breach.get('threshold', 1.0)
            current_value = abs(breach.get('current_value', 0))
            if threshold != 0:
                breach_ratio = current_value / threshold
                breach_score = min((breach_ratio - 1.0) / 0.5 * 100, 100)
                score += breach_score * weights['breach_amount']
            
            # 3. 时间紧急性评分（专家修正：指数衰减）
            time_horizon = breach.get('time_horizon', '1d')
            urgency_map = {
                'immediate': 100,
                '1h': 89,    # 指数衰减
                '4h': 63,
                '1d': 6,
                '1w': 0.4,
                '1m': 0.0003
            }
            urgency_score = urgency_map.get(time_horizon, 50)
            score += urgency_score * weights['time_horizon']
            
            # 4. 级联影响评分（专家确认：第2轮咨询）
            # 作为5个维度之一，总权重恒为100%
            cascading_impact = self._analyze_cascading_impact(breach, [])
            cascading_score = min(cascading_impact['impact_score'], 100)
            score += cascading_score * weights['cascading_impact']
            
            # 5. 监管影响评分
            regulatory_impact = 50
            if 'leverage' in limit_type:
                regulatory_impact = 90
            elif 'liquidity' in limit_type:
                regulatory_impact = 80
            elif 'concentration' in limit_type:
                regulatory_impact = 70
            score += regulatory_impact * weights['regulatory_impact']
            
            # 确保总分不超过100（专家确认）
            return min(score, 100)
            
        except Exception as e:
            logger.error(f"优先级评分计算失败: {e}")
            return 50.0
    
    def _analyze_cascading_impact(self, breach: Dict[str, Any], all_breaches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """基于风险传导网络的级联影响分析（专家补充）"""
        impact = {
            'affected_limits': [],
            'impact_score': 0.0,
            'chain_reaction': [],
            'systemic_risk_level': 'low'
        }
        
        # 风险传导网络定义（专家提供）
        cascading_network = {
            'leverage_ratio': {
                'direct': ['margin_call', 'liquidity_risk'],
                'indirect': ['forced_liquidation', 'counterparty_risk'],
                'weight': 0.8  # 传导强度
            },
            'value_at_risk': {
                'direct': ['regulatory_capital', 'liquidity_risk'],
                'indirect': ['funding_cost', 'reputation_risk'],
                'weight': 0.7
            },
            'liquidity_risk': {
                'direct': ['funding_liquidity', 'market_liquidity'],
                'indirect': ['fire_sale', 'systemic_risk'],
                'weight': 0.9  # 流动性风险传导性强
            },
            'concentration': {
                'direct': ['specific_risk', 'liquidity'],
                'indirect': ['market_impact', 'execution_cost'],
                'weight': 0.6
            }
        }
        
        limit_type = breach.get('limit_type', '')
        if limit_type in cascading_network:
            network = cascading_network[limit_type]
            
            # 分析直接传导
            for direct_impact in network['direct']:
                if any(direct_impact in b.get('limit_type', '') for b in all_breaches):
                    impact['affected_limits'].append(direct_impact)
                    impact['chain_reaction'].append(f"{limit_type} → {direct_impact}")
                    impact['impact_score'] += 25 * network['weight']
            
            # 分析间接传导
            for indirect_impact in network['indirect']:
                impact['chain_reaction'].append(f"{limit_type} → ... → {indirect_impact}")
                impact['impact_score'] += 15 * network['weight']
            
            # 系统性风险评估
            if impact['impact_score'] >= 50:
                impact['systemic_risk_level'] = 'high'
            elif impact['impact_score'] >= 30:
                impact['systemic_risk_level'] = 'medium'
        
        return impact
    
    def _determine_priority_level(self, priority_score: float, systemic_risk: str = 'low') -> str:
        """考虑系统性风险的优先级判定（专家修正）"""
        # 基础分界点（专家调整）
        base_thresholds = {
            'P0-紧急': 85,
            'P1-高优先级': 65,
            'P2-中优先级': 45,
            'P3-低优先级': 25
        }
        
        # 系统性风险调整（专家确认：第2轮咨询）
        # 高系统性风险应升级（更高优先级），使用正数调整
        systemic_adjustments = {
            'high': 15,     # 高风险：升级处理（增加15分，P2→P1）
            'medium': 5,    # 中风险：轻微升级（增加5分）
            'low': 0        # 低风险：不变
        }
        
        adjusted_score = priority_score + systemic_adjustments.get(systemic_risk, 0)
        
        if adjusted_score >= base_thresholds['P0-紧急']:
            return 'P0-紧急'
        elif adjusted_score >= base_thresholds['P1-高优先级']:
            return 'P1-高优先级'
        elif adjusted_score >= base_thresholds['P2-中优先级']:
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
        """A股市场特定规则（基于专家确认的监管框架）"""
        breaches = []
        
        # 1. ST股票限额检查（专家确认：机构内部风控要求）
        st_limit = self.market_limits.get('st_stock_max_weight', 0.05)
        st_exposure = 0
        
        # 2. 创业板/科创板单股限额（专家新增）
        cyb_limit = self.market_limits.get('创业板_stock_max_weight', 0.08)
        kcb_limit = self.market_limits.get('科创板_stock_max_weight', 0.08)
        
        # 3. 日换手率限额（专家调整：0.20→0.15）
        turnover_limit = self.market_limits.get('daily_turnover_limit', 0.15)
        
        # 4. 跌停风险敞口（专家调整：0.15→0.10）
        limit_down_limit = self.market_limits.get('limit_down_exposure_max', 0.10)
        
        if hasattr(portfolio_state, 'allocations'):
            for symbol, allocation in portfolio_state.allocations.items():
                # ST股票检查
                if symbol.startswith('ST') or symbol.startswith('*ST'):
                    st_exposure += allocation.weight
                    if allocation.weight > st_limit:
                        breaches.append({
                            'limit_type': 'CN_st_stock_limit',
                            'symbol': symbol,
                            'current_value': allocation.weight,
                            'threshold': st_limit,
                            'severity': 'high',
                            'suggested_action': 'ST股票单只不超5%（机构风控要求）',
                            'regulatory_framework': 'Internal Risk Control'
                        })
                
                # 创业板检查（专家新增）
                if symbol.startswith('30'):  # 创业板代码
                    if allocation.weight > cyb_limit:
                        breaches.append({
                            'limit_type': 'CN_cyb_stock_limit',
                            'symbol': symbol,
                            'current_value': allocation.weight,
                            'threshold': cyb_limit,
                            'severity': 'medium',
                            'suggested_action': '创业板单股不超8%（专家建议）',
                            'regulatory_framework': 'Best Practice'
                        })
                
                # 科创板检查（专家新增）
                if symbol.startswith('688'):  # 科创板代码
                    if allocation.weight > kcb_limit:
                        breaches.append({
                            'limit_type': 'CN_kcb_stock_limit',
                            'symbol': symbol,
                            'current_value': allocation.weight,
                            'threshold': kcb_limit,
                            'severity': 'medium',
                            'suggested_action': '科创板单股不超8%（专家建议）',
                            'regulatory_framework': 'Best Practice'
                        })
        
        # 日换手率检查
        if hasattr(portfolio_state, 'daily_turnover'):
            if portfolio_state.daily_turnover > turnover_limit:
                breaches.append({
                    'limit_type': 'CN_daily_turnover',
                    'current_value': portfolio_state.daily_turnover,
                    'threshold': turnover_limit,
                    'severity': 'medium',
                    'suggested_action': f'日换手率控制在{turnover_limit:.0%}以内（专家调整）'
                })
        
        return breaches
    
    def _check_us_specific_rules(self, portfolio_state) -> List[Dict[str, Any]]:
        """美股市场特定规则（专家确认：SEC/FINRA规则）"""
        breaches = []
        
        # 1. PDT规则检查（专家补充完整）
        min_equity = self.market_limits.get('day_trading_min_equity', 25000)
        pdt_limit = self.market_limits.get('pattern_day_trader_limit', 4.0)
        
        # 2. OTC股票限额（专家调整：0.08→0.05）
        otc_limit = self.market_limits.get('otc_stock_max_weight', 0.05)
        
        # 3. 仙股限额（专家调整：0.05→0.03）
        penny_limit = self.market_limits.get('penny_stock_max_weight', 0.03)
        
        if hasattr(portfolio_state, 'allocations'):
            for symbol, allocation in portfolio_state.allocations.items():
                # OTC股票检查
                if hasattr(allocation, 'market') and allocation.market == 'OTC':
                    if allocation.weight > otc_limit:
                        breaches.append({
                            'limit_type': 'US_otc_stock_limit',
                            'symbol': symbol,
                            'current_value': allocation.weight,
                            'threshold': otc_limit,
                            'severity': 'high',
                            'suggested_action': 'OTC股票单只不超5%（专家调整）',
                            'regulatory_framework': 'SEC Rule 15c2-11'
                        })
                
                # 仙股检查（股价<$5）
                if hasattr(allocation, 'price') and allocation.price < 5.0:
                    if allocation.weight > penny_limit:
                        breaches.append({
                            'limit_type': 'US_penny_stock_limit',
                            'symbol': symbol,
                            'current_value': allocation.weight,
                            'threshold': penny_limit,
                            'severity': 'high',
                            'suggested_action': '仙股单只不超3%（专家调整）',
                            'regulatory_framework': 'SEC Rule 3a51-1'
                        })
        
        # PDT规则检查
        if hasattr(portfolio_state, 'equity') and hasattr(portfolio_state, 'day_trades'):
            if portfolio_state.day_trades >= 4 and portfolio_state.equity < min_equity:
                breaches.append({
                    'limit_type': 'US_pdt_violation',
                    'current_value': portfolio_state.equity,
                    'threshold': min_equity,
                    'severity': 'critical',
                    'suggested_action': f'Pattern Day Trader规则要求最低${min_equity:,}净值（FINRA规则4210）',
                    'regulatory_framework': 'FINRA Rule 4210'
                })
        
        return breaches
    
    def _check_hk_specific_rules(self, portfolio_state) -> List[Dict[str, Any]]:
        """港股市场特定规则（专家确认并调整）"""
        breaches = []
        
        # 专家调整的限额
        mainland_limit = self.market_limits.get('mainland_stock_max_weight', 0.10)
        small_cap_limit = self.market_limits.get('small_cap_max_weight', 0.05)  # 调整：0.08→0.05
        
        if hasattr(portfolio_state, 'allocations'):
            for symbol, allocation in portfolio_state.allocations.items():
                # 沪深港通股票检查
                if hasattr(allocation, 'connect_type') and allocation.connect_type in ['SH-HK', 'SZ-HK']:
                    if allocation.weight > mainland_limit:
                        breaches.append({
                            'limit_type': 'HK_mainland_stock_limit',
                            'symbol': symbol,
                            'current_value': allocation.weight,
                            'threshold': mainland_limit,
                            'severity': 'medium',
                            'suggested_action': '沪深港通股票单只不超10%（专家确认）',
                            'regulatory_framework': 'SFC'
                        })
                
                # 小盘股检查（市值<50亿港币）
                if hasattr(allocation, 'market_cap') and allocation.market_cap < 5_000_000_000:
                    if allocation.weight > small_cap_limit:
                        breaches.append({
                            'limit_type': 'HK_small_cap_limit',
                            'symbol': symbol,
                            'current_value': allocation.weight,
                            'threshold': small_cap_limit,
                            'severity': 'medium',
                            'suggested_action': '小盘股单只不超5%（专家调整）',
                            'regulatory_framework': 'Best Practice'
                        })
        
        return breaches


# =============================================================================
# P1-3-E: 可插拔设计与配置化接口（专家建议）
# =============================================================================

@dataclass
class EnhancedLimitsConfig:
    """增强功能配置（特性开关）"""
    # 特性开关（专家建议）
    enable_smart_threshold: bool = True      # 智能阈值分层
    enable_portfolio_advisor: bool = True    # 投资组合建议（仅建议）
    enable_breach_prioritizer: bool = True   # 违规优先级
    enable_market_specific: bool = True      # 市场差异化限额
    
    # 智能阈值配置（专家审核后的参数）
    threshold_tiers: Optional[Dict[str, float]] = None
    severity_exponential: bool = True        # 使用指数评分
    
    # 违规优先级配置（专家确认：第2轮咨询）
    priority_weights: Optional[Dict[str, float]] = None
    enable_cascading_analysis: bool = True   # 级联影响分析
    
    # 动态权重配置档案（专家建议）
    weight_profile: str = 'normal'           # normal/high_volatility/regulatory_scrutiny
    
    # 动态调整参数（专家确认）
    max_weight_adjustment: float = 0.05      # 单次调整上限：±5%（不是±10%）
    weight_bounds: tuple = (0.10, 0.40)      # 权重范围：[10%, 40%]
    
    # 市场配置
    default_market: str = 'CN'
    
    # 性能配置（专家建议）
    enable_caching: bool = True
    cache_ttl_seconds: int = 300             # 缓存5分钟
    
    def __post_init__(self):
        # 默认阈值层级（专家审核后）
        if self.threshold_tiers is None:
            self.threshold_tiers = {
                'GREEN': 0.85,
                'YELLOW': 1.0,
                'ORANGE': 1.15,
                'RED': 1.3
            }
        
        # 动态权重配置档案（专家建议：第2轮咨询）
        DYNAMIC_WEIGHT_PROFILES = {
            'normal': {
                'severity': 0.30, 'breach_amount': 0.25, 'time_horizon': 0.20,
                'cascading_impact': 0.15, 'regulatory_impact': 0.10
            },
            'high_volatility': {
                'severity': 0.25, 'breach_amount': 0.20, 'time_horizon': 0.30,  # 时间紧迫性提升
                'cascading_impact': 0.15, 'regulatory_impact': 0.10
            },
            'regulatory_scrutiny': {
                'severity': 0.25, 'breach_amount': 0.20, 'time_horizon': 0.15,
                'cascading_impact': 0.15, 'regulatory_impact': 0.25  # 监管影响提升
            }
        }
        
        # 默认优先级权重（专家确认）
        if self.priority_weights is None:
            # 根据当前配置档案选择权重
            self.priority_weights = DYNAMIC_WEIGHT_PROFILES.get(
                self.weight_profile, 
                DYNAMIC_WEIGHT_PROFILES['normal']
            )


class EnhancedRiskLimitsManager:
    """
    增强型风险限额管理器（P1-3集成）
    
    职责边界（专家强调）：
    - ✅ 提供智能化风险评估和建议
    - ❌ 不执行实际的投资组合优化（由portfolio模块负责）
    - ❌ 不做市场状态判定（由strategy模块负责）
    """
    
    def __init__(self, base_manager, config: Optional[EnhancedLimitsConfig] = None):
        """
        Args:
            base_manager: 基础RiskLimitsManager实例（来自risk_limits.py）
            config: 增强功能配置（可选）
        """
        self.base_manager = base_manager
        self.config = config or EnhancedLimitsConfig()
        
        # 根据配置初始化组件
        self.smart_threshold = None
        self.portfolio_advisor = None
        self.breach_prioritizer = None
        self.market_checker = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """动态加载组件（专家建议的可插拔设计）"""
        try:
            if self.config.enable_smart_threshold:
                self.smart_threshold = SmartThresholdChecker()
                logger.info("✅ 智能阈值分层系统已启用")
            
            if self.config.enable_portfolio_advisor:
                # TODO: PortfolioBasedAdvisor在portfolio模块中，暂时禁用
                # self.portfolio_advisor = PortfolioBasedAdvisor()
                self.portfolio_advisor = None  # 需要跨模块集成
                logger.info("⚠️ 投资组合建议系统需要portfolio模块支持，暂时禁用")
            
            if self.config.enable_breach_prioritizer:
                self.breach_prioritizer = BreachPrioritizer()
                logger.info("✅ 违规优先级处理器已启用")
            
            if self.config.enable_market_specific:
                self.market_checker = MarketSpecificLimitsChecker(
                    market_type=self.config.default_market
                )
                logger.info(f"✅ 市场特定限额检查已启用（{self.config.default_market}）")
        
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            # 优雅降级：如果增强功能失败，仍可使用基础功能
            logger.warning("⚠️ 部分增强功能不可用，降级到基础模式")
    
    def check_all_limits(self, portfolio_state, risk_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        全面的限额检查（整合基础+增强功能）
        
        Returns:
            {
                'base_breaches': [...],       # 基础限额违规
                'enhanced_breaches': [...],    # 增强检查发现的违规
                'prioritized_breaches': [...], # 按优先级排序的所有违规
                'portfolio_recommendations': [...], # 投资组合建议（仅建议）
                'market_specific_issues': [...] # 市场特定问题
            }
        """
        result = {
            'base_breaches': [],
            'enhanced_breaches': [],
            'prioritized_breaches': [],
            'portfolio_recommendations': [],
            'market_specific_issues': []
        }
        
        try:
            # 1. 基础限额检查（使用原有RiskLimitsManager）
            result['base_breaches'] = self.base_manager.check_all_limits(
                portfolio_state, risk_metrics
            )
            
            # 2. 智能阈值检查（增强）
            if self.smart_threshold:
                enhanced = self._run_smart_threshold_checks(portfolio_state, risk_metrics)
                result['enhanced_breaches'].extend(enhanced)
            
            # 3. 市场特定限额检查
            if self.market_checker:
                market_issues = self.market_checker.check_market_limits(portfolio_state)
                result['market_specific_issues'] = market_issues
            
            # 4. 整合所有违规并排序
            all_breaches = (
                result['base_breaches'] + 
                result['enhanced_breaches'] + 
                result['market_specific_issues']
            )
            
            if self.breach_prioritizer and all_breaches:
                result['prioritized_breaches'] = self.breach_prioritizer.prioritize_breaches(
                    all_breaches
                )
            else:
                result['prioritized_breaches'] = all_breaches
            
            # 5. 生成投资组合建议（仅建议，不执行）
            if self.portfolio_advisor and result['prioritized_breaches']:
                result['portfolio_recommendations'] = self.portfolio_advisor.generate_recommendations(
                    portfolio_state, risk_metrics, result['prioritized_breaches']
                )
                # 添加免责声明（专家强调）
                result['portfolio_recommendations'].insert(0, {
                    'type': 'disclaimer',
                    'message': '⚠️ 以下为智能建议，不构成自动交易指令，需由投资组合管理模块审核执行'
                })
            
            return result
            
        except Exception as e:
            logger.error(f"限额检查失败: {e}")
            # 优雅降级
            return {
                'base_breaches': result.get('base_breaches', []),
                'error': str(e)
            }
    
    def _run_smart_threshold_checks(self, portfolio_state, risk_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """执行智能阈值检查"""
        breaches = []
        
        if not self.smart_threshold:
            return breaches
        
        # 检查关键风险指标
        key_metrics = [
            ('value_at_risk', risk_metrics.get('var_95', 0), self.base_manager.limits.get('var_limit', 0.05)),
            ('max_drawdown', risk_metrics.get('max_drawdown', 0), self.base_manager.limits.get('max_drawdown', 0.20)),
            ('volatility', risk_metrics.get('volatility', 0), self.base_manager.limits.get('volatility_limit', 0.30))
        ]
        
        for metric_name, current_value, base_threshold in key_metrics:
            if base_threshold > 0:
                breach = self.smart_threshold.check_smart_threshold(
                    metric_name, current_value, base_threshold
                )
                if breach:
                    breaches.append(breach.__dict__)
        
        return breaches
    
    def update_config(self, **kwargs):
        """动态更新配置（专家建议的热更新能力）"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"配置已更新: {key} = {value}")
        
        # 重新初始化组件
        self._initialize_components()
    
    def get_feature_status(self) -> Dict[str, bool]:
        """获取特性开关状态（用于监控和调试）"""
        return {
            'smart_threshold': self.config.enable_smart_threshold and self.smart_threshold is not None,
            'portfolio_advisor': self.config.enable_portfolio_advisor and self.portfolio_advisor is not None,
            'breach_prioritizer': self.config.enable_breach_prioritizer and self.breach_prioritizer is not None,
            'market_specific': self.config.enable_market_specific and self.market_checker is not None
        }
