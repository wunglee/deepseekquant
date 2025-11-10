"""
高级风险度量模块
从 core_bak/risk_manager.py 提取的真实业务逻辑实现
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import math


@dataclass
class VaRResult:
    """VaR计算结果"""
    value_at_risk: float
    confidence_level: float
    method: str
    timestamp: str


@dataclass
class StressTestResult:
    """压力测试结果"""
    scenario_name: str
    portfolio_impact: float
    breached_limits: List[str]
    recommendations: List[str]


class AdvancedRiskMetrics:
    """高级风险度量"""

    @staticmethod
    def calculate_var(returns: List[float], confidence_level: float = 0.95, method: str = "historical") -> Optional[VaRResult]:
        """
        计算在险价值（Value at Risk）
        从 core_bak/risk_manager.py 提取 (line 698-787)
        支持方法：historical, parametric, monte_carlo
        """
        if not returns or len(returns) < 2:
            return None
        
        from datetime import datetime
        
        if method == "historical":
            # 历史模拟法 (line 724-736)
            if len(returns) < 20:
                return VaRResult(
                    value_at_risk=-0.1,
                    confidence_level=confidence_level,
                    method="historical",
                    timestamp=datetime.now().isoformat()
                )
            
            # 计算指定置信水平的分位数
            sorted_returns = sorted(returns)
            percentile_index = int((1 - confidence_level) * len(sorted_returns))
            var = sorted_returns[percentile_index] if percentile_index < len(sorted_returns) else sorted_returns[0]
            
            return VaRResult(
                value_at_risk=float(var),
                confidence_level=confidence_level,
                method="historical",
                timestamp=datetime.now().isoformat()
            )
        
        elif method == "parametric":
            # 参数法（正态分布假设）(line 738-753)
            if len(returns) < 20:
                return VaRResult(
                    value_at_risk=-0.1,
                    confidence_level=confidence_level,
                    method="parametric",
                    timestamp=datetime.now().isoformat()
                )
            
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            std_return = math.sqrt(variance)
            
            # 使用标准正态分布的分位数
            # 简化实现：95% -> z=1.645, 99% -> z=2.326
            if abs(confidence_level - 0.95) < 0.01:
                z_score = -1.645  # 95%置信水平
            elif abs(confidence_level - 0.99) < 0.01:
                z_score = -2.326  # 99%置信水平
            else:
                # 粗略估计
                z_score = -1.645
            
            var = mean_return + z_score * std_return
            
            return VaRResult(
                value_at_risk=float(var),
                confidence_level=confidence_level,
                method="parametric",
                timestamp=datetime.now().isoformat()
            )
        
        elif method == "monte_carlo":
            # 蒙特卡洛模拟法 (line 755-787)
            # TODO：补充了蒙特卡洛模拟实现，待确认（需要多变量随机数生成）
            # 回退到历史法
            return AdvancedRiskMetrics.calculate_var(returns, confidence_level, "historical")
        
        else:
            # 默认使用历史法
            return AdvancedRiskMetrics.calculate_var(returns, confidence_level, "historical")

    @staticmethod
    def calculate_expected_shortfall(returns: List[float], confidence_level: float = 0.95) -> Optional[float]:
        """
        计算预期短缺（Expected Shortfall / CVaR）
        从 core_bak/risk_manager.py:_calculate_expected_shortfall 提取 (line 789-813)
        """
        if not returns or len(returns) < 20:
            return -0.15  # 默认值
        
        # 先计算VaR作为阈值
        var_result = AdvancedRiskMetrics.calculate_var(returns, confidence_level, "historical")
        if not var_result:
            return -0.15
        
        var = var_result.value_at_risk
        
        # 计算超过VaR的平均损失
        tail_returns = [r for r in returns if r <= var]
        
        if len(tail_returns) > 0:
            es = sum(tail_returns) / len(tail_returns)
        else:
            es = var * 1.2  # 保守估计
        
        return float(es)

    @staticmethod
    def calculate_max_drawdown(returns: List[float]) -> Optional[float]:
        """
        计算最大回撤
        从 core_bak/risk_manager.py:_calculate_max_drawdown 提取 (line 815-835)
        """
        if not returns or len(returns) < 20:
            return -0.2  # 默认值
        
        # 计算累积收益
        cumulative_returns = []
        cumulative = 1.0
        for r in returns:
            cumulative *= (1 + r)
            cumulative_returns.append(cumulative - 1)
        
        # 计算回撤
        max_drawdown = 0.0
        peak = cumulative_returns[0]
        
        for cum_ret in cumulative_returns:
            if cum_ret > peak:
                peak = cum_ret
            
            drawdown = (cum_ret - peak) / (1 + peak) if (1 + peak) > 0 else 0
            if drawdown < max_drawdown:
                max_drawdown = drawdown
        
        return float(max_drawdown)

    @staticmethod
    def stress_test(portfolio_weights: Dict[str, float],
                    scenario: Dict[str, float],
                    scenario_name: str = "market_crash") -> StressTestResult:
        """
        压力测试：模拟极端市场情景下的组合表现
        从 core_bak/risk_manager.py:_run_single_stress_test 提取 (line 952-972)
        TODO：补充了压力测试实现，待确认（core_bak中逻辑较复杂，需要更多上下文）
        """
        # scenario: {symbol: shock_return}
        impact = 0.0
        for symbol, weight in portfolio_weights.items():
            shock = scenario.get(symbol, 0.0)
            impact += weight * shock
        
        breached = []
        if impact < -0.15:
            breached.append("STRESS_MAX_LOSS_EXCEEDED")
        if impact < -0.3:
            breached.append("STRESS_CRITICAL_LOSS")
        
        recommendations = []
        if impact < -0.1:
            recommendations.append("建议减少高风险资产配置")
        if impact < -0.2:
            recommendations.append("建议增加对冲策略")
        if impact < -0.3:
            recommendations.append("紧急减仓建议")
        
        return StressTestResult(
            scenario_name=scenario_name,
            portfolio_impact=round(impact, 6),
            breached_limits=breached,
            recommendations=recommendations
        )

    @staticmethod
    def tail_risk_analysis(returns: List[float], tail_percentile: float = 0.05) -> Dict[str, float]:
        """
        尾部风险分析
        TODO：补充了尾部风险分析实现，待确认（core_bak中未找到明确独立方法）
        """
        if not returns:
            return {}
        
        sorted_returns = sorted(returns)
        tail_index = int(tail_percentile * len(sorted_returns))
        tail_returns = sorted_returns[:max(tail_index, 1)]
        
        tail_mean = sum(tail_returns) / len(tail_returns) if tail_returns else 0.0
        tail_max = min(tail_returns) if tail_returns else 0.0
        
        # 计算尾部波动率
        if len(tail_returns) > 1:
            tail_mean_calc = sum(tail_returns) / len(tail_returns)
            tail_var = sum((r - tail_mean_calc) ** 2 for r in tail_returns) / len(tail_returns)
            tail_volatility = math.sqrt(tail_var)
        else:
            tail_volatility = 0.0
        
        return {
            'tail_mean': round(tail_mean, 6),
            'tail_worst': round(tail_max, 6),
            'tail_count': len(tail_returns),
            'tail_volatility': round(tail_volatility, 6)
        }

    @staticmethod
    def concentration_risk(weights: Dict[str, float], hhi_threshold: float = 0.25) -> Dict[str, Any]:
        """
        集中度风险评估（HHI - 赫芬达尔-赫希曼指数）
        从 core_bak/risk_manager.py:_calculate_concentration_risk 提取 (line 864-880)
        """
        if not weights:
            return {
                'hhi': 0.0,
                'max_weight': 0.0,
                'exceeded': False,
                'diversification_ratio': 0.0,
                'effective_n': 0.0
            }
        
        # 计算赫芬达尔-赫希曼指数 (HHI)
        hhi = sum(w ** 2 for w in weights.values())
        
        # 标准化到0-1范围
        concentration_risk_value = min(hhi, 1.0)
        
        max_weight = max(weights.values()) if weights else 0.0
        
        # 有效资产数量
        effective_n = 1.0 / hhi if hhi > 0 else 0.0
        
        return {
            'hhi': round(float(concentration_risk_value), 6),
            'max_weight': round(float(max_weight), 6),
            'exceeded': concentration_risk_value > hhi_threshold,
            'diversification_ratio': round(effective_n, 6),
            'effective_n': round(effective_n, 2)
        }
