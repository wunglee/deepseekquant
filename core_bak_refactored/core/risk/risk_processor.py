"""
风险处理器核心 - 业务层
从 core_bak/risk_manager.py 拆分
职责: 风险处理流程协调、状态管理
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .risk_models import RiskAssessment, RiskLevel
from .risk_calculator import RiskCalculator
from .risk_limits import RiskLimitsManager
from .stress_testing import StressTester
from .portfolio_risk import PortfolioRiskAnalyzer
from .position_risk import PositionRiskAnalyzer

logger = logging.getLogger('DeepSeekQuant.RiskProcessor')


class RiskProcessor:
    """风险处理器 - 协调各个风险分析组件"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 初始化各个组件
        self.calculator = RiskCalculator(config)
        self.limits_manager = RiskLimitsManager(config)
        self.stress_tester = StressTester(config)
        self.portfolio_analyzer = PortfolioRiskAnalyzer(config)
        self.position_analyzer = PositionRiskAnalyzer(config)
        
        logger.info("风险处理器初始化完成")
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理风险评估请求
        
        Args:
            data: 输入数据
            
        Returns:
            风险评估结果
        """
        try:
            # 1. 计算核心风险指标
            risk_metrics = self.calculator.calculate_all_metrics(data)
            
            # 提取组合状态与市场数据
            portfolio_state = data.get('portfolio_state')
            market_data = data.get('market_data') or {}
            
            # 2. 检查限额
            limit_breaches = self.limits_manager.check_limits(portfolio_state, risk_metrics)
            
            # 3. 运行压力测试
            stress_results = self.stress_tester.run_stress_tests(portfolio_state, market_data)
            scenario_results = self.stress_tester.run_scenario_analysis(portfolio_state, market_data)
            
            # 4. 分析组合风险
            portfolio_risk = self.portfolio_analyzer.analyze(data, risk_metrics)
            
            # 5. 生成风险评估报告
            assessment = self._create_risk_assessment(
                risk_metrics,
                limit_breaches,
                stress_results,
                portfolio_risk,
                scenario_results
            )
            
            return {
                'success': True,
                'assessment': assessment,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"风险处理失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_risk_assessment(self,
                                risk_metrics: Dict,
                                limit_breaches: List,
                                stress_results: Dict,
                                portfolio_risk: Dict,
                                scenario_results: Dict) -> RiskAssessment:
        """创建风险评估结果"""
        # 计算综合风险评分
        risk_score = self._calculate_overall_risk_score(
            risk_metrics,
            len(limit_breaches),
            stress_results
        )
        
        # 确定风险等级
        risk_level = self._determine_risk_level(risk_score)
        
        return RiskAssessment(
            timestamp=datetime.now().isoformat(),
            portfolio_id=self.config.get('portfolio_id', 'default'),
            overall_risk_level=risk_level,
            risk_score=risk_score,
            value_at_risk=risk_metrics.get('var_95', 0.0),
            expected_shortfall=risk_metrics.get('cvar_95', 0.0),
            max_drawdown=risk_metrics.get('max_drawdown', 0.0),
            liquidity_risk=risk_metrics.get('liquidity_risk', 0.0),
            concentration_risk=risk_metrics.get('concentration_risk', 0.0),
            leverage_risk=risk_metrics.get('leverage_risk', 0.0),
            stress_test_results=stress_results,
            scenario_analysis=scenario_results,
            risk_contributions=portfolio_risk.get('risk_contributions', {}),
            limit_breaches=limit_breaches,
            recommendations=self._generate_recommendations(limit_breaches, risk_score)
        )
    
    def _calculate_overall_risk_score(self,
                                      risk_metrics: Dict,
                                      breach_count: int,
                                      stress_results: Dict) -> float:
        """计算综合风险评分 (0-100)"""
        # 基础分数
        base_score = 50.0
        
        # VaR贡献
        var = abs(risk_metrics.get('var_95', 0.0))
        var_score = min(var * 100, 30)
        
        # 限额违规贡献
        breach_score = min(breach_count * 10, 20)
        
        # 压力测试贡献
        stress_score = min(len([v for v in stress_results.values() if v < -0.1]) * 10, 20)
        
        total = base_score + var_score + breach_score + stress_score
        return min(total, 100.0)
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """根据评分确定风险等级"""
        if risk_score < 20:
            return RiskLevel.VERY_LOW
        elif risk_score < 40:
            return RiskLevel.LOW
        elif risk_score < 60:
            return RiskLevel.MODERATE
        elif risk_score < 80:
            return RiskLevel.HIGH
        elif risk_score < 95:
            return RiskLevel.VERY_HIGH
        else:
            return RiskLevel.EXTREME
    
    def _generate_recommendations(self,
                                  limit_breaches: List,
                                  risk_score: float) -> List[Dict[str, Any]]:
        """生成风险建议"""
        recommendations = []
        
        if limit_breaches:
            recommendations.append({
                'priority': 'high',
                'action': 'review_limits',
                'description': f'发现{len(limit_breaches)}处限额违规，需要立即处理'
            })
        
        if risk_score > 80:
            recommendations.append({
                'priority': 'high',
                'action': 'reduce_exposure',
                'description': '风险评分过高，建议减少市场敞口'
            })
        
        return recommendations
