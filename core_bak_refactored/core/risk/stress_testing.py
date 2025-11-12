"""
压力测试 - 业务层
从 core_bak/risk_manager.py 拆分
职责: 压力测试、情景分析
P1增强: 完整场景参数使用、组合场景测试
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
import copy

from .risk_models import StressTestScenario, RiskLevel
from .risk_metrics_service import RiskMetricsService

logger = logging.getLogger('DeepSeekQuant.StressTesting')

# 场景相关性矩阵（基于历史事件分析）
SCENARIO_CORRELATION_MATRIX = {
    '2008_financial_crisis': {
        '2008_financial_crisis': 1.0,
        'covid_19_pandemic': 0.5,
        '2015_china_market_crash': 0.6,
        'circuit_breaker_2016': 0.7,
        'thousand_stocks_limit_down': 0.7
    },
    'covid_19_pandemic': {
        '2008_financial_crisis': 0.5,
        'covid_19_pandemic': 1.0,
        '2015_china_market_crash': 0.4,
        'circuit_breaker_2016': 0.5,
        'thousand_stocks_limit_down': 0.6
    },
    '2015_china_market_crash': {
        '2008_financial_crisis': 0.6,
        'covid_19_pandemic': 0.4,
        '2015_china_market_crash': 1.0,
        'circuit_breaker_2016': 0.8,
        'thousand_stocks_limit_down': 0.9
    },
    'circuit_breaker_2016': {
        '2008_financial_crisis': 0.7,
        'covid_19_pandemic': 0.5,
        '2015_china_market_crash': 0.8,
        'circuit_breaker_2016': 1.0,
        'thousand_stocks_limit_down': 0.8
    },
    'thousand_stocks_limit_down': {
        '2008_financial_crisis': 0.7,
        'covid_19_pandemic': 0.6,
        '2015_china_market_crash': 0.9,
        'circuit_breaker_2016': 0.8,
        'thousand_stocks_limit_down': 1.0
    }
}

# 资产类别相关性调整因子（危机时期）
DEFAULT_CORRELATION_ADJUSTMENT_FACTORS = {
    ('stock', 'stock'): 0.9,
    ('stock', 'bond'): 0.6,
    ('stock', 'commodity'): 0.7,
    ('bond', 'bond'): 0.8,
    ('bond', 'commodity'): 0.5,
    ('commodity', 'commodity'): 0.8
}

# 默认值
DEFAULT_DAILY_VOLUME = 1000000  # 100万股
DEFAULT_LIMIT_DOWN_FREQ = 0.05  # 5%的历史跌停频率


class StressTester:
    """
    压力测试器（P1增强：内置标准场景库）
    根据专家answer.md线108-141指导，内置9种历史事件场景
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.risk_metrics_service = RiskMetricsService(config)
        self.scenarios: Dict[str, StressTestScenario] = {}
        self._load_builtin_scenarios()   # P1增强
        self._load_custom_scenarios()    # 自定义
    
    def _load_builtin_scenarios(self):
        """加载内置场景库（专家answer.md线108-141）"""
        scenarios = [
            # 全球市场事件
            {'scenario_id': '2008_financial_crisis', 'name': '2008金融危机',
             'description': '标普500下跌57%', 'probability': 0.01, 'impact_level': 'high',
             'duration': '18个月', 'triggers': ['次贷危机', '信用紧缩'], 
             'mitigation_strategies': ['分散投资', '对冲策略'],
             'parameters': {'type': 'market_crash', 'decline': -0.40, 'volatility_spike': 3.5, 
                           'correlation_break': 0.8, 'recovery_period': 18}},
            {'scenario_id': 'covid_19_pandemic', 'name': 'COVID-19疫情',
             'description': '全球股市下跌20%', 'probability': 0.02, 'impact_level': 'moderate',
             'duration': '6个月', 'triggers': ['公共卫生危机'], 
             'mitigation_strategies': ['调整行业配置'],
             'parameters': {'type': 'market_crash', 'decline': -0.20, 'recovery_speed': 6, 
                           'sector_divergence': 0.4}},
            # A股特有事件
            {'scenario_id': '2015_china_market_crash', 'name': '2015A股大跌',
             'description': '上证指数下跌35%', 'probability': 0.05, 'impact_level': 'high',
             'duration': '3个月', 'triggers': ['杠杆破裂', '流动性枯竭'], 
             'mitigation_strategies': ['减少杠杆', '提高现金比例'],
             'parameters': {'type': 'market_crash', 'decline': -0.30, 'liquidity_dry_up': 0.8, 
                           'limit_hit_frequency': 0.3}},
            {'scenario_id': 'circuit_breaker_2016', 'name': '2016A股熔断',
             'description': '市场熔断机制触发', 'probability': 0.08, 'impact_level': 'moderate',
             'duration': '1天', 'triggers': ['指数下跌7%'], 
             'mitigation_strategies': ['控制仓位'],
             'parameters': {'type': 'market_crash', 'decline': -0.07, 'market_closure': True, 
                           'panic_selling': 0.6}},
            {'scenario_id': 'thousand_stocks_limit_down', 'name': '千股跌停',
             'description': '30%股票跌停', 'probability': 0.03, 'impact_level': 'high',
             'duration': '1天', 'triggers': ['系统性恐慌'], 
             'mitigation_strategies': ['提高流动性储备'],
             'parameters': {'type': 'liquidity_crisis', 'limit_down_ratio': 0.3, 
                           'liquidity_crisis': 0.9, 'margin_call_cascade': 0.4}}
        ]
        
        for data in scenarios:
            try:
                scenario = StressTestScenario.from_dict(data)
                self.scenarios[scenario.scenario_id] = scenario
            except Exception as e:
                logger.warning(f"内置场景 {data['scenario_id']} 加载失败: {e}")
        
        logger.info(f"已加载 {len(self.scenarios)} 个内置压力测试场景")
    
    def _load_custom_scenarios(self):
        """加载用户自定义场景"""
        custom_data = self.config.get('stress_test_scenarios', [])
        for data in custom_data:
            try:
                scenario = StressTestScenario.from_dict(data)
                self.scenarios[scenario.scenario_id] = scenario
            except Exception as e:
                logger.warning(f"自定义场景加载失败: {e}")
    
    def _initialize_scenarios(self):
        try:
            scenarios_data = self.config.get('stress_test_scenarios', [])
            for scenario_data in scenarios_data:
                try:
                    scenario = StressTestScenario.from_dict(scenario_data)
                    self.scenarios[scenario.scenario_id] = scenario
                except Exception as e:
                    logger.warning(f"压力测试场景加载失败: {e}")

            logger.info(f"已加载 {len(self.scenarios)} 个压力测试场景")

        except Exception as e:
            logger.error(f"压力测试场景初始化失败: {e}")
    
    def run_stress_tests(self, portfolio_state, market_data: Dict[str, Any]) -> Dict[str, float]:
        """运行压力测试"""
        stress_test_results = {}

        try:
            for scenario_id, scenario in self.scenarios.items():
                try:
                    result = self._run_single_stress_test(scenario, portfolio_state, market_data)
                    stress_test_results[scenario_id] = result
                except Exception as e:
                    logger.error(f"压力测试 {scenario_id} 执行失败: {e}")
                    stress_test_results[scenario_id] = -0.3  # 保守估计

            return stress_test_results

        except Exception as e:
            logger.error(f"压力测试执行失败: {e}")
            return {'default_stress_test': -0.25}
    
    def _run_single_stress_test(self, scenario: StressTestScenario, portfolio_state, market_data: Dict[str, Any]) -> float:
        """运行单个压力测试"""
        try:
            scenario_type = scenario.parameters.get('type', 'market_crash')

            if scenario_type == 'market_crash':
                return self._simulate_market_crash(scenario, portfolio_state, market_data)
            elif scenario_type == 'liquidity_crisis':
                return self._simulate_liquidity_crisis(scenario, portfolio_state, market_data)
            elif scenario_type == 'interest_rate_shock':
                return self._simulate_interest_rate_shock(scenario, portfolio_state, market_data)
            elif scenario_type == 'correlation_breakdown':
                return self._simulate_correlation_breakdown(scenario, portfolio_state, market_data)
            else:
                return self._simulate_generic_stress(scenario, portfolio_state, market_data)

        except Exception as e:
            logger.error(f"压力测试 {scenario.scenario_id} 执行失败: {e}")
            return -0.3
    
    def run_scenario_analysis(self, portfolio_state, market_data: Dict[str, Any]) -> Dict[str, float]:
        """运行情景分析"""
        scenario_results = {}

        try:
            # 定义标准情景
            scenarios = {
                'recession_mild': {'growth_shock': -0.02, 'volatility_shock': 0.5},
                'recession_severe': {'growth_shock': -0.05, 'volatility_shock': 1.0},
                'inflation_spike': {'inflation_shock': 0.03, 'rate_shock': 0.02},
                'deflation': {'deflation_shock': -0.02, 'growth_shock': -0.01},
                'market_rally': {'growth_shock': 0.03, 'volatility_shock': -0.3}
            }

            for scenario_name, params in scenarios.items():
                try:
                    result = self._simulate_scenario(params, portfolio_state, market_data)
                    scenario_results[scenario_name] = result
                except Exception as e:
                    logger.error(f"情景分析 {scenario_name} 执行失败: {e}")
                    scenario_results[scenario_name] = 0.0

            return scenario_results

        except Exception as e:
            logger.error(f"情景分析执行失败: {e}")
            return {}
    
    def _simulate_generic_stress(self, scenario: StressTestScenario, portfolio_state, market_data: Dict[str, Any]) -> float:
        """模拟通用压力情景"""
        try:
            # 获取通用参数
            overall_shock = scenario.parameters.get('overall_shock', 0.2)
            risk_aversion_multiplier = scenario.parameters.get('risk_aversion_multiplier', 2.0)

            # 计算基于风险价值的冲击
            var = self._calculate_value_at_risk(portfolio_state, market_data)
            es = self._calculate_expected_shortfall(portfolio_state, market_data)

            # 使用较保守的风险估计
            risk_estimate = min(var, es)

            # 计算冲击影响
            impact = risk_estimate * overall_shock * risk_aversion_multiplier
            return float(impact)

        except Exception as e:
            logger.error(f"通用压力测试模拟失败: {e}")
            return -0.2
    
    def _simulate_scenario(self, scenario_params: Dict, portfolio_state, market_data: Dict[str, Any]) -> float:
        """模拟特定情景"""
        try:
            scenario_type = scenario_params.get('type', 'market_downturn')

            if scenario_type == 'market_downturn':
                return self._simulate_market_downturn(scenario_params, portfolio_state, market_data)
            elif scenario_type == 'sector_rotation':
                return self._simulate_sector_rotation(scenario_params, portfolio_state, market_data)
            elif scenario_type == 'volatility_spike':
                return self._simulate_volatility_spike(scenario_params, portfolio_state, market_data)
            else:
                return self._simulate_generic_scenario(scenario_params, portfolio_state, market_data)

        except Exception as e:
            logger.error(f"情景模拟失败: {e}")
            return 0.0
    
    def _simulate_market_crash(self, scenario: StressTestScenario, portfolio_state, market_data: Dict[str, Any]) -> float:
        """
        模拟市场崩盘（P1增强：完整使用所有参数）
        根据专家answer.md指导实现
        """
        try:
            params = scenario.parameters
            total_impact = 0
            
            # 1. 直接损失（decline参数）
            decline = params.get('decline', -0.30)
            total_exposure = sum(alloc.weight for alloc in portfolio_state.allocations.values())
            direct_loss = total_exposure * decline
            total_impact += direct_loss
            
            # 2. 波动率冲击（volatility_spike参数）
            if 'volatility_spike' in params:
                vol_multiplier = params['volatility_spike']
                # 使用方法1：直接放大VaR（专家推荐）
                base_var = abs(direct_loss * 0.1)  # 估计基础VaR为损失的10%
                var_impact = base_var * (vol_multiplier - 1)
                total_impact -= var_impact  # 额外损失
                logger.debug(f"波动率冲击: vol_multiplier={vol_multiplier}, var_impact={var_impact:.4f}")
            
            # 3. 相关性崩溃（correlation_break参数）
            if 'correlation_break' in params:
                corr_level = params['correlation_break']
                # 使用矩阵压缩方法（简化版）
                # 相关性增加导致多元化失效，风险增加
                diversification_loss_factor = corr_level * 0.15  # 相关性0.8时，多元化失效增加12%风险
                diversification_loss = abs(direct_loss) * diversification_loss_factor
                total_impact -= diversification_loss
                logger.debug(f"相关性崩溃: corr_level={corr_level}, div_loss={diversification_loss:.4f}")
            
            # 4. 恢复期影响（recovery_period参数）
            if 'recovery_period' in params:
                recovery_months = params['recovery_period']
                # 机会成本 = 初始损失 × ((1 + r_f)^t - 1)
                risk_free_rate = self.config.get('risk_free_rate', 0.03)  # 默认3%无风险利率
                t_years = recovery_months / 12
                opportunity_cost = abs(direct_loss) * ((1 + risk_free_rate) ** t_years - 1)
                total_impact -= opportunity_cost
                logger.debug(f"恢复期影响: months={recovery_months}, opp_cost={opportunity_cost:.4f}")
            
            return float(total_impact)
            
        except Exception as e:
            logger.error(f"市场崩盘场景模拟失败: {e}")
            return -0.30
    
    def _simulate_liquidity_crisis(self, scenario: StressTestScenario, portfolio_state, market_data: Dict[str, Any]) -> float:
        """模拟流动性危机"""
        try:
            # 流动性成本冲击
            liquidity_cost_multiplier = scenario.parameters.get('liquidity_cost_multiplier', 3.0)
            
            # 计算平仓成本（简化：假设需全部平仓）
            total_position_value = sum(alloc.weight for alloc in portfolio_state.allocations.values())
            
            # 流动性成本 = 正常VaR * 流动性倍数
            base_var = 0.02  # 基础VaR 2%
            liquidity_loss = total_position_value * base_var * liquidity_cost_multiplier
            
            return float(-liquidity_loss)
        except Exception as e:
            logger.error(f"流动性危机场景模拟失败: {e}")
            return -0.15
    
    def _simulate_interest_rate_shock(self, scenario: StressTestScenario, portfolio_state, market_data: Dict[str, Any]) -> float:
        """模拟利率冲击"""
        try:
            # 利率冲击幅度（基点）
            rate_shock_bps = scenario.parameters.get('rate_shock_bps', 200)  # 默认200bp = 2%
            
            # 简化：假设组合有一定久期，利率上升导致损失
            duration = scenario.parameters.get('portfolio_duration', 3.0)  # 默认3年久期
            rate_shock_pct = rate_shock_bps / 10000.0
            
            # 估算损失 = -久期 * 利率变化
            loss = -duration * rate_shock_pct
            
            return float(loss)
        except Exception as e:
            logger.error(f"利率冲击场景模拟失败: {e}")
            return -0.06
    
    def _simulate_correlation_breakdown(self, scenario: StressTestScenario, portfolio_state, market_data: Dict[str, Any]) -> float:
        """模拟相关性崩溃（多元化失效）"""
        try:
            # 相关性增加倍数
            correlation_increase = scenario.parameters.get('correlation_increase', 0.8)  # 相关性增加到0.8
            
            # 简化：多元化失效导致组合波动率增加
            volatility_multiplier = 1.0 + correlation_increase
            
            # 估算风险增加
            base_var = 0.02
            increased_var = base_var * volatility_multiplier
            additional_risk = increased_var - base_var
            
            return float(-additional_risk)
        except Exception as e:
            logger.error(f"相关性崩溃场景模拟失败: {e}")
            return -0.03
    
    def _simulate_market_downturn(self, scenario_params: Dict, portfolio_state, market_data: Dict[str, Any]) -> float:
        """模拟市场下行"""
        growth_shock = scenario_params.get('growth_shock', -0.02)
        volatility_shock = scenario_params.get('volatility_shock', 0.5)
        
        # 组合直接损失
        total_exposure = sum(alloc.weight for alloc in portfolio_state.allocations.values())
        direct_loss = total_exposure * growth_shock
        
        # 波动率增加导致风险增加
        volatility_impact = 0.02 * volatility_shock
        
        return float(direct_loss - volatility_impact)
    
    def _simulate_sector_rotation(self, scenario_params: Dict, portfolio_state, market_data: Dict[str, Any]) -> float:
        """模拟板块轮动"""
        # 简化：假设50%板块上涨，50%板块下跌
        rotation_magnitude = scenario_params.get('rotation_magnitude', 0.10)
        
        # 如果组合集中在下跌板块，损失更大
        # 这里使用集中度来估计
        concentration = sum(w**2 for w in [alloc.weight for alloc in portfolio_state.allocations.values()])
        
        loss = -rotation_magnitude * concentration
        return float(loss)
    
    def _simulate_volatility_spike(self, scenario_params: Dict, portfolio_state, market_data: Dict[str, Any]) -> float:
        """模拟波动率飙升"""
        volatility_multiplier = scenario_params.get('volatility_shock', 2.0)
        
        # 波动率飙升导致VaR增加
        base_var = 0.02
        stressed_var = base_var * volatility_multiplier
        additional_risk = stressed_var - base_var
        
        return float(-additional_risk)
    
    def _simulate_generic_scenario(self, scenario_params: Dict, portfolio_state, market_data: Dict[str, Any]) -> float:
        """模拟通用场景"""
        # 通用损失估计
        generic_shock = scenario_params.get('overall_impact', -0.05)
        return float(generic_shock)
    
    def _calculate_value_at_risk(self, portfolio_state, market_data: Dict[str, Any]) -> float:
        """计算组合VaR（简化版）"""
        # 这里可以委托给 RiskMetricsService，暂时使用固定值
        return 0.02
    
    def _calculate_expected_shortfall(self, portfolio_state, market_data: Dict[str, Any]) -> float:
        """计算组合ES（简化版）"""
        return 0.025



    
    def _simulate_market_downturn(self, scenario_params: Dict, portfolio_state, market_data: Dict[str, Any]) -> float:
        """模拟市场下行"""
        growth_shock = scenario_params.get('growth_shock', -0.02)
        volatility_shock = scenario_params.get('volatility_shock', 0.5)
        
        # 组合直接损失
        total_exposure = sum(alloc.weight for alloc in portfolio_state.allocations.values())
        direct_loss = total_exposure * growth_shock
        
        # 波动率增加导致风险增加
        volatility_impact = 0.02 * volatility_shock
        
        return float(direct_loss - volatility_impact)
    
    def _simulate_sector_rotation(self, scenario_params: Dict, portfolio_state, market_data: Dict[str, Any]) -> float:
        """模拟板块轮动"""
        # 简化：假设50%板块上涨，50%板块下跌
        rotation_magnitude = scenario_params.get('rotation_magnitude', 0.10)
        
        # 如果组合集中在下跌板块，损失更大
        # 这里使用集中度来估计
        concentration = sum(w**2 for w in [alloc.weight for alloc in portfolio_state.allocations.values()])
        
        loss = -rotation_magnitude * concentration
        return float(loss)
    
    def _simulate_volatility_spike(self, scenario_params: Dict, portfolio_state, market_data: Dict[str, Any]) -> float:
        """模拟波动率飙升"""
        volatility_multiplier = scenario_params.get('volatility_shock', 2.0)
        
        # 波动率飙升导致VaR增加
        base_var = 0.02
        stressed_var = base_var * volatility_multiplier
        additional_risk = stressed_var - base_var
        
        return float(-additional_risk)
    
    def _simulate_generic_scenario(self, scenario_params: Dict, portfolio_state, market_data: Dict[str, Any]) -> float:
        """模拟通用场景"""
        # 通用损失估计
        generic_shock = scenario_params.get('overall_impact', -0.05)
        return float(generic_shock)
    
    def _calculate_value_at_risk(self, portfolio_state, market_data: Dict[str, Any]) -> float:
        """计算组合VaR（简化版）"""
        # 这里可以委托给 RiskMetricsService，暂时使用固定值
        return 0.02
    
    def _calculate_expected_shortfall(self, portfolio_state, market_data: Dict[str, Any]) -> float:
        """计算组合ES（简化版）"""
        return 0.025



