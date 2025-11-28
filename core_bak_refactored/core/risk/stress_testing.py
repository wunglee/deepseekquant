"""
压力测试 - 业务层
从 core_bak/risk_manager.py 拆分
职责: 压力测试、情景分析
P1增强: 完整场景参数使用、组合场景测试（基于专家answer.md指导）
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
import copy
import random

from .risk_models import StressTestScenario, RiskLevel
from .risk_metrics_service import RiskMetricsService, RiskMetricsEngine
from . import calculate_hhi

logger = logging.getLogger('DeepSeekQuant.StressTesting')

# =============================================================================
# 常量定义（基于专家answer.md指导）
# =============================================================================

# 场景相关性矩阵（第6轮更新：扩展至11×11矩阵，新增2011欧债危机、2013钱荒、2011美国债务上限危机）
SCENARIO_CORRELATION_MATRIX = {
    '2008_financial_crisis': {
        '2008_financial_crisis': 1.0,
        'covid_19_pandemic': 0.5,
        '2015_china_market_crash': 0.3,
        'circuit_breaker_2016': 0.7,
        'thousand_stocks_limit_down': 0.7,
        'currency_crisis': 0.45,
        '1997_asian_financial_crisis': 0.4,
        '2022_russia_ukraine_conflict': 0.3,
        '2011_eurozone_debt_crisis': 0.75,
        '2013_china_credit_crunch': 0.5,
        '2011_us_debt_ceiling_crisis': 0.6
    },
    'covid_19_pandemic': {
        '2008_financial_crisis': 0.5,
        'covid_19_pandemic': 1.0,
        '2015_china_market_crash': 0.4,
        'circuit_breaker_2016': 0.5,
        'thousand_stocks_limit_down': 0.6,
        'currency_crisis': 0.2,
        '1997_asian_financial_crisis': 0.2,
        '2022_russia_ukraine_conflict': 0.4,
        '2011_eurozone_debt_crisis': 0.3,
        '2013_china_credit_crunch': 0.3,
        '2011_us_debt_ceiling_crisis': 0.3
    },
    '2015_china_market_crash': {
        '2008_financial_crisis': 0.3,
        'covid_19_pandemic': 0.4,
        '2015_china_market_crash': 1.0,
        'circuit_breaker_2016': 0.8,
        'thousand_stocks_limit_down': 0.9,
        'currency_crisis': 0.25,
        '1997_asian_financial_crisis': 0.3,
        '2022_russia_ukraine_conflict': 0.2,
        '2011_eurozone_debt_crisis': 0.25,
        '2013_china_credit_crunch': 0.7,
        '2011_us_debt_ceiling_crisis': 0.2
    },
    'circuit_breaker_2016': {
        '2008_financial_crisis': 0.7,
        'covid_19_pandemic': 0.5,
        '2015_china_market_crash': 0.8,
        'circuit_breaker_2016': 1.0,
        'thousand_stocks_limit_down': 0.8,
        'currency_crisis': 0.3,
        '1997_asian_financial_crisis': 0.25,
        '2022_russia_ukraine_conflict': 0.2,
        '2011_eurozone_debt_crisis': 0.3,
        '2013_china_credit_crunch': 0.6,
        '2011_us_debt_ceiling_crisis': 0.25
    },
    'thousand_stocks_limit_down': {
        '2008_financial_crisis': 0.7,
        'covid_19_pandemic': 0.6,
        '2015_china_market_crash': 0.9,
        'circuit_breaker_2016': 0.8,
        'thousand_stocks_limit_down': 1.0,
        'currency_crisis': 0.35,
        '1997_asian_financial_crisis': 0.3,
        '2022_russia_ukraine_conflict': 0.25,
        '2011_eurozone_debt_crisis': 0.35,
        '2013_china_credit_crunch': 0.75,
        '2011_us_debt_ceiling_crisis': 0.3
    },
    'currency_crisis': {
        '2008_financial_crisis': 0.45,
        'covid_19_pandemic': 0.2,
        '2015_china_market_crash': 0.25,
        'circuit_breaker_2016': 0.3,
        'thousand_stocks_limit_down': 0.35,
        'currency_crisis': 1.0,
        '1997_asian_financial_crisis': 0.8,
        '2022_russia_ukraine_conflict': 0.35,
        '2011_eurozone_debt_crisis': 0.5,
        '2013_china_credit_crunch': 0.3,
        '2011_us_debt_ceiling_crisis': 0.3
    },
    '1997_asian_financial_crisis': {
        '2008_financial_crisis': 0.4,
        'covid_19_pandemic': 0.2,
        '2015_china_market_crash': 0.3,
        'circuit_breaker_2016': 0.25,
        'thousand_stocks_limit_down': 0.3,
        'currency_crisis': 0.8,
        '1997_asian_financial_crisis': 1.0,
        '2022_russia_ukraine_conflict': 0.25,
        '2011_eurozone_debt_crisis': 0.4,
        '2013_china_credit_crunch': 0.25,
        '2011_us_debt_ceiling_crisis': 0.3
    },
    '2022_russia_ukraine_conflict': {
        '2008_financial_crisis': 0.3,
        'covid_19_pandemic': 0.4,
        '2015_china_market_crash': 0.2,
        'circuit_breaker_2016': 0.2,
        'thousand_stocks_limit_down': 0.25,
        'currency_crisis': 0.35,
        '1997_asian_financial_crisis': 0.25,
        '2022_russia_ukraine_conflict': 1.0,
        '2011_eurozone_debt_crisis': 0.3,
        '2013_china_credit_crunch': 0.2,
        '2011_us_debt_ceiling_crisis': 0.25
    },
    '2011_eurozone_debt_crisis': {
        '2008_financial_crisis': 0.75,
        'covid_19_pandemic': 0.3,
        '2015_china_market_crash': 0.25,
        'circuit_breaker_2016': 0.3,
        'thousand_stocks_limit_down': 0.35,
        'currency_crisis': 0.5,
        '1997_asian_financial_crisis': 0.4,
        '2022_russia_ukraine_conflict': 0.3,
        '2011_eurozone_debt_crisis': 1.0,
        '2013_china_credit_crunch': 0.4,
        '2011_us_debt_ceiling_crisis': 0.6
    },
    '2013_china_credit_crunch': {
        '2008_financial_crisis': 0.5,
        'covid_19_pandemic': 0.3,
        '2015_china_market_crash': 0.7,
        'circuit_breaker_2016': 0.6,
        'thousand_stocks_limit_down': 0.75,
        'currency_crisis': 0.3,
        '1997_asian_financial_crisis': 0.25,
        '2022_russia_ukraine_conflict': 0.2,
        '2011_eurozone_debt_crisis': 0.4,
        '2013_china_credit_crunch': 1.0,
        '2011_us_debt_ceiling_crisis': 0.4
    },
    '2011_us_debt_ceiling_crisis': {
        '2008_financial_crisis': 0.6,
        'covid_19_pandemic': 0.3,
        '2015_china_market_crash': 0.2,
        'circuit_breaker_2016': 0.25,
        'thousand_stocks_limit_down': 0.3,
        'currency_crisis': 0.3,
        '1997_asian_financial_crisis': 0.3,
        '2022_russia_ukraine_conflict': 0.25,
        '2011_eurozone_debt_crisis': 0.6,
        '2013_china_credit_crunch': 0.4,
        '2011_us_debt_ceiling_crisis': 1.0
    }
}

# 资产类别相关性调整因子（answer.md 123-131行）
DEFAULT_CORRELATION_ADJUSTMENT_FACTORS = {
    ('stock', 'stock'): 0.9,
    ('stock', 'bond'): 0.6,
    ('stock', 'commodity'): 0.7,
    ('bond', 'bond'): 0.8,
    ('bond', 'commodity'): 0.5,
    ('commodity', 'commodity'): 0.8
}

# 默认值（answer.md 495-498行）
DEFAULT_DAILY_VOLUME = 1000000  # 100万股
DEFAULT_LIMIT_DOWN_FREQ = 0.05  # 5%的历史跌停频率


class StressTester:
    """
    压力测试器（P1增强：内置标准场景库）
    根据专家answer.md线108-141指导，内置9种历史事件场景
    """
    
    def __init__(self, config: Dict, metrics_engine: Optional[RiskMetricsEngine] = None):
        self.config = config
        self.risk_metrics_service: RiskMetricsEngine = metrics_engine or RiskMetricsService(config)
        self.scenarios: Dict[str, StressTestScenario] = {}
        self._load_builtin_scenarios()   # P1增强
        self._load_custom_scenarios()    # 自定义
    
    def _load_builtin_scenarios(self):
        """加载内置场景库（第6轮更新：补全至10个场景，覆盖率100%）"""
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
            {'scenario_id': 'currency_crisis', 'name': '货币危机',
             'description': '汇率波动剧烈导致资产缩水', 'probability': 0.03, 'impact_level': 'high',
             'duration': '3-6个月', 'triggers': ['外汇储备不足', '资本外流'], 
             'mitigation_strategies': ['货币对冲', '减少外币敞口'],
             'parameters': {'type': 'market_crash', 'decline': -0.25, 'currency_volatility': 2.5, 
                           'capital_flight_intensity': 0.6}},
            # 第5轮新增：1997亚洲金融危机（专家answer.md 2.3节参数）
            {'scenario_id': '1997_asian_financial_crisis', 'name': '1997亚洲金融危机',
             'description': '恒生指数跌60%，对A股影响约-35%', 'probability': 0.01, 'impact_level': 'high',
             'duration': '24个月', 'triggers': ['泰铢崩盘', '区域传导'], 
             'mitigation_strategies': ['货币对冲', '区域多元化'],
             'parameters': {'type': 'currency_crisis', 'decline': -0.35, 'currency_volatility': 3.0, 
                           'regional_contagion': 0.6, 'recovery_period': 24, 'liquidity_dry_up': 0.7}},
            # 第5轮新增：2022俄乌冲突（专家answer.md 2.3节参数）
            {'scenario_id': '2022_russia_ukraine_conflict', 'name': '2022俄乌冲突',
             'description': '地缘政治风险，全球指数平均跌20%', 'probability': 0.02, 'impact_level': 'moderate',
             'duration': '12个月', 'triggers': ['地缘政治', '供应链中断'], 
             'mitigation_strategies': ['避险资产配置', '商品对冲'],
             'parameters': {'type': 'geopolitical_risk', 'decline': -0.20, 'commodity_shock': 0.8, 
                           'sanction_impact': 0.6, 'flight_to_quality': 0.7, 'recovery_period': 12}},
            # 第6轮新增：2011欧债危机
            {'scenario_id': '2011_eurozone_debt_crisis', 'name': '2011欧债危机',
             'description': '欧洲主权债务危机，全球股市平均跌25%', 'probability': 0.015, 'impact_level': 'high',
             'duration': '18个月', 'triggers': ['主权债务违约风险', '银行业危机'], 
             'mitigation_strategies': ['减少欧洲敞口', '增持避险资产'],
             'parameters': {'type': 'sovereign_debt_crisis', 'decline': -0.25, 'credit_spread_widening': 3.0, 
                           'contagion_risk': 0.7, 'recovery_period': 18, 'banking_sector_stress': 0.8}},
            # 第6轮新增：2013钱荒
            {'scenario_id': '2013_china_credit_crunch', 'name': '2013中国钱荒',
             'description': '银行间市场流动性骤紧，隔夜利率飙升至13%', 'probability': 0.04, 'impact_level': 'moderate',
             'duration': '2周', 'triggers': ['央行去杠杆', '流动性收紧'], 
             'mitigation_strategies': ['增加现金储备', '缩短资金久期'],
             'parameters': {'type': 'liquidity_crisis', 'decline': -0.08, 'interest_rate_spike': 6.5, 
                           'interbank_freeze': 0.6, 'recovery_period': 0.5, 'margin_pressure': 0.5}},
            # 第6轮新增：2011美国债务上限危机
            {'scenario_id': '2011_us_debt_ceiling_crisis', 'name': '2011美国债务上限危机',
             'description': '政策不确定性上升，全球指数跌幅约12-20%', 'probability': 0.03, 'impact_level': 'moderate',
             'duration': '3个月', 'triggers': ['财政悬崖', '评级下调'], 
             'mitigation_strategies': ['提升现金', '降低风险资产'],
             'parameters': {'type': 'sovereign_debt_crisis', 'decline': -0.12, 'credit_spread_widening': 1.5, 
                           'contagion_risk': 0.4, 'recovery_period': 6, 'banking_sector_stress': 0.4}},
            # A股特有事件
            {'scenario_id': '2015_china_market_crash', 'name': '2015A股大跌',
             'description': '上证指数下跌43%', 'probability': 0.05, 'impact_level': 'high',
             'duration': '3个月', 'triggers': ['杠杆破裂', '流动性枯竭'], 
             'mitigation_strategies': ['减少杠杆', '提高现金比例'],
             'parameters': {'type': 'market_crash', 'decline': -0.43, 'liquidity_dry_up': 0.8, 
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
        """运行单个压力测试（第6轮更新：支持sovereign_debt_crisis类型）"""
        try:
            scenario_type = scenario.parameters.get('type', 'market_crash')

            if scenario_type == 'market_crash':
                return self._simulate_market_crash(scenario, portfolio_state, market_data)
            elif scenario_type == 'liquidity_crisis':
                return self._simulate_liquidity_crisis(scenario, portfolio_state, market_data)
            elif scenario_type == 'currency_crisis':  # 第5轮新增：1997亚洲金融危机
                return self._simulate_currency_crisis(scenario, portfolio_state, market_data)
            elif scenario_type == 'geopolitical_risk':  # 第5轮新增：2022俄乌冲突
                return self._simulate_geopolitical_risk(scenario, portfolio_state, market_data)
            elif scenario_type == 'sovereign_debt_crisis':  # 第6轮新增：2011欧债危机
                return self._simulate_sovereign_debt_crisis(scenario, portfolio_state, market_data)
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
            total_exposure = self._calculate_total_exposure(portfolio_state)
            direct_loss = total_exposure * decline
            total_impact += direct_loss
            
            # 2. 波动率冲击（volatility_spike参数）
            if 'volatility_spike' in params:
                vol_multiplier = params['volatility_spike']
                # 使用方法1：直接放大VaR（专家推荐，answer.md 13-26行）
                base_var = abs(direct_loss * 0.1)  # 估计基础VaR为损失的10%
                var_impact = self._calculate_var_amplification(base_var, vol_multiplier)
                total_impact -= var_impact  # 额外损失
                logger.info(f"波动率冲击: vol_multiplier={vol_multiplier}, var_impact={var_impact:.4f}")
            
            # 3. 相关性崩溃（correlation_break参数）
            if 'correlation_break' in params:
                corr_level = params['correlation_break']
                # 使用矩阵压缩方法（answer.md 74-88行）
                # 相关性增加导致多元化失效，风险增加
                diversification_loss_factor = corr_level * 0.15  # 相关性0.8时，多元化失效增加12%风险
                diversification_loss = abs(direct_loss) * diversification_loss_factor
                total_impact -= diversification_loss
                logger.info(f"相关性崩溃: corr_level={corr_level}, div_loss={diversification_loss:.4f}")
            
            # 4. 恢复期影响（recovery_period参数）
            if 'recovery_period' in params:
                recovery_months = params['recovery_period']
                opportunity_cost = self._calculate_recovery_opportunity_cost(direct_loss, recovery_months)
                total_impact -= opportunity_cost
                logger.info(f"恢复期影响: months={recovery_months}, opp_cost={opportunity_cost:.4f}")
            
            return float(total_impact)
            
        except Exception as e:
            logger.error(f"市场崩盘场景模拟失败: {e}", exc_info=True)
            return -0.30
    
    def _simulate_liquidity_crisis(self, scenario: StressTestScenario, portfolio_state, market_data: Dict[str, Any]) -> float:
        """
        模拟流动性危机（P1增强：处理A股特有参数）
        根据专家answer.md 310-365行指导实现
        """
        try:
            params = scenario.parameters
            total_impact = 0
            
            # 1. 基础流动性成本
            liquidity_cost_multiplier = params.get('liquidity_cost_multiplier', 3.0)
            total_position_value = self._calculate_total_exposure(portfolio_state)
            base_var = 0.02
            liquidity_loss = total_position_value * base_var * liquidity_cost_multiplier
            total_impact -= liquidity_loss
            
            # 2. liquidity_dry_up（成交量下降，answer.md 310-327行）
            if 'liquidity_dry_up' in params:
                liquidity_ratio = params['liquidity_dry_up']
                # 变现时间急剧增加、冲击成本放大
                for symbol, alloc in portfolio_state.allocations.items():
                    daily_volume = self._get_daily_volume(symbol, market_data)
                    position_size = alloc.weight
                    
                    available_liquidity = daily_volume * (1 - liquidity_ratio)
                    if available_liquidity <= 0:
                        # 完全无法变现，使用极端损失估计
                        total_impact -= position_size * 0.5  # 50%损失
                    else:
                        # 冲击成本放大
                        normal_days = position_size / (daily_volume * 0.1) if daily_volume > 0 else 10
                        crisis_days = position_size / available_liquidity if available_liquidity > 0 else 100
                        impact_multiplier = (crisis_days / normal_days) ** 0.5 if normal_days > 0 else 5.0
                        base_impact = 0.02
                        crisis_impact = base_impact * impact_multiplier
                        total_impact -= position_size * crisis_impact
                
                logger.debug(f"流动性枯竭: liquidity_ratio={liquidity_ratio}")
            
            # 3. limit_hit_frequency（跌停频率，answer.md 329-347行）
            if 'limit_hit_frequency' in params:
                limit_frequency = params['limit_hit_frequency']
                # 随机选择一定比例的资产标记为跌停
                assets = list(portfolio_state.allocations.keys())
                n_limit_hit = int(len(assets) * limit_frequency)
                if n_limit_hit > 0 and len(assets) > 0:
                    limit_assets = random.sample(assets, min(n_limit_hit, len(assets)))
                    for asset in limit_assets:
                        position_size = portfolio_state.allocations[asset].weight
                        # 跌停资产无法交易，损失10%
                        total_impact -= position_size * 0.10
                
                logger.debug(f"跌停频率: limit_frequency={limit_frequency}, n_hit={n_limit_hit}")
            
            # 4. margin_call_cascade（融资盘平仓cascade，answer.md 349-365行）
            if 'margin_call_cascade' in params:
                margin_ratio = params['margin_call_cascade']
                leveraged_position = self._get_leveraged_position(portfolio_state)
                
                # 第一轮平仓
                initial_margin_call = leveraged_position * margin_ratio
                # 平仓导致额夦30%下跌
                additional_decline = initial_margin_call * 0.3
                # 第二轮平仓
                second_margin_call = (leveraged_position - initial_margin_call) * margin_ratio
                total_cascade_impact = initial_margin_call + second_margin_call + additional_decline
                total_impact -= total_cascade_impact
                
                logger.debug(f"融资cascade: margin_ratio={margin_ratio}, cascade_impact={total_cascade_impact:.4f}")
            
            return float(total_impact)
            
        except Exception as e:
            logger.error(f"流动性危机场景模拍失败: {e}", exc_info=True)
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
    
    def _simulate_currency_crisis(self, scenario: StressTestScenario, portfolio_state, market_data: Dict[str, Any]) -> float:
        """
        模拟货币危机（第5轮新增：1997亚洲金融危机）
        基于专家answer.md 2.3节参数
        """
        try:
            params = scenario.parameters
            total_impact = 0
            total_exposure = self._calculate_total_exposure(portfolio_state)
            
            # 1. 直接损失（decline参数）
            decline = params.get('decline', -0.35)
            direct_loss = total_exposure * decline
            total_impact += direct_loss
            
            # 2. 汇率波动（currency_volatility参数）
            if 'currency_volatility' in params:
                currency_vol = params['currency_volatility']
                # 汇率波动导致额外VaR
                base_var = abs(direct_loss * 0.1)
                currency_var_impact = self._calculate_var_amplification(base_var, currency_vol)
                total_impact -= currency_var_impact
                logger.info(f"汇率波动: currency_vol={currency_vol}, var_impact={currency_var_impact:.4f}")
            
            # 3. 区域传导（regional_contagion参数）
            if 'regional_contagion' in params:
                contagion_factor = params['regional_contagion']
                # 区域传导导致额外损失
                contagion_loss = self._calculate_proportional_impact(direct_loss, contagion_factor, 0.3)
                total_impact -= contagion_loss
                logger.info(f"区域传导: contagion_factor={contagion_factor}, loss={contagion_loss:.4f}")
            
            # 4. 流动性枯竭（liquidity_dry_up参数）
            if 'liquidity_dry_up' in params:
                liquidity_ratio = params['liquidity_dry_up']
                liquidity_impact = self._calculate_proportional_impact(direct_loss, liquidity_ratio, 0.2)
                total_impact -= liquidity_impact
                logger.info(f"流动性枯竭: liquidity_ratio={liquidity_ratio}, impact={liquidity_impact:.4f}")
            
            # 5. 恢复期影响（recovery_period参数）
            if 'recovery_period' in params:
                recovery_months = params['recovery_period']
                opportunity_cost = self._calculate_recovery_opportunity_cost(direct_loss, recovery_months)
                total_impact -= opportunity_cost
                logger.info(f"恢复期影响: months={recovery_months}, opp_cost={opportunity_cost:.4f}")
            
            return float(total_impact)
            
        except Exception as e:
            logger.error(f"货币危机场景模拟失败: {e}", exc_info=True)
            return -0.35
    
    def _simulate_geopolitical_risk(self, scenario: StressTestScenario, portfolio_state, market_data: Dict[str, Any]) -> float:
        """
        模拟地缘政治风险（第5轮新增：2022俄乌冲突）
        基于专家answer.md 2.3节参数
        """
        try:
            params = scenario.parameters
            total_impact = 0
            total_exposure = self._calculate_total_exposure(portfolio_state)
            
            # 1. 直接损失（decline参数）
            decline = params.get('decline', -0.20)
            direct_loss = total_exposure * decline
            total_impact += direct_loss
            
            # 2. 商品价格冲击（commodity_shock参数）
            if 'commodity_shock' in params:
                commodity_shock = params['commodity_shock']
                # 商品价格冲击导致成本上升
                commodity_impact = self._calculate_proportional_impact(direct_loss, commodity_shock, 0.15)
                total_impact -= commodity_impact
                logger.info(f"商品冲击: commodity_shock={commodity_shock}, impact={commodity_impact:.4f}")
            
            # 3. 制裁影响（sanction_impact参数）
            if 'sanction_impact' in params:
                sanction_level = params['sanction_impact']
                # 制裁导致贸易受限
                sanction_loss = self._calculate_proportional_impact(direct_loss, sanction_level, 0.1)
                total_impact -= sanction_loss
                logger.info(f"制裁影响: sanction_level={sanction_level}, loss={sanction_loss:.4f}")
            
            # 4. 避险情绪（flight_to_quality参数）
            if 'flight_to_quality' in params:
                flight_intensity = params['flight_to_quality']
                # 避险情绪导致流动性枯竭
                flight_impact = self._calculate_proportional_impact(direct_loss, flight_intensity, 0.12)
                total_impact -= flight_impact
                logger.info(f"避险情绪: flight_intensity={flight_intensity}, impact={flight_impact:.4f}")
            
            # 5. 恢复期影响（recovery_period参数）
            if 'recovery_period' in params:
                recovery_months = params['recovery_period']
                opportunity_cost = self._calculate_recovery_opportunity_cost(direct_loss, recovery_months)
                total_impact -= opportunity_cost
                logger.info(f"恢复期影响: months={recovery_months}, opp_cost={opportunity_cost:.4f}")
            
            return float(total_impact)
            
        except Exception as e:
            logger.error(f"地缘政治风险场景模拟失败: {e}", exc_info=True)
            return -0.20
    
    def _simulate_sovereign_debt_crisis(self, scenario: StressTestScenario, portfolio_state, market_data: Dict[str, Any]) -> float:
        """
        模拟主权债务危机（第6轮新增：2011欧债危机）
        基于欧债危机实证研究
        """
        try:
            params = scenario.parameters
            total_impact = 0
            total_exposure = self._calculate_total_exposure(portfolio_state)
            
            # 1. 直接损失（decline参数）
            decline = params.get('decline', -0.25)
            direct_loss = total_exposure * decline
            total_impact += direct_loss
            
            # 2. 信用利差扩大（credit_spread_widening参数）
            if 'credit_spread_widening' in params:
                spread_multiplier = params['credit_spread_widening']
                # 信用利差扩大导致债券资产价值下降
                # 假设组合中20%为债券资产
                bond_exposure = total_exposure * 0.2
                # 利差扩大300bp，久期约7年，价格下降约21%
                # spread_multiplier=3.0时，损失 = 7% * 3 = 21%
                spread_impact = bond_exposure * 0.07 * spread_multiplier
                total_impact -= spread_impact
                logger.info(f"信用利差冲击: spread_multiplier={spread_multiplier}, impact={spread_impact:.4f}")
            
            # 3. 传染风险（contagion_risk参数）
            if 'contagion_risk' in params:
                contagion_level = params['contagion_risk']
                # 危机传染导致其他市场受影响
                # 使用相关性衰减模型：影响 = 直接损失 × 传染系数 × 30%
                contagion_impact = self._calculate_proportional_impact(direct_loss, contagion_level, 0.3)
                total_impact -= contagion_impact
                logger.info(f"传染风险: contagion_level={contagion_level}, impact={contagion_impact:.4f}")
            
            # 4. 银行业压力（banking_sector_stress参数）
            if 'banking_sector_stress' in params:
                banking_stress = params['banking_sector_stress']
                # 银行业压力导致信贷紧缩
                # 假设对股票资产产生20%的额外下行压力
                banking_impact = self._calculate_proportional_impact(direct_loss, banking_stress, 0.2)
                total_impact -= banking_impact
                logger.info(f"银行业压力: banking_stress={banking_stress}, impact={banking_impact:.4f}")
            
            # 5. 恢复期影响（recovery_period参数）
            if 'recovery_period' in params:
                recovery_months = params['recovery_period']
                opportunity_cost = self._calculate_recovery_opportunity_cost(direct_loss, recovery_months)
                total_impact -= opportunity_cost
                logger.info(f"恢复期影响: months={recovery_months}, opp_cost={opportunity_cost:.4f}")
            
            return float(total_impact)
            
        except Exception as e:
            logger.error(f"主权债务危机场景模拟失败: {e}", exc_info=True)
            return -0.25
    
    def _simulate_market_downturn(self, scenario_params: Dict, portfolio_state, market_data: Dict[str, Any]) -> float:
        """模拟市场下行"""
        growth_shock = scenario_params.get('growth_shock', -0.02)
        volatility_shock = scenario_params.get('volatility_shock', 0.5)
        
        # 组合直接损失
        total_exposure = self._calculate_total_exposure(portfolio_state)
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
        concentration = calculate_hhi([alloc.weight for alloc in portfolio_state.allocations.values()])
        
        loss = -rotation_magnitude * concentration
        return float(loss)
    
    def _simulate_volatility_spike(self, scenario_params: Dict, portfolio_state, market_data: Dict[str, Any]) -> float:
        """模拟波动率飙升"""
        volatility_multiplier = scenario_params.get('volatility_shock', 2.0)
        
        # 波动率飙升导致VaR增加
        base_var = 0.02
        additional_risk = self._calculate_var_amplification(base_var, volatility_multiplier)
        
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
    
    def _calculate_total_exposure(self, portfolio_state) -> float:
        """
        计算组合总敞口（公共方法，消除重复代码）
        
        Args:
            portfolio_state: 组合状态
        
        Returns:
            float: 组合总敞口（所有持仓权重之和）
        
        说明：
            - 统一的敞口计算逻辑，避免多处重复
            - 如果未来需要调整计算方式（如考虑杠杆、净值等），只需修改此处
            - 出现位置：6个场景模拟方法中
        
        示例：
            >>> exposure = self._calculate_total_exposure(portfolio_state)
            >>> # 返回所有持仓权重之和，如 1.0（满仓）或 0.8（80%仓位）
        """
        return sum(alloc.weight for alloc in portfolio_state.allocations.values())
    
    def _calculate_proportional_impact(
        self, 
        base_loss: float, 
        impact_factor: float, 
        coefficient: float = 1.0
    ) -> float:
        """
        计算基于基础损失的比例影响（公共方法，消除重复代码）
        
        Args:
            base_loss: 基础损失金额（可正可负，内部会取绝对值）
            impact_factor: 影响因子（如传导因子、流动性比率等）
            coefficient: 影响系数（默认1.0）
        
        Returns:
            float: 额外影响金额（始终为正值）
        
        说明：
            - 统一的比例影响计算逻辑
            - 适用于区域传导、流动性冲击、商品价格冲击等场景
            - 计算公式：abs(base_loss) × impact_factor × coefficient
        
        示例：
            >>> # 区域传导：30%系数
            >>> impact = self._calculate_proportional_impact(direct_loss, 0.8, 0.3)
            >>> # 等价于：abs(direct_loss) * 0.8 * 0.3
            
            >>> # 流动性冲击：20%系数
            >>> impact = self._calculate_proportional_impact(direct_loss, 0.7, 0.2)
        """
        return abs(base_loss) * impact_factor * coefficient
    
    def _calculate_recovery_opportunity_cost(self, direct_loss: float, recovery_months: int) -> float:
        """
        计算恢复期机会成本（公共方法，消除重复代码）
        
        Args:
            direct_loss: 直接损失金额（可正可负，内部会取绝对值）
            recovery_months: 恢复期（月数）
        
        Returns:
            float: 恢复期机会成本（始终为正值）
        
        边界保护：
            - recovery_months限制在0-120月（0-10年），避免指数爆炸
            - 对于极端值自动截断并记录警告日志
        
        示例：
            >>> cost = self._calculate_recovery_opportunity_cost(-0.30, 36)
            >>> # 恢复期3年，机会成本 = 0.30 * ((1+0.03)^3 - 1) ≈ 0.0278
        """
        # 数值稳定性：限制恢复期范围在0-120月（0-10年）
        original_months = recovery_months
        recovery_months = max(0, min(recovery_months, 120))
        
        # 边界值警告
        if original_months != recovery_months:
            logger.warning(
                f"恢复期超出合理范围: 原值={original_months}月, 已截断为={recovery_months}月 (最大120月)"
            )
        
        # 获取无风险利率（默认3%）
        risk_free_rate = self.config.get('risk_free_rate', 0.03)
        
        # 计算机会成本
        t_years = recovery_months / 12.0
        try:
            opportunity_cost = abs(direct_loss) * ((1 + risk_free_rate) ** t_years - 1)
        except OverflowError:
            # 极端情况下的数值溢出保护
            logger.error(
                f"恢复期机会成本计算溢出: direct_loss={direct_loss}, months={recovery_months}, rate={risk_free_rate}"
            )
            # 使用线性近似：cost ≈ abs(direct_loss) * rate * t_years
            opportunity_cost = abs(direct_loss) * risk_free_rate * t_years
        
        return float(opportunity_cost)
    
    def _calculate_var_amplification(self, base_var: float, multiplier: float) -> float:
        """
        计算VaR放大后的额外风险
        
        Args:
            base_var: 基础VaR水平（正值）
            multiplier: 放大倍数（>=1）
        
        Returns:
            float: 额外风险增量（始终为正值）
        
        说明：
            - 统一VaR放大计算逻辑，避免在各处使用 base_var * (multiplier - 1)
            - 技术性统一，不改变业务口径；若各场景需不同系数，可通过base_var来源或multiplier配置体现
        """
        try:
            multiplier = float(multiplier)
            if multiplier <= 1:
                return 0.0
            return float(base_var * (multiplier - 1))
        except Exception:
            return 0.0    
    # =========================================================================
    # P1增强：组合场景测试方法（基于专家answer.md 295-396行指导）
    # =========================================================================
    
    def run_combined_stress_tests(self, portfolio_state, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行组合场景压力测试
        基于专家answer.md 295-396行指导
        """
        combined_results = {}
        config = self.config.get('stress_testing', {})
        
        try:
            # 1. 顺序冲击测试
            if config.get('enable_sequential_test', True):
                scenario_sequences = config.get('sequential_scenarios', [
                    ['2008_financial_crisis', '2015_china_market_crash']
                ])
                sequential_results = {}
                for seq in scenario_sequences:
                    if all(s in self.scenarios for s in seq):
                        result = self._simulate_sequential_impact(seq, portfolio_state, market_data)
                        sequential_results['_'.join(seq)] = result
                combined_results['sequential'] = sequential_results
            
            # 2. 并发冲击测试
            if config.get('enable_concurrent_test', True):
                scenario_groups = config.get('concurrent_scenarios', [
                    ['2008_financial_crisis', '2015_china_market_crash']
                ])
                concurrent_results = {}
                for group in scenario_groups:
                    if all(s in self.scenarios for s in group):
                        result = self._simulate_concurrent_shock(group, portfolio_state, market_data)
                        concurrent_results['_'.join(group)] = result
                combined_results['concurrent'] = concurrent_results
            
            # 3. 反馈循环测试
            if config.get('enable_feedback_loop_test', True):
                feedback_scenarios = config.get('feedback_loop_scenarios', ['2008_financial_crisis'])
                feedback_results = {}
                for scenario_id in feedback_scenarios:
                    if scenario_id in self.scenarios:
                        result = self._simulate_feedback_loop(
                            self.scenarios[scenario_id], 
                            portfolio_state, 
                            market_data
                        )
                        feedback_results[scenario_id] = result
                combined_results['feedback_loop'] = feedback_results
            
            return combined_results
            
        except Exception as e:
            logger.error(f"组合场景测试执行失败: {e}")
            return {}
    
    def _simulate_sequential_impact(self, scenario_sequence: List[str], portfolio_state, market_data: Dict[str, Any]) -> float:
        """
        顺序冲击测试（危机传导）
        基于专家answer.md 148-178行指导
        """
        try:
            total_impact = 0
            previous_impact = 0
            propagation_factor = self.config.get('stress_testing', {}).get('propagation_factor', 0.3)
            
            for i, scenario_id in enumerate(scenario_sequence):
                scenario = self.scenarios[scenario_id]
                # 基础冲击
                base_impact = self._run_single_stress_test(scenario, portfolio_state, market_data)
                
                # 传导效应：前一个场景的30%传导到下一个
                propagated_impact = previous_impact * propagation_factor
                
                # 当前场景总影响
                scenario_impact = base_impact + propagated_impact
                total_impact += scenario_impact
                previous_impact = scenario_impact
                
                logger.debug(f"顺序冲击 {i} - {scenario_id}: base={base_impact:.4f}, propagated={propagated_impact:.4f}")
            
            return float(total_impact)
            
        except Exception as e:
            logger.error(f"顺序冲击测试失败: {e}")
            return -0.5
    
    def _simulate_concurrent_shock(self, scenarios: List[str], portfolio_state, market_data: Dict[str, Any]) -> float:
        """
        并发冲击测试（系统性风险）
        基于专家answer.md 178-213行指导
        """
        try:
            # 1. 计算各场景独立影响
            impacts = {}
            for scenario_id in scenarios:
                scenario = self.scenarios[scenario_id]
                impact = self._run_single_stress_test(scenario, portfolio_state, market_data)
                impacts[scenario_id] = impact
            
            # 2. 构建影响向量
            impact_vector = np.array([impacts[s] for s in scenarios])
            
            # 3. 获取场景相关性矩阵
            n = len(scenarios)
            corr_matrix = np.zeros((n, n))
            for i, s1 in enumerate(scenarios):
                for j, s2 in enumerate(scenarios):
                    # 自相关强制为1.0
                    if i == j:
                        base_corr = 1.0
                    else:
                        # 先查s1->s2
                        if s1 in SCENARIO_CORRELATION_MATRIX and s2 in SCENARIO_CORRELATION_MATRIX[s1]:
                            base_corr = SCENARIO_CORRELATION_MATRIX[s1][s2]
                        # 再查对称方向s2->s1
                        elif s2 in SCENARIO_CORRELATION_MATRIX and s1 in SCENARIO_CORRELATION_MATRIX[s2]:
                            base_corr = SCENARIO_CORRELATION_MATRIX[s2][s1]
                        else:
                            base_corr = 0.5  # 默认相关性0.5
                    # 根据市场波动率做微调（最高+0.3），不存在则不调整
                    market_vol = market_data.get('market_volatility')
                    if market_vol is not None:
                        try:
                            adj = min(1.0, max(0.0, base_corr + min(0.3, float(market_vol) * 0.1)))
                        except Exception:
                            adj = base_corr
                    else:
                        adj = base_corr
                    corr_matrix[i, j] = adj
            
            # 4. 计算总影响（考虑相关性）
            if len(impact_vector) > 1:
                # 使用相关性调整的平方和公式
                total_variance = impact_vector.T @ corr_matrix @ impact_vector
                # total_variance是损失的平方，取平方根后保持负号
                total_impact = -np.sqrt(abs(total_variance))
            else:
                total_impact = impact_vector[0]
            
            # 5. 系统性风险溢价（25%，第5轮专家调整：20%→25%）
            systemic_premium = self.config.get('stress_testing', {}).get('systemic_premium', 0.25)
            total_impact *= (1 + systemic_premium)
            
            logger.debug(f"并发冲击: impacts={impacts}, total={total_impact:.4f}")
            return float(total_impact)
            
        except Exception as e:
            logger.error(f"并发冲击测试失败: {e}")
            return -0.5
    
    def _simulate_feedback_loop(self, scenario: StressTestScenario, portfolio_state, market_data: Dict[str, Any]) -> float:
        """
        反馈循环测试（风险叠加）
        基于专家answer.md 最新回答（单一反馈因子机制）
        """
        try:
            max_iterations = self.config.get('stress_testing', {}).get('max_feedback_iterations', 5)
            feedback_factor = self.config.get('stress_testing', {}).get('feedback_factor', 0.25)
            
            total_impact = 0
            current_portfolio = copy.deepcopy(portfolio_state)
            
            for iteration in range(max_iterations):
                # 基于当前组合状态计算基础影响
                base_impact = self._run_single_stress_test(scenario, current_portfolio, market_data)
                
                # 计算反馈效应：基础影响的一部分作为额外影响
                feedback_effect = base_impact * feedback_factor
                
                # 本次迭代的总影响
                iteration_impact = base_impact + feedback_effect
                total_impact += iteration_impact
                
                # 更新组合状态：反映本次迭代的损失
                current_portfolio = self._update_portfolio_value(current_portfolio, iteration_impact)
                
                logger.debug(f"反馈循环迭代 {iteration}: base={base_impact:.4f}, feedback={feedback_effect:.4f}, iter={iteration_impact:.4f}")
                
                # 收敛检查：影响小于0.1%时停止
                if abs(iteration_impact) < 0.001:
                    break
            
            return float(total_impact)
            
        except Exception as e:
            logger.error(f"反馈循环测试失败: {e}")
            return -0.5
    
    def _update_portfolio_value(self, portfolio_state, loss_amount: float):
        """
        更新组合价值（损失按比例减少所有资产的价值）
        基于专家answer.md 266-285行指导
        """
        try:
            total_value = self._calculate_total_exposure(portfolio_state)
            if total_value <= 0:
                return portfolio_state
            
            new_total_value = total_value + loss_amount  # loss_amount为负值
            if new_total_value <= 0:
                new_total_value = total_value * 0.1  # 保留至少10%
            
            scale_factor = new_total_value / total_value
            for symbol, alloc in portfolio_state.allocations.items():
                alloc.weight *= scale_factor
            
            return portfolio_state
            
        except Exception as e:
            logger.error(f"组合价值更新失败: {e}")
            return portfolio_state
    
    # =========================================================================
    # P1增强：辅助工具方法（基于专家answer.md 399-500行指导）
    # =========================================================================
    
    def _get_daily_volume(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """
        获取日成交量
        基于专家answer.md 406-434行指导
        """
        try:
            # 优先从volumes字段获取
            if 'volumes' in market_data and symbol in market_data['volumes']:
                volume_data = market_data['volumes'][symbol]
                if isinstance(volume_data, dict):
                    if 'volume' in volume_data:
                        return float(volume_data['volume'])
                    elif 'avg_volume' in volume_data:
                        return float(volume_data['avg_volume'])
                else:
                    return float(volume_data)
            
            # 其次从prices字段推断
            if 'prices' in market_data and symbol in market_data['prices']:
                price_data = market_data['prices'][symbol]
                if isinstance(price_data, dict) and 'volume' in price_data:
                    volumes = price_data['volume']
                    if isinstance(volumes, (list, np.ndarray)) and len(volumes) > 0:
                        return float(volumes[-1])
            
            # 默认值
            logger.warning(f"无法获取{symbol}的成交量数据，使用默认值{DEFAULT_DAILY_VOLUME}")
            return DEFAULT_DAILY_VOLUME
            
        except Exception as e:
            logger.warning(f"获取日成交量失败: {e}，使用默认值")
            return DEFAULT_DAILY_VOLUME
    
    def _get_leveraged_position(self, portfolio_state) -> float:
        """
        获取杠杆仓位规模
        基于专家answer.md 436-455行指导
        """
        try:
            # 如果组合状态有杠杆信息，直接获取
            if hasattr(portfolio_state, 'leveraged_position') and portfolio_state.leveraged_position is not None:
                return float(portfolio_state.leveraged_position)
            
            # 否则，估计为总风险暴露（保守估计）
            total_exposure = self._calculate_total_exposure(portfolio_state)
            return float(total_exposure)
            
        except Exception as e:
            logger.warning(f"获取杠杆仓位失败: {e}，使用默认估计")
            return 0.0

# =========================================================================
# 阶段B：行业参数统计与显著性检验（纯技术实现，遵循专家口径）
# =========================================================================

class IndustryParameterAnalyzer:
    """
    行业参数统计与显著性检验（阶段B）
    - 输入：行业 → 历史冲击系数样本（例如事件窗口中的损失率或冲击值）
    - 输出：行业平均冲击参数、两两行业t检验p值（大样本近似正态）
    - 验证：调用 UATValidator.validate_industry_parameter_difference() 进行业务口径断言
    
    说明：
    - 不在此处新增任何业务参数值；仅提供统计计算能力。
    - t检验p值采用Welch近似与正态近似（样本量≥1000时合理）。
    """
    
    # 专家第2轮参数范围（基准=0.10）
    # 职责：在risk模块维护业务参数，不应放在测试中
    INDUSTRY_CONFIGS = {
        'financial': {
            'mean': -0.150,
            'std': 0.020,
            'rationale': '系统性风险敏感度高'
        },
        'technology': {
            'mean': -0.120,
            'std': 0.022,
            'rationale': '成长性高但波动性大'
        },
        'cyclical': {
            'mean': -0.135,
            'std': 0.025,
            'rationale': '经济周期敏感性强'
        },
        'defensive': {
            'mean': -0.085,
            'std': 0.015,
            'rationale': '风险抵御能力强'
        }
    }
    
    def __init__(self):
        pass
    
    @classmethod
    def generate_test_samples(cls, n_samples: int = 1200, seed: int = 42) -> Dict[str, List[float]]:
        """
        生成测试用行业样本数据
        
        职责：业务逻辑归位到risk模块，供测试调用
        
        Args:
            n_samples: 样本量（默认1200，满足≥1000要求）
            seed: 随机种子
        
        Returns:
            行业样本字典 {industry_name: samples}
        """
        np.random.seed(seed)
        
        industry_samples = {}
        for industry, config in cls.INDUSTRY_CONFIGS.items():
            samples = np.random.normal(
                loc=config['mean'],
                scale=config['std'],
                size=n_samples
            )
            # 限制在业务合理范围（-50%至+20%）
            samples = np.clip(samples, -0.5, 0.2)
            industry_samples[industry] = samples.tolist()
        
        return industry_samples
    
    @staticmethod
    def compute_industry_parameters(samples: Dict[str, List[float]]) -> Dict[str, float]:
        """
        计算各行业平均冲击参数
        Args:
            samples: {industry: [values...]}
        Returns:
            {industry: mean_value}
        """
        params: Dict[str, float] = {}
        for industry, values in samples.items():
            arr = np.array(values, dtype=float)
            if arr.size == 0:
                params[industry] = 0.0
            else:
                params[industry] = float(np.mean(arr))
        return params
    
    @staticmethod
    def _welch_t_stat(arr1: np.ndarray, arr2: np.ndarray) -> float:
        """计算Welch t统计量（不返回df）"""
        n1, n2 = arr1.size, arr2.size
        m1, m2 = float(np.mean(arr1)), float(np.mean(arr2))
        v1, v2 = float(np.var(arr1, ddof=1)), float(np.var(arr2, ddof=1))
        denom = np.sqrt(v1 / n1 + v2 / n2)
        if denom == 0:
            return 0.0
        return (m1 - m2) / denom
    
    @staticmethod
    def _normal_approx_p_value(t_stat: float) -> float:
        """
        对于大样本（n>=1000）采用正态近似计算双侧p值
        p = 2 * (1 - Phi(|t|)); Phi为标准正态CDF
        采用误差函数erf近似：Phi(x) ≈ 0.5 * [1 + erf(x / sqrt(2))]
        """
        try:
            from math import erf, sqrt
            abs_t = abs(t_stat)
            phi = 0.5 * (1.0 + erf(abs_t / sqrt(2.0)))
            p = 2.0 * (1.0 - phi)
            # 限制范围
            p = max(0.0, min(1.0, p))
            return p
        except Exception:
            # 回退：极端保守值
            return 1.0
    
    def compute_t_tests(self, samples: Dict[str, List[float]]) -> Dict[tuple, float]:
        """
        计算两两行业t检验p值（Welch + 正态近似）
        Args:
            samples: {industry: [values...]}
        Returns:
            { (industry_a, industry_b): p_value }
        """
        industries = list(samples.keys())
        p_values: Dict[tuple, float] = {}
        for i in range(len(industries)):
            for j in range(i + 1, len(industries)):
                a, b = industries[i], industries[j]
                arr1 = np.array(samples[a], dtype=float)
                arr2 = np.array(samples[b], dtype=float)
                # 为空回退
                if arr1.size < 2 or arr2.size < 2:
                    p_values[(a, b)] = 1.0
                    continue
                t_stat = self._welch_t_stat(arr1, arr2)
                p_val = self._normal_approx_p_value(t_stat)
                p_values[(a, b)] = float(p_val)
        return p_values
    
    def analyze_and_validate(self, samples: Dict[str, List[float]]):
        """
        端到端：统计→t检验→UAT验证
        Returns:
            UATResult
        """
        from core_bak_refactored.core.backtest._fragments.uat_validator import UATValidator
        params = self.compute_industry_parameters(samples)
        t_tests = self.compute_t_tests(samples)
        validator = UATValidator()
        return validator.validate_industry_parameter_difference(params, t_tests)
