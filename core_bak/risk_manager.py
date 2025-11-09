"""
DeepSeekQuant 风险管理系统
负责交易风险监控、头寸规模计算、风险限额管理和实时风险控制
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import time
import json
import scipy.stats as stats
from scipy import optimize
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import copy
import traceback
from collections import deque, defaultdict
import heapq
import statistics
from decimal import Decimal
import math

# 导入内部模块
from .base_processor import BaseProcessor
from ..utils.helpers import validate_data, calculate_returns, normalize_data
from ..utils.validators import validate_risk_parameters
from ..utils.performance import calculate_risk_adjusted_metrics
from ..core.portfolio_manager import PortfolioState, AssetAllocation

logger = logging.getLogger('DeepSeekQuant.RiskManager')


class RiskLevel(Enum):
    """风险等级枚举"""
    VERY_LOW = "very_low"  # 极低风险
    LOW = "low"  # 低风险
    MODERATE = "moderate"  # 中等风险
    HIGH = "high"  # 高风险
    VERY_HIGH = "very_high"  # 极高风险
    EXTREME = "extreme"  # 极端风险
    BLACK_SWAN = "black_swan"  # 黑天鹅风险


class RiskType(Enum):
    """风险类型枚举"""
    MARKET_RISK = "market_risk"  # 市场风险
    CREDIT_RISK = "credit_risk"  # 信用风险
    LIQUIDITY_RISK = "liquidity_risk"  # 流动性风险
    OPERATIONAL_RISK = "operational_risk"  # 操作风险
    SYSTEMIC_RISK = "systemic_risk"  # 系统性风险
    CONCENTRATION_RISK = "concentration_risk"  # 集中度风险
    LEVERAGE_RISK = "leverage_risk"  # 杠杆风险
    COUNTERPARTY_RISK = "counterparty_risk"  # 对手方风险
    REGULATORY_RISK = "regulatory_risk"  # 监管风险
    MODEL_RISK = "model_risk"  # 模型风险


class RiskMetric(Enum):
    """风险指标枚举"""
    VOLATILITY = "volatility"  # 波动率
    VALUE_AT_RISK = "value_at_risk"  # 在险价值
    EXPECTED_SHORTFALL = "expected_shortfall"  # 预期短缺
    BETA = "beta"  # Beta系数
    CORRELATION = "correlation"  # 相关性
    DRAWDOWN = "drawdown"  # 回撤
    STRESS_TEST = "stress_test"  # 压力测试
    SCENARIO_ANALYSIS = "scenario_analysis"  # 情景分析
    LIQUIDITY_GAP = "liquidity_gap"  # 流动性缺口
    LEVERAGE_RATIO = "leverage_ratio"  # 杠杆比率
    RISK_CONTRIBUTION = "risk_contribution"  # 风险贡献度
    MARGINAL_RISK = "marginal_risk"  # 边际风险
    TAIL_RISK = "tail_risk"  # 尾部风险
    MAX_POSITION_SIZE = "max_position_size"  # 最大头寸规模


class RiskControlAction(Enum):
    """风险控制动作枚举"""
    ALLOW = "allow"  # 允许交易
    WARN = "warn"  # 警告但允许
    REDUCE = "reduce"  # 减少头寸
    REJECT = "reject"  # 拒绝交易
    HEDGE = "hedge"  # 对冲风险
    LIQUIDATE = "liquidate"  # 平仓
    SUSPEND = "suspend"  # 暂停交易
    CIRCUIT_BREAKER = "circuit_breaker"  # 熔断机制


@dataclass
class RiskLimit:
    """风险限额配置"""
    risk_type: RiskType
    metric: RiskMetric
    threshold: float
    time_horizon: str = "1d"  # 时间范围: 1d, 1w, 1m, 1y
    confidence_level: float = 0.95  # 置信水平
    calculation_method: str = "historical"  # 计算方法: historical, parametric, monte_carlo
    action: RiskControlAction = RiskControlAction.WARN
    grace_period: int = 0  # 宽限期（分钟）
    escalation_level: int = 1  # 升级级别
    is_hard_limit: bool = False  # 是否为硬性限额
    notification_channels: List[str] = field(default_factory=lambda: ["email", "dashboard"])
    review_required: bool = False  # 是否需要人工审核

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskLimit':
        return cls(**data)


@dataclass
class PositionLimit:
    """头寸限额配置"""
    symbol: str
    max_notional: float  # 最大名义价值
    max_quantity: float  # 最大数量
    max_weight: float  # 最大权重
    min_liquidity_ratio: float = 0.1  # 最小流动性比率
    max_leverage: float = 1.0  # 最大杠杆
    concentration_limit: float = 0.2  # 集中度限制
    sector_limit: float = 0.3  # 行业限制
    region_limit: float = 0.4  # 地区限制
    var_limit: float = -0.05  # VaR限制
    stress_test_limit: float = -0.15  # 压力测试限制

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PositionLimit':
        return cls(**data)


@dataclass
class RiskAssessment:
    """风险评估结果"""
    timestamp: str
    portfolio_id: str
    overall_risk_level: RiskLevel
    risk_score: float  # 0-100风险评分
    value_at_risk: float  # 在险价值
    expected_shortfall: float  # 预期短缺
    max_drawdown: float  # 最大回撤
    liquidity_risk: float  # 流动性风险
    concentration_risk: float  # 集中度风险
    leverage_risk: float  # 杠杆风险
    stress_test_results: Dict[str, float]  # 压力测试结果
    scenario_analysis: Dict[str, float]  # 情景分析结果
    risk_contributions: Dict[str, float]  # 风险贡献度
    limit_breaches: List[Dict[str, Any]]  # 限额违反情况
    recommendations: List[Dict[str, Any]]  # 风险建议
    confidence_level: float = 0.95  # 评估置信度

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskAssessment':
        return cls(**data)


@dataclass
class RiskEvent:
    """风险事件记录"""
    event_id: str
    event_type: RiskType
    severity: RiskLevel
    timestamp: str
    description: str
    triggered_by: str  # 触发因素
    impact_assessment: Dict[str, Any]  # 影响评估
    action_taken: RiskControlAction  # 采取的措施
    resolved: bool = False  # 是否已解决
    resolution_time: Optional[str] = None  # 解决时间
    root_cause: Optional[str] = None  # 根本原因
    prevention_measures: List[str] = field(default_factory=list)  # 预防措施

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskEvent':
        return cls(**data)


@dataclass
class StressTestScenario:
    """压力测试场景"""
    scenario_id: str
    name: str
    description: str
    parameters: Dict[str, Any]  # 场景参数
    probability: float  # 发生概率
    impact_level: RiskLevel  # 影响程度
    duration: str  # 持续时间
    triggers: List[str]  # 触发条件
    mitigation_strategies: List[str]  # 缓解策略
    historical_precedent: Optional[str] = None  # 历史先例
    recovery_time: Optional[str] = None  # 恢复时间

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StressTestScenario':
        return cls(**data)


class RiskManager(BaseProcessor):
    """风险管理器 - 负责实时风险监控和控制"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化风险管理器

        Args:
            config: 配置字典
        """
        super().__init__(config)

        # 风险管理配置
        self.risk_config = config.get('risk_management', {})
        self.limits_config = self.risk_config.get('limits', {})
        self.monitoring_config = self.risk_config.get('monitoring', {})
        self.control_config = self.risk_config.get('controls', {})

        # 风险状态
        self.current_risk_assessment: Optional[RiskAssessment] = None
        self.risk_history: List[RiskAssessment] = []
        self.active_risk_events: List[RiskEvent] = []
        self.resolved_risk_events: List[RiskEvent] = []

        # 风险限额
        self.risk_limits: Dict[str, RiskLimit] = {}
        self.position_limits: Dict[str, PositionLimit] = {}
        self._initialize_risk_limits()

        # 压力测试场景
        self.stress_test_scenarios: Dict[str, StressTestScenario] = {}
        self._initialize_stress_test_scenarios()

        # 风险模型参数
        self.risk_models: Dict[str, Any] = {}
        self._initialize_risk_models()

        # 监控状态
        self.last_monitoring_time: Optional[datetime] = None
        self.monitoring_interval: int = self.monitoring_config.get('interval_seconds', 60)
        self.alert_thresholds: Dict[str, float] = self.monitoring_config.get('alert_thresholds', {})

        # 性能统计
        self.performance_stats = {
            'risk_assessments_performed': 0,
            'limit_checks_performed': 0,
            'risk_events_detected': 0,
            'preventive_actions_taken': 0,
            'false_positives': 0,
            'avg_assessment_time': 0.0,
            'max_response_time': 0.0
        }

        # 缓存和状态
        self._market_data_cache: Dict[str, Any] = {}
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._volatility_estimates: Dict[str, float] = {}
        self._liquidity_metrics: Dict[str, float] = {}

        # 线程池
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.risk_config.get('max_monitoring_threads', 8)
        )

        # 初始化完成
        logger.info("风险管理系统初始化完成")

    def _initialize_risk_limits(self):
        """初始化风险限额"""
        try:
            # 从配置加载风险限额
            limits_data = self.limits_config.get('risk_limits', [])
            for limit_data in limits_data:
                try:
                    risk_limit = RiskLimit.from_dict(limit_data)
                    limit_key = f"{risk_limit.risk_type.value}_{risk_limit.metric.value}"
                    self.risk_limits[limit_key] = risk_limit
                except Exception as e:
                    logger.warning(f"风险限额加载失败: {e}")

            # 从配置加载头寸限额
            position_limits_data = self.limits_config.get('position_limits', {})
            for symbol, limit_data in position_limits_data.items():
                try:
                    position_limit = PositionLimit(symbol=symbol, **limit_data)
                    self.position_limits[symbol] = position_limit
                except Exception as e:
                    logger.warning(f"头寸限额加载失败 {symbol}: {e}")

            logger.info(f"已加载 {len(self.risk_limits)} 个风险限额和 {len(self.position_limits)} 个头寸限额")

        except Exception as e:
            logger.error(f"风险限额初始化失败: {e}")

    def _initialize_stress_test_scenarios(self):
        """初始化压力测试场景"""
        try:
            scenarios_data = self.risk_config.get('stress_test_scenarios', [])
            for scenario_data in scenarios_data:
                try:
                    scenario = StressTestScenario.from_dict(scenario_data)
                    self.stress_test_scenarios[scenario.scenario_id] = scenario
                except Exception as e:
                    logger.warning(f"压力测试场景加载失败: {e}")

            logger.info(f"已加载 {len(self.stress_test_scenarios)} 个压力测试场景")

        except Exception as e:
            logger.error(f"压力测试场景初始化失败: {e}")

    def _initialize_risk_models(self):
        """初始化风险模型"""
        try:
            # 风险模型配置
            model_config = self.risk_config.get('risk_models', {})

            # 波动率模型
            self.risk_models['volatility'] = {
                'method': model_config.get('volatility_method', 'garch'),
                'lookback_period': model_config.get('volatility_lookback', 252),
                'ewma_lambda': model_config.get('ewma_lambda', 0.94)
            }

            # 相关性模型
            self.risk_models['correlation'] = {
                'method': model_config.get('correlation_method', 'ledoit_wolf'),
                'lookback_period': model_config.get('correlation_lookback', 126),
                'shrinkage_intensity': model_config.get('shrinkage_intensity', 0.5)
            }

            # VaR模型
            self.risk_models['var'] = {
                'method': model_config.get('var_method', 'historical'),
                'confidence_level': model_config.get('var_confidence', 0.95),
                'lookback_period': model_config.get('var_lookback', 504),
                'monte_carlo_simulations': model_config.get('monte_carlo_sims', 10000)
            }

            # ES模型
            self.risk_models['expected_shortfall'] = {
                'method': model_config.get('es_method', 'historical'),
                'confidence_level': model_config.get('es_confidence', 0.975),
                'lookback_period': model_config.get('es_lookback', 504)
            }

            logger.info("风险模型初始化完成")

        except Exception as e:
            logger.error(f"风险模型初始化失败: {e}")

    def process(self, portfolio_state: PortfolioState,
                market_data: Dict[str, Any],
                pending_trades: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        处理风险监控请求

        Args:
            portfolio_state: 当前组合状态
            market_data: 市场数据
            pending_trades: 待处理交易列表

        Returns:
            风险评估和控制结果
        """
        start_time = time.time()

        try:
            # 验证输入数据
            if not self._validate_inputs(portfolio_state, market_data):
                logger.warning("输入数据验证失败")
                return {'error': 'Invalid input data'}

            # 更新市场数据缓存
            self._update_market_data_cache(market_data)

            # 执行全面风险评估
            risk_assessment = self._perform_comprehensive_risk_assessment(
                portfolio_state, market_data
            )

            # 检查风险限额
            limit_breaches = self._check_all_risk_limits(portfolio_state, risk_assessment)

            # 检查待处理交易风险
            trade_risks = {}
            if pending_trades:
                trade_risks = self._assess_trade_risks(pending_trades, portfolio_state, market_data)

            # 执行压力测试
            stress_test_results = self._run_stress_tests(portfolio_state, market_data)

            # 生成风险控制建议
            recommendations = self._generate_risk_recommendations(
                portfolio_state, risk_assessment, limit_breaches, stress_test_results
            )

            # 更新风险状态
            self._update_risk_state(risk_assessment, limit_breaches, recommendations)

            # 执行实时风险控制
            control_actions = self._execute_risk_controls(
                portfolio_state, risk_assessment, limit_breaches, trade_risks
            )

            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time, len(limit_breaches))

            logger.info(f"风险评估完成: 等级={risk_assessment.overall_risk_level.value}, "
                        f"耗时={processing_time:.3f}s, 限额违反={len(limit_breaches)}")

            return {
                'risk_assessment': risk_assessment.to_dict(),
                'limit_breaches': limit_breaches,
                'trade_risks': trade_risks,
                'stress_test_results': stress_test_results,
                'recommendations': recommendations,
                'control_actions': control_actions,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"风险处理失败: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def _validate_inputs(self, portfolio_state: PortfolioState,
                         market_data: Dict[str, Any]) -> bool:
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

    def _update_market_data_cache(self, market_data: Dict[str, Any]):
        """更新市场数据缓存"""
        try:
            timestamp = market_data['timestamp']

            # 更新价格数据
            for symbol, price_data in market_data['prices'].items():
                if symbol not in self._market_data_cache:
                    self._market_data_cache[symbol] = {}

                self._market_data_cache[symbol][timestamp] = {
                    'close': price_data['close'][-1] if price_data['close'] else 0,
                    'high': price_data['high'][-1] if price_data['high'] else 0,
                    'low': price_data['low'][-1] if price_data['low'] else 0,
                    'open': price_data['open'][-1] if price_data['open'] else 0
                }

            # 更新成交量数据
            for symbol, volume_data in market_data.get('volumes', {}).items():
                if symbol in self._market_data_cache and timestamp in self._market_data_cache[symbol]:
                    self._market_data_cache[symbol][timestamp]['volume'] = volume_data.get('volume', 0)
                    self._market_data_cache[symbol][timestamp]['avg_volume'] = volume_data.get('avg_volume', 0)

            # 限制缓存大小
            max_cache_size = self.risk_config.get('max_market_data_cache', 1000)
            for symbol in list(self._market_data_cache.keys()):
                if len(self._market_data_cache[symbol]) > max_cache_size:
                    # 移除最旧的数据
                    oldest_timestamp = min(self._market_data_cache[symbol].keys())
                    del self._market_data_cache[symbol][oldest_timestamp]

            # 更新风险指标
            self._update_risk_metrics(market_data)

        except Exception as e:
            logger.error(f"市场数据缓存更新失败: {e}")

    def _update_risk_metrics(self, market_data: Dict[str, Any]):
        """更新风险指标"""
        try:
            # 计算波动率估计
            self._volatility_estimates = self._calculate_volatility_estimates(market_data)

            # 计算相关性矩阵
            self._correlation_matrix = self._calculate_correlation_matrix(market_data)

            # 计算流动性指标
            self._liquidity_metrics = self._calculate_liquidity_metrics(market_data)

        except Exception as e:
            logger.error(f"风险指标更新失败: {e}")

    def _calculate_volatility_estimates(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """计算波动率估计"""
        volatility_estimates = {}

        try:
            for symbol, price_data in market_data['prices'].items():
                if 'close' in price_data and len(price_data['close']) >= 20:
                    closes = price_data['close']
                    returns = np.diff(np.log(closes))

                    # 使用GARCH模型或简单历史波动率
                    if len(returns) >= 60 and self.risk_models['volatility']['method'] == 'garch':
                        try:
                            # 简化实现：使用历史波动率
                            volatility = np.std(returns) * np.sqrt(252)
                        except:
                            volatility = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0.3
                    else:
                        volatility = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0.3

                    volatility_estimates[symbol] = float(volatility)
                else:
                    volatility_estimates[symbol] = 0.3  # 默认波动率

            return volatility_estimates

        except Exception as e:
            logger.error(f"波动率计算失败: {e}")
            return {symbol: 0.3 for symbol in market_data['prices'].keys()}

    def _calculate_correlation_matrix(self, market_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """计算相关性矩阵"""
        try:
            symbols = list(market_data['prices'].keys())
            if len(symbols) < 2:
                return None

            # 收集收益数据
            returns_data = {}
            for symbol in symbols:
                if 'close' in market_data['prices'][symbol]:
                    closes = market_data['prices'][symbol]['close']
                    if len(closes) >= 20:
                        returns = np.diff(np.log(closes))
                        returns_data[symbol] = returns[-60:]  # 使用最近60个数据点

            if len(returns_data) < 2:
                return None

            # 创建DataFrame并计算相关性
            df = pd.DataFrame(returns_data)
            correlation_matrix = df.corr()

            # 处理缺失值
            correlation_matrix = correlation_matrix.fillna(0)

            return correlation_matrix

        except Exception as e:
            logger.error(f"相关性矩阵计算失败: {e}")
            return None

    def _calculate_liquidity_metrics(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """计算流动性指标"""
        liquidity_metrics = {}

        try:
            for symbol in market_data.get('volumes', {}).keys():
                volume_data = market_data['volumes'][symbol]
                current_volume = volume_data.get('volume', 0)
                avg_volume = volume_data.get('avg_volume', current_volume)

                if avg_volume > 0:
                    # 流动性比率：当前成交量/平均成交量
                    liquidity_ratio = current_volume / avg_volume
                    # 流动性评分（0-1，1表示最好）
                    liquidity_score = min(max(liquidity_ratio, 0.1), 2.0) / 2.0  # 标准化到0-1
                else:
                    liquidity_score = 0.1  # 默认低流动性

                liquidity_metrics[symbol] = liquidity_score

            return liquidity_metrics

        except Exception as e:
            logger.error(f"流动性指标计算失败: {e}")
            return {}

    def _perform_comprehensive_risk_assessment(self, portfolio_state: PortfolioState,
                                               market_data: Dict[str, Any]) -> RiskAssessment:
        """执行全面风险评估"""
        start_time = time.time()

        try:
            # 并行计算各种风险指标
            with ThreadPoolExecutor(max_workers=4) as executor:
                # 提交各种风险计算任务
                future_var = executor.submit(self._calculate_value_at_risk, portfolio_state, market_data)
                future_es = executor.submit(self._calculate_expected_shortfall, portfolio_state, market_data)
                future_drawdown = executor.submit(self._calculate_max_drawdown, portfolio_state, market_data)
                future_liquidity = executor.submit(self._calculate_liquidity_risk, portfolio_state, market_data)
                future_concentration = executor.submit(self._calculate_concentration_risk, portfolio_state)
                future_leverage = executor.submit(self._calculate_leverage_risk, portfolio_state)
                future_contributions = executor.submit(self._calculate_risk_contributions, portfolio_state, market_data)

                # 等待所有任务完成
                var_result = future_var.result()
                es_result = future_es.result()
                drawdown_result = future_drawdown.result()
                liquidity_risk = future_liquidity.result()
                concentration_risk = future_concentration.result()
                leverage_risk = future_leverage.result()
                risk_contributions = future_contributions.result()

            # 执行压力测试和情景分析
            stress_test_results = self._run_stress_tests(portfolio_state, market_data)
            scenario_analysis = self._run_scenario_analysis(portfolio_state, market_data)

            # 计算综合风险评分
            risk_score = self._calculate_overall_risk_score(
                var_result, es_result, drawdown_result, liquidity_risk,
                concentration_risk, leverage_risk, stress_test_results
            )

            # 确定风险等级
            risk_level = self._determine_risk_level(risk_score)

            # 检查限额违反情况
            limit_breaches = self._check_all_risk_limits(portfolio_state, {
                'value_at_risk': var_result,
                'expected_shortfall': es_result,
                'max_drawdown': drawdown_result,
                'liquidity_risk': liquidity_risk,
                'concentration_risk': concentration_risk,
                'leverage_risk': leverage_risk
            })

            # 生成风险建议
            recommendations = self._generate_risk_recommendations(
                risk_score, risk_level, limit_breaches, stress_test_results
            )

            # 创建风险评估对象
            assessment = RiskAssessment(
                timestamp=datetime.now().isoformat(),
                portfolio_id=portfolio_state.portfolio_id,
                overall_risk_level=risk_level,
                risk_score=risk_score,
                value_at_risk=var_result,
                expected_shortfall=es_result,
                max_drawdown=drawdown_result,
                liquidity_risk=liquidity_risk,
                concentration_risk=concentration_risk,
                leverage_risk=leverage_risk,
                stress_test_results=stress_test_results,
                scenario_analysis=scenario_analysis,
                risk_contributions=risk_contributions,
                limit_breaches=limit_breaches,
                recommendations=recommendations,
                confidence_level=0.95
            )

            processing_time = time.time() - start_time
            logger.info(
                f"全面风险评估完成: 风险等级={risk_level.value}, 评分={risk_score:.2f}, 耗时={processing_time:.3f}s")

            return assessment

        except Exception as e:
            logger.error(f"全面风险评估失败: {e}")
            # 返回默认风险评估
            return self._create_default_risk_assessment(portfolio_state)

    def _calculate_value_at_risk(self, portfolio_state: PortfolioState,
                                 market_data: Dict[str, Any]) -> float:
        """计算在险价值(VaR)"""
        try:
            # 获取组合收益数据
            portfolio_returns = self._calculate_portfolio_returns(portfolio_state, market_data)
            if len(portfolio_returns) < 20:
                return -0.1  # 默认值

            # 根据配置选择计算方法
            method = self.risk_models['var']['method']
            confidence_level = self.risk_models['var']['confidence_level']

            if method == 'historical':
                return self._calculate_historical_var(portfolio_returns, confidence_level)
            elif method == 'parametric':
                return self._calculate_parametric_var(portfolio_returns, confidence_level)
            elif method == 'monte_carlo':
                return self._calculate_monte_carlo_var(portfolio_state, market_data, confidence_level)
            else:
                return self._calculate_historical_var(portfolio_returns, confidence_level)

        except Exception as e:
            logger.error(f"VaR计算失败: {e}")
            return -0.1  # 保守估计

    def _calculate_historical_var(self, returns: pd.Series, confidence_level: float) -> float:
        """计算历史VaR"""
        try:
            if len(returns) < 20:
                return -0.1

            # 计算指定置信水平的分位数
            var = np.percentile(returns, (1 - confidence_level) * 100)
            return float(var)

        except Exception as e:
            logger.error(f"历史VaR计算失败: {e}")
            return -0.1

    def _calculate_parametric_var(self, returns: pd.Series, confidence_level: float) -> float:
        """计算参数法VaR"""
        try:
            if len(returns) < 20:
                return -0.1

            # 正态分布假设
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            z_score = stats.norm.ppf(1 - confidence_level)
            var = mean_return + z_score * std_return
            return float(var)

        except Exception as e:
            logger.error(f"参数法VaR计算失败: {e}")
            return -0.1

    def _calculate_monte_carlo_var(self, portfolio_state: PortfolioState,
                                   market_data: Dict[str, Any], confidence_level: float) -> float:
        """计算蒙特卡洛模拟VaR"""
        try:
            n_simulations = self.risk_models['var']['monte_carlo_simulations']
            if n_simulations < 1000:
                n_simulations = 1000

            # 获取资产收益和协方差矩阵
            asset_returns = self._get_asset_returns(portfolio_state, market_data)
            if asset_returns is None or asset_returns.empty:
                return -0.1

            cov_matrix = asset_returns.cov()
            mean_returns = asset_returns.mean()

            # 生成随机收益
            np.random.seed(42)  # 可重复性
            simulated_returns = np.random.multivariate_normal(
                mean_returns, cov_matrix, n_simulations
            )

            # 计算组合收益
            weights = np.array([alloc.weight for alloc in portfolio_state.allocations.values()])
            portfolio_simulated_returns = simulated_returns @ weights

            # 计算VaR
            var = np.percentile(portfolio_simulated_returns, (1 - confidence_level) * 100)
            return float(var)

        except Exception as e:
            logger.error(f"蒙特卡洛VaR计算失败: {e}")
            return -0.1

    def _calculate_expected_shortfall(self, portfolio_state: PortfolioState,
                                      market_data: Dict[str, Any]) -> float:
        """计算预期短缺(ES)"""
        try:
            # 获取组合收益
            portfolio_returns = self._calculate_portfolio_returns(portfolio_state, market_data)
            if len(portfolio_returns) < 20:
                return -0.15

            # 计算VaR作为阈值
            confidence_level = self.risk_models['expected_shortfall']['confidence_level']
            var = self._calculate_value_at_risk(portfolio_state, market_data)

            # 计算超过VaR的平均损失
            tail_returns = portfolio_returns[portfolio_returns <= var]
            if len(tail_returns) > 0:
                es = np.mean(tail_returns)
            else:
                es = var * 1.2  # 保守估计

            return float(es)

        except Exception as e:
            logger.error(f"预期短缺计算失败: {e}")
            return -0.15

    def _calculate_max_drawdown(self, portfolio_state: PortfolioState,
                                market_data: Dict[str, Any]) -> float:
        """计算最大回撤"""
        try:
            portfolio_returns = self._calculate_portfolio_returns(portfolio_state, market_data)
            if len(portfolio_returns) < 20:
                return -0.2

            # 计算累积收益
            cumulative_returns = np.cumprod(1 + portfolio_returns) - 1

            # 计算回撤
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / (1 + peak)

            max_drawdown = np.min(drawdown)
            return float(max_drawdown)

        except Exception as e:
            logger.error(f"最大回撤计算失败: {e}")
            return -0.2

    def _calculate_liquidity_risk(self, portfolio_state: PortfolioState,
                                  market_data: Dict[str, Any]) -> float:
        """计算流动性风险"""
        try:
            liquidity_scores = []
            total_value = portfolio_state.total_value

            for symbol, allocation in portfolio_state.allocations.items():
                # 获取流动性评分
                liquidity_score = self._liquidity_metrics.get(symbol, 0.5)

                # 考虑头寸规模
                position_value = allocation.current_value
                position_liquidity_risk = (1 - liquidity_score) * (position_value / total_value)

                liquidity_scores.append(position_liquidity_risk)

            if liquidity_scores:
                overall_liquidity_risk = sum(liquidity_scores)
                return float(min(overall_liquidity_risk, 1.0))
            else:
                return 0.3  # 默认中等流动性风险

        except Exception as e:
            logger.error(f"流动性风险计算失败: {e}")
            return 0.3

    def _calculate_concentration_risk(self, portfolio_state: PortfolioState) -> float:
        """计算集中度风险"""
        try:
            weights = [allocation.weight for allocation in portfolio_state.allocations.values()]
            if not weights:
                return 0.0

            # 计算赫芬达尔-赫希曼指数 (HHI)
            hhi = sum(w ** 2 for w in weights)

            # 标准化到0-1范围
            concentration_risk = min(hhi, 1.0)
            return float(concentration_risk)

        except Exception as e:
            logger.error(f"集中度风险计算失败: {e}")
            return 0.5

    def _calculate_leverage_risk(self, portfolio_state: PortfolioState) -> float:
        """计算杠杆风险"""
        try:
            if portfolio_state.total_value > 0:
                leverage_ratio = portfolio_state.leveraged_value / portfolio_state.total_value

                # 风险评分：杠杆比率超过1的部分
                leverage_risk = max(leverage_ratio - 1, 0) / 2.0  # 假设最大杠杆为3倍
                return float(min(leverage_risk, 1.0))
            return 0.0

        except Exception as e:
            logger.error(f"杠杆风险计算失败: {e}")
            return 0.0

    def _calculate_risk_contributions(self, portfolio_state: PortfolioState,
                                      market_data: Dict[str, Any]) -> Dict[str, float]:
        """计算风险贡献度"""
        try:
            risk_contributions = {}

            # 获取协方差矩阵
            if self._correlation_matrix is None:
                return risk_contributions

            # 计算组合方差
            weights = np.array([alloc.weight for alloc in portfolio_state.allocations.values()])
            symbols = list(portfolio_state.allocations.keys())

            # 确保协方差矩阵维度匹配
            if len(weights) != self._correlation_matrix.shape[0]:
                return risk_contributions

            portfolio_variance = weights.T @ self._correlation_matrix @ weights

            if portfolio_variance > 0:
                # 计算边际风险贡献
                marginal_risk = self._correlation_matrix @ weights / np.sqrt(portfolio_variance)

                # 计算风险贡献
                for i, symbol in enumerate(symbols):
                    risk_contribution = weights[i] * marginal_risk[i]
                    risk_contributions[symbol] = float(risk_contribution)

            return risk_contributions

        except Exception as e:
            logger.error(f"风险贡献度计算失败: {e}")
            return {}

    def _run_stress_tests(self, portfolio_state: PortfolioState,
                          market_data: Dict[str, Any]) -> Dict[str, float]:
        """运行压力测试"""
        stress_test_results = {}

        try:
            for scenario_id, scenario in self.stress_test_scenarios.items():
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

    def _run_single_stress_test(self, scenario: StressTestScenario,
                                portfolio_state: PortfolioState,
                                market_data: Dict[str, Any]) -> float:
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

    def _run_scenario_analysis(self, portfolio_state: PortfolioState,
                               market_data: Dict[str, Any]) -> Dict[str, float]:
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

    def _calculate_overall_risk_score(self, var: float, es: float, max_drawdown: float,
                                      liquidity_risk: float, concentration_risk: float,
                                      leverage_risk: float, stress_test_results: Dict[str, float]) -> float:
        """计算综合风险评分"""
        try:
            # 权重配置
            weights = {
                'var': 0.25,
                'es': 0.20,
                'max_drawdown': 0.15,
                'liquidity_risk': 0.10,
                'concentration_risk': 0.10,
                'leverage_risk': 0.10,
                'stress_test': 0.10
            }

            # 标准化各项风险指标
            var_score = min(abs(var) / 0.2, 1.0)  # VaR最大20%
            es_score = min(abs(es) / 0.25, 1.0)  # ES最大25%
            drawdown_score = min(abs(max_drawdown) / 0.3, 1.0)  # 回撤最大30%

            # 计算压力测试平均得分
            stress_scores = [min(abs(result) / 0.4, 1.0) for result in stress_test_results.values()]
            avg_stress_score = np.mean(stress_scores) if stress_scores else 0.5

            # 计算加权风险评分
            risk_score = (
                    weights['var'] * var_score +
                    weights['es'] * es_score +
                    weights['max_drawdown'] * drawdown_score +
                    weights['liquidity_risk'] * liquidity_risk +
                    weights['concentration_risk'] * concentration_risk +
                    weights['leverage_risk'] * leverage_risk +
                    weights['stress_test'] * avg_stress_score
            )

            return float(min(risk_score * 100, 100))  # 转换为0-100分

        except Exception as e:
            logger.error(f"综合风险评分计算失败: {e}")
            return 50.0  # 中等风险

    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """确定风险等级"""
        if risk_score >= 90:
            return RiskLevel.BLACK_SWAN
        elif risk_score >= 75:
            return RiskLevel.EXTREME
        elif risk_score >= 60:
            return RiskLevel.VERY_HIGH
        elif risk_score >= 45:
            return RiskLevel.HIGH
        elif risk_score >= 30:
            return RiskLevel.MODERATE
        elif risk_score >= 15:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW

    def _check_all_risk_limits(self, portfolio_state: PortfolioState,
                               risk_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
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

    def _generate_risk_recommendations(self, risk_score: float, risk_level: RiskLevel,
                                       limit_breaches: List[Dict[str, Any]],
                                       stress_test_results: Dict[str, float]) -> List[Dict[str, Any]]:
        """生成风险建议"""
        recommendations = []

        try:
            # 基于风险等级的建议
            if risk_level in [RiskLevel.VERY_HIGH, RiskLevel.EXTREME, RiskLevel.BLACK_SWAN]:
                recommendations.append({
                    'type': 'risk_level',
                    'priority': 'critical',
                    'recommendation': '立即降低整体风险暴露',
                    'action': '减少高风险资产头寸，增加对冲策略',
                    'expected_impact': '高'
                })

            # 基于限额违反的建议
            for breach in limit_breaches:
                if breach['severity'] == 'critical':
                    recommendations.append({
                        'type': 'limit_breach',
                        'priority': 'high',
                        'recommendation': f"紧急处理限额违反: {breach['limit_type']}",
                        'action': breach['suggested_action'],
                        'expected_impact': '高'
                    })

            # 基于压力测试的建议
            worst_stress_test = min(stress_test_results.values()) if stress_test_results else 0
            if worst_stress_test < -0.25:
                recommendations.append({
                    'type': 'stress_test',
                    'priority': 'medium',
                    'recommendation': '压力测试显示极端情景下损失较大',
                    'action': '增加压力测试保护策略，考虑尾部风险对冲',
                    'expected_impact': '中'
                })

            return recommendations

        except Exception as e:
            logger.error(f"风险建议生成失败: {e}")
            return []

    def _create_default_risk_assessment(self, portfolio_state: PortfolioState) -> RiskAssessment:
        """创建默认风险评估"""
        return RiskAssessment(
            timestamp=datetime.now().isoformat(),
            portfolio_id=portfolio_state.portfolio_id,
            overall_risk_level=RiskLevel.MODERATE,
            risk_score=50.0,
            value_at_risk=-0.1,
            expected_shortfall=-0.15,
            max_drawdown=-0.2,
            liquidity_risk=0.3,
            concentration_risk=0.5,
            leverage_risk=0.0,
            stress_test_results={'default': -0.25},
            scenario_analysis={},
            risk_contributions={},
            limit_breaches=[],
            recommendations=[],
            confidence_level=0.8
        )

    def _calculate_portfolio_returns(self, portfolio_state: PortfolioState,
                                     market_data: Dict[str, Any]) -> pd.Series:
        """计算组合收益序列"""
        try:
            # 获取组合中所有资产
            symbols = list(portfolio_state.allocations.keys())
            if not symbols:
                logger.warning("组合中没有资产")
                return pd.Series()

            # 获取价格数据并确保时间对齐
            price_data = {}
            min_length = float('inf')

            for symbol in symbols:
                if symbol in market_data['prices']:
                    closes = market_data['prices'][symbol].get('close', [])
                    if len(closes) > 0:
                        price_data[symbol] = closes
                        min_length = min(min_length, len(closes))
                    else:
                        logger.warning(f"符号 {symbol} 没有价格数据")
                        return pd.Series()
                else:
                    logger.warning(f"市场数据中缺少符号 {symbol}")
                    return pd.Series()

            # 确保所有价格序列长度一致
            if min_length < 2:
                logger.warning("价格数据不足")
                return pd.Series()

            # 截取相同长度的价格序列
            aligned_prices = {}
            for symbol, prices in price_data.items():
                aligned_prices[symbol] = prices[-min_length:]

            # 计算对数收益
            returns_data = {}
            for symbol, prices in aligned_prices.items():
                log_returns = np.diff(np.log(prices))
                returns_data[symbol] = log_returns

            # 创建DataFrame
            returns_df = pd.DataFrame(returns_data)

            # 获取权重
            weights = np.array([portfolio_state.allocations[symbol].weight for symbol in symbols])

            # 计算加权组合收益
            portfolio_returns = returns_df.dot(weights)

            # 转换为Series并设置时间索引
            if 'timestamp' in market_data and len(market_data['timestamp']) >= len(portfolio_returns):
                # 使用最后一个时间戳作为索引（假设时间戳是升序排列）
                timestamps = market_data['timestamp'][-len(portfolio_returns):]
                portfolio_returns = pd.Series(portfolio_returns.values, index=timestamps)
            else:
                # 使用数值索引作为备选
                portfolio_returns = pd.Series(portfolio_returns.values)

            logger.debug(f"组合收益计算完成: 数据点={len(portfolio_returns)}, 期间={len(portfolio_returns)}天")
            return portfolio_returns

        except Exception as e:
            logger.error(f"组合收益计算失败: {e}")
            return pd.Series()

    def _get_asset_returns(self, portfolio_state: PortfolioState,
                           market_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """获取资产收益数据"""
        try:
            symbols = list(portfolio_state.allocations.keys())
            returns_data = {}

            for symbol in symbols:
                if symbol in market_data['prices']:
                    prices = market_data['prices'][symbol].get('close', [])
                    if len(prices) >= 20:  # 至少需要20个数据点
                        returns = np.diff(np.log(prices))
                        returns_data[symbol] = returns
                    else:
                        logger.warning(f"符号 {symbol} 价格数据不足")
                        return None
                else:
                    logger.warning(f"市场数据中缺少符号 {symbol}")
                    return None

            if not returns_data:
                return None

            # 找到最小长度
            min_length = min(len(returns) for returns in returns_data.values())
            if min_length < 10:
                logger.warning("收益数据不足")
                return None

            # 对齐数据长度
            aligned_returns = {}
            for symbol, returns in returns_data.items():
                aligned_returns[symbol] = returns[-min_length:]

            # 创建DataFrame
            returns_df = pd.DataFrame(aligned_returns)

            return returns_df

        except Exception as e:
            logger.error(f"资产收益数据获取失败: {e}")
            return None

    def _simulate_market_crash(self, scenario: StressTestScenario,
                               portfolio_state: PortfolioState,
                               market_data: Dict[str, Any]) -> float:
        """模拟市场崩盘情景"""
        try:
            # 获取场景参数
            crash_severity = scenario.parameters.get('severity', 0.3)  # 默认30%下跌
            duration_days = scenario.parameters.get('duration_days', 10)
            recovery_days = scenario.parameters.get('recovery_days', 30)

            # 计算受影响资产
            equity_assets = []
            for symbol, allocation in portfolio_state.allocations.items():
                if allocation.asset_class == 'equity':
                    equity_assets.append(symbol)

            if not equity_assets:
                return 0.0

            # 计算股票资产暴露
            equity_exposure = sum(
                portfolio_state.allocations[symbol].weight
                for symbol in equity_assets
            )

            # 计算预期损失
            # 假设崩盘期间线性下跌，恢复期间部分恢复
            crash_loss = equity_exposure * crash_severity
            recovery_gain = crash_loss * 0.3  # 恢复30%
            net_loss = crash_loss - recovery_gain

            return float(-net_loss)

        except Exception as e:
            logger.error(f"市场崩盘模拟失败: {e}")
            return -0.25

    def _simulate_liquidity_crisis(self, scenario: StressTestScenario,
                                   portfolio_state: PortfolioState,
                                   market_data: Dict[str, Any]) -> float:
        """模拟流动性危机情景"""
        try:
            # 获取场景参数
            liquidity_shock = scenario.parameters.get('liquidity_shock', 0.5)  # 流动性减少50%
            bid_ask_spread_increase = scenario.parameters.get('bid_ask_spread_increase', 0.02)  # 点差增加2%

            # 计算流动性风险
            liquidity_impact = 0.0
            for symbol, allocation in portfolio_state.allocations.items():
                # 获取流动性评分
                liquidity_score = self._liquidity_metrics.get(symbol, 0.5)

                # 计算冲击后的流动性
                shocked_liquidity = liquidity_score * (1 - liquidity_shock)

                # 计算交易成本增加
                spread_impact = bid_ask_spread_increase * allocation.weight

                # 总影响
                asset_impact = (1 - shocked_liquidity) * allocation.weight + spread_impact
                liquidity_impact += asset_impact

            return float(-liquidity_impact)

        except Exception as e:
            logger.error(f"流动性危机模拟失败: {e}")
            return -0.15

    def _simulate_interest_rate_shock(self, scenario: StressTestScenario,
                                      portfolio_state: PortfolioState,
                                      market_data: Dict[str, Any]) -> float:
        """模拟利率冲击情景"""
        try:
            rate_shock = scenario.parameters.get('rate_shock', 0.02)  # 利率上升2%

            # 识别对利率敏感的资产
            rate_sensitive_assets = []
            for symbol, allocation in portfolio_state.allocations.items():
                asset_class = allocation.asset_class
                if asset_class in ['bond', 'fixed_income', 'real_estate', 'utilities']:
                    rate_sensitive_assets.append(symbol)

            if not rate_sensitive_assets:
                return 0.0

            # 计算敏感资产暴露
            sensitive_exposure = sum(
                portfolio_state.allocations[symbol].weight
                for symbol in rate_sensitive_assets
            )

            # 计算利率冲击影响（简化模型）
            # 假设久期为5年，利率上升2%导致价格下跌10%
            duration = scenario.parameters.get('duration', 5)
            price_impact = -duration * rate_shock

            total_impact = sensitive_exposure * price_impact
            return float(total_impact)

        except Exception as e:
            logger.error(f"利率冲击模拟失败: {e}")
            return -0.1

    def _simulate_correlation_breakdown(self, scenario: StressTestScenario,
                                        portfolio_state: PortfolioState,
                                        market_data: Dict[str, Any]) -> float:
        """模拟相关性断裂情景"""
        try:
            correlation_increase = scenario.parameters.get('correlation_increase', 0.5)  # 相关性增加0.5

            # 获取当前相关性矩阵
            if self._correlation_matrix is None:
                return -0.1

            # 计算冲击后的相关性（所有资产相关性增加）
            shocked_correlation = self._correlation_matrix.copy()
            shocked_correlation.values[shocked_correlation.values < 1.0] += correlation_increase
            shocked_correlation = shocked_correlation.clip(upper=0.99)  # 避免完全相关

            # 计算组合方差变化
            weights = np.array([alloc.weight for alloc in portfolio_state.allocations.values()])

            original_variance = weights.T @ self._correlation_matrix @ weights
            shocked_variance = weights.T @ shocked_correlation @ weights

            # 计算波动率增加带来的额外风险
            vol_increase = np.sqrt(shocked_variance) - np.sqrt(original_variance)
            risk_impact = -vol_increase * 2.5  # 转换为损失估计

            return float(risk_impact)

        except Exception as e:
            logger.error(f"相关性断裂模拟失败: {e}")
            return -0.2

    def _simulate_generic_stress(self, scenario: StressTestScenario,
                                 portfolio_state: PortfolioState,
                                 market_data: Dict[str, Any]) -> float:
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

    def _simulate_scenario(self, scenario_params: Dict[str, Any],
                           portfolio_state: PortfolioState,
                           market_data: Dict[str, Any]) -> float:
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

    def _check_market_risk_limits(self, risk_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """检查市场风险限额"""
        breaches = []

        try:
            # VaR限额检查
            var_limit = self.risk_limits.get('market_risk_value_at_risk')
            if var_limit and 'value_at_risk' in risk_metrics:
                var_value = risk_metrics['value_at_risk']
                if var_value < var_limit.threshold:
                    breaches.append({
                        'limit_type': 'value_at_risk',
                        'metric': 'value_at_risk',
                        'current_value': var_value,
                        'threshold': var_limit.threshold,
                        'breach_amount': abs(var_value - var_limit.threshold),
                        'severity': 'critical' if var_value < var_limit.threshold * 1.5 else 'high',
                        'suggested_action': '减少高风险头寸，增加对冲',
                        'time_horizon': var_limit.time_horizon
                    })

            # ES限额检查
            es_limit = self.risk_limits.get('market_risk_expected_shortfall')
            if es_limit and 'expected_shortfall' in risk_metrics:
                es_value = risk_metrics['expected_shortfall']
                if es_value < es_limit.threshold:
                    breaches.append({
                        'limit_type': 'expected_shortfall',
                        'metric': 'expected_shortfall',
                        'current_value': es_value,
                        'threshold': es_limit.threshold,
                        'breach_amount': abs(es_value - es_limit.threshold),
                        'severity': 'critical' if es_value < es_limit.threshold * 1.5 else 'high',
                        'suggested_action': '加强尾部风险保护',
                        'time_horizon': es_limit.time_horizon
                    })

            return breaches

        except Exception as e:
            logger.error(f"市场风险限额检查失败: {e}")
            return []

    def _check_credit_risk_limits(self, portfolio_state: PortfolioState) -> List[Dict[str, Any]]:
        """检查信用风险限额"""
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

    def _check_liquidity_risk_limits(self, portfolio_state: PortfolioState,
                                     risk_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
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

    def _check_concentration_limits(self, portfolio_state: PortfolioState) -> List[Dict[str, Any]]:
        """检查集中度限额"""
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
                sector_limit = self.limits_config.get('sector_limits', {}).get(sector, 0.3)
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

    def _check_leverage_limits(self, portfolio_state: PortfolioState) -> List[Dict[str, Any]]:
        """检查杠杆限额"""
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

    def _update_risk_state(self, risk_assessment: RiskAssessment,
                           limit_breaches: List[Dict[str, Any]],
                           recommendations: List[Dict[str, Any]]):
        """更新风险状态"""
        try:
            # 保存风险评估历史
            self.risk_history.append(risk_assessment)

            # 限制历史记录长度
            max_history = self.risk_config.get('max_risk_history', 1000)
            if len(self.risk_history) > max_history:
                self.risk_history = self.risk_history[-max_history:]

            # 更新当前风险评估
            self.current_risk_assessment = risk_assessment

            # 记录风险事件
            critical_breaches = [b for b in limit_breaches if b['severity'] in ['critical', 'high']]
            if critical_breaches:
                for breach in critical_breaches:
                    risk_event = RiskEvent(
                        event_id=f"risk_event_{int(time.time())}_{hash(breach['limit_type'])}",
                        event_type=RiskType.MARKET_RISK,  # 根据实际情况调整
                        severity=RiskLevel.HIGH if breach['severity'] == 'high' else RiskLevel.CRITICAL,
                        timestamp=datetime.now().isoformat(),
                        description=f"{breach['limit_type']} 限额违反",
                        triggered_by=breach['metric'],
                        impact_assessment={'breach_amount': breach['breach_amount']},
                        action_taken=RiskControlAction.WARN,
                        resolved=False
                    )
                    self.active_risk_events.append(risk_event)

            logger.info(f"风险状态更新: 等级={risk_assessment.overall_risk_level.value}, "
                        f"限额违反={len(limit_breaches)}, 建议={len(recommendations)}")

        except Exception as e:
            logger.error(f"风险状态更新失败: {e}")

    def _update_performance_stats(self, processing_time: float, breach_count: int):
        """更新性能统计"""
        try:
            self.performance_stats['risk_assessments_performed'] += 1
            self.performance_stats['limit_checks_performed'] += breach_count
            self.performance_stats['risk_events_detected'] += breach_count

            # 更新平均处理时间
            prev_total = self.performance_stats['avg_assessment_time'] * (
                    self.performance_stats['risk_assessments_performed'] - 1
            )
            self.performance_stats['avg_assessment_time'] = (prev_total + processing_time) / \
                                                            self.performance_stats['risk_assessments_performed']

            # 更新最大响应时间
            self.performance_stats['max_response_time'] = max(
                self.performance_stats['max_response_time'],
                processing_time
            )

        except Exception as e:
            logger.error(f"性能统计更新失败: {e}")

    def get_risk_dashboard(self) -> Dict[str, Any]:
        """获取风险仪表板数据"""
        try:
            if not self.current_risk_assessment:
                return {'error': 'No risk assessment available'}

            return {
                'current_risk': self.current_risk_assessment.to_dict(),
                'performance_stats': self.performance_stats,
                'active_events': [event.to_dict() for event in self.active_risk_events],
                'recent_assessments': [assessment.to_dict() for assessment in self.risk_history[-10:]],
                'limit_utilization': self._calculate_limit_utilization(),
                'risk_trends': self._calculate_risk_trends(),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"风险仪表板生成失败: {e}")
            return {'error': str(e)}

    def _calculate_limit_utilization(self) -> Dict[str, float]:
        """计算限额使用率"""
        utilization = {}

        try:
            if not self.current_risk_assessment:
                return utilization

            # 计算各种限额的使用率
            risk_metrics = {
                'value_at_risk': self.current_risk_assessment.value_at_risk,
                'expected_shortfall': self.current_risk_assessment.expected_shortfall,
                'liquidity_risk': self.current_risk_assessment.liquidity_risk,
                'concentration_risk': self.current_risk_assessment.concentration_risk,
                'leverage_risk': self.current_risk_assessment.leverage_risk
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

    def _calculate_risk_trends(self) -> Dict[str, Any]:
        """计算风险趋势"""
        trends = {}

        try:
            if len(self.risk_history) < 2:
                return trends

            # 获取最近的风险评估
            recent_assessments = self.risk_history[-10:]  # 最近10次评估

            # 计算各项指标的趋势
            metrics = ['risk_score', 'value_at_risk', 'expected_shortfall',
                       'liquidity_risk', 'concentration_risk', 'leverage_risk']

            for metric in metrics:
                values = [getattr(assessment, metric) for assessment in recent_assessments]
                if len(values) >= 2:
                    # 计算线性趋势
                    x = np.arange(len(values))
                    slope = np.polyfit(x, values, 1)[0]
                    trends[metric] = {
                        'current': values[-1],
                        'trend': 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable',
                        'slope': slope,
                        'volatility': np.std(values)
                    }

            return trends

        except Exception as e:
            logger.error(f"风险趋势计算失败: {e}")
            return {}

    def cleanup(self):
        """清理资源"""
        try:
            # 关闭线程池
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)

            # 清理缓存
            self._market_data_cache.clear()
            self._correlation_matrix = None
            self._volatility_estimates.clear()
            self._liquidity_metrics.clear()

            # 保存风险评估历史
            self._save_risk_history()

            logger.info("风险管理系统资源清理完成")

        except Exception as e:
            logger.error(f"资源清理失败: {e}")

    def _save_risk_history(self):
        """保存风险评估历史"""
        try:
            if not self.risk_history:
                return

            # 保存最近100次评估
            recent_history = self.risk_history[-100:]
            history_data = [assessment.to_dict() for assessment in recent_history]

            # 这里应该保存到数据库或文件
            # 简化实现：记录到日志
            logger.info(f"保存 {len(recent_history)} 个风险评估记录")

        except Exception as e:
            logger.error(f"风险评估历史保存失败: {e}")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.cleanup()

    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except:
            pass  # 避免析构函数中的异常

# 辅助函数
def calculate_risk_metrics(returns: pd.Series,
                           confidence_level: float = 0.95,
                           time_horizon: int = 1) -> Dict[str, float]:
    """计算基本风险指标"""
    try:
        if len(returns) < 20:
            return {}

        metrics = {}

        # 基本统计
        metrics['mean_return'] = float(np.mean(returns))
        metrics['volatility'] = float(np.std(returns) * np.sqrt(252))
        metrics['sharpe_ratio'] = float(
            metrics['mean_return'] / metrics['volatility'] * np.sqrt(252) if metrics['volatility'] > 0 else 0)

        # 风险指标
        metrics['value_at_risk'] = float(np.percentile(returns, (1 - confidence_level) * 100))
        metrics['expected_shortfall'] = float(np.mean(returns[returns <= metrics['value_at_risk']]) if len(
            returns[returns <= metrics['value_at_risk']]) > 0 else metrics['value_at_risk'])
        metrics['max_drawdown'] = float(calculate_max_drawdown(returns))
        metrics['skewness'] = float(stats.skew(returns))
        metrics['kurtosis'] = float(stats.kurtosis(returns))

        # 时间调整
        if time_horizon > 1:
            metrics['value_at_risk'] *= np.sqrt(time_horizon)
            metrics['expected_shortfall'] *= np.sqrt(time_horizon)
            metrics['volatility'] *= np.sqrt(time_horizon)

        return metrics

    except Exception as e:
        logger.error(f"风险指标计算失败: {e}")
        return {}

def calculate_max_drawdown(returns: pd.Series) -> float:
    """计算最大回撤"""
    try:
        cumulative_returns = np.cumprod(1 + returns) - 1
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / (1 + peak)
        return float(np.min(drawdown))
    except:
        return 0.0

def simulate_stress_scenario(portfolio_weights: Dict[str, float],
                             scenario_returns: Dict[str, float]) -> float:
    """模拟压力情景下的组合收益"""
    try:
        portfolio_return = 0.0
        for symbol, weight in portfolio_weights.items():
            asset_return = scenario_returns.get(symbol, 0.0)
            portfolio_return += weight * asset_return
        return portfolio_return
    except:
        return 0.0

def validate_risk_parameters(parameters: Dict[str, Any]) -> bool:
    """验证风险参数有效性"""
    try:
        required_params = ['confidence_level', 'time_horizon', 'calculation_method']
        if not all(param in parameters for param in required_params):
            return False

        if not (0 < parameters['confidence_level'] < 1):
            return False

        if parameters['time_horizon'] <= 0:
            return False

        valid_methods = ['historical', 'parametric', 'monte_carlo']
        if parameters['calculation_method'] not in valid_methods:
            return False

        return True

    except:
        return False

class RiskMonitor:
    """独立风险监控器 - 用于实时风险监控"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_thresholds = config.get('risk_thresholds', {})
        self.alert_handlers = []
        self.is_monitoring = False
        self.monitoring_thread = None

    def add_alert_handler(self, handler):
        """添加警报处理器"""
        self.alert_handlers.append(handler)

    def start_monitoring(self):
        """开始风险监控"""
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """停止风险监控"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 执行风险检查
                risk_status = self._check_risk_status()

                # 触发警报
                if risk_status['alert_level'] > 0:
                    self._trigger_alerts(risk_status)

                # 等待下一次检查
                time.sleep(self.config.get('monitoring_interval', 60))

            except Exception as e:
                logger.error(f"风险监控循环错误: {e}")
                time.sleep(10)  # 错误后等待10秒

if __name__ == "__main__":
    # 测试代码
    config = {
        'risk_management': {
            'limits': {
                'risk_limits': [
                    {
                        'risk_type': 'market_risk',
                        'metric': 'value_at_risk',
                        'threshold': -0.05,
                        'time_horizon': '1d',
                        'confidence_level': 0.95,
                        'action': 'warn'
                    }
                ],
                'position_limits': {
                    'AAPL': {
                        'max_notional': 100000,
                        'max_quantity': 1000,
                        'max_weight': 0.2,
                        'min_liquidity_ratio': 0.1
                    }
                }
            },
            'monitoring': {
                'interval_seconds': 60,
                'alert_thresholds': {
                    'risk_score': 70,
                    'var_breach': 0.1
                }
            }
        }
    }

    # 创建风险管理器实例
    risk_manager = RiskManager(config)

    # 测试风险指标计算
    test_returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    risk_metrics = calculate_risk_metrics(test_returns)

    print("风险指标测试结果:")
    for metric, value in risk_metrics.items():
        print(f"{metric}: {value:.4f}")

    # 清理
    risk_manager.cleanup()