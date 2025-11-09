"""
DeepSeekQuant 组合管理器
负责投资组合的构建、优化、再平衡和风险管理
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import time
import json
import scipy.optimize as opt
from scipy import stats
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib
import copy
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import cvxpy as cp
import riskfolio as rp
import yfinance as yf
import quantstats as qs
from pypfopt import expected_returns, risk_models, EfficientFrontier, objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import LedoitWolf, OAS
import traceback
from collections import deque, defaultdict
import heapq
import pickle
import zlib
import base64

# 导入内部模块
from .base_processor import BaseProcessor
from ..utils.helpers import validate_data, calculate_returns, normalize_data, calculate_correlations
from ..utils.validators import validate_portfolio, validate_allocation
from ..utils.performance import calculate_portfolio_performance, calculate_risk_metrics
from ..core.signal_engine import SignalType, SignalStrength

logger = logging.getLogger('DeepSeekQuant.PortfolioManager')


class AllocationMethod(Enum):
    """资产配置方法枚举"""
    EQUAL_WEIGHT = "equal_weight"  # 等权重
    MARKET_CAP = "market_cap"  # 市值加权
    MIN_VARIANCE = "min_variance"  # 最小方差
    MAX_SHARPE = "max_sharpe"  # 最大夏普比率
    RISK_PARITY = "risk_parity"  # 风险平价
    BLACK_LITTERMAN = "black_litterman"  # Black-Litterman模型
    HRP = "hierarchical_risk_parity"  # 分层风险平价
    CLA = "critical_line_algorithm"  # 关键线算法
    CUSTOM = "custom"  # 自定义配置


class RebalanceFrequency(Enum):
    """再平衡频率枚举"""
    DAILY = "daily"  # 每日
    WEEKLY = "weekly"  # 每周
    MONTHLY = "monthly"  # 每月
    QUARTERLY = "quarterly"  # 每季度
    SEMI_ANNUAL = "semi_annual"  # 每半年
    ANNUAL = "annual"  # 每年
    AD_HOC = "ad_hoc"  # 临时调整
    SIGNAL_BASED = "signal_based"  # 基于信号


class RiskModel(Enum):
    """风险模型枚举"""
    SAMPLE_COVARIANCE = "sample_covariance"  # 样本协方差
    LEDOIT_WOLF = "ledoit_wolf"  # Ledoit-Wolf收缩
    ORACLE_APPROXIMATING = "oracle_approximating"  # Oracle近似收缩
    CONSTANT_CORRELATION = "constant_correlation"  # 常数相关
    EXPONENTIALLY_WEIGHTED = "exponentially_weighted"  # 指数加权
    GARCH = "garch"  # GARCH模型
    DCC_GARCH = "dcc_garch"  # DCC-GARCH模型


class PortfolioObjective(Enum):
    """组合优化目标枚举"""
    MAXIMIZE_RETURN = "maximize_return"  # 最大化收益
    MINIMIZE_RISK = "minimize_risk"  # 最小化风险
    MAXIMIZE_SHARPE = "maximize_sharpe"  # 最大化夏普比率
    MAXIMIZE_SORTINO = "maximize_sortino"  # 最大化索提诺比率
    MAXIMIZE_OMEGA = "maximize_omega"  # 最大化Omega比率
    MINIMIZE_CVAR = "minimize_cvar"  # 最小化条件风险价值
    MAXIMIZE_UTILITY = "maximize_utility"  # 最大化效用函数
    TRACK_ERROR = "track_error"  # 最小化跟踪误差


@dataclass
class PortfolioConstraints:
    """组合约束条件"""
    max_asset_weight: float = 0.2  # 单资产最大权重
    min_asset_weight: float = 0.0  # 单资产最小权重
    max_sector_weight: float = 0.3  # 单行业最大权重
    max_turnover: float = 0.2  # 最大换手率
    leverage_limit: float = 1.0  # 杠杆限制
    short_selling_limit: float = 0.0  # 卖空限制
    liquidity_constraints: Dict[str, float] = field(default_factory=dict)  # 流动性约束
    concentration_limit: float = 0.5  # 集中度限制
    risk_budget: Dict[str, float] = field(default_factory=dict)  # 风险预算
    trading_cost: float = 0.001  # 交易成本
    tax_consideration: bool = False  # 税务考虑
    regulatory_constraints: List[str] = field(default_factory=list)  # 监管约束


@dataclass
class PortfolioMetadata:
    """组合元数据"""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_rebalanced: str = field(default_factory=lambda: datetime.now().isoformat())
    optimization_method: AllocationMethod = AllocationMethod.MAX_SHARPE
    risk_model: RiskModel = RiskModel.LEDOIT_WOLF
    objective: PortfolioObjective = PortfolioObjective.MAXIMIZE_SHARPE
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)
    backtest_period: str = "3y"  # 回看期
    expected_return: float = 0.0  # 预期收益
    expected_risk: float = 0.0  # 预期风险
    sharpe_ratio: float = 0.0  # 夏普比率
    max_drawdown: float = 0.0  # 最大回撤
    diversification: float = 0.0  # 分散度
    turnover_rate: float = 0.0  # 换手率
    risk_parity_score: float = 0.0  # 风险平价得分
    liquidity_score: float = 1.0  # 流动性评分
    stress_test_passed: bool = True  # 压力测试结果
    regulatory_compliant: bool = True  # 监管合规性


@dataclass
class AssetAllocation:
    """资产配置"""
    symbol: str
    weight: float
    target_weight: float
    current_value: float
    target_value: float
    notional: float
    sector: str = ""
    asset_class: str = "equity"
    region: str = "domestic"
    currency: str = "USD"
    liquidity_tier: int = 1  # 流动性分级 1-5, 1最高
    risk_contribution: float = 0.0  # 风险贡献度
    marginal_risk: float = 0.0  # 边际风险
    expected_return: float = 0.0  # 预期收益
    expected_risk: float = 0.0  # 预期风险
    transaction_cost: float = 0.0  # 交易成本
    tax_implication: float = 0.0  # 税务影响
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioState:
    """组合状态"""
    portfolio_id: str
    total_value: float
    cash_balance: float
    leveraged_value: float
    allocations: Dict[str, AssetAllocation]
    metadata: PortfolioMetadata
    performance: Dict[str, float]
    risk_metrics: Dict[str, float]
    constraints: PortfolioConstraints
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "active"
    version: int = 1
    benchmark: str = "SPY"
    tracking_error: float = 0.0
    active_share: float = 0.0
    concentration: float = 0.0
    liquidity_metrics: Dict[str, float] = field(default_factory=dict)
    stress_test_results: Dict[str, Any] = field(default_factory=dict)
    regulatory_compliance: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'portfolio_id': self.portfolio_id,
            'total_value': self.total_value,
            'cash_balance': self.cash_balance,
            'leveraged_value': self.leveraged_value,
            'allocations': {symbol: asdict(alloc) for symbol, alloc in self.allocations.items()},
            'metadata': asdict(self.metadata),
            'performance': self.performance,
            'risk_metrics': self.risk_metrics,
            'constraints': asdict(self.constraints),
            'timestamp': self.timestamp,
            'status': self.status,
            'version': self.version,
            'benchmark': self.benchmark,
            'tracking_error': self.tracking_error,
            'active_share': self.active_share,
            'concentration': self.concentration,
            'liquidity_metrics': self.liquidity_metrics,
            'stress_test_results': self.stress_test_results,
            'regulatory_compliance': self.regulatory_compliance
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PortfolioState':
        """从字典创建组合状态"""
        allocations = {}
        for symbol, alloc_data in data['allocations'].items():
            allocations[symbol] = AssetAllocation(**alloc_data)

        metadata_data = data['metadata']
        constraints_data = data['constraints']

        return cls(
            portfolio_id=data['portfolio_id'],
            total_value=data['total_value'],
            cash_balance=data['cash_balance'],
            leveraged_value=data['leveraged_value'],
            allocations=allocations,
            metadata=PortfolioMetadata(**metadata_data),
            performance=data['performance'],
            risk_metrics=data['risk_metrics'],
            constraints=PortfolioConstraints(**constraints_data),
            timestamp=data['timestamp'],
            status=data['status'],
            version=data['version'],
            benchmark=data['benchmark'],
            tracking_error=data['tracking_error'],
            active_share=data['active_share'],
            concentration=data['concentration'],
            liquidity_metrics=data['liquidity_metrics'],
            stress_test_results=data['stress_test_results'],
            regulatory_compliance=data['regulatory_compliance']
        )


class PortfolioManager(BaseProcessor):
    """组合管理器 - 负责投资组合的构建和优化"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化组合管理器

        Args:
            config: 配置字典
        """
        super().__init__(config)

        # 组合管理配置
        self.portfolio_config = config.get('portfolio_management', {})
        self.optimization_config = self.portfolio_config.get('optimization', {})
        self.risk_config = self.portfolio_config.get('risk_management', {})
        self.rebalance_config = self.portfolio_config.get('rebalancing', {})

        # 组合状态
        self.current_portfolio: Optional[PortfolioState] = None
        self.target_portfolio: Optional[PortfolioState] = None
        self.portfolio_history: List[PortfolioState] = []
        self.optimization_results: Dict[str, Any] = {}

        # 市场数据缓存
        self.market_data_cache: Dict[str, Dict] = {}
        self.covariance_matrix: Optional[pd.DataFrame] = None
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.expected_returns: Optional[pd.Series] = None

        # 性能统计
        self.performance_stats = {
            'optimizations_performed': 0,
            'rebalances_executed': 0,
            'avg_optimization_time': 0.0,
            'avg_rebalance_time': 0.0,
            'total_turnover': 0.0,
            'avg_tracking_error': 0.0,
            'max_drawdown': 0.0,
            'best_sharpe_ratio': 0.0,
            'worst_drawdown': 0.0
        }

        # 优化器缓存
        self._optimizer_cache: Dict[str, Any] = {}
        self._constraint_cache: Dict[str, Any] = {}
        self._risk_model_cache: Dict[str, Any] = {}

        # 线程池
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.portfolio_config.get('max_optimization_workers', 4)
        )

        # 初始化优化器
        self._initialize_optimizers()

        logger.info("组合管理器初始化完成")

    def _initialize_optimizers(self):
        """初始化优化器"""
        self.optimizers = {
            AllocationMethod.EQUAL_WEIGHT: self._equal_weight_optimization,
            AllocationMethod.MARKET_CAP: self._market_cap_optimization,
            AllocationMethod.MIN_VARIANCE: self._min_variance_optimization,
            AllocationMethod.MAX_SHARPE: self._max_sharpe_optimization,
            AllocationMethod.RISK_PARITY: self._risk_parity_optimization,
            AllocationMethod.BLACK_LITTERMAN: self._black_litterman_optimization,
            AllocationMethod.HRP: self._hierarchical_risk_parity,
            AllocationMethod.CLA: self._critical_line_algorithm,
            AllocationMethod.CUSTOM: self._custom_optimization
        }

        # 初始化风险模型
        self.risk_models = {
            RiskModel.SAMPLE_COVARIANCE: risk_models.sample_cov,
            RiskModel.LEDOIT_WOLF: risk_models.ledoit_wolf,
            RiskModel.ORACLE_APPROXIMATING: risk_models.oracle_approximating,
            RiskModel.CONSTANT_CORRELATION: risk_models.constant_correlation,
            RiskModel.EXPONENTIALLY_WEIGHTED: risk_models.exp_cov,
            RiskModel.GARCH: self._garch_covariance,
            RiskModel.DCC_GARCH: self._dcc_garch_covariance
        }

        logger.info(f"已加载 {len(self.optimizers)} 种优化方法和 {len(self.risk_models)} 种风险模型")

    def process(self, signals: Dict[str, Any], market_data: Dict[str, Any],
                current_positions: Dict[str, float]) -> Dict[str, Any]:
        """
        处理信号和市场数据，生成优化后的组合

        Args:
            signals: 交易信号字典
            market_data: 市场数据
            current_positions: 当前持仓

        Returns:
            优化后的组合配置
        """
        start_time = time.time()

        try:
            # 验证输入数据
            if not self._validate_inputs(signals, market_data, current_positions):
                logger.warning("输入数据验证失败")
                return {}

            # 更新市场数据缓存
            self._update_market_data(market_data)

            # 计算预期收益和风险
            self._calculate_expected_returns(market_data)
            self._calculate_covariance_matrix(market_data)

            # 构建初始组合
            initial_portfolio = self._build_initial_portfolio(current_positions, market_data)

            # 优化组合
            optimized_portfolio = self._optimize_portfolio(initial_portfolio, signals, market_data)

            # 应用约束条件
            constrained_portfolio = self._apply_constraints(optimized_portfolio, market_data)

            # 风险检查
            risk_assessment = self._assess_portfolio_risk(constrained_portfolio, market_data)
            if not risk_assessment['approved']:
                logger.warning(f"组合风险检查未通过: {risk_assessment['reason']}")
                # 应用风险控制
                constrained_portfolio = self._apply_risk_control(constrained_portfolio, risk_assessment)

            # 生成调仓指令
            rebalance_instructions = self._generate_rebalance_instructions(
                self.current_portfolio, constrained_portfolio, market_data
            )

            # 更新当前组合状态
            self._update_portfolio_state(constrained_portfolio, rebalance_instructions)

            processing_time = time.time() - start_time
            logger.info(
                f"组合优化完成: 耗时 {processing_time:.3f}s, 换手率: {rebalance_instructions['turnover_rate']:.2%}")

            return {
                'optimized_portfolio': constrained_portfolio.to_dict(),
                'rebalance_instructions': rebalance_instructions,
                'risk_assessment': risk_assessment,
                'performance_metrics': self._calculate_performance_metrics(constrained_portfolio)
            }

        except Exception as e:
            logger.error(f"组合处理失败: {e}")
            self._handle_processing_error(e)
            return {}

    def _validate_inputs(self, signals: Dict, market_data: Dict,
                         current_positions: Dict) -> bool:
        """验证输入数据"""
        try:
            # 检查必要字段
            required_market_fields = ['timestamp', 'symbols', 'prices']
            if not all(field in market_data for field in required_market_fields):
                logger.warning("市场数据缺少必要字段")
                return False

            # 检查价格数据完整性
            for symbol in market_data['symbols']:
                if symbol not in market_data['prices']:
                    logger.warning(f"缺少价格数据: {symbol}")
                    return False

                price_data = market_data['prices'][symbol]
                if not all(key in price_data for key in ['open', 'high', 'low', 'close']):
                    logger.warning(f"价格数据不完整: {symbol}")
                    return False

            # 检查持仓数据
            if not isinstance(current_positions, dict):
                logger.warning("持仓数据格式错误")
                return False

            # 检查信号数据
            if signals and not isinstance(signals, dict):
                logger.warning("信号数据格式错误")
                return False

            return True

        except Exception as e:
            logger.error(f"输入数据验证失败: {e}")
            return False

    def _update_market_data(self, market_data: Dict):
        """更新市场数据缓存"""
        try:
            timestamp = market_data['timestamp']

            # 更新个股数据
            for symbol in market_data['symbols']:
                if symbol not in self.market_data_cache:
                    self.market_data_cache[symbol] = {}

                # 存储价格数据
                price_data = market_data['prices'][symbol]
                self.market_data_cache[symbol][timestamp] = {
                    'open': price_data['open'],
                    'high': price_data['high'],
                    'low': price_data['low'],
                    'close': price_data['close'],
                    'volume': market_data.get('volumes', {}).get(symbol, {}).get('volume', 0)
                }

            # 限制缓存大小
            max_cache_size = self.portfolio_config.get('max_market_data_cache', 1000)
            if len(self.market_data_cache) > max_cache_size:
                # 移除最旧的数据
                oldest_symbol = min(self.market_data_cache.keys(),
                                    key=lambda s: len(self.market_data_cache[s]))
                del self.market_data_cache[oldest_symbol]

        except Exception as e:
            logger.error(f"市场数据更新失败: {e}")

    def _calculate_expected_returns(self, market_data: Dict):
        """计算预期收益"""
        try:
            symbols = market_data['symbols']
            returns_data = {}

            for symbol in symbols:
                if symbol in market_data['prices']:
                    closes = market_data['prices'][symbol]['close']
                    if len(closes) > 1:
                        # 计算对数收益
                        log_returns = np.diff(np.log(closes))
                        returns_data[symbol] = np.mean(log_returns) * 252  # 年化收益

            self.expected_returns = pd.Series(returns_data)

        except Exception as e:
            logger.error(f"预期收益计算失败: {e}")
            self.expected_returns = None

    def _calculate_covariance_matrix(self, market_data: Dict):
        """计算协方差矩阵"""
        try:
            symbols = market_data['symbols']
            returns_matrix = []
            valid_symbols = []

            # 构建收益矩阵
            for symbol in symbols:
                if symbol in market_data['prices']:
                    closes = market_data['prices'][symbol]['close']
                    if len(closes) > 2:  # 至少需要3个点计算收益
                        log_returns = np.diff(np.log(closes))
                        returns_matrix.append(log_returns)
                        valid_symbols.append(symbol)

            if len(returns_matrix) < 2:
                logger.warning("不足够的数据计算协方差矩阵")
                return

            returns_matrix = np.array(returns_matrix)

            # 选择风险模型
            risk_model_name = self.optimization_config.get('risk_model', RiskModel.LEDOIT_WOLF)
            risk_model_func = self.risk_models.get(risk_model_name, risk_models.ledoit_wolf)

            # 计算协方差矩阵
            if risk_model_name in [RiskModel.SAMPLE_COVARIANCE, RiskModel.LEDOIT_WOLF,
                                   RiskModel.ORACLE_APPROXIMATING, RiskModel.CONSTANT_CORRELATION]:
                cov_matrix = risk_model_func(pd.DataFrame(returns_matrix.T, columns=valid_symbols))
            elif risk_model_name == RiskModel.EXPONENTIALLY_WEIGHTED:
                span = self.optimization_config.get('ewm_span', 60)
                cov_matrix = risk_model_func(pd.DataFrame(returns_matrix.T, columns=valid_symbols), span=span)
            else:
                # 使用默认的Ledoit-Wolf模型
                cov_matrix = risk_models.ledoit_wolf(pd.DataFrame(returns_matrix.T, columns=valid_symbols))

            self.covariance_matrix = cov_matrix
            self.correlation_matrix = self._calculate_correlation_matrix(cov_matrix)

        except Exception as e:
            logger.error(f"协方差矩阵计算失败: {e}")
            self.covariance_matrix = None
            self.correlation_matrix = None

    def _calculate_correlation_matrix(self, cov_matrix: pd.DataFrame) -> pd.DataFrame:
        """从协方差矩阵计算相关矩阵"""
        try:
            std_dev = np.sqrt(np.diag(cov_matrix))
            correlation = cov_matrix / np.outer(std_dev, std_dev)
            return pd.DataFrame(correlation, index=cov_matrix.index, columns=cov_matrix.columns)
        except Exception as e:
            logger.error(f"相关矩阵计算失败: {e}")
            return None

    def _build_initial_portfolio(self, current_positions: Dict,
                                 market_data: Dict) -> PortfolioState:
        """构建初始组合状态"""
        try:
            total_value = sum(
                current_positions[symbol] * market_data['prices'][symbol]['close'][-1]
                for symbol in current_positions if symbol in market_data['prices']
            )

            # 添加现金余额
            cash_balance = current_positions.get('CASH', 0)
            total_value += cash_balance

            # 构建资产配置
            allocations = {}
            for symbol, shares in current_positions.items():
                if symbol == 'CASH':
                    continue

                if symbol in market_data['prices']:
                    current_price = market_data['prices'][symbol]['close'][-1]
                    current_value = shares * current_price
                    weight = current_value / total_value if total_value > 0 else 0

                    allocation = AssetAllocation(
                        symbol=symbol,
                        weight=weight,
                        target_weight=weight,
                        current_value=current_value,
                        target_value=current_value,
                        notional=current_value,
                        sector=self._get_asset_sector(symbol),
                        asset_class=self._get_asset_class(symbol),
                        region=self._get_asset_region(symbol),
                        currency=self._get_asset_currency(symbol),
                        liquidity_tier=self._get_liquidity_tier(symbol, market_data),
                        risk_contribution=0.0,
                        marginal_risk=0.0,
                        expected_return=self._get_expected_return(symbol),
                        expected_risk=self._get_expected_risk(symbol),
                        transaction_cost=self._calculate_transaction_cost(symbol, shares),
                        tax_implication=self._calculate_tax_implication(symbol, shares, current_price),
                        constraints=self._get_asset_constraints(symbol)
                    )

                    allocations[symbol] = allocation

            # 创建组合状态
            portfolio = PortfolioState(
                portfolio_id=f"portfolio_{int(time.time())}",
                total_value=total_value,
                cash_balance=cash_balance,
                leveraged_value=total_value,
                allocations=allocations,
                metadata=PortfolioMetadata(),
                performance=self._calculate_initial_performance(allocations),
                risk_metrics=self._calculate_initial_risk_metrics(allocations),
                constraints=PortfolioConstraints(),
                benchmark=self.portfolio_config.get('benchmark', 'SPY')
            )

            return portfolio

        except Exception as e:
            logger.error(f"初始组合构建失败: {e}")
            raise

    def _optimize_portfolio(self, portfolio: PortfolioState, signals: Dict,
                            market_data: Dict) -> PortfolioState:
        """优化投资组合"""
        start_time = time.time()

        try:
            # 获取优化配置
            optimization_method = AllocationMethod(
                self.optimization_config.get('method', 'max_sharpe')
            )
            objective = PortfolioObjective(
                self.optimization_config.get('objective', 'maximize_sharpe')
            )

            # 选择优化方法
            if optimization_method in self.optimizers:
                optimizer_func = self.optimizers[optimization_method]
                optimized_weights = optimizer_func(portfolio, signals, market_data, objective)
            else:
                logger.warning(f"未知的优化方法: {optimization_method}, 使用默认的最大夏普比率优化")
                optimized_weights = self._max_sharpe_optimization(portfolio, signals, market_data, objective)

            # 应用优化权重
            optimized_portfolio = self._apply_optimized_weights(portfolio, optimized_weights, market_data)

            # 计算优化后的风险指标
            optimized_portfolio.risk_metrics = self._calculate_portfolio_risk_metrics(
                optimized_portfolio, market_data
            )

            # 更新元数据
            optimized_portfolio.metadata.optimization_method = optimization_method
            optimized_portfolio.metadata.objective = objective
            optimized_portfolio.metadata.last_rebalanced = datetime.now().isoformat()
            optimized_portfolio.metadata.expected_return = optimized_portfolio.risk_metrics.get('expected_return', 0.0)
            optimized_portfolio.metadata.expected_risk = optimized_portfolio.risk_metrics.get('volatility', 0.0)
            optimized_portfolio.metadata.sharpe_ratio = optimized_portfolio.risk_metrics.get('sharpe_ratio', 0.0)

            optimization_time = time.time() - start_time
            self.performance_stats['optimizations_performed'] += 1
            self.performance_stats['avg_optimization_time'] = (
                                                                      self.performance_stats[
                                                                          'avg_optimization_time'] * (
                                                                                  self.performance_stats[
                                                                                      'optimizations_performed'] - 1) +
                                                                      optimization_time
                                                              ) / self.performance_stats['optimizations_performed']

            logger.info(f"组合优化完成: 方法={optimization_method.value}, 耗时={optimization_time:.3f}s")

            return optimized_portfolio

        except Exception as e:
            logger.error(f"组合优化失败: {e}")
            # 返回原始组合作为备选
            return portfolio

    def _equal_weight_optimization(self, portfolio: PortfolioState, signals: Dict,
                                   market_data: Dict, objective: PortfolioObjective) -> Dict[str, float]:
        """等权重优化"""
        try:
            symbols = list(portfolio.allocations.keys())
            if not symbols:
                return {}

            # 等权重分配
            n_assets = len(symbols)
            equal_weight = 1.0 / n_assets

            optimized_weights = {symbol: equal_weight for symbol in symbols}

            # 考虑现金余额
            cash_weight = portfolio.cash_balance / portfolio.total_value if portfolio.total_value > 0 else 0
            if cash_weight > 0:
                # 调整权重，考虑现金
                adjustment_factor = 1 - cash_weight
                optimized_weights = {symbol: weight * adjustment_factor for symbol, weight in optimized_weights.items()}
                optimized_weights['CASH'] = cash_weight

            return optimized_weights

        except Exception as e:
            logger.error(f"等权重优化失败: {e}")
            return {}

    def _market_cap_optimization(self, portfolio: PortfolioState, signals: Dict,
                                 market_data: Dict, objective: PortfolioObjective) -> Dict[str, float]:
        """市值加权优化"""
        try:
            symbols = list(portfolio.allocations.keys())
            if not symbols:
                return {}

            # 获取市值数据
            market_caps = {}
            total_market_cap = 0

            for symbol in symbols:
                # 这里应该从市场数据获取实际市值
                # 简化实现：使用价格*假设的流通股本
                if symbol in market_data['prices']:
                    price = market_data['prices'][symbol]['close'][-1]
                    # 假设流通股本（实际中应该从基本面数据获取）
                    shares_outstanding = self._estimate_shares_outstanding(symbol)
                    market_cap = price * shares_outstanding
                    market_caps[symbol] = market_cap
                    total_market_cap += market_cap

            if total_market_cap == 0:
                return self._equal_weight_optimization(portfolio, signals, market_data, objective)

            # 计算市值权重
            optimized_weights = {}
            for symbol, market_cap in market_caps.items():
                optimized_weights[symbol] = market_cap / total_market_cap

            # 考虑现金余额
            cash_weight = portfolio.cash_balance / portfolio.total_value if portfolio.total_value > 0 else 0
            if cash_weight > 0:
                adjustment_factor = 1 - cash_weight
                optimized_weights = {symbol: weight * adjustment_factor for symbol, weight in optimized_weights.items()}
                optimized_weights['CASH'] = cash_weight

            return optimized_weights

        except Exception as e:
            logger.error(f"市值加权优化失败: {e}")
            return self._equal_weight_optimization(portfolio, signals, market_data, objective)

    def _min_variance_optimization(self, portfolio: PortfolioState, signals: Dict,
                                   market_data: Dict, objective: PortfolioObjective) -> Dict[str, float]:
        """最小方差优化"""
        try:
            symbols = list(portfolio.allocations.keys())
            if not symbols or self.covariance_matrix is None:
                return self._equal_weight_optimization(portfolio, signals, market_data, objective)

            # 确保协方差矩阵包含所有符号
            missing_symbols = set(symbols) - set(self.covariance_matrix.columns)
            if missing_symbols:
                logger.warning(f"协方差矩阵缺少符号: {missing_symbols}, 使用等权重优化")
                return self._equal_weight_optimization(portfolio, signals, market_data, objective)

            # 使用PyPortfolioOpt进行最小方差优化
            cov_matrix = self.covariance_matrix.loc[symbols, symbols]

            ef = EfficientFrontier(None, cov_matrix, weight_bounds=(0, 1))
            ef.min_volatility()
            weights = ef.clean_weights()

            optimized_weights = {symbol: weights[symbol] for symbol in symbols}

            # 考虑现金余额
            cash_weight = portfolio.cash_balance / portfolio.total_value if portfolio.total_value > 0 else 0
            if cash_weight > 0:
                adjustment_factor = 1 - cash_weight
                optimized_weights = {symbol: weight * adjustment_factor for symbol, weight in optimized_weights.items()}
                optimized_weights['CASH'] = cash_weight

            return optimized_weights

        except Exception as e:
            logger.error(f"最小方差优化失败: {e}")
            return self._equal_weight_optimization(portfolio, signals, market_data, objective)

    def _max_sharpe_optimization(self, portfolio: PortfolioState, signals: Dict,
                                 market_data: Dict, objective: PortfolioObjective) -> Dict[str, float]:
        """最大夏普比率优化"""
        try:
            symbols = list(portfolio.allocations.keys())
            if not symbols or self.covariance_matrix is None or self.expected_returns is None:
                return self._min_variance_optimization(portfolio, signals, market_data, objective)

            # 确保数据完整性
            missing_cov = set(symbols) - set(self.covariance_matrix.columns)
            missing_returns = set(symbols) - set(self.expected_returns.index)

            if missing_cov or missing_returns:
                logger.warning(f"数据不完整, 使用最小方差优化")
                return self._min_variance_optimization(portfolio, signals, market_data, objective)

            # 使用PyPortfolioOpt进行最大夏普优化
            cov_matrix = self.covariance_matrix.loc[symbols, symbols]
            expected_returns = self.expected_returns[symbols]

            ef = EfficientFrontier(expected_returns, cov_matrix, weight_bounds=(0, 1))
            ef.max_sharpe()
            weights = ef.clean_weights()

            optimized_weights = {symbol: weights[symbol] for symbol in symbols}

            # 考虑现金余额
            cash_weight = portfolio.cash_balance / portfolio.total_value if portfolio.total_value > 0 else 0
            if cash_weight > 0:
                adjustment_factor = 1 - cash_weight
                optimized_weights = {symbol: weight * adjustment_factor for symbol, weight in optimized_weights.items()}
                optimized_weights['CASH'] = cash_weight

            return optimized_weights

        except Exception as e:
            logger.error(f"最大夏普比率优化失败: {e}")
            return self._min_variance_optimization(portfolio, signals, market_data, objective)

    def _risk_parity_optimization(self, portfolio: PortfolioState, signals: Dict,
                                  market_data: Dict, objective: PortfolioObjective) -> Dict[str, float]:
        """风险平价优化"""
        try:
            symbols = list(portfolio.allocations.keys())
            if not symbols or self.covariance_matrix is None:
                return self._max_sharpe_optimization(portfolio, signals, market_data, objective)

            # 确保协方差矩阵包含所有符号
            missing_symbols = set(symbols) - set(self.covariance_matrix.columns)
            if missing_symbols:
                logger.warning(f"协方差矩阵缺少符号: {missing_symbols}, 使用最大夏普优化")
                return self._max_sharpe_optimization(portfolio, signals, market_data, objective)

            # 使用Riskfolio-Lib进行风险平价优化
            cov_matrix = self.covariance_matrix.loc[symbols, symbols]
            returns_data = self._get_historical_returns(symbols, market_data)

            if returns_data is None or returns_data.empty:
                return self._max_sharpe_optimization(portfolio, signals, market_data, objective)

            # 创建风险平价组合
            port = rp.Portfolio(returns=returns_data)
            port.assets_stats(method_mu='hist', method_cov='hist')
            port.rp_optimization(model='Classic', objective='Risk', rm='MV')
            weights = port.w.to_dict()

            optimized_weights = {symbol: weights[symbol] for symbol in symbols if symbol in weights}

            # 考虑现金余额
            cash_weight = portfolio.cash_balance / portfolio.total_value if portfolio.total_value > 0 else 0
            if cash_weight > 0:
                adjustment_factor = 1 - cash_weight
                optimized_weights = {symbol: weight * adjustment_factor for symbol, weight in optimized_weights.items()}
                optimized_weights['CASH'] = cash_weight

            return optimized_weights

        except Exception as e:
            logger.error(f"风险平价优化失败: {e}")
            return self._max_sharpe_optimization(portfolio, signals, market_data, objective)

    def _black_litterman_optimization(self, portfolio: PortfolioState, signals: Dict,
                                      market_data: Dict, objective: PortfolioObjective) -> Dict[str, float]:
        """Black-Litterman模型优化"""
        try:
            symbols = list(portfolio.allocations.keys())
            if not symbols or self.covariance_matrix is None or self.expected_returns is None:
                return self._risk_parity_optimization(portfolio, signals, market_data, objective)

            # 获取市场均衡收益（先验收益）
            market_returns = self.expected_returns[symbols]
            cov_matrix = self.covariance_matrix.loc[symbols, symbols]

            # 基于信号生成观点
            views, confidence = self._generate_black_litterman_views(signals, symbols, market_data)

            if not views:
                logger.info("没有生成有效的观点，使用市场均衡收益")
                return self._max_sharpe_optimization(portfolio, signals, market_data, objective)

            # 使用PyPortfolioOpt的Black-Litterman模型
            from pypfopt import BlackLittermanModel

            bl = BlackLittermanModel(cov_matrix, pi=market_returns, absolute_views=views)
            bl_returns = bl.bl_returns()
            bl_cov = bl.bl_cov()

            # 使用Black-Litterman后的收益和协方差进行优化
            ef = EfficientFrontier(bl_returns, bl_cov, weight_bounds=(0, 1))

            if objective == PortfolioObjective.MAXIMIZE_SHARPE:
                ef.max_sharpe()
            elif objective == PortfolioObjective.MINIMIZE_RISK:
                ef.min_volatility()
            else:
                ef.max_sharpe()

            weights = ef.clean_weights()
            optimized_weights = {symbol: weights[symbol] for symbol in symbols}

            # 考虑现金余额
            cash_weight = portfolio.cash_balance / portfolio.total_value if portfolio.total_value > 0 else 0
            if cash_weight > 0:
                adjustment_factor = 1 - cash_weight
                optimized_weights = {symbol: weight * adjustment_factor for symbol, weight in optimized_weights.items()}
                optimized_weights['CASH'] = cash_weight

            return optimized_weights

        except Exception as e:
            logger.error(f"Black-Litterman优化失败: {e}")
            return self._risk_parity_optimization(portfolio, signals, market_data, objective)

    def _hierarchical_risk_parity(self, portfolio: PortfolioState, signals: Dict,
                                  market_data: Dict, objective: PortfolioObjective) -> Dict[str, float]:
        """分层风险平价优化"""
        try:
            symbols = list(portfolio.allocations.keys())
            if not symbols or self.covariance_matrix is None:
                return self._black_litterman_optimization(portfolio, signals, market_data, objective)

            # 确保协方差矩阵包含所有符号
            missing_symbols = set(symbols) - set(self.covariance_matrix.columns)
            if missing_symbols:
                logger.warning(f"协方差矩阵缺少符号: {missing_symbols}, 使用Black-Litterman优化")
                return self._black_litterman_optimization(portfolio, signals, market_data, objective)

            # 使用Riskfolio-Lib进行HRP优化
            returns_data = self._get_historical_returns(symbols, market_data)

            if returns_data is None or returns_data.empty:
                return self._black_litterman_optimization(portfolio, signals, market_data, objective)

            port = rp.Portfolio(returns=returns_data)
            port.assets_stats(method_mu='hist', method_cov='hist')
            port.hrp_optimization(model='HRP')
            weights = port.w.to_dict()

            optimized_weights = {symbol: weights[symbol] for symbol in symbols if symbol in weights}

            # 考虑现金余额
            cash_weight = portfolio.cash_balance / portfolio.total_value if portfolio.total_value > 0 else 0
            if cash_weight > 0:
                adjustment_factor = 1 - cash_weight
                optimized_weights = {symbol: weight * adjustment_factor for symbol, weight in optimized_weights.items()}
                optimized_weights['CASH'] = cash_weight

            return optimized_weights

        except Exception as e:
            logger.error(f"分层风险平价优化失败: {e}")
            return self._black_litterman_optimization(portfolio, signals, market_data, objective)

    def _critical_line_algorithm(self, portfolio: PortfolioState, signals: Dict,
                                 market_data: Dict, objective: PortfolioObjective) -> Dict[str, float]:
        """关键线算法优化"""
        try:
            symbols = list(portfolio.allocations.keys())
            if not symbols or self.covariance_matrix is None or self.expected_returns is None:
                return self._hierarchical_risk_parity(portfolio, signals, market_data, objective)

            # 使用PyPortfolioOpt的关键线算法
            cov_matrix = self.covariance_matrix.loc[symbols, symbols]
            expected_returns = self.expected_returns[symbols]

            ef = EfficientFrontier(expected_returns, cov_matrix, weight_bounds=(0, 1))

            if objective == PortfolioObjective.MAXIMIZE_SHARPE:
                ef.max_sharpe()
            elif objective == PortfolioObjective.MINIMIZE_RISK:
                ef.min_volatility()
            elif objective == PortfolioObjective.MAXIMIZE_UTILITY:
                risk_aversion = self.optimization_config.get('risk_aversion', 1.0)
                ef.max_quadratic_utility(risk_aversion=risk_aversion)
            else:
                ef.max_sharpe()

            weights = ef.clean_weights()
            optimized_weights = {symbol: weights[symbol] for symbol in symbols}

            # 考虑现金余额
            cash_weight = portfolio.cash_balance / portfolio.total_value if portfolio.total_value > 0 else 0
            if cash_weight > 0:
                adjustment_factor = 1 - cash_weight
                optimized_weights = {symbol: weight * adjustment_factor for symbol, weight in optimized_weights.items()}
                optimized_weights['CASH'] = cash_weight

            return optimized_weights

        except Exception as e:
            logger.error(f"关键线算法优化失败: {e}")
            return self._hierarchical_risk_parity(portfolio, signals, market_data, objective)

    def _custom_optimization(self, portfolio: PortfolioState, signals: Dict,
                             market_data: Dict, objective: PortfolioObjective) -> Dict[str, float]:
        """自定义优化"""
        try:
            # 获取自定义优化配置
            custom_config = self.optimization_config.get('custom_parameters', {})
            optimization_type = custom_config.get('type', 'signal_based')

            if optimization_type == 'signal_based':
                return self._signal_based_optimization(portfolio, signals, market_data)
            elif optimization_type == 'risk_budget':
                return self._risk_budget_optimization(portfolio, signals, market_data)
            elif optimization_type == 'factor_based':
                return self._factor_based_optimization(portfolio, signals, market_data)
            else:
                logger.warning(f"未知的自定义优化类型: {optimization_type}, 使用风险平价优化")
                return self._risk_parity_optimization(portfolio, signals, market_data, objective)

        except Exception as e:
            logger.error(f"自定义优化失败: {e}")
            return self._critical_line_algorithm(portfolio, signals, market_data, objective)

    def _signal_based_optimization(self, portfolio: PortfolioState, signals: Dict,
                                   market_data: Dict) -> Dict[str, float]:
        """基于信号的优化"""
        try:
            symbols = list(portfolio.allocations.keys())
            if not signals or not symbols:
                return self._equal_weight_optimization(portfolio, signals, market_data,
                                                       PortfolioObjective.MAXIMIZE_SHARPE)

            # 基于信号强度计算权重
            signal_weights = {}
            total_signal_strength = 0

            for symbol in symbols:
                symbol_signals = signals.get(symbol, [])
                if symbol_signals:
                    # 计算信号综合强度
                    signal_strength = sum(
                        signal.weight * signal.metadata.confidence
                        for signal in symbol_signals
                    )
                    signal_weights[symbol] = signal_strength
                    total_signal_strength += signal_strength
                else:
                    signal_weights[symbol] = 0

            if total_signal_strength == 0:
                return self._equal_weight_optimization(portfolio, signals, market_data,
                                                       PortfolioObjective.MAXIMIZE_SHARPE)

            # 归一化权重
            optimized_weights = {}
            for symbol, strength in signal_weights.items():
                optimized_weights[symbol] = strength / total_signal_strength

            # 考虑现金余额
            cash_weight = portfolio.cash_balance / portfolio.total_value if portfolio.total_value > 0 else 0
            if cash_weight > 0:
                adjustment_factor = 1 - cash_weight
                optimized_weights = {symbol: weight * adjustment_factor for symbol, weight in optimized_weights.items()}
                optimized_weights['CASH'] = cash_weight

            return optimized_weights

        except Exception as e:
            logger.error(f"信号优化失败: {e}")
            return self._equal_weight_optimization(portfolio, signals, market_data,
                                                   PortfolioObjective.MAXIMIZE_SHARPE)

    def _apply_optimized_weights(self, portfolio: PortfolioState, optimized_weights: Dict[str, float],
                                 market_data: Dict) -> PortfolioState:
        """应用优化权重"""
        try:
            # 创建新的组合状态
            new_portfolio = copy.deepcopy(portfolio)
            new_portfolio.version += 1
            new_portfolio.timestamp = datetime.now().isoformat()

            # 更新资产配置
            total_value = portfolio.total_value
            new_allocations = {}

            for symbol, target_weight in optimized_weights.items():
                if symbol == 'CASH':
                    continue

                current_allocation = portfolio.allocations.get(symbol)
                if current_allocation:
                    target_value = total_value * target_weight

                    new_allocation = AssetAllocation(
                        symbol=symbol,
                        weight=target_weight,
                        target_weight=target_weight,
                        current_value=current_allocation.current_value,
                        target_value=target_value,
                        notional=target_value,
                        sector=current_allocation.sector,
                        asset_class=current_allocation.asset_class,
                        region=current_allocation.region,
                        currency=current_allocation.currency,
                        liquidity_tier=current_allocation.liquidity_tier,
                        risk_contribution=0.0,  # 将在后续计算
                        marginal_risk=0.0,
                        expected_return=current_allocation.expected_return,
                        expected_risk=current_allocation.expected_risk,
                        transaction_cost=self._calculate_transaction_cost(symbol, target_value),
                        tax_implication=current_allocation.tax_implication,
                        constraints=current_allocation.constraints,
                        metadata=current_allocation.metadata
                    )

                    new_allocations[symbol] = new_allocation

            # 处理现金
            cash_weight = optimized_weights.get('CASH', 0)
            new_portfolio.cash_balance = total_value * cash_weight
            new_portfolio.allocations = new_allocations

            return new_portfolio

        except Exception as e:
            logger.error(f"优化权重应用失败: {e}")
            return portfolio

    def _apply_constraints(self, portfolio: PortfolioState, market_data: Dict) -> PortfolioState:
        """应用约束条件"""
        try:
            constraints = self.portfolio_config.get('constraints', {})
            if not constraints:
                return portfolio

            # 创建约束后的组合副本
            constrained_portfolio = copy.deepcopy(portfolio)

            # 应用各种约束
            constrained_portfolio = self._apply_weight_constraints(constrained_portfolio, constraints)
            constrained_portfolio = self._apply_sector_constraints(constrained_portfolio, constraints, market_data)
            constrained_portfolio = self._apply_liquidity_constraints(constrained_portfolio, constraints, market_data)
            constrained_portfolio = self._apply_leverage_constraints(constrained_portfolio, constraints)
            constrained_portfolio = self._apply_regulatory_constraints(constrained_portfolio, constraints)

            # 重新归一化权重
            constrained_portfolio = self._normalize_weights(constrained_portfolio)

            return constrained_portfolio

        except Exception as e:
            logger.error(f"约束应用失败: {e}")
            return portfolio

    def _apply_weight_constraints(self, portfolio: PortfolioState, constraints: Dict) -> PortfolioState:
        """应用权重约束"""
        try:
            max_weight = constraints.get('max_asset_weight', 0.2)
            min_weight = constraints.get('min_asset_weight', 0.0)

            for symbol, allocation in portfolio.allocations.items():
                current_weight = allocation.weight

                # 应用上下限约束
                if current_weight > max_weight:
                    allocation.weight = max_weight
                    allocation.target_weight = max_weight
                elif current_weight < min_weight and current_weight > 0:  # 只调整正权重
                    allocation.weight = min_weight
                    allocation.target_weight = min_weight

            return portfolio

        except Exception as e:
            logger.error(f"权重约束应用失败: {e}")
            return portfolio

    def _apply_sector_constraints(self, portfolio: PortfolioState, constraints: Dict,
                                  market_data: Dict) -> PortfolioState:
        """应用行业约束"""
        try:
            max_sector_weight = constraints.get('max_sector_weight', 0.3)

            # 计算各行业当前权重
            sector_weights = {}
            for allocation in portfolio.allocations.values():
                sector = allocation.sector
                if sector not in sector_weights:
                    sector_weights[sector] = 0
                sector_weights[sector] += allocation.weight

            # 检查并调整超限行业
            for sector, weight in sector_weights.items():
                if weight > max_sector_weight:
                    # 计算需要减少的权重
                    excess_weight = weight - max_sector_weight
                    reduction_factor = max_sector_weight / weight

                    # 按比例减少该行业所有资产的权重
                    for allocation in portfolio.allocations.values():
                        if allocation.sector == sector:
                            allocation.weight *= reduction_factor
                            allocation.target_weight *= reduction_factor

            return portfolio

        except Exception as e:
            logger.error(f"行业约束应用失败: {e}")
            return portfolio

    def _assess_portfolio_risk(self, portfolio: PortfolioState, market_data: Dict) -> Dict[str, Any]:
        """评估组合风险"""
        try:
            risk_assessment = {
                'approved': True,
                'reason': '',
                'risk_score': 0.0,
                'violations': [],
                'stress_test_results': {},
                'scenario_analysis': {}
            }

            # 计算风险指标
            risk_metrics = self._calculate_portfolio_risk_metrics(portfolio, market_data)

            # 检查风险限额
            risk_limits = self.risk_config.get('risk_limits', {})

            # 波动率检查
            max_volatility = risk_limits.get('max_volatility', 0.3)
            if risk_metrics.get('volatility', 0) > max_volatility:
                risk_assessment['approved'] = False
                risk_assessment['violations'].append(
                    f"波动率超限: {risk_metrics['volatility']:.2%} > {max_volatility:.2%}")

            # VaR检查
            var_limit = risk_limits.get('var_95', -0.05)
            if risk_metrics.get('var_95', 0) < var_limit:
                risk_assessment['approved'] = False
                risk_assessment['violations'].append(f"VaR超限: {risk_metrics['var_95']:.2%} < {var_limit:.2%}")

            # 最大回撤检查
            max_drawdown_limit = risk_limits.get('max_drawdown', -0.2)
            if risk_metrics.get('max_drawdown', 0) < max_drawdown_limit:
                risk_assessment['approved'] = False
                risk_assessment['violations'].append(
                    f"最大回撤超限: {risk_metrics['max_drawdown']:.2%} < {max_drawdown_limit:.2%}")

            # 集中度检查
            concentration_limit = risk_limits.get('concentration_limit', 0.4)
            concentration = self._calculate_concentration(portfolio)
            if concentration > concentration_limit:
                risk_assessment['approved'] = False
                risk_assessment['violations'].append(f"集中度超限: {concentration:.2%} > {concentration_limit:.2%}")

            # 流动性检查
            liquidity_threshold = risk_limits.get('min_liquidity_score', 0.7)
            liquidity_score = self._calculate_liquidity_score(portfolio, market_data)
            if liquidity_score < liquidity_threshold:
                risk_assessment['approved'] = False
                risk_assessment['violations'].append(f"流动性不足: {liquidity_score:.2f} < {liquidity_threshold:.2f}")

            # 杠杆检查
            leverage_limit = risk_limits.get('max_leverage', 2.0)
            if portfolio.leveraged_value / portfolio.total_value > leverage_limit:
                risk_assessment['approved'] = False
                risk_assessment['violations'].append(
                    f"杠杆超限: {portfolio.leveraged_value / portfolio.total_value:.2f} > {leverage_limit:.2f}")

            # 压力测试
            stress_test_results = self._run_stress_tests(portfolio, market_data)
            risk_assessment['stress_test_results'] = stress_test_results

            # 场景分析
            scenario_analysis = self._run_scenario_analysis(portfolio, market_data)
            risk_assessment['scenario_analysis'] = scenario_analysis

            # 计算综合风险评分
            risk_score = self._calculate_risk_score(risk_metrics, stress_test_results, scenario_analysis)
            risk_assessment['risk_score'] = risk_score

            # 高风险场景额外检查
            if risk_score > risk_limits.get('max_risk_score', 0.7):
                risk_assessment['approved'] = False
                risk_assessment['violations'].append(
                    f"综合风险评分过高: {risk_score:.2f} > {risk_limits['max_risk_score']:.2f}")

            # 如果存在违规，设置拒绝原因
            if not risk_assessment['approved']:
                risk_assessment['reason'] = "; ".join(risk_assessment['violations'])

            return risk_assessment

        except Exception as e:
            logger.error(f"组合风险评估失败: {e}")
            return {
                'approved': False,
                'reason': f'风险评估错误: {str(e)}',
                'risk_score': 1.0,
                'violations': ['风险评估过程出错'],
                'stress_test_results': {},
                'scenario_analysis': {}
            }

    def _calculate_concentration(self, portfolio: PortfolioState) -> float:
        """计算组合集中度"""
        try:
            weights = [alloc.weight for alloc in portfolio.allocations.values()]
            if not weights:
                return 0.0

            # 计算赫芬达尔-赫希曼指数 (HHI)
            hhi = sum(w ** 2 for w in weights)

            # 转换为集中度百分比
            concentration = hhi * 100
            return concentration

        except Exception as e:
            logger.error(f"集中度计算失败: {e}")
            return 0.0

    def _calculate_liquidity_score(self, portfolio: PortfolioState, market_data: Dict) -> float:
        """计算流动性评分"""
        try:
            liquidity_scores = []
            weight_sum = 0

            for symbol, allocation in portfolio.allocations.items():
                if symbol in market_data.get('liquidity', {}):
                    # 获取流动性指标
                    liquidity_data = market_data['liquidity'][symbol]
                    volume = liquidity_data.get('volume', 0)
                    avg_daily_volume = liquidity_data.get('avg_daily_volume', volume)
                    bid_ask_spread = liquidity_data.get('bid_ask_spread', 0.01)

                    # 计算单个资产的流动性评分
                    volume_ratio = volume / avg_daily_volume if avg_daily_volume > 0 else 0
                    spread_score = 1 - min(bid_ask_spread / 0.1, 1.0)  # 假设10%为最大点差

                    asset_liquidity = (volume_ratio + spread_score) / 2
                    liquidity_scores.append(asset_liquidity * allocation.weight)
                    weight_sum += allocation.weight

            if weight_sum > 0:
                return sum(liquidity_scores) / weight_sum
            return 1.0

        except Exception as e:
            logger.error(f"流动性评分计算失败: {e}")
            return 1.0

    def _run_stress_tests(self, portfolio: PortfolioState, market_data: Dict) -> Dict[str, Any]:
        """运行压力测试"""
        stress_scenarios = self.risk_config.get('stress_scenarios', {})
        results = {}

        try:
            # 市场崩盘场景（2008年式）
            if 'market_crash' in stress_scenarios:
                crash_params = stress_scenarios['market_crash']
                results['market_crash'] = self._simulate_market_crash(portfolio, market_data, crash_params)

            # 流动性危机场景
            if 'liquidity_crisis' in stress_scenarios:
                crisis_params = stress_scenarios['liquidity_crisis']
                results['liquidity_crisis'] = self._simulate_liquidity_crisis(portfolio, market_data, crisis_params)

            # 利率冲击场景
            if 'interest_rate_shock' in stress_scenarios:
                shock_params = stress_scenarios['interest_rate_shock']
                results['interest_rate_shock'] = self._simulate_interest_rate_shock(portfolio, market_data,
                                                                                    shock_params)

            # 波动率飙升场景
            if 'volatility_spike' in stress_scenarios:
                spike_params = stress_scenarios['volatility_spike']
                results['volatility_spike'] = self._simulate_volatility_spike(portfolio, market_data, spike_params)

            # 相关性断裂场景
            if 'correlation_breakdown' in stress_scenarios:
                breakdown_params = stress_scenarios['correlation_breakdown']
                results['correlation_breakdown'] = self._simulate_correlation_breakdown(portfolio, market_data,
                                                                                        breakdown_params)

            return results

        except Exception as e:
            logger.error(f"压力测试执行失败: {e}")
            return {}

    def _simulate_market_crash(self, portfolio: PortfolioState, market_data: Dict,
                               params: Dict) -> Dict[str, Any]:
        """模拟市场崩盘场景"""
        try:
            # 默认参数：股市下跌50%，相关性增加到0.9，波动率增加到60%
            equity_decline = params.get('equity_decline', 0.5)
            correlation_increase = params.get('correlation_increase', 0.9)
            volatility_increase = params.get('volatility_increase', 0.6)

            # 计算预期损失
            equity_exposure = sum(
                alloc.weight for alloc in portfolio.allocations.values()
                if alloc.asset_class == 'equity'
            )

            expected_loss = equity_exposure * equity_decline
            max_drawdown = min(-expected_loss, -0.8)  # 至少80%的最大回撤

            return {
                'scenario': 'market_crash',
                'expected_loss': expected_loss,
                'max_drawdown': max_drawdown,
                'liquidity_impact': 'high',
                'recovery_time': '12-24 months',
                'pass': expected_loss > -0.3  # 损失不超过30%为通过
            }

        except Exception as e:
            logger.error(f"市场崩盘模拟失败: {e}")
            return {'scenario': 'market_crash', 'error': str(e)}

    def _simulate_liquidity_crisis(self, portfolio: PortfolioState, market_data: Dict,
                                   params: Dict) -> Dict[str, Any]:
        """模拟流动性危机场景"""
        try:
            # 默认参数：流动性下降80%，交易成本增加5倍
            liquidity_reduction = params.get('liquidity_reduction', 0.8)
            cost_increase = params.get('cost_increase', 5.0)

            # 评估流动性最差的资产
            illiquid_assets = []
            for symbol, allocation in portfolio.allocations.items():
                liquidity_data = market_data.get('liquidity', {}).get(symbol, {})
                volume_ratio = liquidity_data.get('volume', 0) / max(liquidity_data.get('avg_daily_volume', 1), 1)

                if volume_ratio < 0.5:  # 交易量低于平均水平50%
                    illiquid_assets.append({
                        'symbol': symbol,
                        'weight': allocation.weight,
                        'liquidity_score': volume_ratio
                    })

            # 计算潜在损失
            illiquid_exposure = sum(asset['weight'] for asset in illiquid_assets)
            expected_slippage = illiquid_exposure * liquidity_reduction * 0.1  # 假设10%的滑点

            return {
                'scenario': 'liquidity_crisis',
                'illiquid_exposure': illiquid_exposure,
                'expected_slippage': expected_slippage,
                'affected_assets': illiquid_assets,
                'trading_cost_increase': cost_increase,
                'pass': illiquid_exposure < 0.2 and expected_slippage < 0.02  # 非流动性暴露<20%，滑点<2%
            }

        except Exception as e:
            logger.error(f"流动性危机模拟失败: {e}")
            return {'scenario': 'liquidity_crisis', 'error': str(e)}

    def _run_scenario_analysis(self, portfolio: PortfolioState, market_data: Dict) -> Dict[str, Any]:
        """运行场景分析"""
        scenarios = self.risk_config.get('scenario_analysis', {})
        results = {}

        try:
            # 经济增长场景
            if 'economic_growth' in scenarios:
                growth_params = scenarios['economic_growth']
                results['economic_growth'] = self._simulate_economic_growth(portfolio, market_data, growth_params)

            # 经济衰退场景
            if 'economic_recession' in scenarios:
                recession_params = scenarios['economic_recession']
                results['economic_recession'] = self._simulate_economic_recession(portfolio, market_data,
                                                                                  recession_params)

            # 通胀上升场景
            if 'inflation_rise' in scenarios:
                inflation_params = scenarios['inflation_rise']
                results['inflation_rise'] = self._simulate_inflation_rise(portfolio, market_data, inflation_params)

            # 通缩场景
            if 'deflation' in scenarios:
                deflation_params = scenarios['deflation']
                results['deflation'] = self._simulate_deflation(portfolio, market_data, deflation_params)

            # 地缘政治风险场景
            if 'geopolitical_risk' in scenarios:
                geopolitical_params = scenarios['geopolitical_risk']
                results['geopolitical_risk'] = self._simulate_geopolitical_risk(portfolio, market_data,
                                                                                geopolitical_params)

            return results

        except Exception as e:
            logger.error(f"场景分析执行失败: {e}")
            return {}

    def _calculate_risk_score(self, risk_metrics: Dict, stress_test_results: Dict,
                              scenario_analysis: Dict) -> float:
        """计算综合风险评分"""
        try:
            risk_weights = self.risk_config.get('risk_weights', {
                'volatility': 0.2,
                'var': 0.25,
                'max_drawdown': 0.15,
                'concentration': 0.1,
                'liquidity': 0.1,
                'stress_tests': 0.1,
                'scenario_analysis': 0.1
            })

            # 标准化各项风险指标
            volatility_score = min(risk_metrics.get('volatility', 0) / 0.3, 1.0)  # 30%波动率为满分
            var_score = min(abs(risk_metrics.get('var_95', 0)) / 0.05, 1.0)  # 5% VaR为满分
            drawdown_score = min(abs(risk_metrics.get('max_drawdown', 0)) / 0.2, 1.0)  # 20%回撤为满分
            concentration_score = risk_metrics.get('concentration', 0) / 100  # HHI指数
            liquidity_score = 1 - risk_metrics.get('liquidity_score', 1.0)  # 反转流动性评分

            # 压力测试评分（取最差结果）
            stress_test_score = 0
            for test_name, result in stress_test_results.items():
                if 'pass' in result and not result['pass']:
                    stress_test_score = max(stress_test_score, 0.8)  # 任何压力测试失败都得高分

            # 场景分析评分
            scenario_score = 0
            for scenario_name, result in scenario_analysis.items():
                if 'expected_return' in result and result['expected_return'] < -0.1:
                    scenario_score = max(scenario_score, abs(result['expected_return']))

            # 计算加权风险评分
            risk_score = (
                    risk_weights['volatility'] * volatility_score +
                    risk_weights['var'] * var_score +
                    risk_weights['max_drawdown'] * drawdown_score +
                    risk_weights['concentration'] * concentration_score +
                    risk_weights['liquidity'] * liquidity_score +
                    risk_weights['stress_tests'] * stress_test_score +
                    risk_weights['scenario_analysis'] * scenario_score
            )

            return min(max(risk_score, 0.0), 1.0)

        except Exception as e:
            logger.error(f"风险评分计算失败: {e}")
            return 0.5  # 默认中等风险

    def _apply_risk_control(self, portfolio: PortfolioState, risk_assessment: Dict) -> PortfolioState:
        """应用风险控制措施"""
        try:
            if risk_assessment['approved']:
                return portfolio

            logger.warning(f"应用风险控制措施: {risk_assessment['reason']}")

            # 根据风险类型应用不同的控制措施
            controlled_portfolio = copy.deepcopy(portfolio)

            # 降低高风险资产权重
            for violation in risk_assessment['violations']:
                if '波动率超限' in violation or 'VaR超限' in violation:
                    controlled_portfolio = self._reduce_high_risk_assets(controlled_portfolio)
                elif '集中度超限' in violation:
                    controlled_portfolio = self._diversify_portfolio(controlled_portfolio)
                elif '流动性不足' in violation:
                    controlled_portfolio = self._improve_liquidity(controlled_portfolio)
                elif '杠杆超限' in violation:
                    controlled_portfolio = self._reduce_leverage(controlled_portfolio)

            # 确保风险控制后的组合通过检查
            new_risk_assessment = self._assess_portfolio_risk(controlled_portfolio, {})
            if not new_risk_assessment['approved']:
                logger.error("风险控制措施未能有效降低风险，使用最小风险组合")
                controlled_portfolio = self._create_min_risk_portfolio(portfolio)

            return controlled_portfolio

        except Exception as e:
            logger.error(f"风险控制应用失败: {e}")
            return portfolio

    def _reduce_high_risk_assets(self, portfolio: PortfolioState) -> PortfolioState:
        """降低高风险资产权重"""
        try:
            # 识别高风险资产（波动率前20%）
            risk_rankings = []
            for symbol, allocation in portfolio.allocations.items():
                risk_rankings.append((symbol, allocation.expected_risk, allocation.weight))

            # 按风险排序
            risk_rankings.sort(key=lambda x: x[1], reverse=True)
            high_risk_assets = risk_rankings[:max(1, len(risk_rankings) // 5)]  # 前20%

            # 减少高风险资产权重，增加低风险资产权重
            total_reduction = 0
            for symbol, risk, weight in high_risk_assets:
                reduction = min(weight * 0.5, weight - 0.02)  # 最多减少50%，至少保留2%
                portfolio.allocations[symbol].weight -= reduction
                total_reduction += reduction

            # 将减少的权重分配给现金或低风险资产
            cash_weight = portfolio.cash_balance / portfolio.total_value
            portfolio.cash_balance += total_reduction * portfolio.total_value

            return portfolio

        except Exception as e:
            logger.error(f"高风险资产降低失败: {e}")
            return portfolio

    def _generate_rebalance_instructions(self, current_portfolio: PortfolioState,
                                         target_portfolio: PortfolioState,
                                         market_data: Dict) -> Dict[str, Any]:
        """生成再平衡指令"""
        start_time = time.time()

        try:
            if current_portfolio is None:
                return {
                    'instructions': [],
                    'turnover_rate': 0.0,
                    'estimated_cost': 0.0,
                    'tax_implication': 0.0,
                    'execution_priority': 'medium',
                    'status': 'completed'
                }

            instructions = []
            total_turnover = 0.0
            total_cost = 0.0
            total_tax = 0.0

            # 生成每个资产的交易指令
            for symbol, target_alloc in target_portfolio.allocations.items():
                current_alloc = current_portfolio.allocations.get(symbol)
                current_weight = current_alloc.weight if current_alloc else 0
                target_weight = target_alloc.weight

                # 计算权重变化
                weight_change = target_weight - current_weight
                if abs(weight_change) < 0.001:  # 忽略小于0.1%的变化
                    continue

                # 计算交易数量和价值
                current_price = market_data['prices'][symbol]['close'][-1] if symbol in market_data[
                    'prices'] else target_alloc.target_value / target_alloc.notional
                trade_value = abs(weight_change) * target_portfolio.total_value
                trade_quantity = trade_value / current_price

                # 考虑最小交易单位和手数
                lot_size = self._get_lot_size(symbol)
                if lot_size > 1:
                    trade_quantity = round(trade_quantity / lot_size) * lot_size
                    trade_value = trade_quantity * current_price

                # 确定交易方向
                action = 'BUY' if weight_change > 0 else 'SELL'

                # 计算交易成本和税费
                transaction_cost = self._calculate_transaction_cost(symbol, trade_value)
                tax_implication = self._calculate_tax_implication(symbol, trade_quantity, current_price, action)

                instruction = {
                    'symbol': symbol,
                    'action': action,
                    'quantity': trade_quantity,
                    'price': current_price,
                    'value': trade_value,
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'weight_change': weight_change,
                    'transaction_cost': transaction_cost,
                    'tax_implication': tax_implication,
                    'execution_priority': self._determine_execution_priority(symbol, weight_change, market_data),
                    'valid_until': (datetime.now() + timedelta(hours=24)).isoformat(),
                    'constraints': self._get_trading_constraints(symbol)
                }

                instructions.append(instruction)
                total_turnover += trade_value
                total_cost += transaction_cost
                total_tax += tax_implication

            # 现金调整指令
            cash_change = target_portfolio.cash_balance - current_portfolio.cash_balance
            if abs(cash_change) > 1:  # 忽略小于1美元的现金变化
                cash_instruction = {
                    'symbol': 'CASH',
                    'action': 'DEPOSIT' if cash_change > 0 else 'WITHDRAW',
                    'amount': abs(cash_change),
                    'current_balance': current_portfolio.cash_balance,
                    'target_balance': target_portfolio.cash_balance,
                    'transaction_cost': 0.0,
                    'tax_implication': 0.0,
                    'execution_priority': 'low',
                    'valid_until': (datetime.now() + timedelta(hours=24)).isoformat()
                }
                instructions.append(cash_instruction)
                total_turnover += abs(cash_change)

            # 计算总换手率
            turnover_rate = total_turnover / current_portfolio.total_value if current_portfolio.total_value > 0 else 0

            # 确定整体执行优先级
            overall_priority = self._determine_overall_priority(instructions, market_data)

            processing_time = time.time() - start_time

            return {
                'instructions': instructions,
                'turnover_rate': turnover_rate,
                'estimated_cost': total_cost,
                'tax_implication': total_tax,
                'execution_priority': overall_priority,
                'status': 'generated',
                'generated_at': datetime.now().isoformat(),
                'processing_time': processing_time
            }

        except Exception as e:
            logger.error(f"再平衡指令生成失败: {e}")
            return {
                'instructions': [],
                'turnover_rate': 0.0,
                'estimated_cost': 0.0,
                'tax_implication': 0.0,
                'execution_priority': 'medium',
                'status': 'failed',
                'error': str(e)
            }

    def _get_lot_size(self, symbol: str) -> int:
        """获取交易单位（手数）"""
        # 实际实现中应该从配置或市场数据获取
        lot_sizes = {
            'STOCK': 1,
            'ETF': 1,
            'FUTURES': 1,
            'OPTIONS': 100
        }

        # 简化实现
        if symbol.endswith('.O'):  # 期权
            return lot_sizes['OPTIONS']
        elif symbol.endswith('.F'):  # 期货
            return lot_sizes['FUTURES']
        else:
            return lot_sizes['STOCK']

    def _calculate_transaction_cost(self, symbol: str, trade_value: float) -> float:
        """计算交易成本"""
        try:
            # 获取交易成本配置
            cost_config = self.portfolio_config.get('transaction_costs', {})
            commission_rate = cost_config.get('commission_rate', 0.0005)  # 0.05%
            slippage_rate = cost_config.get('slippage_rate', 0.0002)  # 0.02%
            exchange_fee = cost_config.get('exchange_fee', 0.0001)  # 0.01%

            # 计算总成本
            commission = trade_value * commission_rate
            slippage = trade_value * slippage_rate
            exchange_fee_total = trade_value * exchange_fee

            total_cost = commission + slippage + exchange_fee_total

            # 考虑最低佣金
            min_commission = cost_config.get('min_commission', 1.0)
            if commission < min_commission:
                total_cost += (min_commission - commission)

            return total_cost

        except Exception as e:
            logger.error(f"交易成本计算失败: {e}")
            return trade_value * 0.001  # 默认0.1%的成本

        def _calculate_tax_implication(self, symbol: str, quantity: float, price: float,
                                       action: str = 'SELL') -> float:
            """计算税费影响"""
            try:
                if action != 'SELL':
                    return 0.0  # 只有卖出才产生税费

                # 获取税费配置
                tax_config = self.portfolio_config.get('tax_considerations', {})
                capital_gains_rate = tax_config.get('capital_gains_rate', 0.15)  # 15%资本利得税
                holding_period_threshold = tax_config.get('holding_period_threshold', 365)  # 365天为长期持有
                short_term_rate = tax_config.get('short_term_rate', 0.3)  # 30%短期资本利得税
                dividend_tax_rate = tax_config.get('dividend_tax_rate', 0.15)  # 15%股息税
                wash_sale_period = tax_config.get('wash_sale_period', 30)  # 30天洗售规则

                # 获取持仓信息
                position_info = self._get_position_info(symbol)
                if not position_info:
                    return 0.0

                # 计算实现收益
                cost_basis = position_info['average_cost']
                current_value = quantity * price
                original_cost = quantity * cost_basis
                realized_gain = current_value - original_cost

                if realized_gain <= 0:
                    # 亏损可以抵税，但有限制
                    tax_benefit = min(abs(realized_gain) * capital_gains_rate,
                                      tax_config.get('max_loss_deduction', 3000))
                    return -tax_benefit  # 负值表示税务优惠

                # 检查持有期
                holding_period = (datetime.now() - position_info['acquisition_date']).days
                is_long_term = holding_period >= holding_period_threshold

                # 应用相应税率
                if is_long_term:
                    tax_amount = realized_gain * capital_gains_rate
                else:
                    tax_amount = realized_gain * short_term_rate

                # 检查洗售规则
                if self._is_wash_sale(symbol, position_info, wash_sale_period):
                    logger.warning(f"洗售规则触发: {symbol}, 亏损不可抵税")
                    tax_amount = max(tax_amount, 0)  # 洗售规则下亏损不可抵税

                # 考虑州税和地方税
                state_tax_rate = tax_config.get('state_tax_rate', 0.0)
                local_tax_rate = tax_config.get('local_tax_rate', 0.0)
                total_tax_rate = capital_gains_rate + state_tax_rate + local_tax_rate

                # 应用总税率
                tax_amount = realized_gain * total_tax_rate

                # 考虑税务亏损抵扣
                capital_loss_carryover = tax_config.get('capital_loss_carryover', 0)
                if capital_loss_carryover > 0 and realized_gain > 0:
                    deductible_loss = min(capital_loss_carryover, realized_gain)
                    tax_amount -= deductible_loss * total_tax_rate
                    # 更新可抵扣亏损余额
                    tax_config['capital_loss_carryover'] = capital_loss_carryover - deductible_loss

                return tax_amount

            except Exception as e:
                logger.error(f"税费计算失败 {symbol}: {e}")
                # 保守估计，使用最高税率
                return quantity * price * 0.3  # 30%的保守估计

        def _get_position_info(self, symbol: str) -> Optional[Dict[str, Any]]:
            """获取持仓详细信息"""
            try:
                if not self.current_portfolio:
                    return None

                # 从当前组合中获取持仓信息
                allocation = self.current_portfolio.allocations.get(symbol)
                if not allocation:
                    return None

                # 从元数据中提取持仓细节
                metadata = allocation.metadata
                position_info = {
                    'symbol': symbol,
                    'quantity': allocation.notional / allocation.current_value if allocation.current_value > 0 else 0,
                    'average_cost': metadata.get('average_cost', allocation.current_value),
                    'acquisition_date': datetime.fromisoformat(
                        metadata.get('acquisition_date', datetime.now().isoformat())),
                    'acquisition_cost': metadata.get('acquisition_cost', allocation.current_value),
                    'unrealized_gain': metadata.get('unrealized_gain', 0),
                    'dividends_received': metadata.get('dividends_received', 0),
                    'wash_sale_adjustment': metadata.get('wash_sale_adjustment', 0),
                    'tax_lot_method': metadata.get('tax_lot_method', 'FIFO')
                    # FIFO, LIFO, HIFO, Specific Identification
                }

                return position_info

            except Exception as e:
                logger.error(f"持仓信息获取失败 {symbol}: {e}")
                return None

        def _is_wash_sale(self, symbol: str, position_info: Dict[str, Any],
                          wash_sale_period: int) -> bool:
            """检查是否触发洗售规则"""
            try:
                # 检查最近是否有类似交易
                recent_trades = self._get_recent_trades(symbol, wash_sale_period)

                for trade in recent_trades:
                    # 检查是否在30天内买入又卖出同一证券
                    if (trade['action'] == 'BUY' and
                            trade['symbol'] == symbol and
                            (datetime.now() - trade['timestamp']).days <= wash_sale_period):
                        return True

                # 检查是否有未实现的亏损
                if position_info['unrealized_gain'] < 0:
                    # 检查是否在亏损期间有买入操作
                    recent_buys = [t for t in recent_trades if t['action'] == 'BUY']
                    if recent_buys:
                        return True

                return False

            except Exception as e:
                logger.error(f"洗售规则检查失败 {symbol}: {e}")
                return False

        def _get_recent_trades(self, symbol: str, lookback_days: int) -> List[Dict[str, Any]]:
            """获取近期交易记录"""
            try:
                # 从交易历史中筛选相关交易
                recent_trades = []
                cutoff_date = datetime.now() - timedelta(days=lookback_days)

                # 这里应该从数据库或交易日志中获取实际数据
                # 简化实现：返回空列表
                return recent_trades

            except Exception as e:
                logger.error(f"近期交易记录获取失败 {symbol}: {e}")
                return []

        def _determine_execution_priority(self, symbol: str, weight_change: float,
                                          market_data: Dict) -> str:
            """确定执行优先级"""
            try:
                # 基于权重变化幅度
                abs_change = abs(weight_change)
                if abs_change > 0.05:  # 5%以上的权重变化
                    return 'high'
                elif abs_change > 0.02:  # 2%-5%的权重变化
                    return 'medium'
                else:
                    return 'low'

                # 基于市场波动性
                volatility = market_data.get('volatility', {}).get(symbol, 0)
                if volatility > 0.3:  # 高波动性
                    return 'high'

                # 基于流动性
                liquidity = market_data.get('liquidity', {}).get(symbol, {}).get('score', 1)
                if liquidity < 0.7:  # 低流动性
                    return 'high'

                # 基于信号强度
                signals = market_data.get('signals', {}).get(symbol, [])
                if signals:
                    strongest_signal = max(signals, key=lambda s: s.get('confidence', 0))
                    if strongest_signal.get('confidence', 0) > 0.8:
                        return 'high'

                return 'medium'

            except Exception as e:
                logger.error(f"执行优先级确定失败 {symbol}: {e}")
                return 'medium'

        def _determine_overall_priority(self, instructions: List[Dict],
                                        market_data: Dict) -> str:
            """确定整体执行优先级"""
            try:
                # 检查是否有高优先级指令
                high_priority_count = sum(1 for instr in instructions
                                          if instr.get('execution_priority') == 'high')

                if high_priority_count > 0:
                    return 'high'

                # 检查市场条件
                market_volatility = market_data.get('overall_volatility', 0)
                if market_volatility > 0.25:  # 25%以上的市场波动率
                    return 'high'

                # 检查流动性条件
                avg_liquidity = np.mean([mkt.get('liquidity', {}).get('score', 1)
                                         for mkt in market_data.get('liquidity', {}).values()])
                if avg_liquidity < 0.6:  # 平均流动性低于60%
                    return 'high'

                return 'medium'

            except Exception as e:
                logger.error(f"整体优先级确定失败: {e}")
                return 'medium'

        def _get_trading_constraints(self, symbol: str) -> Dict[str, Any]:
            """获取交易约束条件"""
            try:
                constraints_config = self.portfolio_config.get('trading_constraints', {})

                # 通用约束
                constraints = {
                    'max_order_size': constraints_config.get('max_order_size', 1000000),
                    'min_order_size': constraints_config.get('min_order_size', 1000),
                    'price_limits': {
                        'max_deviation': constraints_config.get('max_price_deviation', 0.05),
                        'time_in_force': constraints_config.get('time_in_force', 'GTC')
                    },
                    'quantity_limits': {
                        'lot_size': self._get_lot_size(symbol),
                        'min_shares': constraints_config.get('min_shares', 1)
                    },
                    'timing_constraints': {
                        'market_hours_only': constraints_config.get('market_hours_only', True),
                        'avoid_open_close': constraints_config.get('avoid_open_close', True)
                    }
                }

                # 特定符号的约束
                symbol_constraints = constraints_config.get('symbol_specific', {}).get(symbol, {})
                constraints.update(symbol_constraints)

                return constraints

            except Exception as e:
                logger.error(f"交易约束获取失败 {symbol}: {e}")
                return {}

        def _update_portfolio_state(self, new_portfolio: PortfolioState,
                                    rebalance_instructions: Dict[str, Any]):
            """更新组合状态"""
            try:
                # 保存当前组合到历史
                if self.current_portfolio:
                    self.portfolio_history.append(self.current_portfolio)

                    # 限制历史记录长度
                    max_history = self.portfolio_config.get('max_portfolio_history', 1000)
                    if len(self.portfolio_history) > max_history:
                        self.portfolio_history = self.portfolio_history[-max_history:]

                # 更新当前组合
                self.current_portfolio = new_portfolio

                # 更新性能统计
                self.performance_stats['rebalances_executed'] += 1
                self.performance_stats['total_turnover'] += rebalance_instructions.get('turnover_rate', 0)
                self.performance_stats['avg_turnover'] = (
                        self.performance_stats['total_turnover'] /
                        self.performance_stats['rebalances_executed']
                )

                # 记录再平衡操作
                rebalance_record = {
                    'timestamp': datetime.now().isoformat(),
                    'portfolio_id': new_portfolio.portfolio_id,
                    'previous_value': self.current_portfolio.total_value if self.current_portfolio else 0,
                    'new_value': new_portfolio.total_value,
                    'turnover_rate': rebalance_instructions.get('turnover_rate', 0),
                    'transaction_cost': rebalance_instructions.get('estimated_cost', 0),
                    'tax_impact': rebalance_instructions.get('tax_implication', 0),
                    'instructions_count': len(rebalance_instructions.get('instructions', [])),
                    'performance_impact': self._calculate_rebalance_performance_impact()
                }

                # 保存再平衡记录
                self.rebalance_history.append(rebalance_record)

                logger.info(f"组合状态更新完成: 净值={new_portfolio.total_value:,.2f}, " +
                            f"换手率={rebalance_instructions.get('turnover_rate', 0):.2%}")

            except Exception as e:
                logger.error(f"组合状态更新失败: {e}")

        def _calculate_rebalance_performance_impact(self) -> Dict[str, float]:
            """计算再平衡的性能影响"""
            try:
                if len(self.portfolio_history) < 2:
                    return {}

                # 获取最近两次组合状态
                previous_portfolio = self.portfolio_history[-1]
                current_portfolio = self.current_portfolio

                # 计算各项指标变化
                performance_impact = {
                    'value_change': current_portfolio.total_value - previous_portfolio.total_value,
                    'value_change_pct': (current_portfolio.total_value / previous_portfolio.total_value - 1)
                    if previous_portfolio.total_value > 0 else 0,
                    'risk_change': (current_portfolio.risk_metrics.get('volatility', 0) -
                                    previous_portfolio.risk_metrics.get('volatility', 0)),
                    'sharpe_change': (current_portfolio.risk_metrics.get('sharpe_ratio', 0) -
                                      previous_portfolio.risk_metrics.get('sharpe_ratio', 0)),
                    'diversification_change': (current_portfolio.risk_metrics.get('diversification', 0) -
                                               previous_portfolio.risk_metrics.get('diversification', 0))
                }

                return performance_impact

            except Exception as e:
                logger.error(f"再平衡性能影响计算失败: {e}")
                return {}

        def get_portfolio_performance(self, period: str = '30d') -> Dict[str, Any]:
            """获取组合性能报告"""
            try:
                if not self.portfolio_history:
                    return {'error': 'No portfolio history available'}

                # 筛选指定期间的数据
                if period.endswith('d'):
                    days = int(period[:-1])
                    cutoff_date = datetime.now() - timedelta(days=days)
                else:
                    cutoff_date = datetime.now() - timedelta(days=30)  # 默认30天

                relevant_history = [
                    p for p in self.portfolio_history
                    if datetime.fromisoformat(p.timestamp) >= cutoff_date
                ]

                if not relevant_history:
                    return {'error': f'No data available for period: {period}'}

                # 计算性能指标
                performance_metrics = self._calculate_performance_metrics(relevant_history)
                risk_metrics = self._calculate_risk_metrics(relevant_history)
                attribution_analysis = self._perform_attribution_analysis(relevant_history)

                return {
                    'period': period,
                    'start_date': relevant_history[0].timestamp,
                    'end_date': relevant_history[-1].timestamp,
                    'performance_metrics': performance_metrics,
                    'risk_metrics': risk_metrics,
                    'attribution_analysis': attribution_analysis,
                    'rebalance_stats': self._get_rebalance_statistics(period),
                    'comparison_benchmark': self._compare_with_benchmark(relevant_history)
                }

            except Exception as e:
                logger.error(f"组合性能报告生成失败: {e}")
                return {'error': str(e)}

        def _calculate_performance_metrics(self, portfolio_history: List[PortfolioState]) -> Dict[str, float]:
            """计算性能指标"""
            try:
                if len(portfolio_history) < 2:
                    return {}

                # 提取净值序列
                values = [p.total_value for p in portfolio_history]
                timestamps = [datetime.fromisoformat(p.timestamp) for p in portfolio_history]

                # 计算收益
                returns = np.diff(values) / values[:-1]

                # 计算基本指标
                total_return = values[-1] / values[0] - 1
                annualized_return = (1 + total_return) ** (365.25 / (timestamps[-1] - timestamps[0]).days) - 1
                volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
                sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

                # 计算回撤
                drawdowns = self._calculate_drawdowns(values)
                max_drawdown = min(drawdowns) if drawdowns else 0

                return {
                    'total_return': total_return,
                    'annualized_return': annualized_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'sortino_ratio': self._calculate_sortino_ratio(returns),
                    'max_drawdown': max_drawdown,
                    'calmar_ratio': annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0,
                    'win_rate': np.mean([1 for r in returns if r > 0]) if returns else 0,
                    'profit_factor': abs(np.sum([r for r in returns if r > 0]) /
                                         np.sum([r for r in returns if r < 0])) if any(
                        r < 0 for r in returns) else float('inf')
                }

            except Exception as e:
                logger.error(f"性能指标计算失败: {e}")
                return {}

        def _calculate_risk_metrics(self, portfolio_history: List[PortfolioState]) -> Dict[str, float]:
            """计算风险指标"""
            try:
                if len(portfolio_history) < 2:
                    return {}

                values = [p.total_value for p in portfolio_history]
                returns = np.diff(values) / values[:-1]

                if len(returns) < 10:  # 需要足够的数据点
                    return {}

                # 计算风险指标
                risk_metrics = {
                    'var_95': np.percentile(returns, 5),
                    'cvar_95': np.mean([r for r in returns if r <= np.percentile(returns, 5)]),
                    'expected_shortfall': self._calculate_expected_shortfall(returns),
                    'tail_risk': self._calculate_tail_risk(returns),
                    'beta': self._calculate_beta(returns),
                    'alpha': self._calculate_alpha(returns),
                    'tracking_error': self._calculate_tracking_error(returns),
                    'information_ratio': self._calculate_information_ratio(returns)
                }

                return risk_metrics

            except Exception as e:
                logger.error(f"风险指标计算失败: {e}")
                return {}

        def _perform_attribution_analysis(self, portfolio_history: List[PortfolioState]) -> Dict[str, Any]:
            """执行归因分析"""
            try:
                if len(portfolio_history) < 2:
                    return {}

                # 简化实现 - 实际中应该使用专业的归因模型
                attribution = {
                    'asset_allocation_effect': 0.0,
                    'security_selection_effect': 0.0,
                    'currency_effect': 0.0,
                    'timing_effect': 0.0,
                    'interaction_effect': 0.0
                }

                return attribution

            except Exception as e:
                logger.error(f"归因分析失败: {e}")
                return {}

        def optimize_portfolio_allocation(self, objective: PortfolioObjective = None,
                                          constraints: PortfolioConstraints = None) -> Dict[str, Any]:
            """优化投资组合配置"""
            start_time = time.time()

            try:
                if not self.current_portfolio:
                    return {'error': 'No current portfolio available'}

                # 使用配置的目标和约束
                optimization_objective = objective or PortfolioObjective(
                    self.optimization_config.get('objective', 'maximize_sharpe')
                )
                optimization_constraints = constraints or PortfolioConstraints(
                    **self.portfolio_config.get('constraints', {})
                )

                # 执行优化
                optimized_weights = self._critical_line_algorithm(
                    self.current_portfolio, {}, {}, optimization_objective
                )

                # 应用约束
                constrained_weights = self._apply_optimization_constraints(
                    optimized_weights, optimization_constraints
                )

                # 创建优化后的组合
                optimized_portfolio = self._apply_optimized_weights(
                    self.current_portfolio, constrained_weights, {}
                )

                # 计算优化效果
                improvement_metrics = self._calculate_optimization_improvement(
                    self.current_portfolio, optimized_portfolio
                )

                optimization_time = time.time() - start_time

                return {
                    'optimization_time': optimization_time,
                    'original_portfolio': self.current_portfolio.to_dict(),
                    'optimized_portfolio': optimized_portfolio.to_dict(),
                    'weight_changes': self._calculate_weight_changes(
                        self.current_portfolio.allocations,
                        optimized_portfolio.allocations
                    ),
                    'improvement_metrics': improvement_metrics,
                    'constraints_violations': self._check_constraints_violations(
                        optimized_portfolio, optimization_constraints
                    ),
                    'optimization_parameters': {
                        'objective': optimization_objective.value,
                        'constraints': asdict(optimization_constraints)
                    }
                }

            except Exception as e:
                logger.error(f"组合配置优化失败: {e}")
                return {'error': str(e)}

        def run_what_if_analysis(self, scenario_type: str, scenario_params: Dict) -> Dict[str, Any]:
            """运行假设情景分析"""
            try:
                if not self.current_portfolio:
                    return {'error': 'No current portfolio available'}

                scenario_analyzers = {
                    'market_crash': self._analyze_market_crash_scenario,
                    'interest_rate_change': self._analyze_interest_rate_scenario,
                    'sector_rotation': self._analyze_sector_rotation_scenario,
                    'liquidity_crisis': self._analyze_liquidity_crisis_scenario,
                    'black_swan': self._analyze_black_swan_scenario
                }

                analyzer = scenario_analyzers.get(scenario_type)
                if not analyzer:
                    return {'error': f'Unknown scenario type: {scenario_type}'}

                analysis_result = analyzer(self.current_portfolio, scenario_params)

                return {
                    'scenario_type': scenario_type,
                    'scenario_params': scenario_params,
                    'analysis_result': analysis_result,
                    'portfolio_impact': self._calculate_scenario_impact(
                        self.current_portfolio, analysis_result
                    ),
                    'recommendations': self._generate_scenario_recommendations(analysis_result)
                }

            except Exception as e:
                logger.error(f"假设情景分析失败: {e}")
                return {'error': str(e)}

        def save_portfolio_state(self, filepath: str) -> bool:
            """保存组合状态到文件"""
            try:
                portfolio_data = {
                    'current_portfolio': self.current_portfolio.to_dict() if self.current_portfolio else None,
                    'portfolio_history': [p.to_dict() for p in self.portfolio_history],
                    'performance_stats': self.performance_stats,
                    'timestamp': datetime.now().isoformat()
                }

                with open(filepath, 'w') as f:
                    json.dump(portfolio_data, f, indent=2, default=str)

                logger.info(f"组合状态保存成功: {filepath}")
                return True

            except Exception as e:
                logger.error(f"组合状态保存失败: {e}")
                return False

        def load_portfolio_state(self, filepath: str) -> bool:
            """从文件加载组合状态"""
            try:
                with open(filepath, 'r') as f:
                    portfolio_data = json.load(f)

                # 恢复当前组合
                if portfolio_data['current_portfolio']:
                    self.current_portfolio = PortfolioState.from_dict(portfolio_data['current_portfolio'])

                # 恢复历史记录
                self.portfolio_history = [
                    PortfolioState.from_dict(p) for p in portfolio_data['portfolio_history']
                ]

                # 恢复性能统计
                self.performance_stats = portfolio_data['performance_stats']

                logger.info(f"组合状态加载成功: {filepath}")
                return True

            except Exception as e:
                logger.error(f"组合状态加载失败: {e}")
                return False

        def cleanup(self):
            """清理资源"""
            try:
                # 关闭线程池
                if hasattr(self, 'thread_pool'):
                    self.thread_pool.shutdown(wait=True)

                # 清理缓存
                if hasattr(self, '_optimizer_cache'):
                    self._optimizer_cache.clear()

                if hasattr(self, '_constraint_cache'):
                    self._constraint_cache.clear()

                if hasattr(self, '_risk_model_cache'):
                    self._risk_model_cache.clear()

                # 保存当前状态
                if hasattr(self, 'current_portfolio') and self.current_portfolio:
                    self.save_portfolio_state('portfolio_state_backup.json')

                logger.info("组合管理器资源清理完成")

            except Exception as e:
                logger.error(f"组合管理器资源清理失败: {e}")

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
    def validate_portfolio_state(portfolio_state: Dict) -> bool:
        """验证组合状态有效性"""
        try:
            required_fields = ['portfolio_id', 'total_value', 'allocations', 'timestamp']
            if not all(field in portfolio_state for field in required_fields):
                return False

            # 验证数值有效性
            if portfolio_state['total_value'] <= 0:
                return False

            # 验证时间戳格式
            try:
                datetime.fromisoformat(portfolio_state['timestamp'])
            except ValueError:
                return False

            # 验证权重总和
            total_weight = sum(alloc['weight'] for alloc in portfolio_state['allocations'].values())
            if abs(total_weight - 1.0) > 0.01:  # 允许1%的误差
                return False

            # 验证现金余额一致性
            cash_balance = portfolio_state.get('cash_balance', 0)
            if cash_balance < 0 or cash_balance > portfolio_state['total_value']:
                return False

            # 验证每个资产配置
            for symbol, allocation in portfolio_state['allocations'].items():
                if not validate_asset_allocation(allocation):
                    return False

            return True

        except Exception as e:
            logger.error(f"组合状态验证失败: {e}")
            return False

    def validate_asset_allocation(allocation: Dict) -> bool:
        """验证资产配置有效性"""
        try:
            required_fields = ['symbol', 'weight', 'current_value', 'target_value']
            if not all(field in allocation for field in required_fields):
                return False

            # 验证权重范围
            if not 0 <= allocation['weight'] <= 1:
                return False

            # 验证价值非负
            if allocation['current_value'] < 0 or allocation['target_value'] < 0:
                return False

            # 验证风险指标
            if 'risk_contribution' in allocation and allocation['risk_contribution'] < 0:
                return False

            if 'expected_return' in allocation and not -1 <= allocation['expected_return'] <= 1:
                return False

            if 'expected_risk' in allocation and not 0 <= allocation['expected_risk'] <= 1:
                return False

            return True

        except Exception as e:
            logger.error(f"资产配置验证失败: {e}")
            return False

    def calculate_portfolio_metrics(portfolio_state: Dict, market_data: Dict) -> Dict[str, Any]:
        """计算组合性能指标"""
        try:
            if not validate_portfolio_state(portfolio_state):
                raise ValueError("无效的组合状态")

            metrics = {
                'total_value': portfolio_state['total_value'],
                'cash_weight': portfolio_state.get('cash_balance', 0) / portfolio_state['total_value'],
                'equity_exposure': 0.0,
                'fixed_income_exposure': 0.0,
                'alternative_exposure': 0.0,
                'sector_concentration': {},
                'geographic_concentration': {},
                'risk_metrics': {},
                'performance_metrics': {}
            }

            # 计算资产类别暴露
            for allocation in portfolio_state['allocations'].values():
                asset_class = allocation.get('asset_class', 'equity')
                if asset_class == 'equity':
                    metrics['equity_exposure'] += allocation['weight']
                elif asset_class == 'fixed_income':
                    metrics['fixed_income_exposure'] += allocation['weight']
                else:
                    metrics['alternative_exposure'] += allocation['weight']

                # 计算行业集中度
                sector = allocation.get('sector', 'unknown')
                if sector not in metrics['sector_concentration']:
                    metrics['sector_concentration'][sector] = 0
                metrics['sector_concentration'][sector] += allocation['weight']

                # 计算地域集中度
                region = allocation.get('region', 'unknown')
                if region not in metrics['geographic_concentration']:
                    metrics['geographic_concentration'][region] = 0
                metrics['geographic_concentration'][region] += allocation['weight']

            # 计算风险指标
            metrics['risk_metrics'] = calculate_risk_metrics(portfolio_state, market_data)

            # 计算性能指标
            metrics['performance_metrics'] = calculate_performance_metrics(portfolio_state, market_data)

            return metrics

        except Exception as e:
            logger.error(f"组合指标计算失败: {e}")
            return {}

    def calculate_risk_metrics(portfolio_state: Dict, market_data: Dict) -> Dict[str, float]:
        """计算组合风险指标"""
        try:
            # 获取协方差矩阵
            symbols = list(portfolio_state['allocations'].keys())
            if not symbols or 'covariance_matrix' not in market_data:
                return {}

            cov_matrix = market_data['covariance_matrix']
            weights = np.array([alloc['weight'] for alloc in portfolio_state['allocations'].values()])

            # 计算组合波动率
            portfolio_variance = weights.T @ cov_matrix @ weights
            portfolio_volatility = np.sqrt(portfolio_variance)

            # 计算在险价值 (VaR)
            returns = market_data.get('returns', {})
            if returns:
                portfolio_returns = self._calculate_portfolio_returns(portfolio_state, returns)
                var_95 = np.percentile(portfolio_returns, 5)
                cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])
            else:
                var_95 = -2.33 * portfolio_volatility  # 正态分布假设
                cvar_95 = -2.64 * portfolio_volatility

            # 计算风险贡献
            risk_contributions = self._calculate_risk_contributions(weights, cov_matrix)

            return {
                'volatility': portfolio_volatility,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'max_drawdown': self._calculate_max_drawdown(portfolio_state, market_data),
                'beta': self._calculate_portfolio_beta(portfolio_state, market_data),
                'sharpe_ratio': self._calculate_sharpe_ratio(portfolio_state, market_data),
                'sortino_ratio': self._calculate_sortino_ratio(portfolio_state, market_data),
                'risk_contributions': risk_contributions,
                'diversification_ratio': self._calculate_diversification_ratio(weights, cov_matrix)
            }

        except Exception as e:
            logger.error(f"风险指标计算失败: {e}")
            return {}

    def calculate_performance_metrics(portfolio_state: Dict, market_data: Dict) -> Dict[str, float]:
        """计算组合性能指标"""
        try:
            # 获取历史收益数据
            returns = market_data.get('returns', {})
            if not returns:
                return {}

            portfolio_returns = self._calculate_portfolio_returns(portfolio_state, returns)

            # 计算基本性能指标
            total_return = np.prod(1 + portfolio_returns) - 1
            annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            volatility = np.std(portfolio_returns) * np.sqrt(252)

            # 计算风险调整后收益
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
            calmar_ratio = annualized_return / abs(self._calculate_max_drawdown(portfolio_returns))

            # 计算其他指标
            win_rate = np.mean(portfolio_returns > 0)
            profit_factor = abs(np.sum(portfolio_returns[portfolio_returns > 0]) /
                                np.sum(portfolio_returns[portfolio_returns < 0]))

            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
                'upside_capture': self._calculate_upside_capture(portfolio_returns, market_data),
                'downside_capture': self._calculate_downside_capture(portfolio_returns, market_data)
            }

        except Exception as e:
            logger.error(f"性能指标计算失败: {e}")
            return {}

    def optimize_portfolio_with_constraints(weights: Dict[str, float],
                                            constraints: Dict[str, Any],
                                            market_data: Dict) -> Dict[str, float]:
        """在约束条件下优化组合权重"""
        try:
            symbols = list(weights.keys())
            if not symbols:
                return weights

            # 转换权重为numpy数组
            initial_weights = np.array([weights[symbol] for symbol in symbols])

            # 定义优化问题
            def objective_function(x):
                # 目标函数：最小化风险或最大化夏普比率
                if constraints.get('objective', 'min_risk') == 'min_risk':
                    return x.T @ market_data['covariance_matrix'] @ x
                else:
                    returns = market_data.get('expected_returns', np.zeros(len(x)))
                    return -(returns @ x) / np.sqrt(x.T @ market_data['covariance_matrix'] @ x)

            # 定义约束条件
            constraints_list = []

            # 权重总和为1
            constraints_list.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

            # 单个资产权重约束
            for i, symbol in enumerate(symbols):
                min_weight = constraints.get('min_asset_weight', {}).get(symbol, 0)
                max_weight = constraints.get('max_asset_weight', {}).get(symbol, 1)
                constraints_list.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - min_weight})
                constraints_list.append({'type': 'ineq', 'fun': lambda x, i=i: max_weight - x[i]})

            # 行业权重约束
            sector_constraints = constraints.get('sector_constraints', {})
            for sector, max_weight in sector_constraints.items():
                sector_indices = [i for i, symbol in enumerate(symbols)
                                  if market_data['assets'][symbol].get('sector') == sector]
                if sector_indices:
                    constraints_list.append({'type': 'ineq', 'fun': lambda x, indices=sector_indices:
                    max_weight - np.sum(x[indices])})

            # 风险预算约束
            risk_budget = constraints.get('risk_budget', {})
            if risk_budget:
                def risk_budget_constraint(x):
                    risk_contributions = self._calculate_risk_contributions(x, market_data['covariance_matrix'])
                    return np.array([budget - risk_contributions[symbol]
                                     for symbol, budget in risk_budget.items()])

                constraints_list.append({'type': 'ineq', 'fun': risk_budget_constraint})

            # 执行优化
            result = opt.minimize(
                objective_function,
                initial_weights,
                method='SLSQP',
                constraints=constraints_list,
                bounds=[(0, 1) for _ in range(len(symbols))],
                options={'maxiter': 1000, 'ftol': 1e-6}
            )

            if result.success:
                optimized_weights = {symbol: result.x[i] for i, symbol in enumerate(symbols)}
                return optimized_weights
            else:
                logger.warning(f"优化失败: {result.message}")
                return weights

        except Exception as e:
            logger.error(f"约束优化失败: {e}")
            return weights

    def generate_rebalance_report(current_portfolio: Dict, target_portfolio: Dict,
                                  market_data: Dict) -> Dict[str, Any]:
        """生成再平衡报告"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'current_value': current_portfolio['total_value'],
                'target_value': target_portfolio['total_value'],
                'value_change': target_portfolio['total_value'] - current_portfolio['total_value'],
                'weight_changes': {},
                'turnover_analysis': {},
                'cost_analysis': {},
                'risk_impact': {},
                'performance_impact': {},
                'recommendations': []
            }

            # 分析权重变化
            for symbol in set(current_portfolio['allocations'].keys()) | set(target_portfolio['allocations'].keys()):
                current_weight = current_portfolio['allocations'].get(symbol, {}).get('weight', 0)
                target_weight = target_portfolio['allocations'].get(symbol, {}).get('weight', 0)
                weight_change = target_weight - current_weight

                report['weight_changes'][symbol] = {
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'weight_change': weight_change,
                    'change_pct': weight_change / current_weight if current_weight > 0 else float('inf')
                }

            # 分析换手率
            total_turnover = sum(abs(change['weight_change']) * current_portfolio['total_value']
                                 for change in report['weight_changes'].values())
            report['turnover_analysis'] = {
                'total_turnover': total_turnover,
                'turnover_rate': total_turnover / current_portfolio['total_value'],
                'largest_changes': sorted(report['weight_changes'].items(),
                                          key=lambda x: abs(x[1]['weight_change']), reverse=True)[:5]
            }

            # 分析成本影响
            transaction_costs = sum(
                self._calculate_transaction_cost(symbol,
                                                 abs(change['weight_change']) * current_portfolio['total_value'])
                for symbol, change in report['weight_changes'].items()
            )
            tax_implications = sum(
                self._calculate_tax_implication(symbol, abs(change['weight_change']) * current_portfolio['total_value'])
                for symbol, change in report['weight_changes'].items()
            )

            report['cost_analysis'] = {
                'transaction_costs': transaction_costs,
                'tax_implications': tax_implications,
                'total_cost': transaction_costs + tax_implications,
                'cost_as_pct': (transaction_costs + tax_implications) / current_portfolio['total_value']
            }

            # 分析风险影响
            current_risk = calculate_risk_metrics(current_portfolio, market_data)
            target_risk = calculate_risk_metrics(target_portfolio, market_data)

            report['risk_impact'] = {
                'volatility_change': target_risk.get('volatility', 0) - current_risk.get('volatility', 0),
                'var_change': target_risk.get('var_95', 0) - current_risk.get('var_95', 0),
                'diversification_change': target_risk.get('diversification_ratio', 0) - current_risk.get(
                    'diversification_ratio', 0),
                'risk_contribution_changes': self._compare_risk_contributions(current_risk, target_risk)
            }

            # 分析性能影响
            current_perf = calculate_performance_metrics(current_portfolio, market_data)
            target_perf = calculate_performance_metrics(target_portfolio, market_data)

            report['performance_impact'] = {
                'expected_return_change': target_perf.get('annualized_return', 0) - current_perf.get(
                    'annualized_return', 0),
                'sharpe_change': target_perf.get('sharpe_ratio', 0) - current_perf.get('sharpe_ratio', 0),
                'drawdown_change': target_perf.get('max_drawdown', 0) - current_perf.get('max_drawdown', 0)
            }

            # 生成建议
            report['recommendations'] = generate_rebalance_recommendations(report, market_data)

            return report

        except Exception as e:
            logger.error(f"再平衡报告生成失败: {e}")
            return {'error': str(e)}

    def generate_rebalance_recommendations(report: Dict, market_data: Dict) -> List[Dict]:
        """生成再平衡建议"""
        recommendations = []

        try:
            # 高换手率警告
            if report['turnover_analysis']['turnover_rate'] > 0.1:
                recommendations.append({
                    'type': 'warning',
                    'priority': 'high',
                    'message': f'高换手率: {report["turnover_analysis"]["turnover_rate"]:.2%}',
                    'suggestion': '考虑分批执行或延长再平衡周期'
                })

            # 高交易成本警告
            if report['cost_analysis']['cost_as_pct'] > 0.005:
                recommendations.append({
                    'type': 'warning',
                    'priority': 'high',
                    'message': f'高交易成本: {report["cost_analysis"]["cost_as_pct"]:.2%}',
                    'suggestion': '优化执行策略，使用限价单和算法交易'
                })

            # 风险增加警告
            if report['risk_impact']['volatility_change'] > 0.05:
                recommendations.append({
                    'type': 'warning',
                    'priority': 'medium',
                    'message': f'波动率增加: {report["risk_impact"]["volatility_change"]:.2%}',
                    'suggestion': '重新评估风险预算和约束条件'
                })

            # 分散度改善建议
            if report['risk_impact']['diversification_change'] > 0.1:
                recommendations.append({
                    'type': 'improvement',
                    'priority': 'low',
                    'message': f'分散度改善: {report["risk_impact"]["diversification_change"]:.2f}',
                    'suggestion': '继续保持当前的分散化策略'
                })

            # 性能改善预期
            if report['performance_impact']['expected_return_change'] > 0.01:
                recommendations.append({
                    'type': 'improvement',
                    'priority': 'medium',
                    'message': f'预期收益提升: {report["performance_impact"]["expected_return_change"]:.2%}',
                    'suggestion': '执行再平衡以获取预期收益提升'
                })

            # 执行优先级建议
            large_changes = report['turnover_analysis']['largest_changes'][:3]
            for symbol, change in large_changes:
                if abs(change['weight_change']) > 0.05:
                    recommendations.append({
                        'type': 'execution',
                        'priority': 'high',
                        'message': f'大额调整: {symbol} {change["weight_change"]:+.2%}',
                        'suggestion': '优先执行，考虑使用算法交易减少市场冲击'
                    })

            return recommendations

        except Exception as e:
            logger.error(f"建议生成失败: {e}")
            return []

    def backtest_portfolio_strategy(historical_data: Dict, strategy_config: Dict) -> Dict[str, Any]:
        """回测组合策略"""
        try:
            # 初始化回测参数
            initial_capital = strategy_config.get('initial_capital', 1000000)
            rebalance_frequency = strategy_config.get('rebalance_frequency', 'monthly')
            transaction_cost = strategy_config.get('transaction_cost', 0.001)

            # 准备回测数据
            dates = sorted(historical_data.keys())
            portfolio_values = [initial_capital]
            weights_history = []
            trades_history = []

            current_weights = strategy_config.get('initial_weights', {})
            if not current_weights:
                # 等权重初始化
                symbols = list(historical_data[dates[0]]['prices'].keys())
                current_weights = {symbol: 1.0 / len(symbols) for symbol in symbols}

            # 执行回测
            for i in range(1, len(dates)):
                current_date = dates[i]
                prev_date = dates[i - 1]

                # 计算期间收益
                period_returns = self._calculate_period_returns(
                    historical_data[prev_date]['prices'],
                    historical_data[current_date]['prices']
                )

                # 更新组合价值
                portfolio_return = sum(current_weights[symbol] * period_returns.get(symbol, 0)
                                       for symbol in current_weights.keys())
                new_value = portfolio_values[-1] * (1 + portfolio_return)
                portfolio_values.append(new_value)

                # 检查再平衡条件
                if self._should_rebalance(current_date, rebalance_frequency, strategy_config):
                    # 生成新权重
                    target_weights = self._generate_target_weights(
                        current_weights, historical_data[current_date], strategy_config
                    )

                    # 计算换手成本和交易
                    turnover, trades, costs = self._calculate_rebalance_costs(
                        current_weights, target_weights, new_value, transaction_cost
                    )

                    # 应用成本和更新权重
                    new_value -= costs
                    portfolio_values[-1] = new_value
                    current_weights = target_weights

                    # 记录交易
                    trades_history.append({
                        'date': current_date,
                        'trades': trades,
                        'costs': costs,
                        'turnover': turnover
                    })

                weights_history.append(current_weights.copy())

            # 计算回测结果
            returns = np.diff(portfolio_values) / portfolio_values[:-1]

            result = {
                'final_value': portfolio_values[-1],
                'total_return': portfolio_values[-1] / initial_capital - 1,
                'annualized_return': (portfolio_values[-1] / initial_capital) ** (252 / len(dates)) - 1,
                'volatility': np.std(returns) * np.sqrt(252),
                'max_drawdown': self._calculate_max_drawdown(portfolio_values),
                'sharpe_ratio': (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252)) if np.std(
                    returns) > 0 else 0,
                'sortino_ratio': self._calculate_sortino_ratio(returns),
                'turnover': sum(trade['turnover'] for trade in trades_history) / len(trades_history),
                'total_costs': sum(trade['costs'] for trade in trades_history),
                'number_of_trades': sum(len(trade['trades']) for trade in trades_history),
                'weights_history': weights_history,
                'trades_history': trades_history,
                'portfolio_values': portfolio_values
            }

            return result

        except Exception as e:
            logger.error(f"策略回测失败: {e}")
            return {'error': str(e)}

    # 风险平价优化相关函数
    def risk_parity_optimization(cov_matrix: pd.DataFrame,
                                 risk_budget: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """风险平价优化"""
        try:
            n = cov_matrix.shape[0]
            symbols = cov_matrix.index.tolist()

            # 默认等风险贡献
            if risk_budget is None:
                risk_budget = {symbol: 1.0 / n for symbol in symbols}

            # 定义优化问题
            def objective_function(weights):
                portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
                risk_contributions = (weights * (cov_matrix @ weights)) / portfolio_risk
                budget_deviation = np.sum([(risk_contributions[i] - risk_budget[symbols[i]]) ** 2
                                           for i in range(n)])
                return budget_deviation

            # 初始权重（等权重）
            x0 = np.ones(n) / n

            # 约束条件：权重和为1，权重非负
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            bounds = [(0, 1) for _ in range(n)]

            # 执行优化
            result = opt.minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-6}
            )

            if result.success:
                optimized_weights = {symbols[i]: result.x[i] for i in range(n)}
                return optimized_weights
            else:
                logger.warning(f"风险平价优化失败: {result.message}")
                return {symbol: 1.0 / n for symbol in symbols}

        except Exception as e:
            logger.error(f"风险平价优化错误: {e}")
            return {symbol: 1.0 / n for symbol in symbols}

    # Black-Litterman模型相关函数
    def black_litterman_optimization(market_weights: Dict[str, float],
                                     cov_matrix: pd.DataFrame,
                                     views: Dict[str, Tuple[float, float]],
                                     tau: float = 0.05) -> Dict[str, float]:
        """Black-Litterman模型优化"""
        try:
            symbols = list(market_weights.keys())
            n = len(symbols)

            # 验证输入数据
            if not symbols or cov_matrix.empty or not views:
                logger.warning("Black-Litterman优化: 输入数据不完整")
                return market_weights

            # 检查协方差矩阵维度
            if cov_matrix.shape != (n, n):
                logger.warning("Black-Litterman优化: 协方差矩阵维度不匹配")
                return market_weights

            # 转换市场权重为向量
            pi = np.array([market_weights[symbol] for symbol in symbols])

            # 构建观点矩阵
            P = np.zeros((len(views), n))
            Q = np.zeros(len(views))
            Omega = np.zeros((len(views), len(views)))

            for i, (symbol, (return_view, confidence)) in enumerate(views.items()):
                if symbol in symbols:
                    j = symbols.index(symbol)
                    P[i, j] = 1
                    Q[i] = return_view
                    Omega[i, i] = confidence ** 2
                else:
                    logger.warning(f"Black-Litterman优化: 观点中的符号 {symbol} 不在组合中")

            # 如果没有有效观点，返回市场权重
            if np.all(P == 0):
                logger.warning("Black-Litterman优化: 没有有效的观点")
                return market_weights

            # 移除全零行
            valid_rows = np.any(P != 0, axis=1)
            P = P[valid_rows]
            Q = Q[valid_rows]
            Omega = Omega[valid_rows][:, valid_rows]

            # 检查Omega是否可逆
            if np.linalg.det(Omega) == 0:
                logger.warning("Black-Litterman优化: Omega矩阵奇异，添加正则化")
                Omega += np.eye(len(Omega)) * 1e-6

            # 计算Black-Litterman后验收益
            pi = pi.reshape(-1, 1)
            Q = Q.reshape(-1, 1)

            # 计算后验收益
            cov_matrix_inv = np.linalg.inv(cov_matrix)
            term1 = np.linalg.inv(tau * cov_matrix_inv + P.T @ np.linalg.inv(Omega) @ P)
            term2 = tau * cov_matrix_inv @ pi + P.T @ np.linalg.inv(Omega) @ Q
            posterior_returns = term1 @ term2

            # 计算后验协方差矩阵
            posterior_cov = cov_matrix + term1

            # 使用后验收益和协方差进行均值-方差优化
            ef = EfficientFrontier(posterior_returns.flatten(), posterior_cov)
            ef.max_sharpe()
            weights = ef.clean_weights()

            # 转换为字典格式
            optimized_weights = {symbol: weights[symbol] for symbol in symbols}

            # 确保权重和为1
            total_weight = sum(optimized_weights.values())
            if abs(total_weight - 1.0) > 1e-6:
                # 重新归一化权重
                optimized_weights = {symbol: weight / total_weight
                                     for symbol, weight in optimized_weights.items()}

            logger.info("Black-Litterman优化完成")
            return optimized_weights

        except np.linalg.LinAlgError as e:
            logger.error(f"Black-Litterman优化线性代数错误: {e}")
            return market_weights
        except Exception as e:
            logger.error(f"Black-Litterman优化失败: {e}")
            return market_weights

    def hierarchical_risk_parity_optimization(returns_data: pd.DataFrame,
                                              covariance_matrix: pd.DataFrame,
                                              linkage_method: str = 'ward') -> Dict[str, float]:
        """分层风险平价优化"""
        try:
            if returns_data.empty or covariance_matrix.empty:
                logger.warning("HRP优化: 输入数据为空")
                return {}

            # 使用Riskfolio-Lib进行HRP优化
            port = rp.Portfolio(returns=returns_data)

            # 计算资产统计
            port.assets_stats(method_mu='hist', method_cov='hist')

            # 选择链接方法
            if linkage_method not in ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']:
                linkage_method = 'ward'

            # 执行HRP优化
            port.hrp_optimization(model='HRP', linkage=linkage_method, max_k=10)

            if port.w is None:
                logger.warning("HRP优化: 无法计算权重")
                return {}

            # 获取优化权重
            weights = port.w.to_dict()

            # 确保所有权重非负且和为1
            total_weight = sum(weights.values())
            if total_weight <= 0:
                logger.warning("HRP优化: 权重总和非正")
                return {symbol: 1.0 / len(weights) for symbol in weights.keys()}

            # 归一化权重
            normalized_weights = {symbol: weight / total_weight for symbol, weight in weights.items()}

            logger.info(f"分层风险平价优化完成: 链接方法={linkage_method}")
            return normalized_weights

        except Exception as e:
            logger.error(f"分层风险平价优化失败: {e}")
            return {}

    def critical_line_algorithm_optimization(expected_returns: pd.Series,
                                             covariance_matrix: pd.DataFrame,
                                             risk_free_rate: float = 0.02,
                                             constraints: Dict[str, Any] = None) -> Dict[str, float]:
        """关键线算法优化"""
        try:
            if expected_returns.empty or covariance_matrix.empty:
                logger.warning("CLA优化: 输入数据为空")
                return {}

            symbols = expected_returns.index.tolist()

            # 使用PyPortfolioOpt的关键线算法
            ef = EfficientFrontier(expected_returns, covariance_matrix)

            # 应用约束条件
            if constraints:
                # 权重上下限
                weight_bounds = constraints.get('weight_bounds', (0, 1))
                ef.set_weights_bounds(weight_bounds[0], weight_bounds[1])

                # 行业约束
                if 'sector_constraints' in constraints:
                    for sector, max_weight in constraints['sector_constraints'].items():
                        sector_symbols = [s for s in symbols if self._get_asset_sector(s) == sector]
                        if sector_symbols:
                            ef.add_sector_constraints({sector: sector_symbols},
                                                      [max_weight], [max_weight])

            # 执行优化
            ef.efficient_return(target_return=expected_returns.mean())
            weights = ef.clean_weights()

            # 计算有效前沿上的最优组合
            optimal_weights = {}
            for symbol in symbols:
                optimal_weights[symbol] = weights.get(symbol, 0)

            # 确保权重和为1
            total_weight = sum(optimal_weights.values())
            if abs(total_weight - 1.0) > 1e-6:
                optimal_weights = {symbol: weight / total_weight
                                   for symbol, weight in optimal_weights.items()}

            logger.info("关键线算法优化完成")
            return optimal_weights

        except Exception as e:
            logger.error(f"关键线算法优化失败: {e}")
            return {}

    def risk_budget_optimization(covariance_matrix: pd.DataFrame,
                                 risk_budgets: Dict[str, float]) -> Dict[str, float]:
        """风险预算优化"""
        try:
            if covariance_matrix.empty or not risk_budgets:
                logger.warning("风险预算优化: 输入数据为空")
                return {}

            symbols = covariance_matrix.index.tolist()
            n = len(symbols)

            # 验证风险预算
            total_budget = sum(risk_budgets.values())
            if total_budget <= 0:
                logger.warning("风险预算优化: 风险预算总和非正")
                return {symbol: 1.0 / n for symbol in symbols}

            # 归一化风险预算
            normalized_budgets = {symbol: budget / total_budget
                                  for symbol, budget in risk_budgets.items()}

            # 定义优化问题
            def objective(weights):
                portfolio_risk = np.sqrt(weights.T @ covariance_matrix @ weights)
                risk_contributions = (weights * (covariance_matrix @ weights)) / portfolio_risk
                budget_deviation = sum((risk_contributions[i] - normalized_budgets[symbols[i]]) ** 2
                                       for i in range(n))
                return budget_deviation

            # 初始权重（等权重）
            x0 = np.ones(n) / n

            # 约束条件
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'ineq', 'fun': lambda x: x}  # 非负约束
            ]

            # 执行优化
            result = opt.minimize(
                objective,
                x0,
                method='SLSQP',
                constraints=constraints,
                bounds=[(0, 1) for _ in range(n)],
                options={'maxiter': 1000, 'ftol': 1e-6}
            )

            if result.success:
                optimized_weights = {symbols[i]: result.x[i] for i in range(n)}

                # 验证风险贡献
                final_risk_contributions = self._calculate_risk_contributions(
                    np.array(list(optimized_weights.values())), covariance_matrix
                )

                # 计算预算偏差
                budget_deviation = sum(abs(final_risk_contributions[symbols[i]] - normalized_budgets[symbols[i]])
                                       for i in range(n))

                logger.info(f"风险预算优化完成: 预算偏差={budget_deviation:.6f}")
                return optimized_weights
            else:
                logger.warning(f"风险预算优化失败: {result.message}")
                return {symbol: 1.0 / n for symbol in symbols}

        except Exception as e:
            logger.error(f"风险预算优化失败: {e}")
            return {}

    def mean_variance_optimization(expected_returns: pd.Series,
                                   covariance_matrix: pd.DataFrame,
                                   objective: str = 'max_sharpe',
                                   constraints: Dict[str, Any] = None) -> Dict[str, float]:
        """均值-方差优化"""
        try:
            if expected_returns.empty or covariance_matrix.empty:
                logger.warning("均值-方差优化: 输入数据为空")
                return {}

            # 使用PyPortfolioOpt进行均值-方差优化
            ef = EfficientFrontier(expected_returns, covariance_matrix)

            # 应用约束条件
            if constraints:
                # 权重约束
                weight_bounds = constraints.get('weight_bounds', (0, 1))
                ef.set_weights_bounds(weight_bounds[0], weight_bounds[1])

                # 位置限制
                if 'position_limit' in constraints:
                    ef.add_objective(objective_functions.L2_reg, gamma=constraints['position_limit'])

                # 市场中性
                if constraints.get('market_neutral', False):
                    ef.add_constraint(lambda w: np.sum(w) == 0)

            # 根据目标执行优化
            if objective == 'max_sharpe':
                ef.max_sharpe()
            elif objective == 'min_volatility':
                ef.min_volatility()
            elif objective == 'efficient_risk':
                target_risk = constraints.get('target_risk', expected_returns.std())
                ef.efficient_risk(target_risk)
            elif objective == 'efficient_return':
                target_return = constraints.get('target_return', expected_returns.mean())
                ef.efficient_return(target_return)
            elif objective == 'max_quadratic_utility':
                risk_aversion = constraints.get('risk_aversion', 1.0)
                ef.max_quadratic_utility(risk_aversion=risk_aversion)
            else:
                ef.max_sharpe()  # 默认最大化夏普比率

            weights = ef.clean_weights()

            # 转换为字典格式
            optimized_weights = {symbol: weight for symbol, weight in weights.items()}

            # 计算优化后的性能指标
            if objective == 'max_sharpe':
                performance = ef.portfolio_performance()
                logger.info(f"均值-方差优化完成: 预期收益={performance[0]:.2%}, " +
                            f"波动率={performance[1]:.2%}, 夏普比率={performance[2]:.2f}")
            else:
                logger.info("均值-方差优化完成")

            return optimized_weights

        except Exception as e:
            logger.error(f"均值-方差优化失败: {e}")
            return {}

    def robust_optimization(expected_returns: pd.Series,
                            covariance_matrix: pd.DataFrame,
                            uncertainty_sets: Dict[str, Any],
                            objective: str = 'worst_case') -> Dict[str, float]:
        """鲁棒优化"""
        try:
            if expected_returns.empty or covariance_matrix.empty:
                logger.warning("鲁棒优化: 输入数据为空")
                return {}

            symbols = expected_returns.index.tolist()
            n = len(symbols)

            # 定义鲁棒优化问题
            w = cp.Variable(n)

            # 不确定集参数
            return_uncertainty = uncertainty_sets.get('return_uncertainty', 0.1)
            cov_uncertainty = uncertainty_sets.get('covariance_uncertainty', 0.1)

            # 最坏情况收益
            worst_case_returns = expected_returns - return_uncertainty * np.abs(expected_returns)

            # 最坏情况协方差
            worst_case_cov = covariance_matrix + cov_uncertainty * np.eye(n)

            # 组合收益和风险
            portfolio_return = worst_case_returns @ w
            portfolio_risk = cp.quad_form(w, worst_case_cov)

            # 约束条件
            constraints = [
                cp.sum(w) == 1,
                w >= 0
            ]

            # 额外约束
            if 'sector_constraints' in uncertainty_sets:
                for sector, max_weight in uncertainty_sets['sector_constraints'].items():
                    sector_symbols = [i for i, symbol in enumerate(symbols)
                                      if self._get_asset_sector(symbol) == sector]
                    if sector_symbols:
                        constraints.append(cp.sum(w[sector_symbols]) <= max_weight)

            # 目标函数
            if objective == 'worst_case':
                # 最大化最坏情况夏普比率
                risk_free = uncertainty_sets.get('risk_free_rate', 0.02)
                objective_func = cp.Maximize((portfolio_return - risk_free) / cp.sqrt(portfolio_risk))
            elif objective == 'min_max_var':
                # 最小化最大在险价值
                confidence = uncertainty_sets.get('var_confidence', 0.95)
                z_score = stats.norm.ppf(1 - confidence)
                objective_func = cp.Minimize(z_score * cp.sqrt(portfolio_risk) - portfolio_return)
            else:
                objective_func = cp.Maximize(
                    portfolio_return - uncertainty_sets.get('risk_aversion', 1.0) * portfolio_risk)

            # 求解优化问题
            problem = cp.Problem(objective_func, constraints)
            problem.solve()

            if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                logger.warning(f"鲁棒优化求解失败: {problem.status}")
                return {}

            # 获取优化权重
            optimized_weights = {symbols[i]: w.value[i] for i in range(n)}

            # 处理数值误差
            optimized_weights = {symbol: max(0, weight) for symbol, weight in optimized_weights.items()}
            total_weight = sum(optimized_weights.values())
            if total_weight > 0:
                optimized_weights = {symbol: weight / total_weight for symbol, weight in optimized_weights.items()}

            logger.info("鲁棒优化完成")
            return optimized_weights

        except Exception as e:
            logger.error(f"鲁棒优化失败: {e}")
            return {}

    def factor_based_optimization(factor_exposures: pd.DataFrame,
                                  factor_covariance: pd.DataFrame,
                                  specific_risk: pd.Series,
                                  expected_returns: pd.Series) -> Dict[str, float]:
        """基于因子的优化"""
        try:
            if factor_exposures.empty or factor_covariance.empty or specific_risk.empty:
                logger.warning("因子优化: 输入数据为空")
                return {}

            symbols = factor_exposures.index.tolist()
            n = len(symbols)
            k = factor_exposures.shape[1]  # 因子数量

            # 计算总协方差矩阵
            total_covariance = factor_exposures @ factor_covariance @ factor_exposures.T + np.diag(specific_risk)

            # 使用均值-方差优化
            optimized_weights = mean_variance_optimization(
                expected_returns,
                total_covariance,
                objective='max_sharpe'
            )

            # 计算因子暴露
            factor_weights = np.array([optimized_weights[symbol] for symbol in symbols])
            portfolio_factor_exposure = factor_exposures.T @ factor_weights

            logger.info(f"因子优化完成: 因子暴露={portfolio_factor_exposure}")
            return optimized_weights

        except Exception as e:
            logger.error(f"因子优化失败: {e}")
            return {}

    def multi_period_optimization(historical_data: Dict[str, pd.DataFrame],
                                  optimization_horizon: int = 5,
                                  objective: str = 'terminal_wealth') -> Dict[str, float]:
        """多期优化"""
        try:
            if not historical_data:
                logger.warning("多期优化: 历史数据为空")
                return {}

            # 提取收益数据
            returns_data = {}
            for symbol, data in historical_data.items():
                if 'returns' in data:
                    returns_data[symbol] = data['returns']

            if not returns_data:
                logger.warning("多期优化: 没有收益数据")
                return {}

            # 转换为DataFrame
            returns_df = pd.DataFrame(returns_data)

            # 使用滚动窗口优化
            if objective == 'terminal_wealth':
                # 最大化终端财富
                optimized_weights = self._maximize_terminal_wealth(returns_df, optimization_horizon)
            elif objective == 'consumption_utility':
                # 最大化消费效用
                optimized_weights = self._maximize_consumption_utility(returns_df, optimization_horizon)
            else:
                # 默认：均值-方差多期优化
                optimized_weights = self._multi_period_mean_variance(returns_df, optimization_horizon)

            logger.info(f"多期优化完成: 期限={optimization_horizon}, 目标={objective}")
            return optimized_weights

        except Exception as e:
            logger.error(f"多期优化失败: {e}")
            return {}

    def _maximize_terminal_wealth(self, returns_df: pd.DataFrame, horizon: int) -> Dict[str, float]:
        """最大化终端财富"""
        try:
            n_assets = returns_df.shape[1]
            symbols = returns_df.columns.tolist()

            # 使用动态规划
            # 简化实现：使用几何平均收益作为预期收益
            geometric_returns = np.exp(np.log(1 + returns_df).mean()) - 1

            # 使用几何平均收益进行均值-方差优化
            covariance_matrix = returns_df.cov()
            optimized_weights = mean_variance_optimization(
                geometric_returns,
                covariance_matrix,
                objective='max_sharpe'
            )

            return optimized_weights

        except Exception as e:
            logger.error(f"终端财富最大化失败: {e}")
            return {symbol: 1.0 / n_assets for symbol in symbols}

    def _maximize_consumption_utility(self, returns_df: pd.DataFrame, horizon: int) -> Dict[str, float]:
        """最大化消费效用"""
        try:
            n_assets = returns_df.shape[1]
            symbols = returns_df.columns.tolist()

            # 简化实现：使用幂效用函数
            risk_aversion = 2.0  # 风险厌恶系数
            expected_returns = returns_df.mean()
            covariance_matrix = returns_df.cov()

            # 最大化期望效用
            def expected_utility(weights):
                portfolio_return = weights @ expected_returns
                portfolio_risk = weights @ covariance_matrix @ weights
                return portfolio_return - 0.5 * risk_aversion * portfolio_risk

            # 优化问题
            x0 = np.ones(n_assets) / n_assets
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            bounds = [(0, 1) for _ in range(n_assets)]

            result = opt.minimize(
                lambda x: -expected_utility(x),  # 最小化负效用
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if result.success:
                optimized_weights = {symbols[i]: result.x[i] for i in range(n_assets)}
                return optimized_weights
            else:
                return {symbol: 1.0 / n_assets for symbol in symbols}

        except Exception as e:
            logger.error(f"消费效用最大化失败: {e}")
            return {symbol: 1.0 / n_assets for symbol in symbols}

    def _multi_period_mean_variance(self, returns_df: pd.DataFrame, horizon: int) -> Dict[str, float]:
        """多期均值-方差优化"""
        try:
            n_assets = returns_df.shape[1]
            symbols = returns_df.columns.tolist()

            # 使用滚动估计
            rolling_returns = returns_df.rolling(window=horizon).mean().dropna()
            rolling_cov = returns_df.rolling(window=horizon).cov().dropna()

            if rolling_returns.empty or rolling_cov.empty:
                logger.warning("多期均值-方差优化: 滚动窗口数据不足")
                return {symbol: 1.0 / n_assets for symbol in symbols}

            # 使用最近期的估计
            latest_returns = rolling_returns.iloc[-1]
            latest_cov = rolling_cov.xs(rolling_cov.index[-1][0])

            optimized_weights = mean_variance_optimization(
                latest_returns,
                latest_cov,
                objective='max_sharpe'
            )

            return optimized_weights

        except Exception as e:
            logger.error(f"多期均值-方差优化失败: {e}")
            return {symbol: 1.0 / n_assets for symbol in symbols}

    # 组合分析工具函数
    def calculate_portfolio_risk_decomposition(weights: Dict[str, float],
                                               covariance_matrix: pd.DataFrame) -> Dict[str, Any]:
        """计算组合风险分解"""
        try:
            symbols = list(weights.keys())
            n = len(symbols)

            if n == 0 or covariance_matrix.empty:
                return {}

            # 转换为向量
            w = np.array([weights[symbol] for symbol in symbols])

            # 计算组合风险
            portfolio_variance = w.T @ covariance_matrix @ w
            portfolio_risk = np.sqrt(portfolio_variance)

            if portfolio_risk == 0:
                return {symbol: 0 for symbol in symbols}

            # 计算边际风险贡献
            marginal_risk = covariance_matrix @ w / portfolio_risk

            # 计算风险贡献
            risk_contributions = w * marginal_risk

            # 计算百分比贡献
            percent_contributions = risk_contributions / portfolio_risk

            # 转换为字典格式
            decomposition = {}
            for i, symbol in enumerate(symbols):
                decomposition[symbol] = {
                    'marginal_risk': marginal_risk[i],
                    'risk_contribution': risk_contributions[i],
                    'percent_contribution': percent_contributions[i],
                    'standalone_risk': np.sqrt(covariance_matrix.iloc[i, i])
                }

            # 总体风险指标
            decomposition['portfolio'] = {
                'total_risk': portfolio_risk,
                'diversification_ratio': sum(decomposition[symbol]['standalone_risk'] * weights[symbol]
                                             for symbol in symbols) / portfolio_risk,
                'concentration_ratio': sum(contrib['percent_contribution'] ** 2
                                           for contrib in decomposition.values())
            }

            return decomposition

        except Exception as e:
            logger.error(f"风险分解计算失败: {e}")
            return {}

    def calculate_performance_attribution(portfolio_returns: pd.Series,
                                          benchmark_returns: pd.Series,
                                          factor_returns: pd.DataFrame) -> Dict[str, Any]:
        """计算绩效归因"""
        try:
            if portfolio_returns.empty or benchmark_returns.empty:
                return {}

            # 对齐数据时间索引
            common_index = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_index) < 2:
                logger.warning("绩效归因: 数据时间索引不匹配")
                return {}

            portfolio_returns = portfolio_returns.loc[common_index]
            benchmark_returns = benchmark_returns.loc[common_index]

            # 计算超额收益
            excess_returns = portfolio_returns - benchmark_returns

            # 使用回归分析进行归因
            attribution_results = {}

            # 市场择时能力
            market_timing = self._calculate_market_timing(portfolio_returns, benchmark_returns)
            attribution_results['market_timing'] = market_timing

            # 证券选择能力
            stock_selection = self._calculate_stock_selection(portfolio_returns, benchmark_returns)
            attribution_results['stock_selection'] = stock_selection

            # 因子暴露归因
            if not factor_returns.empty:
                factor_attribution = self._calculate_factor_attribution(excess_returns, factor_returns)
                attribution_results['factor_attribution'] = factor_attribution

            # 行业归因
            sector_attribution = self._calculate_sector_attribution(portfolio_returns, benchmark_returns)
            attribution_results['sector_attribution'] = sector_attribution

            # 风格归因
            style_attribution = self._calculate_style_attribution(portfolio_returns, benchmark_returns)
            attribution_results['style_attribution'] = style_attribution

            # 计算总归因效果
            total_attribution = self._calculate_total_attribution(attribution_results, excess_returns)
            attribution_results['total_attribution'] = total_attribution

            # 计算统计显著性
            significance_tests = self._calculate_attribution_significance(attribution_results, excess_returns)
            attribution_results['significance_tests'] = significance_tests

            logger.info("绩效归因分析完成")
            return attribution_results

        except Exception as e:
            logger.error(f"绩效归因计算失败: {e}")
            return {}

    def _calculate_market_timing(self, portfolio_returns: pd.Series,
                                 benchmark_returns: pd.Series) -> Dict[str, Any]:
        """计算市场择时能力"""
        try:
            # 使用Treynor-Mazuy模型
            benchmark_squared = benchmark_returns ** 2

            # 准备回归数据
            X = sm.add_constant(np.column_stack([benchmark_returns, benchmark_squared]))
            y = portfolio_returns

            # 执行回归
            model = sm.OLS(y, X).fit()

            # 解释回归结果
            alpha = model.params[0]  # 选股能力
            beta = model.params[1]  # 系统性风险暴露
            timing_coeff = model.params[2]  # 择时能力系数

            # 计算择时收益
            timing_return = timing_coeff * np.var(benchmark_returns)

            return {
                'alpha': alpha,
                'beta': beta,
                'timing_coefficient': timing_coeff,
                'timing_return': timing_return,
                't_statistic': model.tvalues[2],
                'p_value': model.pvalues[2],
                'r_squared': model.rsquared,
                'significant': model.pvalues[2] < 0.05,
                'model_summary': str(model.summary())
            }

        except Exception as e:
            logger.error(f"市场择时能力计算失败: {e}")
            return {}

    def _calculate_stock_selection(self, portfolio_returns: pd.Series,
                                   benchmark_returns: pd.Series) -> Dict[str, Any]:
        """计算证券选择能力"""
        try:
            # 使用Jensen's Alpha模型
            X = sm.add_constant(benchmark_returns)
            y = portfolio_returns

            model = sm.OLS(y, X).fit()

            # 选股能力（Alpha）
            selection_alpha = model.params[0]

            # 计算信息比率
            tracking_error = np.std(portfolio_returns - benchmark_returns)
            information_ratio = selection_alpha / tracking_error if tracking_error > 0 else 0

            return {
                'selection_alpha': selection_alpha,
                'beta': model.params[1],
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                't_statistic': model.tvalues[0],
                'p_value': model.pvalues[0],
                'r_squared': model.rsquared,
                'significant': model.pvalues[0] < 0.05,
                'model_summary': str(model.summary())
            }

        except Exception as e:
            logger.error(f"证券选择能力计算失败: {e}")
            return {}

    def _calculate_factor_attribution(self, excess_returns: pd.Series,
                                      factor_returns: pd.DataFrame) -> Dict[str, Any]:
        """计算因子暴露归因"""
        try:
            # 对齐数据
            common_index = excess_returns.index.intersection(factor_returns.index)
            if len(common_index) < 2:
                return {}

            excess_returns = excess_returns.loc[common_index]
            factor_returns = factor_returns.loc[common_index]

            # 多因子回归分析
            X = sm.add_constant(factor_returns)
            y = excess_returns

            model = sm.OLS(y, X).fit()

            # 提取因子暴露和贡献
            factor_exposures = {}
            for i, factor in enumerate(factor_returns.columns, 1):
                factor_exposures[factor] = {
                    'exposure': model.params[i],
                    't_statistic': model.tvalues[i],
                    'p_value': model.pvalues[i],
                    'significant': model.pvalues[i] < 0.05,
                    'contribution': model.params[i] * factor_returns[factor].mean()
                }

            return {
                'alpha': model.params[0],
                'factor_exposures': factor_exposures,
                'r_squared': model.rsquared,
                'adjusted_r_squared': model.rsquared_adj,
                'f_statistic': model.fvalue,
                'f_p_value': model.f_pvalue,
                'model_significant': model.f_pvalue < 0.05,
                'residual_risk': np.std(model.resid),
                'model_summary': str(model.summary())
            }

        except Exception as e:
            logger.error(f"因子暴露归因计算失败: {e}")
            return {}

    def _calculate_sector_attribution(self, portfolio_returns: pd.Series,
                                      benchmark_returns: pd.Series) -> Dict[str, Any]:
        """计算行业归因"""
        try:
            # 这里需要实际的行业权重数据
            # 简化实现：使用Brinson模型的基本框架

            # 假设我们有行业权重数据
            portfolio_sector_weights = self._get_portfolio_sector_weights()
            benchmark_sector_weights = self._get_benchmark_sector_weights()
            sector_returns = self._get_sector_returns()

            if not all([portfolio_sector_weights, benchmark_sector_weights, sector_returns]):
                return {}

            # 计算Brinson归因
            allocation_effect = 0
            selection_effect = 0
            interaction_effect = 0

            for sector in portfolio_sector_weights.keys():
                # 配置效应：(Wp - Wb) * (Rb - Rbenchmark)
                allocation_effect += (portfolio_sector_weights[sector] - benchmark_sector_weights[sector]) * \
                                     (sector_returns[sector] - benchmark_returns.mean())

                # 选择效应：Wb * (Rp - Rb)
                selection_effect += benchmark_sector_weights[sector] * \
                                    (portfolio_returns.mean() - sector_returns[sector])

                # 交互效应：(Wp - Wb) * (Rp - Rb)
                interaction_effect += (portfolio_sector_weights[sector] - benchmark_sector_weights[sector]) * \
                                      (portfolio_returns.mean() - sector_returns[sector])

            total_effect = allocation_effect + selection_effect + interaction_effect

            return {
                'allocation_effect': allocation_effect,
                'selection_effect': selection_effect,
                'interaction_effect': interaction_effect,
                'total_effect': total_effect,
                'portfolio_sector_weights': portfolio_sector_weights,
                'benchmark_sector_weights': benchmark_sector_weights,
                'sector_returns': sector_returns
            }

        except Exception as e:
            logger.error(f"行业归因计算失败: {e}")
            return {}

    def _calculate_style_attribution(self, portfolio_returns: pd.Series,
                                     benchmark_returns: pd.Series) -> Dict[str, Any]:
        """计算风格归因"""
        try:
            # 使用风格因子进行归因
            style_factors = self._get_style_factors()  # 价值、成长、规模等因子

            if style_factors.empty:
                return {}

            # 风格因子回归
            X = sm.add_constant(style_factors)
            y = portfolio_returns - benchmark_returns

            model = sm.OLS(y, X).fit()

            style_attribution = {}
            for i, factor in enumerate(style_factors.columns, 1):
                style_attribution[factor] = {
                    'exposure': model.params[i],
                    'contribution': model.params[i] * style_factors[factor].mean(),
                    't_statistic': model.tvalues[i],
                    'p_value': model.pvalues[i],
                    'significant': model.pvalues[i] < 0.05
                }

            return {
                'style_alpha': model.params[0],
                'style_exposures': style_attribution,
                'r_squared': model.rsquared,
                'model_significant': model.f_pvalue < 0.05
            }

        except Exception as e:
            logger.error(f"风格归因计算失败: {e}")
            return {}

    def _calculate_total_attribution(self, attribution_results: Dict[str, Any],
                                     excess_returns: pd.Series) -> Dict[str, Any]:
        """计算总归因效果"""
        try:
            total_attribution = {
                'total_excess_return': excess_returns.mean(),
                'explained_excess_return': 0,
                'unexplained_excess_return': 0,
                'attribution_breakdown': {}
            }

            # 汇总各归因来源
            explained_return = 0

            # 市场择时贡献
            if 'market_timing' in attribution_results:
                timing_contrib = attribution_results['market_timing'].get('timing_return', 0)
                explained_return += timing_contrib
                total_attribution['attribution_breakdown']['market_timing'] = timing_contrib

            # 证券选择贡献
            if 'stock_selection' in attribution_results:
                selection_contrib = attribution_results['stock_selection'].get('selection_alpha', 0)
                explained_return += selection_contrib
                total_attribution['attribution_breakdown']['stock_selection'] = selection_contrib

            # 因子暴露贡献
            if 'factor_attribution' in attribution_results:
                factor_contrib = sum(attr['contribution'] for attr in
                                     attribution_results['factor_attribution']['factor_exposures'].values())
                explained_return += factor_contrib
                total_attribution['attribution_breakdown']['factor_exposure'] = factor_contrib

            # 行业配置贡献
            if 'sector_attribution' in attribution_results:
                sector_contrib = attribution_results['sector_attribution'].get('total_effect', 0)
                explained_return += sector_contrib
                total_attribution['attribution_breakdown']['sector_allocation'] = sector_contrib

            total_attribution['explained_excess_return'] = explained_return
            total_attribution['unexplained_excess_return'] = excess_returns.mean() - explained_return
            total_attribution[
                'attribution_ratio'] = explained_return / excess_returns.mean() if excess_returns.mean() != 0 else 0

            return total_attribution

        except Exception as e:
            logger.error(f"总归因效果计算失败: {e}")
            return {}

    def _calculate_attribution_significance(self, attribution_results: Dict[str, Any],
                                            excess_returns: pd.Series) -> Dict[str, Any]:
        """计算归因统计显著性"""
        try:
            significance_tests = {}

            # 市场择时显著性
            if 'market_timing' in attribution_results:
                timing_test = attribution_results['market_timing']
                significance_tests['market_timing'] = {
                    'significant': timing_test.get('significant', False),
                    'confidence_level': 1 - timing_test.get('p_value', 1),
                    'effect_size': timing_test.get('timing_coefficient', 0)
                }

            # 证券选择显著性
            if 'stock_selection' in attribution_results:
                selection_test = attribution_results['stock_selection']
                significance_tests['stock_selection'] = {
                    'significant': selection_test.get('significant', False),
                    'confidence_level': 1 - selection_test.get('p_value', 1),
                    'effect_size': selection_test.get('selection_alpha', 0)
                }

            # 因子暴露显著性
            if 'factor_attribution' in attribution_results:
                factor_tests = {}
                for factor, exposure in attribution_results['factor_attribution']['factor_exposures'].items():
                    factor_tests[factor] = {
                        'significant': exposure.get('significant', False),
                        'confidence_level': 1 - exposure.get('p_value', 1),
                        'effect_size': exposure.get('exposure', 0)
                    }
                significance_tests['factor_exposures'] = factor_tests

            # 整体模型显著性
            overall_significance = any(
                test.get('significant', False)
                for test in significance_tests.values()
                if isinstance(test, dict) and 'significant' in test
            )

            significance_tests['overall_significance'] = overall_significance

            return significance_tests

        except Exception as e:
            logger.error(f"归因显著性检验失败: {e}")
            return {}

    # 辅助数据获取方法（需要根据实际数据源实现）
    def _get_portfolio_sector_weights(self) -> Dict[str, float]:
        """获取组合行业权重"""
        # 这里需要实际的行业权重数据
        # 简化实现：返回空字典
        return {}

    def _get_benchmark_sector_weights(self) -> Dict[str, float]:
        """获取基准行业权重"""
        # 这里需要实际的行业权重数据
        # 简化实现：返回空字典
        return {}

    def _get_sector_returns(self) -> Dict[str, float]:
        """获取行业收益"""
        # 这里需要实际的行业收益数据
        # 简化实现：返回空字典
        return {}

    def _get_style_factors(self) -> pd.DataFrame:
        """获取风格因子数据"""
        # 这里需要实际的风格因子数据
        # 简化实现：返回空DataFrame
        return pd.DataFrame()

    def calculate_risk_adjusted_metrics(portfolio_returns: pd.Series,
                                        risk_free_rate: float = 0.02,
                                        benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """计算风险调整后绩效指标"""
        try:
            if portfolio_returns.empty:
                return {}

            metrics = {}

            # 基本收益指标
            total_return = np.prod(1 + portfolio_returns) - 1
            annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            volatility = np.std(portfolio_returns) * np.sqrt(252)

            metrics.update({
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'downside_volatility': self._calculate_downside_volatility(portfolio_returns),
                'upside_volatility': self._calculate_upside_volatility(portfolio_returns)
            })

            # 风险调整后收益指标
            if volatility > 0:
                sharpe_ratio = (annualized_return - risk_free_rate) / volatility
                sortino_ratio = self._calculate_sortino_ratio(portfolio_returns, risk_free_rate)
                calmar_ratio = self._calculate_calmar_ratio(portfolio_returns)
                omega_ratio = self._calculate_omega_ratio(portfolio_returns, risk_free_rate)

                metrics.update({
                    'sharpe_ratio': sharpe_ratio,
                    'sortino_ratio': sortino_ratio,
                    'calmar_ratio': calmar_ratio,
                    'omega_ratio': omega_ratio
                })

            # 相对于基准的指标
            if benchmark_returns is not None and not benchmark_returns.empty:
                common_index = portfolio_returns.index.intersection(benchmark_returns.index)
                if len(common_index) > 1:
                    portfolio_aligned = portfolio_returns.loc[common_index]
                    benchmark_aligned = benchmark_returns.loc[common_index]

                    tracking_error = np.std(portfolio_aligned - benchmark_aligned) * np.sqrt(252)
                    information_ratio = (
                                                    portfolio_aligned.mean() - benchmark_aligned.mean()) / tracking_error if tracking_error > 0 else 0
                    beta = self._calculate_beta(portfolio_aligned, benchmark_aligned)
                    alpha = portfolio_aligned.mean() - beta * benchmark_aligned.mean()

                    metrics.update({
                        'tracking_error': tracking_error,
                        'information_ratio': information_ratio,
                        'beta': beta,
                        'alpha': alpha,
                        'active_share': self._calculate_active_share(portfolio_aligned, benchmark_aligned)
                    })

            # 风险指标
            metrics.update({
                'value_at_risk_95': self._calculate_var(portfolio_returns, 0.95),
                'conditional_var_95': self._calculate_cvar(portfolio_returns, 0.95),
                'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
                'ulcer_index': self._calculate_ulcer_index(portfolio_returns),
                'pain_index': self._calculate_pain_index(portfolio_returns)
            })

            # 收益分布指标
            metrics.update({
                'skewness': stats.skew(portfolio_returns),
                'kurtosis': stats.kurtosis(portfolio_returns),
                'jarque_bera_stat': stats.jarque_bera(portfolio_returns)[0],
                'jarque_bera_pvalue': stats.jarque_bera(portfolio_returns)[1],
                'positive_ratio': len(portfolio_returns[portfolio_returns > 0]) / len(portfolio_returns)
            })

            return metrics

        except Exception as e:
            logger.error(f"风险调整后指标计算失败: {e}")
            return {}

    def _calculate_downside_volatility(self, returns: pd.Series) -> float:
        """计算下行波动率"""
        try:
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 1:
                return np.std(downside_returns) * np.sqrt(252)
            return 0
        except:
            return 0

    def _calculate_upside_volatility(self, returns: pd.Series) -> float:
        """计算上行波动率"""
        try:
            upside_returns = returns[returns > 0]
            if len(upside_returns) > 1:
                return np.std(upside_returns) * np.sqrt(252)
            return 0
        except:
            return 0

    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算索提诺比率"""
        try:
            downside_risk = self._calculate_downside_volatility(returns)
            excess_return = np.mean(returns) * 252 - risk_free_rate
            return excess_return / downside_risk if downside_risk > 0 else 0
        except:
            return 0

    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """计算卡尔玛比率"""
        try:
            annualized_return = np.prod(1 + returns) ** (252 / len(returns)) - 1
            max_drawdown = self._calculate_max_drawdown(returns)
            return annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
        except:
            return 0

    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.02) -> float:
        """计算Omega比率"""
        try:
            excess_returns = returns - threshold
            positive_excess = excess_returns[excess_returns > 0].sum()
            negative_excess = abs(excess_returns[excess_returns < 0].sum())
            return positive_excess / negative_excess if negative_excess > 0 else float('inf')
        except:
            return 0

    def _calculate_beta(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """计算Beta系数"""
        try:
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            return covariance / benchmark_variance if benchmark_variance > 0 else 0
        except:
            return 0

    def _calculate_active_share(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """计算主动份额"""
        try:
            return 0.5 * np.sum(np.abs(portfolio_returns - benchmark_returns))
        except:
            return 0

    def _calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """计算在险价值"""
        try:
            return np.percentile(returns, (1 - confidence_level) * 100)
        except:
            return 0

    def _calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """计算条件在险价值"""
        try:
            var = self._calculate_var(returns, confidence_level)
            tail_returns = returns[returns <= var]
            return np.mean(tail_returns) if len(tail_returns) > 0 else var
        except:
            return 0

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        try:
            cumulative_returns = (1 + returns).cumprod()
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            return drawdown.min()
        except:
            return 0

    def _calculate_ulcer_index(self, returns: pd.Series) -> float:
        """计算溃疡指数"""
        try:
            cumulative_returns = (1 + returns).cumprod()
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            return np.sqrt(np.mean(drawdown ** 2))
        except:
            return 0

    def _calculate_pain_index(self, returns: pd.Series) -> float:
        """计算痛苦指数"""
        try:
            cumulative_returns = (1 + returns).cumprod()
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            return np.mean(drawdown[drawdown < 0])
        except:
            return 0

    # 组合监控和报告功能
    def generate_performance_report(portfolio_state: PortfolioState,
                                    market_data: Dict[str, Any],
                                    period: str = '30d') -> Dict[str, Any]:
        """生成绩效报告"""
        try:
            report = {
                'report_date': datetime.now().isoformat(),
                'period': period,
                'portfolio_summary': self._get_portfolio_summary(portfolio_state),
                'performance_metrics': self._get_performance_metrics(portfolio_state, market_data, period),
                'risk_metrics': self._get_risk_metrics(portfolio_state, market_data, period),
                'attribution_analysis': self._get_attribution_analysis(portfolio_state, market_data, period),
                'sector_analysis': self._get_sector_analysis(portfolio_state, market_data),
                'risk_decomposition': self._get_risk_decomposition(portfolio_state, market_data),
                'comparison_benchmark': self._get_benchmark_comparison(portfolio_state, market_data),
                'recommendations': self._generate_performance_recommendations(portfolio_state, market_data)
            }

            return report

        except Exception as e:
            logger.error(f"绩效报告生成失败: {e}")
            return {'error': str(e)}

    def _get_portfolio_summary(self, portfolio_state: PortfolioState) -> Dict[str, Any]:
        """获取组合摘要信息"""
        try:
            return {
                'total_value': portfolio_state.total_value,
                'cash_balance': portfolio_state.cash_balance,
                'leveraged_value': portfolio_state.leveraged_value,
                'number_of_assets': len(portfolio_state.allocations),
                'effective_leverage': portfolio_state.leveraged_value / portfolio_state.total_value,
                'cash_weight': portfolio_state.cash_balance / portfolio_state.total_value,
                'concentration_ratio': self._calculate_concentration_ratio(portfolio_state),
                'turnover_rate': portfolio_state.metadata.turnover_rate,
                'diversification_score': portfolio_state.metadata.diversification
            }
        except Exception as e:
            logger.error(f"组合摘要获取失败: {e}")
            return {}

    def _get_performance_metrics(self, portfolio_state: PortfolioState,
                                 market_data: Dict[str, Any], period: str) -> Dict[str, Any]:
        """获取性能指标"""
        try:
            if not portfolio_state or not market_data:
                return {}

            # 根据期间获取历史数据
            historical_data = self._get_historical_data_for_period(portfolio_state, market_data, period)
            if not historical_data:
                logger.warning(f"无法获取 {period} 的历史数据")
                return {}

            # 计算基本收益指标
            returns_data = self._calculate_portfolio_returns(portfolio_state, historical_data)
            if len(returns_data) < 2:
                return {}

            metrics = {
                'period': period,
                'start_date': returns_data.index[0] if hasattr(returns_data.index,
                                                               '__getitem__') else returns_data.index.min(),
                'end_date': returns_data.index[-1] if hasattr(returns_data.index,
                                                              '__getitem__') else returns_data.index.max(),
                'number_of_periods': len(returns_data),
                'total_return': self._calculate_total_return(returns_data),
                'annualized_return': self._calculate_annualized_return(returns_data),
                'volatility': self._calculate_volatility(returns_data),
                'sharpe_ratio': self._calculate_sharpe_ratio(returns_data),
                'sortino_ratio': self._calculate_sortino_ratio(returns_data),
                'max_drawdown': self._calculate_max_drawdown(returns_data),
                'calmar_ratio': self._calculate_calmar_ratio(returns_data),
                'omega_ratio': self._calculate_omega_ratio(returns_data),
                'value_at_risk_95': self._calculate_value_at_risk(returns_data, 0.95),
                'conditional_var_95': self._calculate_conditional_var(returns_data, 0.95),
                'beta': self._calculate_beta(returns_data, market_data.get('benchmark_returns')),
                'alpha': self._calculate_alpha(returns_data, market_data.get('benchmark_returns')),
                'tracking_error': self._calculate_tracking_error(returns_data,
                                                                 market_data.get('benchmark_returns')),
                'information_ratio': self._calculate_information_ratio(returns_data,
                                                                       market_data.get('benchmark_returns')),
                'upside_capture': self._calculate_upside_capture(returns_data,
                                                                 market_data.get('benchmark_returns')),
                'downside_capture': self._calculate_downside_capture(returns_data,
                                                                     market_data.get('benchmark_returns')),
                'win_rate': self._calculate_win_rate(returns_data),
                'profit_factor': self._calculate_profit_factor(returns_data),
                'average_win': self._calculate_average_win(returns_data),
                'average_loss': self._calculate_average_loss(returns_data),
                'skewness': self._calculate_skewness(returns_data),
                'kurtosis': self._calculate_kurtosis(returns_data),
                'var_skewness': self._calculate_var_skewness(returns_data),
                'modified_var': self._calculate_modified_var(returns_data),
                'ulcer_index': self._calculate_ulcer_index(returns_data),
                'martin_ratio': self._calculate_martin_ratio(returns_data),
                'pain_index': self._calculate_pain_index(returns_data),
                'gain_loss_ratio': self._calculate_gain_loss_ratio(returns_data),
                'tail_ratio': self._calculate_tail_ratio(returns_data),
                'common_sense_ratio': self._calculate_common_sense_ratio(returns_data),
                'risk_adjusted_return': self._calculate_risk_adjusted_return(returns_data),
                'monthly_returns': self._calculate_monthly_returns(returns_data),
                'quarterly_returns': self._calculate_quarterly_returns(returns_data),
                'yearly_returns': self._calculate_yearly_returns(returns_data),
                'rolling_metrics': self._calculate_rolling_metrics(returns_data),
                'periodic_returns': self._calculate_periodic_returns(returns_data, period)
            }

            # 添加基准比较指标（如果有基准数据）
            if market_data.get('benchmark_returns') is not None:
                benchmark_metrics = self._calculate_benchmark_comparison(returns_data,
                                                                         market_data['benchmark_returns'])
                metrics.update(benchmark_metrics)

            logger.info(f"性能指标计算完成: 期间={period}, 数据点={len(returns_data)}")
            return metrics

        except Exception as e:
            logger.error(f"性能指标计算失败: {e}")
            return {}

    def _get_historical_data_for_period(self, portfolio_state: PortfolioState,
                                        market_data: Dict[str, Any], period: str) -> Optional[pd.DataFrame]:
        """获取指定期间的历史数据"""
        try:
            # 解析期间字符串（如 '30d', '3m', '1y', 'all'）
            end_date = datetime.now()

            if period == 'all':
                start_date = datetime.now() - timedelta(days=365 * 5)  # 默认5年
            elif period.endswith('d'):
                days = int(period[:-1])
                start_date = end_date - timedelta(days=days)
            elif period.endswith('m'):
                months = int(period[:-1])
                start_date = end_date - timedelta(days=months * 30)
            elif period.endswith('y'):
                years = int(period[:-1])
                start_date = end_date - timedelta(days=years * 365)
            else:
                # 默认30天
                start_date = end_date - timedelta(days=30)

            # 从市场数据中筛选指定期间的数据
            historical_prices = {}
            for symbol, price_data in market_data.get('prices', {}).items():
                if 'close' in price_data and len(price_data['close']) > 0:
                    # 简化实现：假设价格数据是时间序列
                    # 实际中应该根据时间戳过滤
                    prices = price_data['close']
                    if len(prices) > 0:
                        historical_prices[symbol] = prices

            if not historical_prices:
                return None

            # 转换为DataFrame
            df = pd.DataFrame(historical_prices)

            # 确保数据按时间排序（最新的在前）
            df = df.sort_index(ascending=False)

            # 截取指定期间的数据（简化实现）
            max_points = min(252 * 5, len(df))  # 最多5年数据
            if len(df) > max_points:
                df = df.head(max_points)

            return df

        except Exception as e:
            logger.error(f"历史数据获取失败: {e}")
            return None

    def _calculate_portfolio_returns(self, portfolio_state: PortfolioState,
                                     historical_data: pd.DataFrame) -> pd.Series:
        """计算组合收益序列"""
        try:
            if historical_data.empty:
                return pd.Series()

            # 获取组合权重
            weights = {}
            for symbol, allocation in portfolio_state.allocations.items():
                if symbol in historical_data.columns:
                    weights[symbol] = allocation.weight

            if not weights:
                return pd.Series()

            # 计算加权收益
            returns = historical_data.pct_change().dropna()
            portfolio_returns = pd.Series(index=returns.index)

            for date in returns.index:
                daily_return = 0
                for symbol, weight in weights.items():
                    if symbol in returns.columns and not pd.isna(returns.loc[date, symbol]):
                        daily_return += weight * returns.loc[date, symbol]
                portfolio_returns[date] = daily_return

            return portfolio_returns.dropna()

        except Exception as e:
            logger.error(f"组合收益计算失败: {e}")
            return pd.Series()

    def _calculate_total_return(self, returns: pd.Series) -> float:
        """计算总收益"""
        try:
            if returns.empty:
                return 0.0
            return float(np.prod(1 + returns) - 1)
        except:
            return 0.0

    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """计算年化收益"""
        try:
            if returns.empty:
                return 0.0

            total_return = self._calculate_total_return(returns)
            periods_per_year = 252  # 假设日数据

            if len(returns) > 1:
                years = len(returns) / periods_per_year
                annualized_return = (1 + total_return) ** (1 / years) - 1
                return float(annualized_return)
            return 0.0
        except:
            return 0.0

    def _calculate_volatility(self, returns: pd.Series) -> float:
        """计算年化波动率"""
        try:
            if len(returns) < 2:
                return 0.0
            return float(np.std(returns) * np.sqrt(252))
        except:
            return 0.0

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        try:
            annualized_return = self._calculate_annualized_return(returns)
            volatility = self._calculate_volatility(returns)

            if volatility > 0:
                sharpe = (annualized_return - risk_free_rate) / volatility
                return float(sharpe)
            return 0.0
        except:
            return 0.0

    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算索提诺比率"""
        try:
            if returns.empty:
                return 0.0

            annualized_return = self._calculate_annualized_return(returns)

            # 计算下行偏差
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 1:
                downside_deviation = np.std(downside_returns) * np.sqrt(252)
                if downside_deviation > 0:
                    sortino = (annualized_return - risk_free_rate) / downside_deviation
                    return float(sortino)
            return 0.0
        except:
            return 0.0

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        try:
            if returns.empty:
                return 0.0

            cumulative_returns = (1 + returns).cumprod()
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            return float(drawdown.min())
        except:
            return 0.0

    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """计算卡尔玛比率"""
        try:
            annualized_return = self._calculate_annualized_return(returns)
            max_drawdown = self._calculate_max_drawdown(returns)

            if max_drawdown < 0:
                calmar = annualized_return / abs(max_drawdown)
                return float(calmar)
            return 0.0
        except:
            return 0.0

    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """计算Omega比率"""
        try:
            if returns.empty:
                return 0.0

            gains = returns[returns > threshold].sum()
            losses = abs(returns[returns < threshold].sum())

            if losses > 0:
                return float(gains / losses)
            return float('inf') if gains > 0 else 0.0
        except:
            return 0.0

    def _calculate_value_at_risk(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """计算在险价值(VaR)"""
        try:
            if returns.empty:
                return 0.0
            return float(np.percentile(returns, (1 - confidence_level) * 100))
        except:
            return 0.0

    def _calculate_conditional_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """计算条件在险价值(CVaR)"""
        try:
            if returns.empty:
                return 0.0

            var = self._calculate_value_at_risk(returns, confidence_level)
            tail_returns = returns[returns <= var]

            if len(tail_returns) > 0:
                return float(tail_returns.mean())
            return float(var)
        except:
            return 0.0

    def _calculate_beta(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """计算Beta系数"""
        try:
            if portfolio_returns.empty or benchmark_returns is None or benchmark_returns.empty:
                return 0.0

            # 对齐数据
            common_index = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_index) < 2:
                return 0.0

            portfolio_aligned = portfolio_returns.loc[common_index]
            benchmark_aligned = benchmark_returns.loc[common_index]

            covariance = np.cov(portfolio_aligned, benchmark_aligned)[0, 1]
            benchmark_variance = np.var(benchmark_aligned)

            if benchmark_variance > 0:
                return float(covariance / benchmark_variance)
            return 0.0
        except:
            return 0.0

    def _calculate_alpha(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """计算Alpha"""
        try:
            if portfolio_returns.empty or benchmark_returns is None or benchmark_returns.empty:
                return 0.0

            portfolio_annual_return = self._calculate_annualized_return(portfolio_returns)
            benchmark_annual_return = self._calculate_annualized_return(benchmark_returns)
            beta = self._calculate_beta(portfolio_returns, benchmark_returns)

            alpha = portfolio_annual_return - (0.02 + beta * (benchmark_annual_return - 0.02))
            return float(alpha)
        except:
            return 0.0

    def _calculate_tracking_error(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """计算跟踪误差"""
        try:
            if portfolio_returns.empty or benchmark_returns is None or benchmark_returns.empty:
                return 0.0

            common_index = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_index) < 2:
                return 0.0

            portfolio_aligned = portfolio_returns.loc[common_index]
            benchmark_aligned = benchmark_returns.loc[common_index]

            active_returns = portfolio_aligned - benchmark_aligned
            tracking_error = np.std(active_returns) * np.sqrt(252)
            return float(tracking_error)
        except:
            return 0.0

    def _calculate_information_ratio(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """计算信息比率"""
        try:
            if portfolio_returns.empty or benchmark_returns is None or benchmark_returns.empty:
                return 0.0

            common_index = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_index) < 2:
                return 0.0

            portfolio_aligned = portfolio_returns.loc[common_index]
            benchmark_aligned = benchmark_returns.loc[common_index]

            active_returns = portfolio_aligned - benchmark_aligned
            tracking_error = self._calculate_tracking_error(portfolio_returns, benchmark_returns)
            average_active_return = active_returns.mean() * 252

            if tracking_error > 0:
                return float(average_active_return / tracking_error)
            return 0.0
        except:
            return 0.0

    def _calculate_upside_capture(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """计算上行捕获比率"""
        try:
            if portfolio_returns.empty or benchmark_returns is None or benchmark_returns.empty:
                return 0.0

            common_index = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_index) < 2:
                return 0.0

            portfolio_aligned = portfolio_returns.loc[common_index]
            benchmark_aligned = benchmark_returns.loc[common_index]

            # 基准上行时期
            up_periods = benchmark_aligned > 0
            if up_periods.sum() > 0:
                portfolio_up_returns = portfolio_aligned[up_periods]
                benchmark_up_returns = benchmark_aligned[up_periods]

                portfolio_up_performance = np.prod(1 + portfolio_up_returns) - 1
                benchmark_up_performance = np.prod(1 + benchmark_up_returns) - 1

                if benchmark_up_performance > 0:
                    return float(portfolio_up_performance / benchmark_up_performance)
            return 0.0
        except:
            return 0.0

    def _calculate_downside_capture(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """计算下行捕获比率"""
        try:
            if portfolio_returns.empty or benchmark_returns is None or benchmark_returns.empty:
                return 0.0

            common_index = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_index) < 2:
                return 0.0

            portfolio_aligned = portfolio_returns.loc[common_index]
            benchmark_aligned = benchmark_returns.loc[common_index]

            # 基准下行时期
            down_periods = benchmark_aligned < 0
            if down_periods.sum() > 0:
                portfolio_down_returns = portfolio_aligned[down_periods]
                benchmark_down_returns = benchmark_aligned[down_periods]

                portfolio_down_performance = np.prod(1 + portfolio_down_returns) - 1
                benchmark_down_performance = np.prod(1 + benchmark_down_returns) - 1

                if benchmark_down_performance < 0:
                    return float(portfolio_down_performance / benchmark_down_performance)
            return 0.0
        except:
            return 0.0

    def _calculate_win_rate(self, returns: pd.Series) -> float:
        """计算胜率"""
        try:
            if returns.empty:
                return 0.0
            win_trades = (returns > 0).sum()
            total_trades = len(returns)
            return float(win_trades / total_trades) if total_trades > 0 else 0.0
        except:
            return 0.0

    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """计算盈利因子"""
        try:
            if returns.empty:
                return 0.0

            gross_profit = returns[returns > 0].sum()
            gross_loss = abs(returns[returns < 0].sum())

            if gross_loss > 0:
                return float(gross_profit / gross_loss)
            return float('inf') if gross_profit > 0 else 0.0
        except:
            return 0.0

    def _calculate_benchmark_comparison(self, portfolio_returns: pd.Series,
                                        benchmark_returns: pd.Series) -> Dict[str, Any]:
        """计算基准比较指标"""
        try:
            if portfolio_returns.empty or benchmark_returns.empty:
                return {}

            common_index = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_index) < 2:
                return {}

            portfolio_aligned = portfolio_returns.loc[common_index]
            benchmark_aligned = benchmark_returns.loc[common_index]

            comparison = {
                'benchmark_correlation': float(portfolio_aligned.corr(benchmark_aligned)),
                'active_return': float((portfolio_aligned - benchmark_aligned).mean() * 252),
                'relative_volatility': float(np.std(portfolio_aligned) / np.std(benchmark_aligned)),
                'relative_sharpe': float(self._calculate_sharpe_ratio(portfolio_aligned) /
                                         self._calculate_sharpe_ratio(benchmark_aligned)),
                'outperformance_frequency': float((portfolio_aligned > benchmark_aligned).mean()),
                'cumulative_outperformance': float(np.prod(1 + portfolio_aligned) - np.prod(1 + benchmark_aligned))
            }

            return comparison

        except Exception as e:
            logger.error(f"基准比较计算失败: {e}")
            return {}

    def _calculate_rolling_metrics(self, returns: pd.Series, window: int = 252) -> Dict[str, Any]:
        """计算滚动指标"""
        try:
            if len(returns) < window:
                return {}

            rolling_returns = returns.rolling(window=window)
            rolling_volatility = rolling_returns.std() * np.sqrt(252)
            rolling_sharpe = rolling_returns.mean() * np.sqrt(252) / rolling_returns.std()

            rolling_metrics = {
                'rolling_volatility_mean': float(rolling_volatility.mean()),
                'rolling_volatility_std': float(rolling_volatility.std()),
                'rolling_sharpe_mean': float(rolling_sharpe.mean()),
                'rolling_sharpe_std': float(rolling_sharpe.std()),
                'rolling_max_drawdowns': [self._calculate_max_drawdown(returns[i:i + window])
                                          for i in range(len(returns) - window + 1)]
            }

            return rolling_metrics

        except Exception as e:
            logger.error(f"滚动指标计算失败: {e}")
            return {}

    def generate_performance_report(self, portfolio_state: PortfolioState,
                                    market_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成完整性能报告"""
        try:
            report_periods = ['30d', '90d', '1y', '3y', '5y', 'all']
            performance_reports = {}

            for period in report_periods:
                metrics = self._get_performance_metrics(portfolio_state, market_data, period)
                if metrics:
                    performance_reports[period] = metrics

            # 生成综合报告
            comprehensive_report = {
                'generated_at': datetime.now().isoformat(),
                'portfolio_id': portfolio_state.portfolio_id,
                'total_value': portfolio_state.total_value,
                'performance_by_period': performance_reports,
                'summary_metrics': self._generate_summary_metrics(performance_reports),
                'trend_analysis': self._analyze_performance_trends(performance_reports),
                'peer_comparison': self._compare_with_peers(portfolio_state, market_data),
                'recommendations': self._generate_performance_recommendations(performance_reports)
            }

            logger.info("性能报告生成完成")
            return comprehensive_report

        except Exception as e:
            logger.error(f"性能报告生成失败: {e}")
            return {}

    def _generate_summary_metrics(self, performance_reports: Dict[str, Dict]) -> Dict[str, Any]:
        """生成摘要指标"""
        try:
            if not performance_reports:
                return {}

            # 使用最近期的数据作为摘要
            recent_period = '30d' if '30d' in performance_reports else list(performance_reports.keys())[0]
            recent_metrics = performance_reports[recent_period]

            summary = {
                'current_period': recent_period,
                'annualized_return': recent_metrics.get('annualized_return', 0),
                'volatility': recent_metrics.get('volatility', 0),
                'sharpe_ratio': recent_metrics.get('sharpe_ratio', 0),
                'max_drawdown': recent_metrics.get('max_drawdown', 0),
                'calmar_ratio': recent_metrics.get('calmar_ratio', 0),
                'win_rate': recent_metrics.get('win_rate', 0),
                'best_performing_period': self._identify_best_period(performance_reports),
                'worst_performing_period': self._identify_worst_period(performance_reports),
                'consistency_score': self._calculate_consistency_score(performance_reports)
            }

            return summary

        except Exception as e:
            logger.error(f"摘要指标生成失败: {e}")
            return {}

    def _analyze_performance_trends(self, performance_reports: Dict[str, Dict]) -> Dict[str, Any]:
        """分析性能趋势"""
        try:
            if len(performance_reports) < 2:
                return {}

            trends = {
                'return_trend': 'improving' if self._assess_return_trend(
                    performance_reports) > 0 else 'deteriorating',
                'risk_trend': 'improving' if self._assess_risk_trend(performance_reports) < 0 else 'deteriorating',
                'sharpe_trend': 'improving' if self._assess_sharpe_trend(
                    performance_reports) > 0 else 'deteriorating',
                'consistency_trend': self._assess_consistency_trend(performance_reports),
                'key_drivers': self._identify_performance_drivers(performance_reports),
                'emerging_risks': self._identify_emerging_risks(performance_reports)
            }

            return trends

        except Exception as e:
            logger.error(f"性能趋势分析失败: {e}")
            return {}

    def _compare_with_peers(self, portfolio_state: PortfolioState,
                            market_data: Dict[str, Any]) -> Dict[str, Any]:
        """与同行比较"""
        try:
            # 这里需要实际的同行数据
            # 简化实现：返回空字典
            peer_universe = market_data.get('peer_returns', {})
            if not peer_universe:
                return {}

            comparison = {
                'peer_ranking': 0,
                'percentile_rank': 0.5,
                'relative_performance': 0.0,
                'competitive_advantages': [],
                'areas_for_improvement': []
            }

            return comparison

        except Exception as e:
            logger.error(f"同行比较失败: {e}")
            return {}

    def _generate_performance_recommendations(self, performance_reports: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """生成性能改进建议"""
        recommendations = []

        try:
            if not performance_reports:
                return recommendations

            recent_metrics = performance_reports.get('30d') or list(performance_reports.values())[0]

            # 基于夏普比率的建议
            sharpe_ratio = recent_metrics.get('sharpe_ratio', 0)
            if sharpe_ratio < 0.5:
                recommendations.append({
                    'category': '风险调整收益',
                    'priority': 'high',
                    'recommendation': '夏普比率较低，建议优化风险调整后收益',
                    'action': '考虑降低组合波动率或提高收益稳定性',
                    'expected_impact': '中高'
                })

            # 基于最大回撤的建议
            max_drawdown = recent_metrics.get('max_drawdown', 0)
            if max_drawdown < -0.15:
                recommendations.append({
                    'category': '风险管理',
                    'priority': 'high',
                    'recommendation': '最大回撤过大，需要加强风险控制',
                    'action': '增加对冲策略或降低高风险资产权重',
                    'expected_impact': '高'
                })

            # 基于胜率的建议
            win_rate = recent_metrics.get('win_rate', 0)
            if win_rate < 0.5:
                recommendations.append({
                    'category': '投资策略',
                    'priority': 'medium',
                    'recommendation': '胜率偏低，建议优化选股策略',
                    'action': '加强基本面分析或技术指标筛选',
                    'expected_impact': '中'
                })

            return recommendations

        except Exception as e:
            logger.error(f"性能建议生成失败: {e}")
            return []

    def save_performance_report(self, report: Dict[str, Any], filepath: str) -> bool:
        """保存性能报告到文件"""
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"性能报告保存成功: {filepath}")
            return True

        except Exception as e:
            logger.error(f"性能报告保存失败: {e}")
            return False

    def load_performance_report(self, filepath: str) -> Optional[Dict[str, Any]]:
        """从文件加载性能报告"""
        try:
            with open(filepath, 'r') as f:
                report = json.load(f)

            logger.info(f"性能报告加载成功: {filepath}")
            return report

        except Exception as e:
            logger.error(f"性能报告加载失败: {e}")
            return None

    # 风险指标计算相关方法
    def _calculate_portfolio_risk_metrics(self, portfolio_state: PortfolioState,
                                          market_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算组合风险指标"""
        try:
            if not portfolio_state or not market_data:
                return {}

            # 获取收益数据
            returns_data = self._calculate_portfolio_returns(portfolio_state, market_data)
            if returns_data.empty:
                return {}

            risk_metrics = {
                'volatility': self._calculate_volatility(returns_data),
                'beta': self._calculate_beta(returns_data, market_data.get('benchmark_returns')),
                'value_at_risk_95': self._calculate_value_at_risk(returns_data, 0.95),
                'conditional_var_95': self._calculate_conditional_var(returns_data, 0.95),
                'expected_shortfall': self._calculate_expected_shortfall(returns_data),
                'tail_risk': self._calculate_tail_risk(returns_data),
                'maximum_drawdown': self._calculate_max_drawdown(returns_data),
                'ulcer_index': self._calculate_ulcer_index(returns_data),
                'pain_index': self._calculate_pain_index(returns_data),
                'risk_contributions': self._calculate_risk_contributions(portfolio_state, market_data),
                'marginal_risk': self._calculate_marginal_risk(portfolio_state, market_data),
                'component_risk': self._calculate_component_risk(portfolio_state, market_data),
                'diversification_benefit': self._calculate_diversification_benefit(portfolio_state, market_data),
                'liquidity_risk': self._calculate_liquidity_risk(portfolio_state, market_data),
                'concentration_risk': self._calculate_concentration_risk(portfolio_state),
                'leverage_risk': self._calculate_leverage_risk(portfolio_state),
                'scenario_risk': self._calculate_scenario_risk(portfolio_state, market_data),
                'stress_test_results': self._run_stress_tests(portfolio_state, market_data)
            }

            return risk_metrics

        except Exception as e:
            logger.error(f"组合风险指标计算失败: {e}")
            return {}

    def _calculate_expected_shortfall(self, returns: pd.Series) -> float:
        """计算预期短缺"""
        try:
            return self._calculate_conditional_var(returns, 0.95)
        except:
            return 0.0

    def _calculate_tail_risk(self, returns: pd.Series) -> float:
        """计算尾部风险"""
        try:
            if returns.empty:
                return 0.0

            # 计算5%分位数的波动率
            tail_threshold = np.percentile(returns, 5)
            tail_returns = returns[returns <= tail_threshold]
            return float(np.std(tail_returns) * np.sqrt(252)) if len(tail_returns) > 1 else 0.0
        except:
            return 0.0

    def _calculate_ulcer_index(self, returns: pd.Series) -> float:
        """计算溃疡指数"""
        try:
            if returns.empty:
                return 0.0

            cumulative_returns = (1 + returns).cumprod()
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            squared_drawdown = drawdown ** 2
            return float(np.sqrt(squared_drawdown.mean()))
        except:
            return 0.0

    def _calculate_pain_index(self, returns: pd.Series) -> float:
        """计算痛苦指数"""
        try:
            if returns.empty:
                return 0.0

            cumulative_returns = (1 + returns).cumprod()
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            return float(drawdown[drawdown < 0].mean())
        except:
            return 0.0

    def _calculate_risk_contributions(self, portfolio_state: PortfolioState,
                                      market_data: Dict[str, Any]) -> Dict[str, float]:
        """计算风险贡献"""
        try:
            symbols = list(portfolio_state.allocations.keys())
            if not symbols or 'covariance_matrix' not in market_data:
                return {}

            weights = np.array([portfolio_state.allocations[symbol].weight for symbol in symbols])
            cov_matrix = market_data['covariance_matrix']

            portfolio_variance = weights.T @ cov_matrix @ weights
            marginal_risk = cov_matrix @ weights / np.sqrt(portfolio_variance)
            risk_contributions = weights * marginal_risk

            return {symbols[i]: float(risk_contributions[i]) for i in range(len(symbols))}

        except Exception as e:
            logger.error(f"风险贡献计算失败: {e}")
            return {}

    def _calculate_marginal_risk(self, portfolio_state: PortfolioState,
                                 market_data: Dict[str, Any]) -> Dict[str, float]:
        """计算边际风险"""
        try:
            symbols = list(portfolio_state.allocations.keys())
            if not symbols or 'covariance_matrix' not in market_data:
                return {}

            weights = np.array([portfolio_state.allocations[symbol].weight for symbol in symbols])
            cov_matrix = market_data['covariance_matrix']

            portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
            marginal_risk = cov_matrix @ weights / portfolio_risk

            return {symbols[i]: float(marginal_risk[i]) for i in range(len(symbols))}

        except Exception as e:
            logger.error(f"边际风险计算失败: {e}")
            return {}

    def _calculate_component_risk(self, portfolio_state: PortfolioState,
                                  market_data: Dict[str, Any]) -> Dict[str, float]:
        """计算成分风险"""
        try:
            symbols = list(portfolio_state.allocations.keys())
            if not symbols or 'covariance_matrix' not in market_data:
                return {}

            weights = np.array([portfolio_state.allocations[symbol].weight for symbol in symbols])
            cov_matrix = market_data['covariance_matrix']

            component_risk = {}
            for i, symbol in enumerate(symbols):
                # 计算去除该资产后的组合风险
                other_weights = np.delete(weights, i)
                other_cov = np.delete(np.delete(cov_matrix, i, 0), i, 1)
                other_risk = np.sqrt(other_weights.T @ other_cov @ other_weights)

                # 计算完整组合风险
                full_risk = np.sqrt(weights.T @ cov_matrix @ weights)

                component_risk[symbol] = float(full_risk - other_risk)

            return component_risk

        except Exception as e:
            logger.error(f"成分风险计算失败: {e}")
            return {}

    def _calculate_diversification_benefit(self, portfolio_state: PortfolioState,
                                           market_data: Dict[str, Any]) -> float:
        """计算分散化收益"""
        try:
            symbols = list(portfolio_state.allocations.keys())
            if not symbols or 'covariance_matrix' not in market_data:
                return 0.0

            weights = np.array([portfolio_state.allocations[symbol].weight for symbol in symbols])
            cov_matrix = market_data['covariance_matrix']

            # 计算组合风险
            portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)

            # 计算加权平均个体风险
            individual_risks = np.sqrt(np.diag(cov_matrix))
            weighted_individual_risk = np.sum(weights * individual_risks)

            if weighted_individual_risk > 0:
                diversification_benefit = 1 - portfolio_risk / weighted_individual_risk
                return float(diversification_benefit)
            return 0.0

        except Exception as e:
            logger.error(f"分散化收益计算失败: {e}")
            return 0.0

    def _calculate_liquidity_risk(self, portfolio_state: PortfolioState,
                                  market_data: Dict[str, Any]) -> float:
        """计算流动性风险"""
        try:
            liquidity_scores = []
            for symbol, allocation in portfolio_state.allocations.items():
                liquidity_data = market_data.get('liquidity', {}).get(symbol, {})
                volume = liquidity_data.get('volume', 0)
                avg_volume = liquidity_data.get('avg_daily_volume', volume)

                if avg_volume > 0:
                    liquidity_ratio = volume / avg_volume
                    # 流动性评分（0-1，1表示最好）
                    liquidity_score = min(liquidity_ratio, 1.0)
                    liquidity_scores.append(liquidity_score * allocation.weight)

            if liquidity_scores:
                return float(1 - np.mean(liquidity_scores))  # 风险评分（0-1，1表示风险最高）
            return 0.5  # 默认中等风险

        except Exception as e:
            logger.error(f"流动性风险计算失败: {e}")
            return 0.5

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
            return 0.0

    def _calculate_leverage_risk(self, portfolio_state: PortfolioState) -> float:
        """计算杠杆风险"""
        try:
            if portfolio_state.total_value > 0:
                leverage_ratio = portfolio_state.leveraged_value / portfolio_state.total_value
                # 风险评分（杠杆越高风险越高）
                leverage_risk = min(max(leverage_ratio - 1, 0) / 2, 1.0)  # 假设杠杆不超过3倍
                return float(leverage_risk)
            return 0.0

        except Exception as e:
            logger.error(f"杠杆风险计算失败: {e}")
            return 0.0

    def _calculate_scenario_risk(self, portfolio_state: PortfolioState,
                                 market_data: Dict[str, Any]) -> Dict[str, float]:
        """计算情景风险"""
        try:
            scenario_risks = {}

            # 市场下跌情景
            scenario_risks['market_down_10'] = self._simulate_scenario(portfolio_state, market_data, -0.1)
            scenario_risks['market_down_20'] = self._simulate_scenario(portfolio_state, market_data, -0.2)
            scenario_risks['market_down_30'] = self._simulate_scenario(portfolio_state, market_data, -0.3)

            # 波动率上升情景
            scenario_risks['volatility_up_50'] = self._simulate_volatility_scenario(portfolio_state, market_data,
                                                                                    1.5)
            scenario_risks['volatility_up_100'] = self._simulate_volatility_scenario(portfolio_state, market_data,
                                                                                     2.0)

            # 相关性上升情景
            scenario_risks['correlation_up'] = self._simulate_correlation_scenario(portfolio_state, market_data,
                                                                                   0.5)

            return scenario_risks

        except Exception as e:
            logger.error(f"情景风险计算失败: {e}")
            return {}

    def _run_stress_tests(self, portfolio_state: PortfolioState,
                          market_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行压力测试"""
        try:
            stress_tests = {}

            # 2008年金融危机情景
            stress_tests['financial_crisis_2008'] = self._simulate_financial_crisis(portfolio_state, market_data)

            # 2020年新冠疫情情景
            stress_tests['covid_2020'] = self._simulate_covid_crisis(portfolio_state, market_data)

            # 利率上升情景
            stress_tests['interest_rate_shock'] = self._simulate_interest_rate_shock(portfolio_state, market_data)

            # 流动性危机情景
            stress_tests['liquidity_crisis'] = self._simulate_liquidity_crisis(portfolio_state, market_data)

            return stress_tests

        except Exception as e:
            logger.error(f"压力测试执行失败: {e}")
            return {}

    def _simulate_scenario(self, portfolio_state: PortfolioState,
                           market_data: Dict[str, Any], market_return: float) -> float:
        """模拟市场情景"""
        try:
            # 简化实现：基于Beta系数计算组合受影响程度
            portfolio_beta = self._calculate_beta(
                self._calculate_portfolio_returns(portfolio_state, market_data),
                market_data.get('benchmark_returns')
            )

            scenario_impact = portfolio_beta * market_return
            return float(scenario_impact)

        except Exception as e:
            logger.error(f"情景模拟失败: {e}")
            return 0.0

    # 性能报告相关的辅助方法
    def _identify_best_period(self, performance_reports: Dict[str, Dict]) -> str:
        """识别最佳表现期间"""
        try:
            best_period = None
            best_return = -float('inf')

            for period, metrics in performance_reports.items():
                annual_return = metrics.get('annualized_return', 0)
                if annual_return > best_return:
                    best_return = annual_return
                    best_period = period

            return best_period or 'N/A'
        except:
            return 'N/A'

    def _identify_worst_period(self, performance_reports: Dict[str, Dict]) -> str:
        """识别最差表现期间"""
        try:
            worst_period = None
            worst_return = float('inf')

            for period, metrics in performance_reports.items():
                annual_return = metrics.get('annualized_return', 0)
                if annual_return < worst_return:
                    worst_return = annual_return
                    worst_period = period

            return worst_period or 'N/A'
        except:
            return 'N/A'

    def _calculate_consistency_score(self, performance_reports: Dict[str, Dict]) -> float:
        """计算表现一致性评分"""
        try:
            if len(performance_reports) < 2:
                return 0.0

            returns = [metrics.get('annualized_return', 0) for metrics in performance_reports.values()]
            consistency = 1 - (np.std(returns) / np.mean(returns)) if np.mean(returns) != 0 else 0
            return float(max(0, consistency))
        except:
            return 0.0

    def _assess_return_trend(self, performance_reports: Dict[str, Dict]) -> float:
        """评估收益趋势"""
        try:
            periods = sorted(performance_reports.keys())
            returns = [performance_reports[period].get('annualized_return', 0) for period in periods]

            if len(returns) > 1:
                # 计算线性回归斜率
                x = np.arange(len(returns))
                slope = np.polyfit(x, returns, 1)[0]
                return float(slope)
            return 0.0
        except:
            return 0.0

    def _assess_risk_trend(self, performance_reports: Dict[str, Dict]) -> float:
        """评估风险趋势"""
        try:
            periods = sorted(performance_reports.keys())
            volatilities = [performance_reports[period].get('volatility', 0) for period in periods]

            if len(volatilities) > 1:
                x = np.arange(len(volatilities))
                slope = np.polyfit(x, volatilities, 1)[0]
                return float(slope)
            return 0.0
        except:
            return 0.0

    def _assess_sharpe_trend(self, performance_reports: Dict[str, Dict]) -> float:
        """评估夏普比率趋势"""
        try:
            periods = sorted(performance_reports.keys())
            sharpe_ratios = [performance_reports[period].get('sharpe_ratio', 0) for period in periods]

            if len(sharpe_ratios) > 1:
                x = np.arange(len(sharpe_ratios))
                slope = np.polyfit(x, sharpe_ratios, 1)[0]
                return float(slope)
            return 0.0
        except:
            return 0.0

    def _assess_consistency_trend(self, performance_reports: Dict[str, Dict]) -> str:
        """评估一致性趋势"""
        try:
            if len(performance_reports) < 3:
                return 'insufficient_data'

            # 计算滚动一致性
            periods = sorted(performance_reports.keys())
            consistency_scores = []

            for i in range(2, len(periods)):
                recent_periods = periods[i - 2:i + 1]
                recent_returns = [performance_reports[p].get('annualized_return', 0) for p in recent_periods]
                consistency = 1 - (np.std(recent_returns) / np.mean(recent_returns)) if np.mean(
                    recent_returns) != 0 else 0
                consistency_scores.append(consistency)

            if len(consistency_scores) > 1:
                trend = 'improving' if consistency_scores[-1] > consistency_scores[0] else 'deteriorating'
                return trend
            return 'stable'
        except:
            return 'unknown'

    def _identify_performance_drivers(self, performance_reports: Dict[str, Dict]) -> List[str]:
        """识别业绩驱动因素"""
        drivers = []

        try:
            if not performance_reports:
                return drivers

            recent_metrics = list(performance_reports.values())[0]

            # 基于风险调整后收益识别驱动因素
            sharpe_ratio = recent_metrics.get('sharpe_ratio', 0)
            if sharpe_ratio > 1.0:
                drivers.append('优秀的风险调整后收益')

            # 基于胜率识别驱动因素
            win_rate = recent_metrics.get('win_rate', 0)
            if win_rate > 0.6:
                drivers.append('高胜率投资策略')

            # 基于风险控制识别驱动因素
            max_drawdown = recent_metrics.get('max_drawdown', 0)
            if max_drawdown > -0.1:
                drivers.append('有效的风险控制')

            return drivers

        except Exception as e:
            logger.error(f"业绩驱动因素识别失败: {e}")
            return []

    def _identify_emerging_risks(self, performance_reports: Dict[str, Dict]) -> List[str]:
        """识别新兴风险"""
        risks = []

        try:
            if len(performance_reports) < 2:
                return risks

            # 检查风险指标趋势
            risk_trend = self._assess_risk_trend(performance_reports)
            if risk_trend > 0.05:
                risks.append('波动率呈上升趋势')

            # 检查回撤情况
            recent_metrics = list(performance_reports.values())[0]
            max_drawdown = recent_metrics.get('max_drawdown', 0)
            if max_drawdown < -0.15:
                risks.append('近期回撤较大')

            # 检查尾部风险
            var_95 = recent_metrics.get('value_at_risk_95', 0)
            if var_95 < -0.08:
                risks.append('尾部风险较高')

            return risks

        except Exception as e:
            logger.error(f"新兴风险识别失败: {e}")
            return []

    def cleanup(self):
        """清理资源"""
        try:
            # 关闭线程池
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)

            # 清理缓存
            if hasattr(self, '_optimizer_cache'):
                self._optimizer_cache.clear()

            if hasattr(self, '_constraint_cache'):
                self._constraint_cache.clear()

            if hasattr(self, '_risk_model_cache'):
                self._risk_model_cache.clear()

            # 保存当前状态
            if hasattr(self, 'current_portfolio') and self.current_portfolio:
                self._save_portfolio_state('portfolio_state_backup.json')

            logger.info("组合管理器资源清理完成")

        except Exception as e:
            logger.error(f"组合管理器资源清理失败: {e}")

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
def validate_portfolio_state(portfolio_state: Dict) -> bool:
    """验证组合状态有效性"""
    try:
        required_fields = ['portfolio_id', 'total_value', 'allocations', 'timestamp']
        if not all(field in portfolio_state for field in required_fields):
            return False

        # 验证数值有效性
        if portfolio_state['total_value'] <= 0:
            return False

        # 验证权重总和
        total_weight = sum(alloc['weight'] for alloc in portfolio_state['allocations'].values())
        if abs(total_weight - 1.0) > 0.01:  # 允许1%的误差
            return False

        return True

    except Exception:
        return False

def calculate_portfolio_performance(portfolio_returns: pd.Series,
                                    benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
    """计算组合性能指标（独立函数版本）"""
    try:
        calculator = PortfolioManager({})
        metrics = calculator._get_performance_metrics(
            PortfolioState('temp', 1000000, 0, 1000000, {}, PortfolioMetadata(), {}, {}, PortfolioConstraints()),
            {'returns': portfolio_returns, 'benchmark_returns': benchmark_returns},
            'all'
        )
        return {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    except Exception:
        return {}

if __name__ == "__main__":
    # 测试代码
    config = {
        'portfolio_management': {
            'optimization': {
                'method': 'max_sharpe',
                'constraints': {
                    'max_asset_weight': 0.2,
                    'min_asset_weight': 0.0,
                    'max_sector_weight': 0.3
                }
            },
            'risk_management': {
                'risk_limits': {
                    'max_volatility': 0.3,
                    'max_drawdown': -0.2,
                    'var_95': -0.05
                }
            }
        }
    }

    # 创建组合管理器实例
    manager = PortfolioManager(config)

    # 测试性能计算
    test_returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    test_benchmark = pd.Series(np.random.normal(0.0008, 0.018, 252))

    performance = calculate_portfolio_performance(test_returns, test_benchmark)
    print("性能测试结果:")
    for metric, value in performance.items():
        print(f"{metric}: {value:.4f}")

    # 清理
    manager.cleanup()