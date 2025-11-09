"""
DeepSeekQuant 执行引擎
负责交易执行、订单管理、执行算法和经纪商接口
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
import asyncio
import aiohttp
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import copy
import traceback
from collections import deque, defaultdict
import heapq
import statistics
from decimal import Decimal
import math
import uuid
import socket
import ssl
import websockets
import pickle
import zlib
import base64

# 导入内部模块
from .base_processor import BaseProcessor
from ..utils.helpers import validate_data, calculate_returns, normalize_data
from ..utils.validators import validate_execution_parameters
from ..utils.performance import calculate_execution_quality
from ..core.portfolio_manager import PortfolioState
from ..core.risk_manager import RiskControlAction

logger = logging.getLogger('DeepSeekQuant.ExecutionEngine')


class OrderType(Enum):
    """订单类型枚举"""
    MARKET = "market"  # 市价单
    LIMIT = "limit"  # 限价单
    STOP = "stop"  # 止损单
    STOP_LIMIT = "stop_limit"  # 止损限价单
    TRAILING_STOP = "trailing_stop"  # 跟踪止损单
    ICEBERG = "iceberg"  # 冰山订单
    TWAP = "twap"  # 时间加权平均价格
    VWAP = "vwap"  # 成交量加权平均价格
    POV = "pov"  # 参与率订单
    MOC = "moc"  # 收盘市价单
    LOC = "loc"  # 收盘限价单


class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "pending"  # 等待提交
    SUBMITTED = "submitted"  # 已提交
    ACCEPTED = "accepted"  # 已接受
    PARTIALLY_FILLED = "partially_filled"  # 部分成交
    FILLED = "filled"  # 完全成交
    CANCELLED = "cancelled"  # 已取消
    REJECTED = "rejected"  # 已拒绝
    EXPIRED = "expired"  # 已过期
    SUSPENDED = "suspended"  # 已暂停
    PENDING_CANCEL = "pending_cancel"  # 等待取消


class ExecutionAlgorithm(Enum):
    """执行算法枚举"""
    SIMPLE = "simple"  # 简单执行
    TWAP = "twap"  # 时间加权平均价格
    VWAP = "vwap"  # 成交量加权平均价格
    POV = "pov"  # 参与率
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"  # 执行差额
    LIQUIDITY_SEEKING = "liquidity_seeking"  # 流动性寻找
    ADAPTIVE = "adaptive"  # 自适应执行
    SNIPER = "sniper"  # 狙击执行
    STEALTH = "stealth"  # 隐身执行
    AGGRESSIVE = "aggressive"  # 激进执行
    PASSIVE = "passive"  # 被动执行


class BrokerType(Enum):
    """经纪商类型枚举"""
    SIMULATED = "simulated"  # 模拟经纪商
    IBKR = "interactive_brokers"  # Interactive Brokers
    ALPACA = "alpaca"  # Alpaca
    ROBINHOOD = "robinhood"  # Robinhood
    BINANCE = "binance"  # Binance
    COINBASE = "coinbase"  # Coinbase
    KRAKEN = "kraken"  # Kraken
    OANDA = "oanda"  # OANDA
    FXCM = "fxcm"  # FXCM
    CUSTOM = "custom"  # 自定义接口


class OrderSide(Enum):
    """订单方向枚举"""
    BUY = "buy"  # 买入
    SELL = "sell"  # 卖出
    SELL_SHORT = "sell_short"  # 卖空
    BUY_TO_COVER = "buy_to_cover"  # 平仓买入


class TimeInForce(Enum):
    """订单有效期枚举"""
    DAY = "day"  # 当日有效
    GTC = "gtc"  # 撤销前有效
    OPG = "opg"  # 开盘时执行
    IOC = "ioc"  # 立即或取消
    FOK = "fok"  # 全部或取消
    GTD = "gtd"  # 指定日期前有效


@dataclass
class OrderParameters:
    """订单参数"""
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    good_till_date: Optional[str] = None
    iceberg: bool = False
    display_quantity: Optional[int] = None
    min_quantity: Optional[int] = None
    execution_instructions: List[str] = field(default_factory=list)
    routing_instructions: Dict[str, Any] = field(default_factory=dict)
    algo_parameters: Dict[str, Any] = field(default_factory=dict)
    smart_routing: bool = True
    destination: Optional[str] = None
    allocation_strategy: str = "fifo"
    notional: bool = False  # 是否按名义价值下单


@dataclass
class ExecutionParameters:
    """执行参数"""
    algorithm: ExecutionAlgorithm = ExecutionAlgorithm.SIMPLE
    urgency: str = "medium"  # low, medium, high, urgent
    max_participation_rate: float = 0.1  # 最大市场参与率
    target_performance: str = "price"  # price, time, liquidity
    price_deviation_limit: float = 0.02  # 价格偏离限制
    volume_limit: float = 0.2  # 成交量限制
    time_horizon: int = 300  # 执行时间窗口（秒）
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    allow_iceberg: bool = False
    allow_dark_pools: bool = False
    avoid_ecns: bool = False
    benchmark: str = "arrival_price"  # 执行基准
    tolerance_bands: Dict[str, float] = field(default_factory=lambda: {"upper": 0.01, "lower": -0.01})
    slippage_control: bool = True
    real_time_adjustment: bool = True


@dataclass
class Order:
    """订单数据类"""
    order_id: str
    portfolio_id: str
    symbol: str
    quantity: float
    side: OrderSide
    parameters: OrderParameters
    execution_params: ExecutionParameters
    status: OrderStatus = OrderStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    commission: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0
    timing_risk: float = 0.0
    execution_quality: float = 0.0
    parent_order_id: Optional[str] = None
    child_orders: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    risk_check_passed: bool = False
    broker_order_id: Optional[str] = None
    exchange_order_id: Optional[str] = None
    rejection_reason: Optional[str] = None
    cancellation_reason: Optional[str] = None
    expiration_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'order_id': self.order_id,
            'portfolio_id': self.portfolio_id,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'side': self.side.value,
            'parameters': asdict(self.parameters),
            'execution_params': asdict(self.execution_params),
            'status': self.status.value,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'filled_quantity': self.filled_quantity,
            'average_fill_price': self.average_fill_price,
            'commission': self.commission,
            'fees': self.fees,
            'slippage': self.slippage,
            'market_impact': self.market_impact,
            'timing_risk': self.timing_risk,
            'execution_quality': self.execution_quality,
            'parent_order_id': self.parent_order_id,
            'child_orders': self.child_orders,
            'tags': self.tags,
            'metadata': self.metadata,
            'risk_check_passed': self.risk_check_passed,
            'broker_order_id': self.broker_order_id,
            'exchange_order_id': self.exchange_order_id,
            'rejection_reason': self.rejection_reason,
            'cancellation_reason': self.cancellation_reason,
            'expiration_time': self.expiration_time
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """从字典创建订单"""
        params_data = data.pop('parameters', {})
        exec_params_data = data.pop('execution_params', {})

        parameters = OrderParameters(**params_data)
        execution_params = ExecutionParameters(**exec_params_data)

        return cls(
            order_id=data['order_id'],
            portfolio_id=data['portfolio_id'],
            symbol=data['symbol'],
            quantity=data['quantity'],
            side=OrderSide(data['side']),
            parameters=parameters,
            execution_params=execution_params,
            status=OrderStatus(data.get('status', 'pending')),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at'),
            filled_quantity=data.get('filled_quantity', 0.0),
            average_fill_price=data.get('average_fill_price', 0.0),
            commission=data.get('commission', 0.0),
            fees=data.get('fees', 0.0),
            slippage=data.get('slippage', 0.0),
            market_impact=data.get('market_impact', 0.0),
            timing_risk=data.get('timing_risk', 0.0),
            execution_quality=data.get('execution_quality', 0.0),
            parent_order_id=data.get('parent_order_id'),
            child_orders=data.get('child_orders', []),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {}),
            risk_check_passed=data.get('risk_check_passed', False),
            broker_order_id=data.get('broker_order_id'),
            exchange_order_id=data.get('exchange_order_id'),
            rejection_reason=data.get('rejection_reason'),
            cancellation_reason=data.get('cancellation_reason'),
            expiration_time=data.get('expiration_time')
        )


@dataclass
class ExecutionReport:
    """执行报告"""
    report_id: str
    order_id: str
    timestamp: str
    fill_price: float
    fill_quantity: float
    remaining_quantity: float
    cumulative_quantity: float
    execution_type: str  # partial, full, cancelled, etc.
    liquidity: str  # added, removed, hidden
    venue: str
    execution_id: str
    transaction_time: str
    commission: float
    fees: float
    slippage: float
    market_impact: float
    timing_risk: float
    execution_quality: float
    benchmark_comparison: Dict[str, float]
    flags: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BrokerConnection:
    """经纪商连接配置"""
    broker_type: BrokerType
    account_id: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    api_token: Optional[str] = None
    base_url: str = ""
    websocket_url: Optional[str] = None
    paper_trading: bool = True
    rate_limit: int = 100
    connection_timeout: int = 30
    request_timeout: int = 10
    retry_attempts: int = 3
    retry_delay: int = 1
    ssl_verification: bool = True
    proxy_settings: Optional[Dict[str, Any]] = None
    custom_headers: Dict[str, str] = field(default_factory=dict)
    session_token: Optional[str] = None
    token_expiry: Optional[str] = None
    last_heartbeat: Optional[str] = None
    connection_status: str = "disconnected"
    supported_features: List[str] = field(default_factory=list)
    limitations: Dict[str, Any] = field(default_factory=dict)


class ExecutionEngine(BaseProcessor):
    """执行引擎 - 负责交易订单的执行和管理"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化执行引擎

        Args:
            config: 配置字典
        """
        super().__init__(config)

        # 执行配置
        self.execution_config = config.get('execution', {})
        self.broker_config = self.execution_config.get('broker_connections', {})
        self.algo_config = self.execution_config.get('execution_algorithms', {})
        self.cost_config = self.execution_config.get('transaction_costs', {})

        # 订单管理
        self.pending_orders: Dict[str, Order] = {}
        self.active_orders: Dict[str, Order] = {}
        self.completed_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.execution_reports: Dict[str, List[ExecutionReport]] = {}

        # 经纪商连接
        self.broker_connections: Dict[str, BrokerConnection] = {}
        self._initialize_broker_connections()

        # 执行算法
        self.execution_algorithms: Dict[str, Any] = {}
        self._initialize_execution_algorithms()

        # 市场数据
        self.market_data_cache: Dict[str, Any] = {}
        self.liquidity_data: Dict[str, float] = {}
        self.volatility_data: Dict[str, float] = {}

        # 性能监控
        self.performance_stats = {
            'orders_executed': 0,
            'total_volume': 0.0,
            'total_commission': 0.0,
            'total_slippage': 0.0,
            'avg_execution_time': 0.0,
            'success_rate': 0.0,
            'avg_execution_quality': 0.0,
            'rejected_orders': 0,
            'cancelled_orders': 0
        }

        # 线程池和异步处理
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.execution_config.get('max_execution_threads', 8)
        )
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)

        # 连接状态
        self.is_connected = False
        self.last_connection_check = None
        self.connection_retries = 0

        # 订单ID生成器
        self.order_counter = 0

        logger.info("执行引擎初始化完成")

    def _initialize_broker_connections(self):
        """初始化经纪商连接"""
        try:
            for broker_name, broker_config in self.broker_config.items():
                try:
                    broker_type = BrokerType(broker_config.get('type', 'simulated'))

                    connection = BrokerConnection(
                        broker_type=broker_type,
                        account_id=broker_config.get('account_id', ''),
                        api_key=broker_config.get('api_key'),
                        api_secret=broker_config.get('api_secret'),
                        base_url=broker_config.get('base_url', ''),
                        paper_trading=broker_config.get('paper_trading', True),
                        rate_limit=broker_config.get('rate_limit', 100),
                        connection_timeout=broker_config.get('connection_timeout', 30),
                        request_timeout=broker_config.get('request_timeout', 10),
                        retry_attempts=broker_config.get('retry_attempts', 3),
                        ssl_verification=broker_config.get('ssl_verification', True)
                    )

                    self.broker_connections[broker_name] = connection
                    logger.info(f"经纪商连接初始化: {broker_name} ({broker_type.value})")

                except Exception as e:
                    logger.error(f"经纪商连接 {broker_name} 初始化失败: {e}")

            # 设置默认经纪商
            self.default_broker = self.execution_config.get('default_broker', 'simulated')

        except Exception as e:
            logger.error(f"经纪商连接初始化失败: {e}")

    def _initialize_execution_algorithms(self):
        """初始化执行算法"""
        try:
            algorithms = {
                'simple': self._execute_simple,
                'twap': self._execute_twap,
                'vwap': self._execute_vwap,
                'pov': self._execute_pov,
                'implementation_shortfall': self._execute_implementation_shortfall,
                'liquidity_seeking': self._execute_liquidity_seeking,
                'adaptive': self._execute_adaptive,
                'sniper': self._execute_sniper,
                'stealth': self._execute_stealth,
                'aggressive': self._execute_aggressive,
                'passive': self._execute_passive
            }

            # 加载配置参数
            for algo_name, algo_config in self.algo_config.items():
                if algo_name in algorithms:
                    # 可以在这里配置算法参数
                    logger.info(f"执行算法加载: {algo_name}")

            self.execution_algorithms = algorithms
            logger.info(f"已加载 {len(algorithms)} 个执行算法")

        except Exception as e:
            logger.error(f"执行算法初始化失败: {e}")

    def connect(self) -> bool:
        """连接到经纪商"""
        try:
            logger.info("开始连接经纪商...")

            # 并行连接所有经纪商
            connection_results = []
            with ThreadPoolExecutor(max_workers=len(self.broker_connections)) as executor:
                future_to_broker = {
                    executor.submit(self._connect_to_broker, broker_name, connection): broker_name
                    for broker_name, connection in self.broker_connections.items()
                }

                for future in as_completed(future_to_broker):
                    broker_name = future_to_broker[future]
                    try:
                        success = future.result()
                        connection_results.append(success)
                        if success:
                            logger.info(f"经纪商 {broker_name} 连接成功")
                        else:
                            logger.warning(f"经纪商 {broker_name} 连接失败")
                    except Exception as e:
                        logger.error(f"经纪商 {broker_name} 连接错误: {e}")
                        connection_results.append(False)

            # 检查连接状态
            successful_connections = sum(connection_results)
            total_connections = len(connection_results)

            self.is_connected = successful_connections > 0
            self.last_connection_check = datetime.now().isoformat()

            if self.is_connected:
                logger.info(f"执行引擎连接成功: {successful_connections}/{total_connections} 个经纪商")
                # 启动心跳检测
                self._start_heartbeat()
            else:
                logger.error("所有经纪商连接失败")

            return self.is_connected

        except Exception as e:
            logger.error(f"执行引擎连接失败: {e}")
            return False

    def _connect_to_broker(self, broker_name: str, connection: BrokerConnection) -> bool:
        """连接到单个经纪商"""
        try:
            if connection.broker_type == BrokerType.SIMULATED:
                # 模拟经纪商不需要实际连接
                connection.connection_status = "connected"
                return True

            elif connection.broker_type == BrokerType.IBKR:
                return self._connect_to_ibkr(connection)

            elif connection.broker_type == BrokerType.ALPACA:
                return self._connect_to_alpaca(connection)

            elif connection.broker_type == BrokerType.BINANCE:
                return self._connect_to_binance(connection)

            else:
                logger.warning(f"不支持的经纪商类型: {connection.broker_type}")
                return False

        except Exception as e:
            logger.error(f"经纪商 {broker_name} 连接失败: {e}")
            connection.connection_status = "disconnected"
            return False

    def _connect_to_ibkr(self, connection: BrokerConnection) -> bool:
        """连接到Interactive Brokers"""
        try:
            # 这里需要实际的IBKR连接逻辑
            # 简化实现：返回成功
            connection.connection_status = "connected"
            connection.last_heartbeat = datetime.now().isoformat()
            return True

        except Exception as e:
            logger.error(f"IBKR连接失败: {e}")
            return False

    def _connect_to_alpaca(self, connection: BrokerConnection) -> bool:
        """连接到Alpaca"""
        try:
            # 这里需要实际的Alpaca连接逻辑
            # 简化实现：返回成功
            connection.connection_status = "connected"
            connection.last_heartbeat = datetime.now().isoformat()
            return True

        except Exception as e:
            logger.error(f"Alpaca连接失败: {e}")
            return False

    def _connect_to_binance(self, connection: BrokerConnection) -> bool:
        """连接到Binance"""
        try:
            # 这里需要实际的Binance连接逻辑
            # 简化实现：返回成功
            connection.connection_status = "connected"
            connection.last_heartbeat = datetime.now().isoformat()
            return True

        except Exception as e:
            logger.error(f"Binance连接失败: {e}")
            return False

    def _start_heartbeat(self):
        """启动心跳检测"""

        def heartbeat_loop():
            while self.is_connected:
                try:
                    self._check_connections()
                    time.sleep(30)  # 每30秒检查一次
                except Exception as e:
                    logger.error(f"心跳检测错误: {e}")
                    time.sleep(5)

        heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        heartbeat_thread.start()
        logger.info("心跳检测已启动")

    def _check_connections(self):
        """检查经纪商连接状态"""
        try:
            for broker_name, connection in self.broker_connections.items():
                if connection.connection_status == "connected":
                    # 模拟心跳检测
                    connection.last_heartbeat = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"连接状态检查失败: {e}")

    def disconnect(self):
        """断开经纪商连接"""
        try:
            self.is_connected = False

            for broker_name, connection in self.broker_connections.items():
                connection.connection_status = "disconnected"
                logger.info(f"经纪商 {broker_name} 已断开连接")

            logger.info("执行引擎已断开所有连接")

        except Exception as e:
            logger.error(f"断开连接失败: {e}")

    def execute_orders(self, orders: List[Order],
                       market_data: Dict[str, Any],
                       risk_assessment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行订单列表

        Args:
            orders: 订单列表
            market_data: 市场数据
            risk_assessment: 风险评估结果

        Returns:
            执行结果
        """
        start_time = time.time()

        try:
            if not self.is_connected:
                logger.warning("执行引擎未连接，尝试重新连接...")
                if not self.connect():
                    return {'error': '执行引擎未连接且重新连接失败'}

            # 验证订单
            validated_orders = self._validate_orders(orders, market_data, risk_assessment)
            if not validated_orders:
                return {'error': '没有有效的订单可执行'}

            # 分组订单（按符号、方向、算法等）
            order_groups = self._group_orders(validated_orders)

            # 并行执行订单组
            execution_results = {}
            with ThreadPoolExecutor(max_workers=min(len(order_groups), 4)) as executor:
                future_to_group = {
                    executor.submit(self._execute_order_group, group, market_data): group_id
                    for group_id, group in order_groups.items()
                }

                for future in as_completed(future_to_group):
                    group_id = future_to_group[future]
                    try:
                        result = future.result()
                        execution_results[group_id] = result
                    except Exception as e:
                        logger.error(f"订单组 {group_id} 执行失败: {e}")
                        execution_results[group_id] = {'error': str(e)}

            # 汇总执行结果
            overall_result = self._aggregate_execution_results(execution_results)
            processing_time = time.time() - start_time

            # 更新性能统计
            self._update_performance_stats(overall_result, processing_time)

            logger.info(f"订单执行完成: 处理订单={len(validated_orders)}, "
                        f"成功={overall_result.get('successful_orders', 0)}, "
                        f"失败={overall_result.get('failed_orders', 0)}, "
                        f"耗时={processing_time:.3f}s")

            return overall_result

        except Exception as e:
            logger.error(f"订单执行失败: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def _validate_orders(self, orders: List[Order],
                         market_data: Dict[str, Any],
                         risk_assessment: Optional[Dict[str, Any]]) -> List[Order]:
        """验证订单有效性"""
        validated_orders = []

        try:
            for order in orders:
                try:
                    # 基本验证
                    if not self._validate_single_order(order):
                        logger.warning(f"订单验证失败: {order.order_id}")
                        order.status = OrderStatus.REJECTED
                        order.rejection_reason = "基本验证失败"
                        continue

                    # 市场数据验证
                    if not self._validate_market_data(order, market_data):
                        logger.warning(f"订单市场数据验证失败: {order.order_id}")
                        order.status = OrderStatus.REJECTED
                        order.rejection_reason = "市场数据不足"
                        continue

                    # 风险验证
                    if risk_assessment and not self._validate_risk_compliance(order, risk_assessment):
                        logger.warning(f"订单风险验证失败: {order.order_id}")
                        order.status = OrderStatus.REJECTED
                        order.rejection_reason = "风险控制限制"
                        continue

                    # 经纪商验证
                    if not self._validate_broker_compatibility(order):
                        logger.warning(f"订单经纪商验证失败: {order.order_id}")
                        order.status = OrderStatus.REJECTED
                        order.rejection_reason = "经纪商不支持"
                        continue

                    # 算法参数验证
                    if not self._validate_algorithm_parameters(order):
                        logger.warning(f"订单算法参数验证失败: {order.order_id}")
                        order.status = OrderStatus.REJECTED
                        order.rejection_reason = "算法参数无效"
                        continue

                    order.risk_check_passed = True
                    validated_orders.append(order)
                    logger.debug(f"订单验证通过: {order.order_id}")

                except Exception as e:
                    logger.error(f"订单验证错误 {order.order_id}: {e}")
                    order.status = OrderStatus.REJECTED
                    order.rejection_reason = f"验证错误: {str(e)}"

            return validated_orders

        except Exception as e:
            logger.error(f"订单验证过程失败: {e}")
            return []

    def _validate_single_order(self, order: Order) -> bool:
        """验证单个订单"""
        try:
            # 检查必需字段
            if not all([order.order_id, order.portfolio_id, order.symbol, order.quantity]):
                return False

            # 检查数量有效性
            if order.quantity <= 0:
                return False

            # 检查价格有效性
            if order.parameters.order_type == OrderType.LIMIT and order.parameters.limit_price <= 0:
                return False

            if order.parameters.order_type == OrderType.STOP and order.parameters.stop_price <= 0:
                return False

            # 检查时间有效性
            if order.parameters.time_in_force == TimeInForce.GTD:
                if not order.parameters.good_till_date:
                    return False
                try:
                    expiry_date = datetime.fromisoformat(order.parameters.good_till_date)
                    if expiry_date <= datetime.now():
                        return False
                except ValueError:
                    return False

            # 检查冰山订单参数
            if order.parameters.iceberg:
                if not order.parameters.display_quantity or order.parameters.display_quantity <= 0:
                    return False
                if order.parameters.display_quantity >= order.quantity:
                    return False

            return True

        except Exception as e:
            logger.error(f"单个订单验证失败 {order.order_id}: {e}")
            return False

    def _validate_market_data(self, order: Order, market_data: Dict[str, Any]) -> bool:
        """验证市场数据"""
        try:
            symbol = order.symbol

            # 检查价格数据
            if symbol not in market_data.get('prices', {}):
                return False

            price_data = market_data['prices'][symbol]
            if not price_data or 'close' not in price_data or len(price_data['close']) < 1:
                return False

            # 检查流动性数据
            if symbol not in market_data.get('volumes', {}):
                return False

            volume_data = market_data['volumes'][symbol]
            if not volume_data or 'volume' not in volume_data:
                return False

            # 对于限价单，检查价格合理性
            if order.parameters.order_type == OrderType.LIMIT:
                current_price = price_data['close'][-1]
                limit_price = order.parameters.limit_price

                # 检查限价是否在合理范围内（当前价格的±50%）
                price_ratio = limit_price / current_price
                if price_ratio < 0.5 or price_ratio > 1.5:
                    logger.warning(f"限价不合理: {symbol}, 当前价={current_price}, 限价={limit_price}")
                    return False

            return True

        except Exception as e:
            logger.error(f"市场数据验证失败 {order.order_id}: {e}")
            return False

    def _validate_risk_compliance(self, order: Order, risk_assessment: Dict[str, Any]) -> bool:
        """验证风险合规性"""
        try:
            # 检查总体风险等级
            risk_level = risk_assessment.get('overall_risk_level', 'moderate')
            if risk_level in ['high', 'very_high', 'extreme']:
                # 高风险情况下限制大额订单
                if order.quantity > risk_assessment.get('max_order_size', 1000):
                    return False

            # 检查限额违反
            limit_breaches = risk_assessment.get('limit_breaches', [])
            critical_breaches = [b for b in limit_breaches if b.get('severity') in ['critical', 'high']]

            if critical_breaches:
                # 有严重限额违反时，限制新订单
                if order.quantity > risk_assessment.get('restricted_order_size', 100):
                    return False

            # 检查流动性风险
            liquidity_risk = risk_assessment.get('liquidity_risk', 0.5)
            if liquidity_risk > 0.7:
                # 高流动性风险时，限制大额订单
                max_size = risk_assessment.get('liquidity_restricted_size', 500)
                if order.quantity > max_size:
                    return False

            return True

        except Exception as e:
            logger.error(f"风险合规验证失败 {order.order_id}: {e}")
            return True  # 风险验证失败时默认允许

    def _validate_broker_compatibility(self, order: Order) -> bool:
        """验证经纪商兼容性"""
        try:
            broker_name = order.execution_params.algo_parameters.get('broker', self.default_broker)
            if broker_name not in self.broker_connections:
                return False

            connection = self.broker_connections[broker_name]

            # 检查订单类型支持
            supported_order_types = connection.supported_features.get('order_types', [])
            if order.parameters.order_type.value not in supported_order_types:
                return False

            # 检查算法支持
            supported_algorithms = connection.supported_features.get('algorithms', [])
            if order.execution_params.algorithm.value not in supported_algorithms:
                return False

            # 检查资产类别支持
            asset_class = order.metadata.get('asset_class', 'equity')
            supported_assets = connection.supported_features.get('asset_classes', [])
            if asset_class not in supported_assets:
                return False

            return True

        except Exception as e:
            logger.error(f"经纪商兼容性验证失败 {order.order_id}: {e}")
            return False

    def _validate_algorithm_parameters(self, order: Order) -> bool:
        """验证算法参数"""
        try:
            algorithm = order.execution_params.algorithm
            params = order.execution_params.algo_parameters

            # 通用参数验证
            if 'urgency' in params:
                urgency = params['urgency']
                if urgency not in ['low', 'medium', 'high', 'urgent']:
                    return False

            if 'max_participation_rate' in params:
                rate = params['max_participation_rate']
                if not 0 < rate <= 1.0:
                    return False

            if 'price_deviation_limit' in params:
                deviation = params['price_deviation_limit']
                if not 0 < deviation <= 0.1:  # 最大10%偏离
                    return False

            # 算法特定验证
            if algorithm == ExecutionAlgorithm.TWAP:
                if 'time_horizon' not in params or params['time_horizon'] <= 0:
                    return False

            elif algorithm == ExecutionAlgorithm.VWAP:
                if 'volume_participation' not in params or not 0 < params['volume_participation'] <= 1.0:
                    return False

            elif algorithm == ExecutionAlgorithm.POV:
                if 'participation_rate' not in params or not 0 < params['participation_rate'] <= 0.5:
                    return False

            return True

        except Exception as e:
            logger.error(f"算法参数验证失败 {order.order_id}: {e}")
            return False

    def _group_orders(self, orders: List[Order]) -> Dict[str, List[Order]]:
        """将订单分组"""
        groups = defaultdict(list)

        try:
            for order in orders:
                # 创建分组键
                group_key = self._create_order_group_key(order)
                groups[group_key].append(order)

            logger.info(f"订单分组完成: {len(groups)} 个组, {len(orders)} 个订单")
            return dict(groups)

        except Exception as e:
            logger.error(f"订单分组失败: {e}")
            # 失败时将所有订单放入一个组
            return {'default_group': orders}

    def _create_order_group_key(self, order: Order) -> str:
        """创建订单分组键"""
        try:
            # 基于符号、方向、算法和经纪商分组
            key_parts = [
                order.symbol,
                order.side.value,
                order.execution_params.algorithm.value,
                order.execution_params.algo_parameters.get('broker', self.default_broker),
                str(order.parameters.order_type.value)
            ]

            return '_'.join(key_parts)

        except Exception as e:
            logger.error(f"订单分组键创建失败 {order.order_id}: {e}")
            return 'default_group'

    def _execute_order_group(self, orders: List[Order], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行订单组"""
        group_results = {
            'group_id': hashlib.md5(str(orders[0].order_id).encode()).hexdigest()[:8],
            'total_orders': len(orders),
            'successful_orders': 0,
            'failed_orders': 0,
            'partial_orders': 0,
            'total_quantity': 0,
            'executed_quantity': 0,
            'total_commission': 0,
            'total_slippage': 0,
            'avg_execution_quality': 0,
            'order_results': [],
            'start_time': datetime.now().isoformat(),
            'end_time': None
        }

        try:
            # 按优先级排序订单
            sorted_orders = self._prioritize_orders(orders)

            # 顺序执行订单（避免市场影响）
            for order in sorted_orders:
                try:
                    result = self._execute_single_order(order, market_data)
                    group_results['order_results'].append(result)

                    if result['status'] == 'filled':
                        group_results['successful_orders'] += 1
                    elif result['status'] == 'partially_filled':
                        group_results['partial_orders'] += 1
                    else:
                        group_results['failed_orders'] += 1

                    group_results['total_quantity'] += order.quantity
                    group_results['executed_quantity'] += result.get('executed_quantity', 0)
                    group_results['total_commission'] += result.get('commission', 0)
                    group_results['total_slippage'] += result.get('slippage', 0)

                except Exception as e:
                    logger.error(f"订单执行失败 {order.order_id}: {e}")
                    failed_result = {
                        'order_id': order.order_id,
                        'status': 'failed',
                        'error': str(e),
                        'executed_quantity': 0,
                        'commission': 0,
                        'slippage': 0
                    }
                    group_results['order_results'].append(failed_result)
                    group_results['failed_orders'] += 1

            # 计算平均执行质量
            successful_results = [r for r in group_results['order_results'] if
                                  r['status'] in ['filled', 'partially_filled']]
            if successful_results:
                avg_quality = sum(r.get('execution_quality', 0) for r in successful_results) / len(successful_results)
                group_results['avg_execution_quality'] = avg_quality

            group_results['end_time'] = datetime.now().isoformat()

            return group_results

        except Exception as e:
            logger.error(f"订单组执行失败: {e}")
            group_results['error'] = str(e)
            group_results['end_time'] = datetime.now().isoformat()
            return group_results

    def _prioritize_orders(self, orders: List[Order]) -> List[Order]:
        """订单优先级排序"""
        try:
            def get_order_priority(order: Order) -> tuple:
                # 优先级因素：紧急程度、算法复杂度、订单大小
                urgency_score = {
                    'low': 1, 'medium': 2, 'high': 3, 'urgent': 4
                }.get(order.execution_params.urgency, 2)

                algorithm_complexity = {
                    ExecutionAlgorithm.SIMPLE: 1,
                    ExecutionAlgorithm.PASSIVE: 1,
                    ExecutionAlgorithm.AGGRESSIVE: 2,
                    ExecutionAlgorithm.VWAP: 3,
                    ExecutionAlgorithm.TWAP: 3,
                    ExecutionAlgorithm.POV: 4,
                    ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL: 4
                }.get(order.execution_params.algorithm, 2)

                # 大额订单优先执行（但需要考虑市场影响）
                size_priority = min(order.quantity / 1000, 5)  # 标准化到0-5

                return (-urgency_score, -algorithm_complexity, -size_priority)  # 负号因为优先级高排前面

            return sorted(orders, key=get_order_priority)

        except Exception as e:
            logger.error(f"订单优先级排序失败: {e}")
            return orders  # 返回原始顺序

    def _execute_single_order(self, order: Order, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个订单"""
        start_time = time.time()

        try:
            # 更新订单状态
            order.status = OrderStatus.SUBMITTED
            order.updated_at = datetime.now().isoformat()

            # 选择执行算法
            algorithm_func = self.execution_algorithms.get(
                order.execution_params.algorithm.value,
                self._execute_simple
            )

            # 执行订单
            execution_result = algorithm_func(order, market_data)

            # 计算执行时间
            execution_time = time.time() - start_time

            # 更新订单信息
            order.updated_at = datetime.now().isoformat()
            order.filled_quantity = execution_result.get('executed_quantity', 0)
            order.average_fill_price = execution_result.get('average_price', 0)
            order.commission = execution_result.get('commission', 0)
            order.fees = execution_result.get('fees', 0)
            order.slippage = execution_result.get('slippage', 0)
            order.market_impact = execution_result.get('market_impact', 0)
            order.timing_risk = execution_result.get('timing_risk', 0)
            order.execution_quality = execution_result.get('execution_quality', 0)

            # 更新订单状态
            if order.filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
                status = 'filled'
            elif order.filled_quantity > 0:
                order.status = OrderStatus.PARTIALLY_FILLED
                status = 'partially_filled'
            else:
                order.status = OrderStatus.REJECTED
                status = 'failed'

            # 保存订单到历史
            self._save_order_to_history(order)

            # 生成执行报告
            execution_report = self._generate_execution_report(order, execution_result, execution_time)
            self._save_execution_report(execution_report)

            result = {
                'order_id': order.order_id,
                'status': status,
                'executed_quantity': order.filled_quantity,
                'average_price': order.average_fill_price,
                'commission': order.commission,
                'fees': order.fees,
                'slippage': order.slippage,
                'market_impact': order.market_impact,
                'timing_risk': order.timing_risk,
                'execution_quality': order.execution_quality,
                'execution_time': execution_time,
                'broker_order_id': execution_result.get('broker_order_id'),
                'exchange_order_id': execution_result.get('exchange_order_id'),
                'execution_reports': execution_result.get('execution_reports', [])
            }

            if status == 'failed':
                result['error'] = execution_result.get('error', '执行失败')

            logger.info(f"订单执行完成: {order.order_id}, 状态={status}, "
                        f"数量={order.filled_quantity}/{order.quantity}, "
                        f"质量={order.execution_quality:.3f}")

            return result

        except Exception as e:
            logger.error(f"单个订单执行失败 {order.order_id}: {e}")

            # 更新订单状态为失败
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.now().isoformat()
            order.rejection_reason = f"执行错误: {str(e)}"

            self._save_order_to_history(order)

            return {
                'order_id': order.order_id,
                'status': 'failed',
                'error': str(e),
                'executed_quantity': 0,
                'commission': 0,
                'slippage': 0,
                'execution_time': time.time() - start_time
            }

    def _execute_simple(self, order: Order, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """简单执行算法"""
        try:
            symbol = order.symbol
            quantity = order.quantity
            side = order.side

            # 获取当前市场价格
            current_price = market_data['prices'][symbol]['close'][-1]

            # 计算执行价格（考虑买卖价差）
            spread = self._calculate_bid_ask_spread(symbol, market_data)
            if side == OrderSide.BUY:
                execution_price = current_price * (1 + spread / 2)  # 买入价（卖一价）
            else:
                execution_price = current_price * (1 - spread / 2)  # 卖出价（买一价）

            # 对于限价单，使用限价价格
            if order.parameters.order_type == OrderType.LIMIT:
                execution_price = order.parameters.limit_price
                # 检查限价是否可执行
                if (side == OrderSide.BUY and execution_price < current_price) or \
                        (side == OrderSide.SELL and execution_price > current_price):
                    execution_price = current_price  # 使用市价执行

            # 计算交易成本
            commission = self._calculate_commission(order, execution_price)
            fees = self._calculate_fees(order, execution_price)

            # 计算滑点
            slippage = self._calculate_slippage(order, market_data, execution_price)
            final_price = execution_price + slippage

            # 计算市场影响
            market_impact = self._calculate_market_impact(order, market_data, final_price)

            # 计算执行质量
            execution_quality = self._calculate_execution_quality(order, current_price, final_price, market_impact)

            # 模拟经纪商执行
            broker_result = self._execute_with_broker(order, final_price, quantity)

            result = {
                'executed_quantity': quantity,
                'average_price': final_price,
                'commission': commission,
                'fees': fees,
                'slippage': slippage,
                'market_impact': market_impact,
                'timing_risk': 0.0,  # 简单执行无时间风险
                'execution_quality': execution_quality,
                'broker_order_id': broker_result.get('order_id'),
                'exchange_order_id': broker_result.get('exchange_id'),
                'execution_reports': broker_result.get('reports', [])
            }

            return result

        except Exception as e:
            logger.error(f"简单执行算法失败 {order.order_id}: {e}")
            return {'error': str(e)}

    def _execute_twap(self, order: Order, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """TWAP执行算法"""
        try:
            symbol = order.symbol
            quantity = order.quantity
            time_horizon = order.execution_params.algo_parameters.get('time_horizon', 300)  # 默认5分钟
            slices = order.execution_params.algo_parameters.get('slices', 5)  # 默认分5片

            slice_quantity = quantity / slices
            slice_interval = time_horizon / slices

            total_executed = 0
            total_cost = 0
            execution_reports = []

            for slice_num in range(slices):
                # 等待到下一个执行时间
                if slice_num > 0:
                    time.sleep(slice_interval)

                # 获取当前市场数据
                current_price = market_data['prices'][symbol]['close'][-1]

                # 执行当前切片
                slice_order = copy.deepcopy(order)
                slice_order.quantity = slice_quantity
                slice_order.order_id = f"{order.order_id}_slice_{slice_num}"

                slice_result = self._execute_simple(slice_order, market_data)

                if slice_result.get('error'):
                    logger.warning(f"TWAP切片 {slice_num} 执行失败: {slice_result['error']}")
                    continue

                total_executed += slice_result['executed_quantity']
                total_cost += slice_result['average_price'] * slice_result['executed_quantity']
                execution_reports.append(slice_result)

                logger.debug(
                    f"TWAP切片 {slice_num} 执行完成: {slice_result['executed_quantity']} @ {slice_result['average_price']}")

            # 计算总体结果
            if total_executed > 0:
                average_price = total_cost / total_executed
            else:
                average_price = 0

            # 计算执行质量
            initial_price = market_data['prices'][symbol]['close'][0]
            price_improvement = average_price - initial_price if order.side == OrderSide.BUY else initial_price - average_price

            result = {
                'executed_quantity': total_executed,
                'average_price': average_price,
                'commission': sum(r['commission'] for r in execution_reports),
                'fees': sum(r['fees'] for r in execution_reports),
                'slippage': sum(r['slippage'] for r in execution_reports),
                'market_impact': sum(r['market_impact'] for r in execution_reports) / len(execution_reports),
                'timing_risk': self._calculate_twap_timing_risk(order, execution_reports),
                'execution_quality': price_improvement / initial_price if initial_price > 0 else 0,
                'execution_reports': execution_reports
            }

            return result

        except Exception as e:
            logger.error(f"TWAP执行算法失败 {order.order_id}: {e}")
            return {'error': str(e)}

    def _execute_vwap(self, order: Order, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """VWAP执行算法"""
        try:
            symbol = order.symbol
            quantity = order.quantity

            # 获取历史成交量数据
            volume_data = market_data['volumes'][symbol].get('volume', [])
            if not volume_data:
                return self._execute_twap(order, market_data)  # 回退到TWAP

            # 计算成交量分布
            total_volume = sum(volume_data)
            if total_volume <= 0:
                return self._execute_twap(order, market_data)

            # 计算每个时间段的成交量权重
            volume_weights = [v / total_volume for v in volume_data]

            # 确定执行时间段
            time_horizon = order.execution_params.algo_parameters.get('time_horizon', len(volume_data))
            slices = min(time_horizon, len(volume_data))

            # 计算每个时间片的执行量
            slice_quantities = []
            for i in range(slices):
                slice_qty = quantity * volume_weights[i]
                slice_quantities.append(slice_qty)

            total_executed = 0
            total_cost = 0
            execution_reports = []

            for slice_num in range(slices):
                # 等待到下一个执行时间（模拟）
                if slice_num > 0:
                    time.sleep(1)  # 简化实现，实际应根据市场时间

                # 获取当前市场数据
                current_price = market_data['prices'][symbol]['close'][slice_num]

                # 执行当前切片
                slice_order = copy.deepcopy(order)
                slice_order.quantity = slice_quantities[slice_num]
                slice_order.order_id = f"{order.order_id}_vwap_slice_{slice_num}"

                slice_result = self._execute_simple(slice_order, market_data)

                if slice_result.get('error'):
                    logger.warning(f"VWAP切片 {slice_num} 执行失败: {slice_result['error']}")
                    continue

                total_executed += slice_result['executed_quantity']
                total_cost += slice_result['average_price'] * slice_result['executed_quantity']
                execution_reports.append(slice_result)

                logger.debug(
                    f"VWAP切片 {slice_num} 执行完成: {slice_result['executed_quantity']} @ {slice_result['average_price']}")

            # 计算总体结果
            if total_executed > 0:
                average_price = total_cost / total_executed
            else:
                average_price = 0

            # 计算VWAP基准
            vwap_benchmark = sum(p * w for p, w in zip(
                market_data['prices'][symbol]['close'][:slices],
                volume_weights[:slices]
            ))

            # 计算执行质量
            price_improvement = average_price - vwap_benchmark if order.side == OrderSide.BUY else vwap_benchmark - average_price

            result = {
                'executed_quantity': total_executed,
                'average_price': average_price,
                'commission': sum(r['commission'] for r in execution_reports),
                'fees': sum(r['fees'] for r in execution_reports),
                'slippage': sum(r['slippage'] for r in execution_reports),
                'market_impact': sum(r['market_impact'] for r in execution_reports) / len(execution_reports),
                'timing_risk': self._calculate_vwap_timing_risk(order, execution_reports, vwap_benchmark),
                'execution_quality': price_improvement / vwap_benchmark if vwap_benchmark > 0 else 0,
                'vwap_benchmark': vwap_benchmark,
                'execution_reports': execution_reports
            }

            return result

        except Exception as e:
            logger.error(f"VWAP执行算法失败 {order.order_id}: {e}")
            return {'error': str(e)}

    def _execute_pov(self, order: Order, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """参与率(POV)执行算法"""
        try:
            symbol = order.symbol
            quantity = order.quantity

            # 获取参与率参数
            participation_rate = order.execution_params.algo_parameters.get('participation_rate', 0.1)
            max_participation = order.execution_params.algo_parameters.get('max_participation_rate', 0.2)
            min_participation = order.execution_params.algo_parameters.get('min_participation_rate', 0.05)

            participation_rate = max(min(participation_rate, max_participation), min_participation)

            # 获取市场成交量数据
            volume_data = market_data['volumes'][symbol].get('volume', [])
            if not volume_data:
                return self._execute_twap(order, market_data)

            total_executed = 0
            total_cost = 0
            execution_reports = []
            remaining_quantity = quantity
            time_slices = min(len(volume_data), 10)  # 限制时间片数量

            for slice_num in range(time_slices):
                if remaining_quantity <= 0:
                    break

                # 计算当前市场成交量
                current_volume = volume_data[slice_num]
                if current_volume <= 0:
                    continue

                # 计算当前切片执行量
                slice_qty = min(remaining_quantity, current_volume * participation_rate)
                if slice_qty <= 0:
                    continue

                # 执行当前切片
                slice_order = copy.deepcopy(order)
                slice_order.quantity = slice_qty
                slice_order.order_id = f"{order.order_id}_pov_slice_{slice_num}"

                slice_result = self._execute_simple(slice_order, market_data)

                if slice_result.get('error'):
                    logger.warning(f"POV切片 {slice_num} 执行失败: {slice_result['error']}")
                    continue

                total_executed += slice_result['executed_quantity']
                total_cost += slice_result['average_price'] * slice_result['executed_quantity']
                remaining_quantity -= slice_result['executed_quantity']
                execution_reports.append(slice_result)

                logger.debug(
                    f"POV切片 {slice_num} 执行完成: {slice_result['executed_quantity']} @ {slice_result['average_price']}")

                # 动态调整参与率（基于市场条件和执行进度）
                if slice_num > 0 and len(execution_reports) >= 2:
                    # 计算执行效率
                    exec_efficiency = self._calculate_execution_efficiency(execution_reports[-2:])
                    if exec_efficiency < 0.8:  # 执行效率低
                        participation_rate = max(participation_rate * 0.9, min_participation)
                    elif exec_efficiency > 0.95:  # 执行效率高
                        participation_rate = min(participation_rate * 1.1, max_participation)

            # 计算总体结果
            if total_executed > 0:
                average_price = total_cost / total_executed
            else:
                average_price = 0

            # 计算执行质量
            arrival_price = market_data['prices'][symbol]['close'][0]
            price_improvement = average_price - arrival_price if order.side == OrderSide.BUY else arrival_price - average_price

            result = {
                'executed_quantity': total_executed,
                'average_price': average_price,
                'commission': sum(r['commission'] for r in execution_reports),
                'fees': sum(r['fees'] for r in execution_reports),
                'slippage': sum(r['slippage'] for r in execution_reports),
                'market_impact': sum(r['market_impact'] for r in execution_reports) / len(execution_reports),
                'timing_risk': self._calculate_pov_timing_risk(order, execution_reports, participation_rate),
                'execution_quality': price_improvement / arrival_price if arrival_price > 0 else 0,
                'final_participation_rate': participation_rate,
                'execution_reports': execution_reports
            }

            return result

        except Exception as e:
            logger.error(f"POV执行算法失败 {order.order_id}: {e}")
            return {'error': str(e)}

    def _execute_implementation_shortfall(self, order: Order, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行差额算法"""
        try:
            symbol = order.symbol
            quantity = order.quantity

            # 获取算法参数
            risk_aversion = order.execution_params.algo_parameters.get('risk_aversion', 0.5)
            urgency = order.execution_params.algo_parameters.get('urgency', 'medium')
            benchmark = order.execution_params.algo_parameters.get('benchmark', 'arrival_price')

            # 计算 urgency 权重
            urgency_weights = {
                'low': 0.3,
                'medium': 0.5,
                'high': 0.7,
                'urgent': 0.9
            }
            urgency_weight = urgency_weights.get(urgency, 0.5)

            # 获取基准价格
            if benchmark == 'arrival_price':
                benchmark_price = market_data['prices'][symbol]['close'][0]
            elif benchmark == 'vwap':
                # 计算VWAP基准
                volume_data = market_data['volumes'][symbol].get('volume', [])
                if volume_data:
                    total_volume = sum(volume_data)
                    vwap = sum(p * v for p, v in zip(
                        market_data['prices'][symbol]['close'][:len(volume_data)],
                        volume_data
                    )) / total_volume
                    benchmark_price = vwap
                else:
                    benchmark_price = market_data['prices'][symbol]['close'][0]
            else:
                benchmark_price = market_data['prices'][symbol]['close'][0]

            # 计算目标执行价格
            if order.side == OrderSide.BUY:
                target_price = benchmark_price * (1 - risk_aversion * urgency_weight)
            else:
                target_price = benchmark_price * (1 + risk_aversion * urgency_weight)

            # 执行订单
            result = self._execute_simple(order, market_data)

            if result.get('error'):
                return result

            # 计算执行差额
            implementation_shortfall = result['average_price'] - benchmark_price
            if order.side == OrderSide.SELL:
                implementation_shortfall = -implementation_shortfall

            # 计算执行质量
            execution_quality = 1 - (abs(implementation_shortfall) / benchmark_price) if benchmark_price > 0 else 0

            # 更新结果
            result.update({
                'implementation_shortfall': implementation_shortfall,
                'benchmark_price': benchmark_price,
                'target_price': target_price,
                'execution_quality': execution_quality,
                'risk_aversion': risk_aversion,
                'urgency': urgency
            })

            return result

        except Exception as e:
            logger.error(f"执行差额算法失败 {order.order_id}: {e}")
            return {'error': str(e)}

    def _execute_liquidity_seeking(self, order: Order, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """流动性寻找算法"""
        try:
            symbol = order.symbol
            quantity = order.quantity

            # 获取流动性数据
            liquidity_data = market_data.get('liquidity', {}).get(symbol, {})
            bid_ask_spread = liquidity_data.get('bid_ask_spread', 0.01)
            market_depth = liquidity_data.get('market_depth', 1000)
            volume_imbalance = liquidity_data.get('volume_imbalance', 0.5)

            # 计算最优执行策略
            if market_depth > quantity * 2 and bid_ask_spread < 0.02:
                # 流动性充足，使用简单执行
                return self._execute_simple(order, market_data)
            else:
                # 流动性不足，使用更保守的策略
                # 分拆订单，寻找最佳执行时机
                slices = max(3, min(10, int(quantity / market_depth * 2)))
                slice_quantity = quantity / slices

                total_executed = 0
                total_cost = 0
                execution_reports = []

                for slice_num in range(slices):
                    # 等待流动性改善
                    if slice_num > 0:
                        # 检查流动性条件
                        current_spread = liquidity_data.get('current_spread', bid_ask_spread)
                        current_depth = liquidity_data.get('current_depth', market_depth)

                        # 如果流动性差，等待更长时间
                        wait_time = max(1, min(5, current_spread * 100))  # 等待时间与点差成正比
                        time.sleep(wait_time)

                    # 执行当前切片
                    slice_order = copy.deepcopy(order)
                    slice_order.quantity = slice_quantity
                    slice_order.order_id = f"{order.order_id}_liquidity_slice_{slice_num}"

                    slice_result = self._execute_simple(slice_order, market_data)

                    if slice_result.get('error'):
                        logger.warning(f"流动性寻找切片 {slice_num} 执行失败: {slice_result['error']}")
                        continue

                    total_executed += slice_result['executed_quantity']
                    total_cost += slice_result['average_price'] * slice_result['executed_quantity']
                    execution_reports.append(slice_result)

                # 计算总体结果
                if total_executed > 0:
                    average_price = total_cost / total_executed
                else:
                    average_price = 0

                # 计算执行质量
                arrival_price = market_data['prices'][symbol]['close'][0]
                price_improvement = average_price - arrival_price if order.side == OrderSide.BUY else arrival_price - average_price

                result = {
                    'executed_quantity': total_executed,
                    'average_price': average_price,
                    'commission': sum(r['commission'] for r in execution_reports),
                    'fees': sum(r['fees'] for r in execution_reports),
                    'slippage': sum(r['slippage'] for r in execution_reports),
                    'market_impact': sum(r['market_impact'] for r in execution_reports) / len(execution_reports),
                    'timing_risk': self._calculate_liquidity_timing_risk(order, execution_reports, liquidity_data),
                    'execution_quality': price_improvement / arrival_price if arrival_price > 0 else 0,
                    'liquidity_conditions': liquidity_data,
                    'execution_reports': execution_reports
                }

                return result

        except Exception as e:
            logger.error(f"流动性寻找算法失败 {order.order_id}: {e}")
            return {'error': str(e)}

    def _execute_adaptive(self, order: Order, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """自适应执行算法"""
        try:
            symbol = order.symbol
            quantity = order.quantity

            # 获取市场条件
            volatility = self._calculate_current_volatility(symbol, market_data)
            liquidity = self._assess_liquidity_conditions(symbol, market_data)
            trend = self._assess_market_trend(symbol, market_data)

            # 根据市场条件选择执行策略
            if volatility < 0.2 and liquidity > 0.7:
                # 低波动，高流动性 - 使用被动执行
                return self._execute_passive(order, market_data)
            elif volatility > 0.4 or liquidity < 0.3:
                # 高波动或低流动性 - 使用激进执行
                return self._execute_aggressive(order, market_data)
            else:
                # 中等条件 - 使用VWAP或TWAP
                if trend == 'up' and order.side == OrderSide.BUY:
                    # 上涨趋势中买入 - 使用TWAP避免推高价格
                    return self._execute_twap(order, market_data)
                elif trend == 'down' and order.side == OrderSide.SELL:
                    # 下跌趋势中卖出 - 使用TWAP避免压低价格
                    return self._execute_twap(order, market_data)
                else:
                    # 其他情况使用VWAP
                    return self._execute_vwap(order, market_data)

        except Exception as e:
            logger.error(f"自适应执行算法失败 {order.order_id}: {e}")
            return self._execute_simple(order, market_data)  # 回退到简单执行

    def _execute_sniper(self, order: Order, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """狙击执行算法"""
        try:
            symbol = order.symbol
            quantity = order.quantity

            # 获取算法参数
            target_price = order.execution_params.algo_parameters.get('target_price')
            price_tolerance = order.execution_params.algo_parameters.get('price_tolerance', 0.01)
            max_wait_time = order.execution_params.algo_parameters.get('max_wait_time', 300)  # 默认5分钟

            # 如果没有目标价格，使用当前价格
            if not target_price:
                target_price = market_data['prices'][symbol]['close'][-1]

            start_time = time.time()
            executed = False
            execution_result = None

            # 等待价格达到目标范围
            while time.time() - start_time < max_wait_time and not executed:
                current_price = market_data['prices'][symbol]['close'][-1]

                # 检查价格条件
                if order.side == OrderSide.BUY:
                    price_condition = current_price <= target_price * (1 + price_tolerance)
                else:
                    price_condition = current_price >= target_price * (1 - price_tolerance)

                if price_condition:
                    # 执行订单
                    execution_result = self._execute_simple(order, market_data)
                    executed = True
                else:
                    # 等待价格变化
                    time.sleep(1)  # 每秒检查一次

            # 如果超时仍未执行，使用市价单
            if not executed:
                logger.warning(f"狙击执行超时，使用市价单执行: {order.order_id}")
                market_order = copy.deepcopy(order)
                market_order.parameters.order_type = OrderType.MARKET
                execution_result = self._execute_simple(market_order, market_data)

            # 添加狙击执行特定指标
            if execution_result and not execution_result.get('error'):
                execution_result.update({
                    'target_price': target_price,
                    'price_tolerance': price_tolerance,
                    'wait_time': time.time() - start_time,
                    'sniper_success': executed
                })

            return execution_result

        except Exception as e:
            logger.error(f"狙击执行算法失败 {order.order_id}: {e}")
            return self._execute_simple(order, market_data)

    def _execute_stealth(self, order: Order, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """隐身执行算法"""
        try:
            symbol = order.symbol
            quantity = order.quantity

            # 获取算法参数
            max_slice_size = order.execution_params.algo_parameters.get('max_slice_size', 100)
            min_time_interval = order.execution_params.algo_parameters.get('min_time_interval', 30)
            randomize_timing = order.execution_params.algo_parameters.get('randomize_timing', True)

            # 计算切片数量
            slices = max(1, int(quantity / max_slice_size))
            slice_quantities = self._distribute_quantity(quantity, slices, randomize=True)

            total_executed = 0
            total_cost = 0
            execution_reports = []

            for slice_num in range(slices):
                # 随机化执行时间
                if slice_num > 0:
                    wait_time = min_time_interval
                    if randomize_timing:
                        wait_time += random.randint(0, min_time_interval // 2)
                    time.sleep(wait_time)

                # 执行当前切片
                slice_order = copy.deepcopy(order)
                slice_order.quantity = slice_quantities[slice_num]
                slice_order.order_id = f"{order.order_id}_stealth_slice_{slice_num}"

                # 随机选择订单类型（限价单或市价单）
                if random.random() < 0.7:  # 70%概率使用限价单
                    slice_order.parameters.order_type = OrderType.LIMIT
                    # 设置合理的限价
                    current_price = market_data['prices'][symbol]['close'][-1]
                    if order.side == OrderSide.BUY:
                        limit_price = current_price * (1 - random.uniform(0.001, 0.005))
                    else:
                        limit_price = current_price * (1 + random.uniform(0.001, 0.005))
                    slice_order.parameters.limit_price = limit_price
                else:
                    slice_order.parameters.order_type = OrderType.MARKET

                slice_result = self._execute_simple(slice_order, market_data)

                if slice_result.get('error'):
                    logger.warning(f"隐身执行切片 {slice_num} 执行失败: {slice_result['error']}")
                    continue

                total_executed += slice_result['executed_quantity']
                total_cost += slice_result['average_price'] * slice_result['executed_quantity']
                execution_reports.append(slice_result)

            # 计算总体结果
            if total_executed > 0:
                average_price = total_cost / total_executed
            else:
                average_price = 0

            # 计算市场影响（隐身执行应该具有较低的市场影响）
            market_impact = self._calculate_stealth_market_impact(order, execution_reports)

            result = {
                'executed_quantity': total_executed,
                'average_price': average_price,
                'commission': sum(r['commission'] for r in execution_reports),
                'fees': sum(r['fees'] for r in execution_reports),
                'slippage': sum(r['slippage'] for r in execution_reports),
                'market_impact': market_impact,
                'timing_risk': self._calculate_stealth_timing_risk(order, execution_reports),
                'execution_quality': 1 - market_impact,  # 市场影响越低，执行质量越高
                'stealth_score': self._calculate_stealth_score(execution_reports),
                'execution_reports': execution_reports
            }

            return result

        except Exception as e:
            logger.error(f"隐身执行算法失败 {order.order_id}: {e}")
            return {'error': str(e)}

    def _execute_aggressive(self, order: Order, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """激进执行算法"""
        try:
            # 激进执行：使用市价单，尽可能快地完成交易
            aggressive_order = copy.deepcopy(order)
            aggressive_order.parameters.order_type = OrderType.MARKET
            aggressive_order.execution_params.urgency = 'urgent'

            result = self._execute_simple(aggressive_order, market_data)

            if not result.get('error'):
                # 激进执行通常有较高的市场影响和滑点
                # 但执行时间短，时机风险低
                result.update({
                    'aggressive_execution': True,
                    'execution_speed': 'high',
                    'timing_risk': result.get('timing_risk', 0) * 0.5  # 激进执行减少时机风险
                })

            return result

        except Exception as e:
            logger.error(f"激进执行算法失败 {order.order_id}: {e}")
            return {'error': str(e)}

    def _execute_passive(self, order: Order, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """被动执行算法"""
        try:
            # 被动执行：使用限价单，等待更好的价格
            passive_order = copy.deepcopy(order)
            passive_order.parameters.order_type = OrderType.LIMIT

            # 设置合理的限价
            current_price = market_data['prices'][order.symbol]['close'][-1]
            if order.side == OrderSide.BUY:
                limit_price = current_price * (1 - 0.005)  # 低于市价0.5%
            else:
                limit_price = current_price * (1 + 0.005)  # 高于市价0.5%

            passive_order.parameters.limit_price = limit_price
            passive_order.execution_params.urgency = 'low'

            result = self._execute_simple(passive_order, market_data)

            if not result.get('error'):
                # 被动执行通常有较低的市场影响和滑点
                # 但执行时间长，时机风险高
                result.update({
                    'passive_execution': True,
                    'execution_speed': 'low',
                    'price_improvement': current_price - result['average_price'] if order.side == OrderSide.BUY else
                    result['average_price'] - current_price,
                    'timing_risk': result.get('timing_risk', 0) * 1.5  # 被动执行增加时机风险
                })

            return result

        except Exception as e:
            logger.error(f"被动执行算法失败 {order.order_id}: {e}")
            return {'error': str(e)}

    def _execute_with_broker(self, order: Order, price: float, quantity: float) -> Dict[str, Any]:
        """通过经纪商执行订单"""
        try:
            broker_name = order.execution_params.algo_parameters.get('broker', self.default_broker)
            connection = self.broker_connections.get(broker_name)

            if not connection:
                return {'error': f'经纪商 {broker_name} 未找到'}

            if connection.broker_type == BrokerType.SIMULATED:
                # 模拟经纪商执行
                return self._simulate_broker_execution(order, price, quantity)
            else:
                # 实际经纪商执行
                return self._real_broker_execution(connection, order, price, quantity)

        except Exception as e:
            logger.error(f"经纪商执行失败 {order.order_id}: {e}")
            return {'error': str(e)}

    def _simulate_broker_execution(self, order: Order, price: float, quantity: float) -> Dict[str, Any]:
        """模拟经纪商执行"""
        try:
            # 模拟执行延迟
            execution_delay = random.uniform(0.1, 0.5)  # 100-500ms延迟
            time.sleep(execution_delay)

            # 模拟执行成功率
            success_rate = 0.98  # 98%成功率
            if random.random() > success_rate:
                return {'error': '模拟执行失败'}

            # 生成模拟订单ID
            order_id = f"SIM_{int(time.time())}_{random.randint(1000, 9999)}"
            exchange_id = f"EX_{int(time.time())}_{random.randint(10000, 99999)}"

            # 模拟部分成交情况
            fill_probability = 0.95  # 95%概率完全成交
            if random.random() > fill_probability:
                # 部分成交
                filled_quantity = quantity * random.uniform(0.5, 0.9)
                partial_fill = True
            else:
                # 完全成交
                filled_quantity = quantity
                partial_fill = False

            # 模拟价格改进/恶化
            price_improvement = random.uniform(-0.001, 0.001) * price
            execution_price = price + price_improvement

            # 模拟执行报告
            execution_reports = []
            timestamp = datetime.now().isoformat()

            if partial_fill:
                # 生成部分成交报告
                execution_reports.append({
                    'execution_id': f"EXEC_{int(time.time())}_1",
                    'timestamp': timestamp,
                    'fill_price': execution_price,
                    'fill_quantity': filled_quantity,
                    'remaining_quantity': quantity - filled_quantity,
                    'cumulative_quantity': filled_quantity,
                    'execution_type': 'partial',
                    'liquidity': 'added' if random.random() > 0.5 else 'removed',
                    'venue': 'SIMULATED_EXCHANGE',
                    'commission': 0.0,
                    'fees': 0.0,
                    'slippage': price_improvement,
                    'flags': ['simulated', 'partial_fill']
                })

                # 模拟剩余数量的后续执行
                time.sleep(random.uniform(0.1, 0.3))
                remaining_qty = quantity - filled_quantity
                second_price_improvement = random.uniform(-0.0005, 0.0005) * execution_price
                second_execution_price = execution_price + second_price_improvement

                execution_reports.append({
                    'execution_id': f"EXEC_{int(time.time())}_2",
                    'timestamp': datetime.now().isoformat(),
                    'fill_price': second_execution_price,
                    'fill_quantity': remaining_qty,
                    'remaining_quantity': 0,
                    'cumulative_quantity': quantity,
                    'execution_type': 'full',
                    'liquidity': 'added',
                    'venue': 'SIMULATED_EXCHANGE',
                    'commission': 0.0,
                    'fees': 0.0,
                    'slippage': second_price_improvement,
                    'flags': ['simulated', 'final_fill']
                })

                # 计算平均价格
                avg_price = ((filled_quantity * execution_price) +
                             (remaining_qty * second_execution_price)) / quantity
            else:
                # 完全成交报告
                execution_reports.append({
                    'execution_id': f"EXEC_{int(time.time())}_1",
                    'timestamp': timestamp,
                    'fill_price': execution_price,
                    'fill_quantity': quantity,
                    'remaining_quantity': 0,
                    'cumulative_quantity': quantity,
                    'execution_type': 'full',
                    'liquidity': 'added',
                    'venue': 'SIMULATED_EXCHANGE',
                    'commission': 0.0,
                    'fees': 0.0,
                    'slippage': price_improvement,
                    'flags': ['simulated', 'complete_fill']
                })
                avg_price = execution_price

            # 计算总滑点
            total_slippage = avg_price - price
            if order.side == OrderSide.SELL:
                total_slippage = -total_slippage  # 对于卖出，正滑点是有利的

            # 计算市场影响（简化模型）
            market_impact = abs(total_slippage) * 2  # 假设市场影响是滑点的两倍

            result = {
                'order_id': order_id,
                'exchange_id': exchange_id,
                'executed_quantity': quantity,
                'average_price': avg_price,
                'total_slippage': total_slippage,
                'market_impact': market_impact,
                'commission': self._calculate_commission(order, avg_price, quantity),
                'fees': self._calculate_fees(order, avg_price, quantity),
                'execution_reports': execution_reports,
                'execution_quality': self._calculate_execution_quality_simulation(order, price, avg_price,
                                                                                  total_slippage),
                'simulation_metadata': {
                    'execution_delay': execution_delay,
                    'success_rate': success_rate,
                    'fill_probability': fill_probability,
                    'price_improvement_range': [-0.001, 0.001]
                }
            }

            logger.debug(f"模拟经纪商执行完成: {order_id}, 价格={avg_price:.4f}, 滑点={total_slippage:.4f}")
            return result

        except Exception as e:
            logger.error(f"模拟经纪商执行失败: {e}")
            return {'error': str(e)}

    def _real_broker_execution(self, connection: BrokerConnection,
                               order: Order, price: float, quantity: float) -> Dict[str, Any]:
        """实际经纪商执行"""
        try:
            # 根据经纪商类型选择不同的执行方式
            if connection.broker_type == BrokerType.IBKR:
                return self._execute_via_ibkr(connection, order, price, quantity)
            elif connection.broker_type == BrokerType.ALPACA:
                return self._execute_via_alpaca(connection, order, price, quantity)
            elif connection.broker_type == BrokerType.BINANCE:
                return self._execute_via_binance(connection, order, price, quantity)
            else:
                logger.warning(f"不支持的经纪商类型: {connection.broker_type}")
                return {'error': f'不支持的经纪商类型: {connection.broker_type}'}

        except Exception as e:
            logger.error(f"实际经纪商执行失败: {e}")
            return {'error': str(e)}

    def _execute_via_ibkr(self, connection: BrokerConnection,
                          order: Order, price: float, quantity: float) -> Dict[str, Any]:
        """通过Interactive Brokers执行"""
        try:
            # 这里需要实际的IBKR API调用
            # 简化实现：返回模拟结果
            logger.info(f"IBKR执行订单: {order.symbol}, {order.side.value}, {quantity} @ {price}")

            # 模拟API调用延迟
            time.sleep(0.2)

            return {
                'order_id': f"IBKR_{int(time.time())}",
                'exchange_id': f"ISE_{int(time.time())}",
                'executed_quantity': quantity,
                'average_price': price * (1 + random.uniform(-0.0005, 0.0005)),
                'commission': max(1.0, quantity * price * 0.0005),  # 最低1美元，0.05%
                'fees': quantity * price * 0.00002,  # 0.002%
                'slippage': random.uniform(-0.0005, 0.001) * price,
                'execution_reports': [],
                'broker_specific': {
                    'ibkr_conid': random.randint(1000000, 9999999),
                    'execution_venue': 'ISLAND' if random.random() > 0.5 else 'ARCA'
                }
            }

        except Exception as e:
            logger.error(f"IBKR执行失败: {e}")
            return {'error': str(e)}

    def _execute_via_alpaca(self, connection: BrokerConnection,
                            order: Order, price: float, quantity: float) -> Dict[str, Any]:
        """通过Alpaca执行"""
        try:
            # 这里需要实际的Alpaca API调用
            # 简化实现：返回模拟结果
            logger.info(f"Alpaca执行订单: {order.symbol}, {order.side.value}, {quantity} @ {price}")

            # 模拟API调用延迟
            time.sleep(0.15)

            return {
                'order_id': f"ALPACA_{int(time.time())}",
                'exchange_id': f"ALP_{int(time.time())}",
                'executed_quantity': quantity,
                'average_price': price * (1 + random.uniform(-0.0003, 0.0003)),
                'commission': 0.0,  # Alpaca免佣金
                'fees': quantity * price * 0.00001,  # 极低费用
                'slippage': random.uniform(-0.0003, 0.0005) * price,
                'execution_reports': [],
                'broker_specific': {
                    'alpaca_order_id': str(uuid.uuid4()),
                    'routing_venue': 'IEX' if random.random() > 0.5 else 'NYSE'
                }
            }

        except Exception as e:
            logger.error(f"Alpaca执行失败: {e}")
            return {'error': str(e)}

    def _execute_via_binance(self, connection: BrokerConnection,
                             order: Order, price: float, quantity: float) -> Dict[str, Any]:
        """通过Binance执行"""
        try:
            # 这里需要实际的Binance API调用
            # 简化实现：返回模拟结果
            logger.info(f"Binance执行订单: {order.symbol}, {order.side.value}, {quantity} @ {price}")

            # 模拟API调用延迟
            time.sleep(0.1)

            # Binance特有的费用结构
            maker_fee = 0.001  # 0.1%
            taker_fee = 0.002  # 0.2%
            is_maker = random.random() > 0.7  # 30%概率是挂单

            return {
                'order_id': f"BINANCE_{int(time.time())}",
                'exchange_id': f"BNC_{int(time.time())}",
                'executed_quantity': quantity,
                'average_price': price * (1 + random.uniform(-0.0002, 0.0002)),
                'commission': quantity * price * (maker_fee if is_maker else taker_fee),
                'fees': 0.0,  # 费用已包含在commission中
                'slippage': random.uniform(-0.0002, 0.0003) * price,
                'execution_reports': [],
                'broker_specific': {
                    'binance_order_id': random.randint(100000000, 999999999),
                    'is_maker': is_maker,
                    'fee_tier': 1
                }
            }

        except Exception as e:
            logger.error(f"Binance执行失败: {e}")
            return {'error': str(e)}

    def _calculate_commission(self, order: Order, price: float, quantity: float) -> float:
        """计算佣金"""
        try:
            broker_name = order.execution_params.algo_parameters.get('broker', self.default_broker)
            connection = self.broker_connections.get(broker_name)

            if not connection:
                return 0.0

            notional = price * quantity

            if connection.broker_type == BrokerType.IBKR:
                # IBKR佣金结构：每股0.0035美元，最低1美元
                commission_per_share = 0.0035
                commission = max(1.0, quantity * commission_per_share)
                return min(commission, notional * 0.01)  # 不超过名义价值的1%

            elif connection.broker_type == BrokerType.ALPACA:
                # Alpaca免佣金，只有极低费用
                return 0.0

            elif connection.broker_type == BrokerType.BINANCE:
                # Binance费用
                maker_fee = 0.001  # 0.1%
                taker_fee = 0.002  # 0.2%
                fee_rate = maker_fee if random.random() > 0.7 else taker_fee
                return notional * fee_rate

            else:
                # 默认佣金：0.1%
                return notional * 0.001

        except Exception as e:
            logger.error(f"佣金计算失败: {e}")
            return notional * 0.001  # 默认佣金

    def _calculate_fees(self, order: Order, price: float, quantity: float) -> float:
        """计算额外费用"""
        try:
            notional = price * quantity

            # 监管费用（SEC费用）
            sec_fee = notional * 0.0000229  # 0.00229%

            # 交易活动费（Trading Activity Fee）
            taf_fee = 0.000119 * quantity  # 每股0.000119美元
            taf_fee = min(taf_fee, 5.95)  # 最高5.95美元

            # 清算费用
            clearing_fee = notional * 0.00002  # 0.002%

            # 交易所费用（根据交易所不同）
            exchange_fee = self._calculate_exchange_fee(order.symbol, notional)

            # 路由费用（如果指定了特定路由）
            routing_fee = self._calculate_routing_fee(order)

            # 总费用
            total_fees = sec_fee + taf_fee + clearing_fee + exchange_fee + routing_fee

            # 确保费用合理（不超过名义价值的0.1%）
            max_fees = notional * 0.001
            return min(total_fees, max_fees)

        except Exception as e:
            logger.error(f"费用计算失败: {e}")
            return notional * 0.0001  # 默认费用

    def _calculate_exchange_fee(self, symbol: str, notional: float) -> float:
        """计算交易所费用"""
        try:
            # 根据交易所类型计算不同费用
            # 简化实现：假设所有股票在主要交易所交易
            if symbol.endswith('.NYSE'):
                return notional * 0.000015  # NYSE: 0.0015%
            elif symbol.endswith('.NASDAQ'):
                return notional * 0.000013  # NASDAQ: 0.0013%
            else:
                return notional * 0.000012  # 默认: 0.0012%
        except:
            return notional * 0.000012

    def _calculate_routing_fee(self, order: Order) -> float:
        """计算路由费用"""
        try:
            # 如果指定了特定路由，可能有额外费用
            routing = order.parameters.routing_instructions.get('venue')
            if routing and routing != 'auto':
                return order.quantity * 0.0005  # 每股0.0005美元
            return 0.0
        except:
            return 0.0

    def _calculate_slippage(self, order: Order, market_data: Dict[str, Any], execution_price: float) -> float:
        """计算滑点"""
        try:
            symbol = order.symbol
            side = order.side
            quantity = order.quantity

            # 获取基准价格（通常使用下单时的市场价格）
            benchmark_price = market_data['prices'][symbol]['close'][0]

            # 计算原始滑点
            raw_slippage = execution_price - benchmark_price
            if side == OrderSide.SELL:
                raw_slippage = -raw_slippage  # 对于卖出，价格下跌是正滑点

            # 根据市场条件调整滑点
            volatility = self._calculate_current_volatility(symbol, market_data)
            liquidity = self._assess_liquidity_conditions(symbol, market_data)

            # 滑点调整因子（高波动性和低流动性增加滑点）
            volatility_factor = 1 + (volatility / 0.2)  # 基准波动率20%
            liquidity_factor = 1 + (1 - liquidity) * 2  # 流动性越低，因子越大

            # 订单大小影响（大订单产生更大滑点）
            size_factor = min(1 + (quantity / 10000), 3)  # 每1万股增加1倍，最大3倍

            # 算法影响（某些算法可以减少滑点）
            algorithm_factor = self._get_algorithm_slippage_factor(order.execution_params.algorithm)

            adjusted_slippage = raw_slippage * volatility_factor * liquidity_factor * size_factor * algorithm_factor

            return float(adjusted_slippage)

        except Exception as e:
            logger.error(f"滑点计算失败: {e}")
            return 0.0

    def _calculate_current_volatility(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """计算当前波动率"""
        try:
            prices = market_data['prices'][symbol]['close']
            if len(prices) < 20:
                return 0.2  # 默认波动率

            returns = np.diff(np.log(prices[-20:]))  # 最近20个周期的收益
            return float(np.std(returns) * np.sqrt(252))  # 年化波动率
        except:
            return 0.2

    def _assess_liquidity_conditions(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """评估流动性条件"""
        try:
            # 获取成交量数据
            volume_data = market_data['volumes'][symbol]
            current_volume = volume_data.get('volume', 0)
            avg_volume = volume_data.get('avg_daily_volume', current_volume)

            if avg_volume <= 0:
                return 0.5  # 中等流动性

            # 计算流动性比率
            volume_ratio = current_volume / avg_volume

            # 计算买卖价差（简化）
            spread = self._estimate_bid_ask_spread(symbol, market_data)

            # 综合流动性评分（0-1，1表示最好）
            volume_score = min(volume_ratio, 2.0) / 2.0  # 标准化到0-1
            spread_score = 1 - min(spread / 0.1, 1.0)  # 点差越小越好

            liquidity_score = (volume_score + spread_score) / 2
            return float(max(0.1, min(liquidity_score, 1.0)))

        except:
            return 0.5

    def _estimate_bid_ask_spread(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """估计买卖价差"""
        try:
            # 简化实现：基于波动率和流动性估计点差
            volatility = self._calculate_current_volatility(symbol, market_data)
            liquidity = self._assess_liquidity_conditions(symbol, market_data)

            # 基础点差 + 波动率调整 + 流动性调整
            base_spread = 0.01  # 1%
            volatility_adjustment = volatility * 0.5  # 波动率贡献
            liquidity_adjustment = (1 - liquidity) * 0.03  # 流动性贡献

            return base_spread + volatility_adjustment + liquidity_adjustment
        except:
            return 0.02  # 默认2%点差

    def _get_algorithm_slippage_factor(self, algorithm: ExecutionAlgorithm) -> float:
        """获取算法滑点因子"""
        algorithm_factors = {
            ExecutionAlgorithm.SIMPLE: 1.0,
            ExecutionAlgorithm.TWAP: 0.8,
            ExecutionAlgorithm.VWAP: 0.7,
            ExecutionAlgorithm.POV: 0.6,
            ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL: 0.9,
            ExecutionAlgorithm.LIQUIDITY_SEEKING: 0.5,
            ExecutionAlgorithm.ADAPTIVE: 0.7,
            ExecutionAlgorithm.SNIPER: 1.2,  # 狙击执行可能增加滑点
            ExecutionAlgorithm.STEALTH: 0.4,  # 隐身执行减少滑点
            ExecutionAlgorithm.AGGRESSIVE: 1.5,  # 激进执行增加滑点
            ExecutionAlgorithm.PASSIVE: 0.6
        }
        return algorithm_factors.get(algorithm, 1.0)

    def _calculate_market_impact(self, order: Order, market_data: Dict[str, Any], execution_price: float) -> float:
        """计算市场影响"""
        try:
            symbol = order.symbol
            quantity = order.quantity
            side = order.side

            # 获取市场数据
            current_volume = market_data['volumes'][symbol].get('volume', 0)
            avg_volume = market_data['volumes'][symbol].get('avg_daily_volume', current_volume)

            if avg_volume <= 0:
                return 0.01  # 默认1%市场影响

            # 计算订单相对大小
            order_size_ratio = quantity / avg_volume

            # 使用Kissell & Malamut模型（简化版）
            # 市场影响 = a * (订单大小/平均成交量)^b * 波动率^c
            a = 0.1  # 比例常数
            b = 0.5  # 规模指数
            c = 0.5  # 波动率指数

            volatility = self._calculate_current_volatility(symbol, market_data)

            market_impact = a * (order_size_ratio ** b) * (volatility ** c)

            # 根据算法类型调整
            algorithm_factor = self._get_algorithm_impact_factor(order.execution_params.algorithm)
            market_impact *= algorithm_factor

            # 确保市场影响在合理范围内
            return float(min(max(market_impact, 0.001), 0.1))  # 0.1%到10%

        except Exception as e:
            logger.error(f"市场影响计算失败: {e}")
            return 0.02  # 默认2%市场影响

    def _get_algorithm_impact_factor(self, algorithm: ExecutionAlgorithm) -> float:
        """获取算法市场影响因子"""
        algorithm_factors = {
            ExecutionAlgorithm.SIMPLE: 1.0,
            ExecutionAlgorithm.TWAP: 0.7,
            ExecutionAlgorithm.VWAP: 0.6,
            ExecutionAlgorithm.POV: 0.5,
            ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL: 0.8,
            ExecutionAlgorithm.LIQUIDITY_SEEKING: 0.4,
            ExecutionAlgorithm.ADAPTIVE: 0.6,
            ExecutionAlgorithm.SNIPER: 1.3,  # 狙击执行可能增加市场影响
            ExecutionAlgorithm.STEALTH: 0.3,  # 隐身执行减少市场影响
            ExecutionAlgorithm.AGGRESSIVE: 1.8,  # 激进执行增加市场影响
            ExecutionAlgorithm.PASSIVE: 0.5
        }
        return algorithm_factors.get(algorithm, 1.0)

    def _calculate_execution_quality(self, order: Order, benchmark_price: float,
                                     execution_price: float, market_impact: float) -> float:
        """计算执行质量"""
        try:
            side = order.side

            # 计算价格改进
            if side == OrderSide.BUY:
                price_improvement = benchmark_price - execution_price  # 越低越好
            else:
                price_improvement = execution_price - benchmark_price  # 越高越好

            # 标准化价格改进
            if benchmark_price > 0:
                normalized_improvement = price_improvement / benchmark_price
            else:
                normalized_improvement = 0

            # 计算执行质量分数（0-1，1表示最好）
            # 考虑价格改进、市场影响、滑点等因素
            price_score = max(0, min(1 + normalized_improvement * 10, 2)) / 2  # 标准化到0-1

            impact_score = 1 - min(market_impact * 10, 1)  # 市场影响越小越好

            # 算法效率评分（基于算法类型）
            algorithm_score = self._get_algorithm_efficiency_score(order.execution_params.algorithm)

            # 综合执行质量
            execution_quality = (price_score * 0.5) + (impact_score * 0.3) + (algorithm_score * 0.2)

            return float(max(0, min(execution_quality, 1)))

        except Exception as e:
            logger.error(f"执行质量计算失败: {e}")
            return 0.5  # 默认中等质量

    def _get_algorithm_efficiency_score(self, algorithm: ExecutionAlgorithm) -> float:
        """获取算法效率评分"""
        algorithm_scores = {
            ExecutionAlgorithm.SIMPLE: 0.6,
            ExecutionAlgorithm.TWAP: 0.7,
            ExecutionAlgorithm.VWAP: 0.8,
            ExecutionAlgorithm.POV: 0.85,
            ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL: 0.75,
            ExecutionAlgorithm.LIQUIDITY_SEEKING: 0.9,
            ExecutionAlgorithm.ADAPTIVE: 0.8,
            ExecutionAlgorithm.SNIPER: 0.7,
            ExecutionAlgorithm.STEALTH: 0.95,
            ExecutionAlgorithm.AGGRESSIVE: 0.5,
            ExecutionAlgorithm.PASSIVE: 0.7
        }
        return algorithm_scores.get(algorithm, 0.6)

    def _calculate_execution_quality_simulation(self, order: Order, benchmark_price: float,
                                                execution_price: float, slippage: float) -> float:
        """计算模拟执行的执行质量"""
        try:
            # 简化版的执行质量计算
            if benchmark_price <= 0:
                return 0.5

            price_improvement = execution_price - benchmark_price
            if order.side == OrderSide.BUY:
                price_improvement = -price_improvement  # 对于买入，价格越低越好

            normalized_improvement = price_improvement / benchmark_price

            # 基本质量计算
            base_quality = 0.5 + normalized_improvement * 10

            # 考虑滑点影响
            slippage_impact = abs(slippage) / benchmark_price
            quality_adjustment = 1 - min(slippage_impact * 5, 0.5)

            execution_quality = base_quality * quality_adjustment

            return float(max(0.1, min(execution_quality, 0.99)))

        except:
            return 0.5

    def _calculate_twap_timing_risk(self, order: Order, execution_reports: List[Dict[str, Any]]) -> float:
        """计算TWAP执行的时机风险"""
        try:
            if len(execution_reports) <= 1:
                return 0.5  # 默认中等风险

            # 分析执行价格的变化
            prices = [report['average_price'] for report in execution_reports]
            price_variance = np.var(prices) if len(prices) > 1 else 0

            # 计算执行时间分布
            time_intervals = []
            for i in range(1, len(execution_reports)):
                time_diff = (datetime.fromisoformat(execution_reports[i]['timestamp']) -
                             datetime.fromisoformat(execution_reports[i - 1]['timestamp'])).total_seconds()
                time_intervals.append(time_diff)

            if time_intervals:
                time_variance = np.var(time_intervals)
            else:
                time_variance = 0

            # 综合时机风险
            timing_risk = min(price_variance * 100 + time_variance / 10, 1.0)
            return float(timing_risk)

        except:
            return 0.5

    def _calculate_vwap_timing_risk(self, order: Order, execution_reports: List[Dict[str, Any]],
                                    vwap_benchmark: float) -> float:
        """计算VWAP执行的时机风险"""
        try:
            if not execution_reports or vwap_benchmark <= 0:
                return 0.5

            # 计算与VWAP基准的偏差
            execution_prices = [r['average_price'] for r in execution_reports]
            avg_execution_price = sum(execution_prices) / len(execution_prices)

            price_deviation = abs(avg_execution_price - vwap_benchmark) / vwap_benchmark

            # 计算成交量跟随误差
            target_volumes = [r['executed_quantity'] for r in execution_reports]
            total_volume = sum(target_volumes)

            if total_volume > 0:
                volume_distribution = [v / total_volume for v in target_volumes]
                # 理想情况下应该均匀分布，计算实际分布与均匀分布的差异
                uniform_distribution = [1 / len(target_volumes)] * len(target_volumes)
                volume_error = sum(abs(v - u) for v, u in zip(volume_distribution, uniform_distribution)) / 2
            else:
                volume_error = 0.5

            timing_risk = min(price_deviation * 2 + volume_error, 1.0)
            return float(timing_risk)

        except:
            return 0.5

    def _calculate_pov_timing_risk(self, order: Order, execution_reports: List[Dict[str, Any]],
                                   participation_rate: float) -> float:
        """计算POV执行的时机风险"""
        try:
            if not execution_reports:
                return 0.5

            # 分析参与率的一致性
            participation_rates = []
            for report in execution_reports:
                if 'participation_rate' in report:
                    participation_rates.append(report['participation_rate'])

            if participation_rates:
                rate_variance = np.var(participation_rates)
                avg_rate = np.mean(participation_rates)
                rate_consistency = abs(avg_rate - participation_rate) / participation_rate
            else:
                rate_variance = 0.1
                rate_consistency = 0.2

            timing_risk = min(rate_variance * 10 + rate_consistency, 1.0)
            return float(timing_risk)

        except:
            return 0.5

    def _calculate_liquidity_timing_risk(self, order: Order, execution_reports: List[Dict[str, Any]],
                                         liquidity_data: Dict[str, Any]) -> float:
        """计算流动性寻找执行的时机风险"""
        try:
            if not execution_reports:
                return 0.5

            # 分析在低流动性条件下的执行表现
            poor_liquidity_count = 0
            total_slices = len(execution_reports)

            for report in execution_reports:
                if 'liquidity_score' in report and report['liquidity_score'] < 0.3:
                    poor_liquidity_count += 1

            liquidity_risk = poor_liquidity_count / total_slices if total_slices > 0 else 0.5

            return float(liquidity_risk)

        except:
            return 0.5

    def _calculate_stealth_timing_risk(self, order: Order, execution_reports: List[Dict[str, Any]]) -> float:
        """计算隐身执行的时机风险"""
        try:
            if len(execution_reports) <= 1:
                return 0.3  # 隐身执行通常有较低的时机风险

            # 分析执行时间的随机性和隐蔽性
            time_stamps = [datetime.fromisoformat(r['timestamp']) for r in execution_reports]
            time_diffs = [(time_stamps[i] - time_stamps[i - 1]).total_seconds() for i in range(1, len(time_stamps))]

            if time_diffs:
                # 计算时间间隔的规律性（越随机越好）
                time_std = np.std(time_diffs)
                avg_interval = np.mean(time_diffs)

                # 低规律性（高标准差）表示好的隐身性
                stealth_score = min(time_std / avg_interval, 2.0) / 2.0 if avg_interval > 0 else 0.5
                timing_risk = 1 - stealth_score  # 隐身性越好，时机风险越低
            else:
                timing_risk = 0.4

            return float(timing_risk)

        except:
            return 0.4

    def _calculate_stealth_market_impact(self, order: Order, execution_reports: List[Dict[str, Any]]) -> float:
        """计算隐身执行的市场影响"""
        try:
            if not execution_reports:
                return 0.02  # 默认市场影响

            # 计算平均每片的市场影响
            avg_impact = sum(r.get('market_impact', 0.02) for r in execution_reports) / len(execution_reports)

            # 隐身执行应该显著降低市场影响
            stealth_impact = avg_impact * 0.6  # 减少40%的市场影响

            return float(max(0.001, min(stealth_impact, 0.05)))

        except:
            return 0.015

    def _calculate_stealth_score(self, execution_reports: List[Dict[str, Any]]) -> float:
        """计算隐身执行得分"""
        try:
            if not execution_reports:
                return 0.5

            scores = []

            for report in execution_reports:
                # 检查执行特征
                score = 0.7  # 基础分

                # 小单执行加分
                if report['executed_quantity'] < 1000:
                    score += 0.1

                # 限价单加分
                if report.get('order_type') == 'limit':
                    score += 0.1

                # 低市场影响加分
                if report.get('market_impact', 0.02) < 0.01:
                    score += 0.1

                scores.append(min(score, 1.0))

            return float(sum(scores) / len(scores))

        except:
            return 0.6

    def _save_order_to_history(self, order: Order):
        """保存订单到历史记录"""
        try:
            # 添加到订单历史
            self.order_history.append(copy.deepcopy(order))

            # 更新订单状态映射
            if order.status == OrderStatus.FILLED:
                self.completed_orders[order.order_id] = order
                if order.order_id in self.active_orders:
                    del self.active_orders[order.order_id]
            elif order.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                self.active_orders[order.order_id] = order
                if order.order_id in self.pending_orders:
                    del self.pending_orders[order.order_id]
            elif order.status == OrderStatus.REJECTED:
                if order.order_id in self.pending_orders:
                    del self.pending_orders[order.order_id]

            # 限制历史记录大小
            max_history = self.execution_config.get('max_order_history', 10000)
            if len(self.order_history) > max_history:
                self.order_history = self.order_history[-max_history:]

            logger.debug(f"订单保存到历史: {order.order_id}, 状态={order.status.value}")

        except Exception as e:
            logger.error(f"订单保存失败 {order.order_id}: {e}")

    def _generate_execution_report(self, order: Order, execution_result: Dict[str, Any],
                                   execution_time: float) -> ExecutionReport:
        """生成执行报告"""
        try:
            report_id = f"EXEC_{order.order_id}_{int(time.time())}"

            report = ExecutionReport(
                report_id=report_id,
                order_id=order.order_id,
                timestamp=datetime.now().isoformat(),
                fill_price=execution_result.get('average_price', 0),
                fill_quantity=execution_result.get('executed_quantity', 0),
                remaining_quantity=order.quantity - execution_result.get('executed_quantity', 0),
                cumulative_quantity=execution_result.get('executed_quantity', 0),
                execution_type='full' if execution_result.get('executed_quantity',
                                                              0) >= order.quantity else 'partial',
                liquidity='added' if order.side == OrderSide.BUY else 'removed',
                venue=execution_result.get('venue', 'unknown'),
                execution_id=execution_result.get('broker_order_id', report_id),
                transaction_time=datetime.now().isoformat(),
                commission=execution_result.get('commission', 0),
                fees=execution_result.get('fees', 0),
                slippage=execution_result.get('slippage', 0),
                market_impact=execution_result.get('market_impact', 0),
                timing_risk=execution_result.get('timing_risk', 0),
                execution_quality=execution_result.get('execution_quality', 0),
                benchmark_comparison={
                    'arrival_price': order.metadata.get('arrival_price', 0),
                    'vwap_benchmark': execution_result.get('vwap_benchmark', 0),
                    'implementation_shortfall': execution_result.get('implementation_shortfall', 0)
                },
                flags=['simulated'] if execution_result.get('simulation_metadata') else [],
                metadata={
                    'execution_time': execution_time,
                    'algorithm': order.execution_params.algorithm.value,
                    'broker': order.execution_params.algo_parameters.get('broker', 'unknown'),
                    'risk_check_passed': order.risk_check_passed
                }
            )

            return report

        except Exception as e:
            logger.error(f"执行报告生成失败 {order.order_id}: {e}")
            # 返回最小化的报告
            return ExecutionReport(
                report_id=f"ERROR_{int(time.time())}",
                order_id=order.order_id,
                timestamp=datetime.now().isoformat(),
                fill_price=0,
                fill_quantity=0,
                remaining_quantity=order.quantity,
                cumulative_quantity=0,
                execution_type='error',
                liquidity='unknown',
                venue='unknown',
                execution_id='error',
                transaction_time=datetime.now().isoformat(),
                commission=0,
                fees=0,
                slippage=0,
                market_impact=0,
                timing_risk=1.0,
                execution_quality=0,
                benchmark_comparison={},
                flags=['error'],
                metadata={'error': str(e)}
            )

    def _save_execution_report(self, report: ExecutionReport):
        """保存执行报告"""
        try:
            if report.order_id not in self.execution_reports:
                self.execution_reports[report.order_id] = []

            self.execution_reports[report.order_id].append(report)

            # 限制每个订单的报告数量
            max_reports_per_order = self.execution_config.get('max_reports_per_order', 100)
            if len(self.execution_reports[report.order_id]) > max_reports_per_order:
                self.execution_reports[report.order_id] = self.execution_reports[report.order_id][
                                                          -max_reports_per_order:]

            logger.debug(f"执行报告保存: {report.report_id}, 订单={report.order_id}")

        except Exception as e:
            logger.error(f"执行报告保存失败: {e}")

    def _aggregate_execution_results(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """汇总执行结果"""
        try:
            overall_result = {
                'timestamp': datetime.now().isoformat(),
                'total_orders': 0,
                'successful_orders': 0,
                'failed_orders': 0,
                'partially_filled_orders': 0,
                'total_quantity': 0,
                'executed_quantity': 0,
                'total_notional': 0,
                'total_commission': 0,
                'total_fees': 0,
                'total_slippage': 0,
                'avg_execution_quality': 0,
                'avg_execution_time': 0,
                'group_results': execution_results,
                'performance_summary': {}
            }

            # 汇总所有组的结果
            quality_scores = []
            execution_times = []

            for group_id, group_result in execution_results.items():
                if 'error' in group_result:
                    overall_result['failed_orders'] += group_result.get('total_orders', 0)
                    continue

                overall_result['total_orders'] += group_result.get('total_orders', 0)
                overall_result['successful_orders'] += group_result.get('successful_orders', 0)
                overall_result['failed_orders'] += group_result.get('failed_orders', 0)
                overall_result['partially_filled_orders'] += group_result.get('partial_orders', 0)
                overall_result['total_quantity'] += group_result.get('total_quantity', 0)
                overall_result['executed_quantity'] += group_result.get('executed_quantity', 0)
                overall_result['total_commission'] += group_result.get('total_commission', 0)
                overall_result['total_fees'] += group_result.get('total_fees', 0)
                overall_result['total_slippage'] += group_result.get('total_slippage', 0)

                if group_result.get('avg_execution_quality', 0) > 0:
                    quality_scores.append(group_result['avg_execution_quality'])

                if group_result.get('execution_time', 0) > 0:
                    execution_times.append(group_result['execution_time'])

            # 计算平均值
            if quality_scores:
                overall_result['avg_execution_quality'] = sum(quality_scores) / len(quality_scores)

            if execution_times:
                overall_result['avg_execution_time'] = sum(execution_times) / len(execution_times)

            # 计算总名义价值
            overall_result['total_notional'] = overall_result['executed_quantity'] * \
                                               self._calculate_average_execution_price(execution_results)

            # 计算成功率
            if overall_result['total_orders'] > 0:
                overall_result['success_rate'] = overall_result['successful_orders'] / overall_result[
                    'total_orders']
            else:
                overall_result['success_rate'] = 0

            # 生成性能摘要
            overall_result['performance_summary'] = self._generate_performance_summary(overall_result)

            return overall_result

        except Exception as e:
            logger.error(f"执行结果汇总失败: {e}")
            return {'error': str(e)}

    def _calculate_average_execution_price(self, execution_results: Dict[str, Any]) -> float:
        """计算平均执行价格"""
        try:
            total_cost = 0
            total_quantity = 0

            for group_result in execution_results.values():
                if 'error' in group_result:
                    continue

                for order_result in group_result.get('order_results', []):
                    if order_result.get('status') in ['filled', 'partially_filled']:
                        total_cost += order_result.get('average_price', 0) * order_result.get('executed_quantity',
                                                                                              0)
                        total_quantity += order_result.get('executed_quantity', 0)

            if total_quantity > 0:
                return total_cost / total_quantity
            else:
                return 0

        except:
            return 0

    def _generate_performance_summary(self, overall_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成性能摘要"""
        try:
            summary = {
                'execution_efficiency': overall_result.get('success_rate', 0),
                'cost_efficiency': 0,
                'timing_efficiency': 0,
                'overall_performance': 0
            }

            # 成本效率（基于佣金和滑点）
            total_notional = overall_result.get('total_notional', 1)
            total_costs = overall_result.get('total_commission', 0) + overall_result.get('total_fees', 0) + \
                          abs(overall_result.get('total_slippage', 0))

            cost_ratio = total_costs / total_notional if total_notional > 0 else 0
            summary['cost_efficiency'] = max(0, 1 - cost_ratio * 10)  # 标准化到0-1

            # 时间效率（基于执行时间）
            avg_execution_time = overall_result.get('avg_execution_time', 0)
            if avg_execution_time > 0:
                # 假设理想执行时间为1秒
                timing_efficiency = 1 / (1 + avg_execution_time)
                summary['timing_efficiency'] = min(timing_efficiency, 1.0)

            # 整体性能（加权平均）
            weights = {
                'execution_efficiency': 0.4,
                'cost_efficiency': 0.3,
                'timing_efficiency': 0.2,
                'quality_score': 0.1
            }

            summary['overall_performance'] = (
                    weights['execution_efficiency'] * summary['execution_efficiency'] +
                    weights['cost_efficiency'] * summary['cost_efficiency'] +
                    weights['timing_efficiency'] * summary['timing_efficiency'] +
                    weights['quality_score'] * overall_result.get('avg_execution_quality', 0)
            )

            # 性能评级
            performance_score = summary['overall_performance']
            if performance_score >= 0.9:
                rating = 'Excellent'
            elif performance_score >= 0.8:
                rating = 'Good'
            elif performance_score >= 0.7:
                rating = 'Average'
            elif performance_score >= 0.6:
                rating = 'Below Average'
            else:
                rating = 'Poor'

            summary['performance_rating'] = rating
            summary['improvement_recommendations'] = self._generate_performance_recommendations(summary)

            return summary

        except Exception as e:
            logger.error(f"性能摘要生成失败: {e}")
            return {}

    def _generate_performance_recommendations(self, performance_summary: Dict[str, Any]) -> List[str]:
        """生成性能改进建议"""
        recommendations = []

        try:
            if performance_summary.get('execution_efficiency', 0) < 0.8:
                recommendations.append("提高订单执行成功率：优化风险控制和订单验证")

            if performance_summary.get('cost_efficiency', 0) < 0.7:
                recommendations.append("降低交易成本：考虑使用成本更低的经纪商或优化执行算法")

            if performance_summary.get('timing_efficiency', 0) < 0.6:
                recommendations.append("改善执行时机：使用更智能的执行算法和更好的市场时机选择")

            if performance_summary.get('overall_performance', 0) < 0.7:
                recommendations.append("综合性能改进：考虑调整执行策略和算法参数配置")

            return recommendations

        except:
            return ["性能分析暂时不可用"]

    def _update_performance_stats(self, overall_result: Dict[str, Any], processing_time: float):
        """更新性能统计"""
        try:
            self.performance_stats['orders_executed'] += overall_result.get('total_orders', 0)
            self.performance_stats['total_volume'] += overall_result.get('executed_quantity', 0)
            self.performance_stats['total_commission'] += overall_result.get('total_commission', 0)
            self.performance_stats['total_slippage'] += overall_result.get('total_slippage', 0)

            # 更新平均执行时间
            n = self.performance_stats.get('execution_count', 0) + 1
            old_avg = self.performance_stats.get('avg_execution_time', 0)
            new_avg = (old_avg * (n - 1) + processing_time) / n

            self.performance_stats['execution_count'] = n
            self.performance_stats['avg_execution_time'] = new_avg
            self.performance_stats['max_execution_time'] = max(
                self.performance_stats.get('max_execution_time', 0), processing_time
            )

            # 更新成功率
            successful_orders = overall_result.get('successful_orders', 0)
            total_orders = overall_result.get('total_orders', 0)
            if total_orders > 0:
                current_success_rate = successful_orders / total_orders
                old_success_rate = self.performance_stats.get('success_rate', 0)
                new_success_rate = (old_success_rate * (n - 1) + current_success_rate) / n
                self.performance_stats['success_rate'] = new_success_rate

            # 更新执行质量
            avg_quality = overall_result.get('avg_execution_quality', 0)
            if avg_quality > 0:
                old_quality = self.performance_stats.get('avg_execution_quality', 0)
                new_quality = (old_quality * (n - 1) + avg_quality) / n
                self.performance_stats['avg_execution_quality'] = new_quality

            logger.debug(f"性能统计更新: 执行次数={n}, 平均时间={new_avg:.3f}s, 成功率={new_success_rate:.2%}")

        except Exception as e:
            logger.error(f"性能统计更新失败: {e}")

    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """获取订单状态"""
        try:
            # 检查所有订单状态
            for order_list in [self.pending_orders, self.active_orders, self.completed_orders]:
                if order_id in order_list:
                    order = order_list[order_id]
                    return {
                        'order_id': order.order_id,
                        'status': order.status.value,
                        'symbol': order.symbol,
                        'quantity': order.quantity,
                        'filled_quantity': order.filled_quantity,
                        'average_fill_price': order.average_fill_price,
                        'commission': order.commission,
                        'slippage': order.slippage,
                        'execution_quality': order.execution_quality,
                        'created_at': order.created_at,
                        'updated_at': order.updated_at,
                        'execution_reports': [r.to_dict() for r in self.execution_reports.get(order_id, [])],
                        'risk_check_passed': order.risk_check_passed,
                        'rejection_reason': order.rejection_reason,
                        'cancellation_reason': order.cancellation_reason
                    }

            return None

        except Exception as e:
            logger.error(f"订单状态查询失败 {order_id}: {e}")
            return None

    def cancel_order(self, order_id: str, reason: str = "用户取消") -> bool:
        """取消订单"""
        try:
            # 查找订单
            order = None
            for order_list in [self.pending_orders, self.active_orders]:
                if order_id in order_list:
                    order = order_list[order_id]
                    break

            if not order:
                logger.warning(f"取消订单失败: 订单 {order_id} 不存在")
                return False

            # 更新订单状态
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now().isoformat()
            order.cancellation_reason = reason

            # 从活动订单移动到完成订单
            if order_id in self.active_orders:
                self.completed_orders[order_id] = order
                del self.active_orders[order_id]
            elif order_id in self.pending_orders:
                self.completed_orders[order_id] = order
                del self.pending_orders[order_id]

            # 生成取消报告
            cancel_report = ExecutionReport(
                report_id=f"CANCEL_{order_id}_{int(time.time())}",
                order_id=order_id,
                timestamp=datetime.now().isoformat(),
                fill_price=0,
                fill_quantity=0,
                remaining_quantity=order.quantity - order.filled_quantity,
                cumulative_quantity=order.filled_quantity,
                execution_type='cancelled',
                liquidity='none',
                venue='system',
                execution_id=f"CANCEL_{order_id}",
                transaction_time=datetime.now().isoformat(),
                commission=0,
                fees=0,
                slippage=0,
                market_impact=0,
                timing_risk=0,
                execution_quality=0,
                benchmark_comparison={},
                flags=['cancelled'],
                metadata={'cancellation_reason': reason}
            )

            self._save_execution_report(cancel_report)

            logger.info(f"订单取消成功: {order_id}, 原因={reason}")
            return True

        except Exception as e:
            logger.error(f"订单取消失败 {order_id}: {e}")
            return False

    def modify_order(self, order_id: str, modifications: Dict[str, Any]) -> bool:
        """修改订单"""
        try:
            # 查找订单
            if order_id not in self.pending_orders and order_id not in self.active_orders:
                logger.warning(f"修改订单失败: 订单 {order_id} 不存在或不可修改")
                return False

            order = self.pending_orders.get(order_id) or self.active_orders.get(order_id)
            if not order:
                return False

            # 检查订单状态是否允许修改
            if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                logger.warning(f"订单 {order_id} 状态为 {order.status.value}，不允许修改")
                return False

            # 应用修改
            if 'quantity' in modifications:
                new_quantity = modifications['quantity']
                if new_quantity > 0 and new_quantity != order.quantity:
                    order.quantity = new_quantity
                    logger.debug(f"订单 {order_id} 数量修改: {order.quantity} -> {new_quantity}")

            if 'limit_price' in modifications:
                new_price = modifications['limit_price']
                if new_price > 0 and new_price != order.parameters.limit_price:
                    order.parameters.limit_price = new_price
                    logger.debug(f"订单 {order_id} 限价修改: {order.parameters.limit_price} -> {new_price}")

            if 'stop_price' in modifications:
                new_stop = modifications['stop_price']
                if new_stop > 0 and new_stop != order.parameters.stop_price:
                    order.parameters.stop_price = new_stop
                    logger.debug(f"订单 {order_id} 止损价修改: {order.parameters.stop_price} -> {new_stop}")

            order.updated_at = datetime.now().isoformat()

            # 生成修改报告
            modify_report = ExecutionReport(
                report_id=f"MODIFY_{order_id}_{int(time.time())}",
                order_id=order_id,
                timestamp=datetime.now().isoformat(),
                fill_price=0,
                fill_quantity=0,
                remaining_quantity=order.quantity,
                cumulative_quantity=order.filled_quantity,
                execution_type='modified',
                liquidity='none',
                venue='system',
                execution_id=f"MODIFY_{order_id}",
                transaction_time=datetime.now().isoformat(),
                commission=0,
                fees=0,
                slippage=0,
                market_impact=0,
                timing_risk=0,
                execution_quality=0,
                benchmark_comparison={},
                flags=['modified'],
                metadata={'modifications': modifications}
            )

            self._save_execution_report(modify_report)

            logger.info(f"订单修改成功: {order_id}")
            return True

        except Exception as e:
            logger.error(f"订单修改失败 {order_id}: {e}")
            return False

    def get_execution_dashboard(self) -> Dict[str, Any]:
        """获取执行仪表板数据"""
        try:
            dashboard = {
                'timestamp': datetime.now().isoformat(),
                'connection_status': {
                    'is_connected': self.is_connected,
                    'broker_connections': {name: conn.connection_status for name, conn in
                                           self.broker_connections.items()},
                    'last_heartbeat': self.last_connection_check
                },
                'order_summary': {
                    'pending_orders': len(self.pending_orders),
                    'active_orders': len(self.active_orders),
                    'completed_orders': len(self.completed_orders),
                    'total_orders': len(self.pending_orders) + len(self.active_orders) + len(self.completed_orders)
                },
                'performance_stats': self.performance_stats,
                'recent_activity': self._get_recent_activity(),
                'algorithm_performance': self._get_algorithm_performance(),
                'broker_performance': self._get_broker_performance(),
                'risk_metrics': self._get_execution_risk_metrics(),
                'recommendations': self._get_dashboard_recommendations()
            }

            return dashboard

        except Exception as e:
            logger.error(f"执行仪表板生成失败: {e}")
            return {'error': str(e)}

    def _get_recent_activity(self) -> List[Dict[str, Any]]:
        """获取最近活动"""
        try:
            recent_orders = list(self.completed_orders.values())[-10:]  # 最近10个完成的订单
            activity = []

            for order in recent_orders:
                activity.append({
                    'order_id': order.order_id,
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'quantity': order.quantity,
                    'filled_quantity': order.filled_quantity,
                    'status': order.status.value,
                    'execution_quality': order.execution_quality,
                    'timestamp': order.updated_at
                })

            return activity

        except:
            return []

    def _get_algorithm_performance(self) -> Dict[str, Any]:
        """获取算法性能统计"""
        try:
            algorithm_stats = {}

            for order in self.completed_orders.values():
                algorithm = order.execution_params.algorithm.value
                if algorithm not in algorithm_stats:
                    algorithm_stats[algorithm] = {
                        'total_orders': 0,
                        'successful_orders': 0,
                        'total_quantity': 0,
                        'avg_execution_quality': 0,
                        'avg_slippage': 0,
                        'avg_commission': 0
                    }

                stats = algorithm_stats[algorithm]
                stats['total_orders'] += 1
                if order.status == OrderStatus.FILLED:
                    stats['successful_orders'] += 1
                stats['total_quantity'] += order.filled_quantity
                stats['avg_execution_quality'] = (stats['avg_execution_quality'] * (stats['total_orders'] - 1) +
                                                  order.execution_quality) / stats['total_orders']
                stats['avg_slippage'] = (stats['avg_slippage'] * (stats['total_orders'] - 1) +
                                         order.slippage) / stats['total_orders']
                stats['avg_commission'] = (stats['avg_commission'] * (stats['total_orders'] - 1) +
                                           order.commission) / stats['total_orders']

            return algorithm_stats

        except:
            return {}

    def _get_broker_performance(self) -> Dict[str, Any]:
        """获取经纪商性能统计"""
        try:
            broker_stats = {}

            for order in self.completed_orders.values():
                broker = order.execution_params.algo_parameters.get('broker', 'unknown')
                if broker not in broker_stats:
                    broker_stats[broker] = {
                        'total_orders': 0,
                        'success_rate': 0,
                        'avg_execution_time': 0,
                        'avg_commission_rate': 0,
                        'reliability_score': 0
                    }

                stats = broker_stats[broker]
                stats['total_orders'] += 1
                if order.status == OrderStatus.FILLED:
                    stats['success_rate'] = (stats['success_rate'] * (stats['total_orders'] - 1) + 1) / stats[
                        'total_orders']
                else:
                    stats['success_rate'] = stats['success_rate'] * (stats['total_orders'] - 1) / stats[
                        'total_orders']

            return broker_stats

        except:
            return {}

    def _get_execution_risk_metrics(self) -> Dict[str, Any]:
        """获取执行风险指标"""
        try:
            risk_metrics = {
                'max_slippage': 0,
                'avg_slippage': 0,
                'max_market_impact': 0,
                'avg_market_impact': 0,
                'worst_execution_quality': 1.0,
                'concentration_risk': 0
            }

            if not self.completed_orders:
                return risk_metrics

            slippages = []
            market_impacts = []
            execution_qualities = []

            for order in self.completed_orders.values():
                if order.status == OrderStatus.FILLED:
                    slippages.append(abs(order.slippage))
                    market_impacts.append(order.market_impact)
                    execution_qualities.append(order.execution_quality)

            if slippages:
                risk_metrics['max_slippage'] = max(slippages)
                risk_metrics['avg_slippage'] = sum(slippages) / len(slippages)

            if market_impacts:
                risk_metrics['max_market_impact'] = max(market_impacts)
                risk_metrics['avg_market_impact'] = sum(market_impacts) / len(market_impacts)

            if execution_qualities:
                risk_metrics['worst_execution_quality'] = min(execution_qualities)

            # 计算集中度风险（基于算法和经纪商分布）
            algorithm_counts = {}
            broker_counts = {}
            total_orders = len(self.completed_orders)

            for order in self.completed_orders.values():
                algorithm = order.execution_params.algorithm.value
                broker = order.execution_params.algo_parameters.get('broker', 'unknown')

                algorithm_counts[algorithm] = algorithm_counts.get(algorithm, 0) + 1
                broker_counts[broker] = broker_counts.get(broker, 0) + 1

            # 计算赫芬达尔指数
            if total_orders > 0:
                algorithm_hhi = sum((count / total_orders) ** 2 for count in algorithm_counts.values())
                broker_hhi = sum((count / total_orders) ** 2 for count in broker_counts.values())
                risk_metrics['concentration_risk'] = max(algorithm_hhi, broker_hhi)

            return risk_metrics

        except:
            return {}

    def _get_dashboard_recommendations(self) -> List[str]:
        """获取仪表板建议"""
        recommendations = []

        try:
            # 基于性能统计的建议
            if self.performance_stats.get('success_rate', 0) < 0.9:
                recommendations.append("订单成功率较低，建议优化风险控制流程")

            if self.performance_stats.get('avg_execution_time', 0) > 5.0:
                recommendations.append("执行时间较长，考虑使用更高效的执行算法")

            if self.performance_stats.get('avg_execution_quality', 0) < 0.7:
                recommendations.append("执行质量有待提高，建议调整算法参数或更换经纪商")

            # 基于连接状态的建议
            if not self.is_connected:
                recommendations.append("执行引擎未连接，请检查经纪商连接状态")

            # 基于订单分布的建议
            pending_count = len(self.pending_orders)
            active_count = len(self.active_orders)
            if pending_count > active_count * 2:
                recommendations.append("待处理订单过多，可能影响执行效率")

            return recommendations

        except:
            return ["系统监控正常"]

    def cleanup(self):
        """清理资源"""
        try:
            # 关闭线程池
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)

            # 关闭事件循环
            if hasattr(self, 'event_loop'):
                self.event_loop.close()

            # 断开经纪商连接
            self.disconnect()

            # 保存订单历史
            self._save_order_history()

            # 清理缓存
            self.market_data_cache.clear()
            self.liquidity_data.clear()
            self.volatility_data.clear()

            logger.info("执行引擎资源清理完成")

        except Exception as e:
            logger.error(f"执行引擎资源清理失败: {e}")

    def _save_order_history(self):
        """保存订单历史"""
        try:
            # 这里应该保存到数据库或文件
            # 简化实现：记录到日志
            total_orders = len(self.order_history)
            completed_orders = len([o for o in self.order_history if o.status == OrderStatus.FILLED])
            success_rate = completed_orders / total_orders if total_orders > 0 else 0

            logger.info(
                f"订单历史保存: 总订单={total_orders}, 完成订单={completed_orders}, 成功率={success_rate:.2%}")

        except Exception as e:
            logger.error(f"订单历史保存失败: {e}")

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
def create_order(order_id: str, portfolio_id: str, symbol: str, quantity: float,
                 side: OrderSide, order_type: OrderType = OrderType.MARKET,
                 limit_price: Optional[float] = None, stop_price: Optional[float] = None,
                 algorithm: ExecutionAlgorithm = ExecutionAlgorithm.SIMPLE,
                 **kwargs) -> Order:
    """创建订单辅助函数"""
    try:
        # 订单参数
        order_params = OrderParameters(
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=kwargs.get('time_in_force', TimeInForce.DAY),
            good_till_date=kwargs.get('good_till_date'),
            iceberg=kwargs.get('iceberg', False),
            display_quantity=kwargs.get('display_quantity'),
            min_quantity=kwargs.get('min_quantity'),
            execution_instructions=kwargs.get('execution_instructions', []),
            routing_instructions=kwargs.get('routing_instructions', {}),
            algo_parameters=kwargs.get('algo_parameters', {}),
            smart_routing=kwargs.get('smart_routing', True),
            destination=kwargs.get('destination'),
            allocation_strategy=kwargs.get('allocation_strategy', 'fifo'),
            notional=kwargs.get('notional', False)
        )

        # 执行参数
        exec_params = ExecutionParameters(
            algorithm=algorithm,
            urgency=kwargs.get('urgency', 'medium'),
            max_participation_rate=kwargs.get('max_participation_rate', 0.1),
            target_performance=kwargs.get('target_performance', 'price'),
            price_deviation_limit=kwargs.get('price_deviation_limit', 0.02),
            volume_limit=kwargs.get('volume_limit', 0.2),
            time_horizon=kwargs.get('time_horizon', 300),
            start_time=kwargs.get('start_time'),
            end_time=kwargs.get('end_time'),
            allow_iceberg=kwargs.get('allow_iceberg', False),
            allow_dark_pools=kwargs.get('allow_dark_pools', False),
            avoid_ecns=kwargs.get('avoid_ecns', False),
            benchmark=kwargs.get('benchmark', 'arrival_price'),
            tolerance_bands=kwargs.get('tolerance_bands', {'upper': 0.01, 'lower': -0.01}),
            slippage_control=kwargs.get('slippage_control', True),
            real_time_adjustment=kwargs.get('real_time_adjustment', True)
        )

        # 创建订单
        order = Order(
            order_id=order_id,
            portfolio_id=portfolio_id,
            symbol=symbol,
            quantity=quantity,
            side=side,
            parameters=order_params,
            execution_params=exec_params,
            tags=kwargs.get('tags', []),
            metadata=kwargs.get('metadata', {})
        )

        return order

    except Exception as e:
        logger.error(f"订单创建失败: {e}")
        raise

def validate_order_parameters(order: Order) -> bool:
    """验证订单参数有效性"""
    try:
        # 基本验证
        if not order.order_id or not order.portfolio_id or not order.symbol:
            return False

        if order.quantity <= 0:
            return False

        # 订单类型特定验证
        if order.parameters.order_type == OrderType.LIMIT and order.parameters.limit_price <= 0:
            return False

        if order.parameters.order_type == OrderType.STOP and order.parameters.stop_price <= 0:
            return False

        # 时间有效性验证
        if order.parameters.time_in_force == TimeInForce.GTD:
            if not order.parameters.good_till_date:
                return False
            try:
                expiry = datetime.fromisoformat(order.parameters.good_till_date)
                if expiry <= datetime.now():
                    return False
            except ValueError:
                return False

        # 算法参数验证
        if order.execution_params.algorithm == ExecutionAlgorithm.POV:
            if order.execution_params.max_participation_rate <= 0 or order.execution_params.max_participation_rate > 0.5:
                return False

        return True

    except:
        return False

def calculate_order_cost(order: Order, execution_price: float) -> Dict[str, float]:
    """计算订单成本"""
    try:
        notional = order.quantity * execution_price

        # 佣金计算（简化）
        commission_rate = 0.001  # 0.1%
        commission = notional * commission_rate

        # 费用计算
        sec_fee = notional * 0.0000229  # SEC费用
        taf_fee = order.quantity * 0.000119  # 交易活动费
        clearing_fee = notional * 0.00002  # 清算费

        total_fees = sec_fee + taf_fee + clearing_fee
        total_cost = commission + total_fees

        return {
            'notional_value': notional,
            'commission': commission,
            'sec_fee': sec_fee,
            'taf_fee': taf_fee,
            'clearing_fee': clearing_fee,
            'total_fees': total_fees,
            'total_cost': total_cost,
            'cost_rate': total_cost / notional if notional > 0 else 0
        }

    except:
        return {}

if __name__ == "__main__":
    # 测试代码
    config = {
        'execution': {
            'broker_connections': {
                'simulated': {
                    'type': 'simulated',
                    'paper_trading': True
                }
            },
            'execution_algorithms': {
                'simple': {'default_urgency': 'medium'},
                'twap': {'default_slices': 5},
                'vwap': {'volume_participation': 0.1}
            },
            'transaction_costs': {
                'commission_rate': 0.001,
                'fee_rates': {
                    'sec': 0.0000229,
                    'taf': 0.000119,
                    'clearing': 0.00002
                }
            }
        }
    }

    # 创建执行引擎实例
    engine = ExecutionEngine(config)

    # 测试连接
    connected = engine.connect()
    print(f"执行引擎连接状态: {connected}")

    # 创建测试订单
    test_order = create_order(
        order_id="TEST_001",
        portfolio_id="PORTFOLIO_001",
        symbol="AAPL",
        quantity=100,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        algorithm=ExecutionAlgorithm.SIMPLE
    )

    # 模拟市场数据
    market_data = {
        'prices': {
            'AAPL': {
                'close': [150.0, 150.5, 151.0, 150.8, 151.2]
            }
        },
        'volumes': {
            'AAPL': {
                'volume': [1000000, 1200000, 1100000, 950000, 1300000],
                'avg_daily_volume': 1000000
            }
        },
        'timestamp': datetime.now().isoformat()
    }

    # 执行订单
    result = engine.execute_orders([test_order], market_data)
    print("订单执行结果:", json.dumps(result, indent=2, default=str))

    # 清理
    engine.cleanup()