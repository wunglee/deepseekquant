"""
DeepSeekQuant 信号引擎
负责生成、验证和管理交易信号
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
import talib
from scipy import stats
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib
import copy
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import traceback
from collections import deque, defaultdict
import heapq
import numba
from numba import jit, prange
import pickle
import zlib
import base64

# 导入内部模块
from .base_processor import BaseProcessor
from ..utils.helpers import validate_data, calculate_returns, normalize_data
from ..utils.validators import validate_signal
from ..utils.performance import calculate_performance_metrics

logger = logging.getLogger('DeepSeekQuant.SignalEngine')


class SignalType(Enum):
    """信号类型枚举"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"
    REVERSE = "reverse"
    SCALP = "scalp"
    SWING = "swing"
    POSITION = "position"
    ARBITRAGE = "arbitrage"
    HEDGE = "hedge"


class SignalStrength(Enum):
    """信号强度枚举"""
    WEAK = "weak"  # 0-25%
    MILD = "mild"  # 25-50%
    STRONG = "strong"  # 50-75%
    VERY_STRONG = "very_strong"  # 75-100%
    EXTREME = "extreme"  # 特殊事件驱动


class SignalSource(Enum):
    """信号来源枚举"""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    QUANTITATIVE = "quantitative"
    MACHINE_LEARNING = "machine_learning"
    SENTIMENT = "sentiment"
    MARKET_MAKER = "market_maker"
    ARBITRAGE = "arbitrage"
    MANUAL = "manual"
    COMPOSITE = "composite"


class SignalStatus(Enum):
    """信号状态枚举"""
    GENERATED = "generated"
    VALIDATED = "validated"
    REJECTED = "rejected"
    EXECUTED = "executed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    CONFLICT = "conflict"


@dataclass
class SignalMetadata:
    """信号元数据"""
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source: SignalSource = SignalSource.TECHNICAL
    confidence: float = 0.0  # 0.0 - 1.0
    strength: SignalStrength = SignalStrength.MILD
    priority: int = 1  # 1-10, 10为最高优先级
    expiration: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    model_version: str = "1.0.0"
    backtest_id: Optional[str] = None
    strategy_name: Optional[str] = None


@dataclass
class TradingSignal:
    """交易信号数据类"""
    id: str
    symbol: str
    signal_type: SignalType
    price: float
    timestamp: str
    metadata: SignalMetadata
    quantity: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timeframe: str = "1d"
    status: SignalStatus = SignalStatus.GENERATED
    reason: str = ""
    weight: float = 1.0  # 信号权重
    correlation: Dict[str, float] = field(default_factory=dict)
    risk_score: float = 0.0  # 0.0-1.0, 越低越好
    expected_return: float = 0.0
    expected_hold_period: int = 0  # 预期持有期（分钟）
    volume_ratio: float = 1.0  # 成交量比率
    volatility: float = 0.0  # 波动率指标
    liquidity_score: float = 1.0  # 流动性评分

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'price': self.price,
            'timestamp': self.timestamp,
            'quantity': self.quantity,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'timeframe': self.timeframe,
            'status': self.status.value,
            'reason': self.reason,
            'weight': self.weight,
            'correlation': self.correlation,
            'risk_score': self.risk_score,
            'expected_return': self.expected_return,
            'expected_hold_period': self.expected_hold_period,
            'volume_ratio': self.volume_ratio,
            'volatility': self.volatility,
            'liquidity_score': self.liquidity_score,
            'metadata': asdict(self.metadata)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingSignal':
        """从字典创建信号"""
        metadata_data = data.pop('metadata', {})
        metadata = SignalMetadata(**metadata_data)

        return cls(
            id=data['id'],
            symbol=data['symbol'],
            signal_type=SignalType(data['signal_type']),
            price=data['price'],
            timestamp=data['timestamp'],
            metadata=metadata,
            quantity=data.get('quantity'),
            stop_loss=data.get('stop_loss'),
            take_profit=data.get('take_profit'),
            timeframe=data.get('timeframe', '1d'),
            status=SignalStatus(data.get('status', 'generated')),
            reason=data.get('reason', ''),
            weight=data.get('weight', 1.0),
            correlation=data.get('correlation', {}),
            risk_score=data.get('risk_score', 0.0),
            expected_return=data.get('expected_return', 0.0),
            expected_hold_period=data.get('expected_hold_period', 0),
            volume_ratio=data.get('volume_ratio', 1.0),
            volatility=data.get('volatility', 0.0),
            liquidity_score=data.get('liquidity_score', 1.0)
        )


class SignalEngine(BaseProcessor):
    """信号引擎 - 生成和管理交易信号"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化信号引擎

        Args:
            config: 配置字典
        """
        super().__init__(config)

        # 信号生成配置
        self.signal_config = config.get('signal_generation', {})
        self.technical_config = self.signal_config.get('technical_indicators', {})
        self.quantitative_config = self.signal_config.get('quantitative_methods', {})
        self.ml_config = self.signal_config.get('machine_learning', {})

        # 信号存储
        self.generated_signals: Dict[str, TradingSignal] = {}
        self.signal_history: List[TradingSignal] = []
        self.signal_queue = deque(maxlen=1000)

        # 性能统计
        self.performance_stats = {
            'signals_generated': 0,
            'signals_executed': 0,
            'signals_rejected': 0,
            'success_rate': 0.0,
            'avg_hold_time': 0.0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }

        # 缓存和状态
        self._indicator_cache = {}
        self._model_cache = {}
        self._market_state = {}
        self._last_processed = {}

        # 线程池
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.signal_config.get('max_workers', 4)
        )

        # 初始化指标库
        self._initialize_indicators()

        logger.info("信号引擎初始化完成")

    def _initialize_indicators(self):
        """初始化技术指标库"""
        self.indicators = {
            # 趋势指标
            'sma': self._calculate_sma,
            'ema': self._calculate_ema,
            'macd': self._calculate_macd,
            'adx': self._calculate_adx,
            'parabolic_sar': self._calculate_parabolic_sar,
            'ichimoku': self._calculate_ichimoku,

            # 动量指标
            'rsi': self._calculate_rsi,
            'stochastic': self._calculate_stochastic,
            'cci': self._calculate_cci,
            'williams_r': self._calculate_williams_r,
            'momentum': self._calculate_momentum,

            # 波动率指标
            'bollinger_bands': self._calculate_bollinger_bands,
            'atr': self._calculate_atr,
            'standard_deviation': self._calculate_standard_deviation,

            # 成交量指标
            'obv': self._calculate_obv,
            'volume_oscillator': self._calculate_volume_oscillator,
            'money_flow': self._calculate_money_flow,

            # 其他指标
            'vwap': self._calculate_vwap,
            'fibonacci': self._calculate_fibonacci,
            'pivot_points': self._calculate_pivot_points
        }

        # 初始化量化方法
        self.quantitative_methods = {
            'mean_reversion': self._mean_reversion_strategy,
            'momentum_strategy': self._momentum_strategy,
            'breakout_strategy': self._breakout_strategy,
            'arbitrage': self._arbitrage_strategy,
            'statistical_arbitrage': self._statistical_arbitrage,
            'volatility_strategy': self._volatility_strategy,
            'correlation_trading': self._correlation_trading
        }

        logger.info(f"已加载 {len(self.indicators)} 个技术指标和 {len(self.quantitative_methods)} 个量化方法")

    def process(self, market_data: Dict[str, Any]) -> Dict[str, List[TradingSignal]]:
        """
        处理市场数据并生成信号

        Args:
            market_data: 市场数据字典

        Returns:
            生成的信号字典
        """
        start_time = time.time()

        try:
            if not self._validate_market_data(market_data):
                logger.warning("市场数据验证失败")
                return {}

            # 更新市场状态
            self._update_market_state(market_data)

            # 生成信号
            signals = self._generate_signals(market_data)

            # 验证和过滤信号
            validated_signals = self._validate_signals(signals)

            # 排序和优先级处理
            prioritized_signals = self._prioritize_signals(validated_signals)

            # 记录信号
            self._record_signals(prioritized_signals)

            processing_time = time.time() - start_time
            logger.info(f"信号生成完成: {len(prioritized_signals)} 个信号, 耗时: {processing_time:.3f}s")

            return prioritized_signals

        except Exception as e:
            logger.error(f"信号处理失败: {e}")
            self._handle_processing_error(e)
            return {}

    def _validate_market_data(self, market_data: Dict) -> bool:
        """验证市场数据"""
        required_keys = ['timestamp', 'symbols', 'prices', 'volumes']

        if not all(key in market_data for key in required_keys):
            logger.warning("市场数据缺少必要字段")
            return False

        if not market_data['symbols'] or not market_data['prices']:
            logger.warning("市场数据为空")
            return False

        # 检查数据完整性
        for symbol in market_data['symbols']:
            if symbol not in market_data['prices']:
                logger.warning(f"缺少价格数据: {symbol}")
                return False

            price_data = market_data['prices'][symbol]
            if not all(key in price_data for key in ['open', 'high', 'low', 'close']):
                logger.warning(f"价格数据不完整: {symbol}")
                return False

        return True

    def _update_market_state(self, market_data: Dict):
        """更新市场状态"""
        current_time = market_data['timestamp']

        for symbol in market_data['symbols']:
            price_data = market_data['prices'][symbol]
            volume_data = market_data.get('volumes', {}).get(symbol, {})

            # 计算基本市场状态
            market_state = {
                'price': price_data['close'],
                'volume': volume_data.get('volume', 0),
                'timestamp': current_time,
                'volatility': self._calculate_volatility(symbol, price_data),
                'trend': self._assess_trend(symbol, price_data),
                'liquidity': self._assess_liquidity(symbol, volume_data),
                'momentum': self._calculate_momentum(symbol, price_data)
            }

            self._market_state[symbol] = market_state

            # 更新最后处理时间
            self._last_processed[symbol] = current_time

    def _generate_signals(self, market_data: Dict) -> Dict[str, List[TradingSignal]]:
        """生成交易信号"""
        signals = {}

        # 并行处理每个品种的信号生成
        symbols = market_data['symbols']
        futures = []

        for symbol in symbols:
            future = self.thread_pool.submit(
                self._generate_symbol_signals, symbol, market_data
            )
            futures.append((symbol, future))

        # 收集结果
        for symbol, future in futures:
            try:
                symbol_signals = future.result(timeout=30)
                if symbol_signals:
                    signals[symbol] = symbol_signals
            except Exception as e:
                logger.error(f"信号生成失败 {symbol}: {e}")

        return signals

    def _generate_symbol_signals(self, symbol: str, market_data: Dict) -> List[TradingSignal]:
        """生成单个品种的交易信号"""
        signals = []

        try:
            price_data = market_data['prices'][symbol]
            volume_data = market_data.get('volumes', {}).get(symbol, {})
            fundamental_data = market_data.get('fundamentals', {}).get(symbol, {})

            # 技术指标信号
            technical_signals = self._generate_technical_signals(symbol, price_data, volume_data)
            signals.extend(technical_signals)

            # 量化策略信号
            quantitative_signals = self._generate_quantitative_signals(symbol, price_data, market_data)
            signals.extend(quantitative_signals)

            # 机器学习信号
            ml_signals = self._generate_ml_signals(symbol, price_data, fundamental_data)
            signals.extend(ml_signals)

            # 复合信号生成
            composite_signals = self._generate_composite_signals(symbol, signals, market_data)
            signals.extend(composite_signals)

            # 过滤重复信号
            unique_signals = self._filter_duplicate_signals(signals)

            return unique_signals

        except Exception as e:
            logger.error(f"品种信号生成失败 {symbol}: {e}")
            return []

    def _generate_technical_signals(self, symbol: str, price_data: Dict,
                                    volume_data: Dict) -> List[TradingSignal]:
        """生成技术指标信号"""
        signals = []

        # 获取配置的指标
        enabled_indicators = self.technical_config.get('enabled_indicators', [])
        indicator_params = self.technical_config.get('indicator_parameters', {})

        for indicator_name in enabled_indicators:
            if indicator_name in self.indicators:
                try:
                    indicator_func = self.indicators[indicator_name]
                    params = indicator_params.get(indicator_name, {})

                    indicator_signals = indicator_func(symbol, price_data, volume_data, params)
                    signals.extend(indicator_signals)

                except Exception as e:
                    logger.error(f"技术指标 {indicator_name} 信号生成失败 {symbol}: {e}")

        return signals

    def _generate_quantitative_signals(self, symbol: str, price_data: Dict,
                                       market_data: Dict) -> List[TradingSignal]:
        """生成量化策略信号"""
        signals = []

        # 获取配置的策略
        enabled_methods = self.quantitative_config.get('enabled_methods', [])
        method_params = self.quantitative_config.get('method_parameters', {})

        for method_name in enabled_methods:
            if method_name in self.quantitative_methods:
                try:
                    method_func = self.quantitative_methods[method_name]
                    params = method_params.get(method_name, {})

                    method_signals = method_func(symbol, price_data, market_data, params)
                    signals.extend(method_signals)

                except Exception as e:
                    logger.error(f"量化策略 {method_name} 信号生成失败 {symbol}: {e}")

        return signals

    def _generate_ml_signals(self, symbol: str, price_data: Dict,
                             fundamental_data: Dict) -> List[TradingSignal]:
        """生成机器学习信号"""
        signals = []

        if not self.ml_config.get('enabled', False):
            return signals

        try:
            # 这里实现机器学习模型预测
            # 简化实现 - 实际中应该加载和运行ML模型

            ml_models = self.ml_config.get('models', [])
            for model_name in ml_models:
                try:
                    # 获取模型预测
                    prediction = self._run_ml_model(model_name, symbol, price_data, fundamental_data)

                    if prediction and abs(prediction) > self.ml_config.get('threshold', 0.1):
                        signal_type = SignalType.BUY if prediction > 0 else SignalType.SELL
                        confidence = min(abs(prediction), 1.0)

                        signal = TradingSignal(
                            id=f"ml_{symbol}_{int(time.time())}",
                            symbol=symbol,
                            signal_type=signal_type,
                            price=price_data['close'],
                            timestamp=datetime.now().isoformat(),
                            metadata=SignalMetadata(
                                source=SignalSource.MACHINE_LEARNING,
                                confidence=confidence,
                                strength=SignalStrength.STRONG if confidence > 0.7 else SignalStrength.MILD,
                                model_version=model_name
                            ),
                            weight=confidence,
                            expected_return=prediction,
                            risk_score=0.3  # 机器学习信号通常风险较低
                        )

                        signals.append(signal)

                except Exception as e:
                    logger.error(f"机器学习模型 {model_name} 预测失败 {symbol}: {e}")

        except Exception as e:
            logger.error(f"机器学习信号生成失败 {symbol}: {e}")

        return signals

    def _generate_composite_signals(self, symbol: str, existing_signals: List[TradingSignal],
                                    market_data: Dict) -> List[TradingSignal]:
        """生成复合信号"""
        composite_signals = []

        try:
            # 信号聚合 - 合并相同方向的信号
            signal_groups = defaultdict(list)
            for signal in existing_signals:
                key = (signal.signal_type, signal.timeframe)
                signal_groups[key].append(signal)

            # 为每个组创建复合信号
            for (signal_type, timeframe), signal_list in signal_groups.items():
                if len(signal_list) >= self.signal_config.get('composite_threshold', 3):
                    composite_signal = self._create_composite_signal(
                        symbol, signal_type, timeframe, signal_list, market_data
                    )
                    if composite_signal:
                        composite_signals.append(composite_signal)

            # 冲突信号处理
            conflict_signals = self._handle_conflicting_signals(existing_signals)
            composite_signals.extend(conflict_signals)

        except Exception as e:
            logger.error(f"复合信号生成失败 {symbol}: {e}")

        return composite_signals

    def _create_composite_signal(self, symbol: str, signal_type: SignalType,
                                 timeframe: str, signals: List[TradingSignal],
                                 market_data: Dict) -> Optional[TradingSignal]:
        """创建复合信号"""
        try:
            # 计算加权平均参数
            total_weight = sum(signal.weight for signal in signals)
            if total_weight <= 0:
                return None

            # 计算加权价格、置信度等
            weighted_price = sum(signal.price * signal.weight for signal in signals) / total_weight
            weighted_confidence = sum(signal.metadata.confidence * signal.weight for signal in signals) / total_weight
            weighted_strength = self._calculate_weighted_strength(signals)

            # 创建复合信号
            composite_id = f"composite_{symbol}_{signal_type.value}_{int(time.time())}"

            signal = TradingSignal(
                id=composite_id,
                symbol=symbol,
                signal_type=signal_type,
                price=weighted_price,
                timestamp=datetime.now().isoformat(),
                metadata=SignalMetadata(
                    source=SignalSource.COMPOSITE,
                    confidence=weighted_confidence,
                    strength=weighted_strength,
                    tags=['composite', 'aggregated']
                ),
                weight=total_weight / len(signals),  # 平均权重
                timeframe=timeframe,
                risk_score=sum(signal.risk_score * signal.weight for signal in signals) / total_weight
            )

            return signal

        except Exception as e:
            logger.error(f"复合信号创建失败 {symbol}: {e}")
            return None

    def _handle_conflicting_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """处理冲突信号"""
        resolved_signals = []

        try:
            # 按信号类型分组
            buy_signals = [s for s in signals if s.signal_type in [SignalType.BUY, SignalType.REVERSE]]
            sell_signals = [s for s in signals if s.signal_type in [SignalType.SELL, SignalType.CLOSE]]

            # 计算多空力量对比
            buy_power = sum(s.weight * s.metadata.confidence for s in buy_signals)
            sell_power = sum(s.weight * s.metadata.confidence for s in sell_signals)

            total_power = buy_power + sell_power
            if total_power == 0:
                return resolved_signals

            buy_ratio = buy_power / total_power
            sell_ratio = sell_power / total_power

            # 根据力量对比决定最终信号
            if abs(buy_ratio - sell_ratio) > self.signal_config.get('conflict_threshold', 0.2):
                if buy_ratio > sell_ratio:
                    # 生成强势买入信号
                    strong_buy = self._create_strong_signal(buy_signals, SignalType.BUY)
                    resolved_signals.append(strong_buy)
                else:
                    # 生成强势卖出信号
                    strong_sell = self._create_strong_signal(sell_signals, SignalType.SELL)
                    resolved_signals.append(strong_sell)
            else:
                # 力量均衡，生成持有或观望信号
                hold_signal = self._create_hold_signal(signals)
                resolved_signals.append(hold_signal)

        except Exception as e:
            logger.error(f"冲突信号处理失败: {e}")

        return resolved_signals

    def _validate_signals(self, signals: Dict[str, List[TradingSignal]]) -> Dict[str, List[TradingSignal]]:
        """验证信号有效性"""
        validated_signals = {}

        for symbol, symbol_signals in signals.items():
            valid_signals = []

            for signal in symbol_signals:
                try:
                    if self._validate_single_signal(signal):
                        valid_signals.append(signal)
                    else:
                        signal.status = SignalStatus.REJECTED
                        self._record_signal(signal)  # 记录被拒绝的信号

                except Exception as e:
                    logger.error(f"信号验证失败 {signal.id}: {e}")
                    signal.status = SignalStatus.REJECTED
                    signal.reason = f"验证错误: {str(e)}"
                    self._record_signal(signal)

            if valid_signals:
                validated_signals[symbol] = valid_signals

        return validated_signals

    def _validate_single_signal(self, signal: TradingSignal) -> bool:
        """验证单个信号"""
        validation_rules = self.signal_config.get('validation_rules', {})

        # 1. 基本验证
        if not all([signal.id, signal.symbol, signal.signal_type, signal.price > 0]):
            signal.reason = "基本验证失败"
            return False

        # 2. 价格合理性验证
        current_price = self._market_state[signal.symbol]['price']
        price_deviation = abs(signal.price - current_price) / current_price

        if price_deviation > validation_rules.get('max_price_deviation', 0.1):
            signal.reason = f"价格偏差过大: {price_deviation:.2%}"
            return False

        # 3. 波动率验证
        volatility = self._market_state[signal.symbol]['volatility']
        if volatility > validation_rules.get('max_volatility', 0.5):
            signal.reason = f"波动率过高: {volatility:.2%}"
            return False

        # 4. 流动性验证
        liquidity = self._market_state[signal.symbol]['liquidity']
        if liquidity < validation_rules.get('min_liquidity', 0.1):
            signal.reason = f"流动性不足: {liquidity:.2f}"
            return False

        # 5. 置信度验证
        if signal.metadata.confidence < validation_rules.get('min_confidence', 0.3):
            signal.reason = f"置信度过低: {signal.metadata.confidence:.2f}"
            return False

        # 6. 风险评分验证
        if signal.risk_score > validation_rules.get('max_risk_score', 0.7):
            signal.reason = f"风险评分过高: {signal.risk_score:.2f}"
            return False

        # 7. 时间有效性验证
        if signal.metadata.expiration and datetime.fromisoformat(signal.metadata.expiration) < datetime.now():
            signal.reason = "信号已过期"
            return False

        # 8. 信号频率验证
        recent_signals = self._get_recent_signals(signal.symbol, signal.signal_type)
        if len(recent_signals) >= validation_rules.get('max_signals_per_period', 5):
            signal.reason = f"信号频率过高: {len(recent_signals)}"
            return False

        # 9. 止损止盈验证
        if signal.stop_loss and signal.take_profit:
            if signal.signal_type == SignalType.BUY:
                if signal.stop_loss >= signal.price or signal.take_profit <= signal.price:
                    signal.reason = "止损止盈设置不合理"
                    return False
            elif signal.signal_type == SignalType.SELL:
                if signal.stop_loss <= signal.price or signal.take_profit >= signal.price:
                    signal.reason = "止损止盈设置不合理"
                    return False

        # 10. 成交量验证
        if signal.volume_ratio < validation_rules.get('min_volume_ratio', 0.8):
            signal.reason = f"成交量比率不足: {signal.volume_ratio:.2f}"
            return False

        # 11. 市场状态验证
        market_trend = self._market_state[signal.symbol]['trend']
        if (signal.signal_type == SignalType.BUY and market_trend == 'downtrend' and
                not validation_rules.get('allow_counter_trend', False)):
            signal.reason = "不允许逆势交易"
            return False

        # 12. 相关性验证
        if signal.correlation:
            max_correlation = max(signal.correlation.values())
            if max_correlation > validation_rules.get('max_correlation', 0.9):
                signal.reason = f"相关性过高: {max_correlation:.2f}"
                return False

        # 所有验证通过
        signal.status = SignalStatus.VALIDATED
        return True

    def _get_recent_signals(self, symbol: str, signal_type: SignalType,
                            period_minutes: int = 60) -> List[TradingSignal]:
        """获取近期同类型信号"""
        cutoff_time = datetime.now() - timedelta(minutes=period_minutes)

        recent_signals = []
        for signal in self.signal_history:
            if (signal.symbol == symbol and
                    signal.signal_type == signal_type and
                    datetime.fromisoformat(signal.timestamp) >= cutoff_time):
                recent_signals.append(signal)

        return recent_signals

    def _prioritize_signals(self, signals: Dict[str, List[TradingSignal]]) -> Dict[str, List[TradingSignal]]:
        """对信号进行优先级排序"""
        prioritized_signals = {}

        for symbol, symbol_signals in signals.items():
            if not symbol_signals:
                continue

            # 计算每个信号的优先级分数
            scored_signals = []
            for signal in symbol_signals:
                score = self._calculate_signal_score(signal)
                scored_signals.append((score, signal))

            # 按分数排序
            scored_signals.sort(key=lambda x: x[0], reverse=True)

            # 应用最大信号数量限制
            max_signals = self.signal_config.get('max_signals_per_symbol', 3)
            prioritized_signals[symbol] = [signal for _, signal in scored_signals[:max_signals]]

        return prioritized_signals

    def _calculate_signal_score(self, signal: TradingSignal) -> float:
        """计算信号优先级分数"""
        score_weights = self.signal_config.get('scoring_weights', {
            'confidence': 0.3,
            'strength': 0.2,
            'risk_score': -0.2,  # 负权重，风险越高分数越低
            'expected_return': 0.15,
            'liquidity': 0.1,
            'volatility': -0.05
        })

        # 计算各项分数
        confidence_score = signal.metadata.confidence * score_weights['confidence']

        # 强度分数映射
        strength_map = {
            SignalStrength.WEAK: 0.25,
            SignalStrength.MILD: 0.5,
            SignalStrength.STRONG: 0.75,
            SignalStrength.VERY_STRONG: 0.9,
            SignalStrength.EXTREME: 1.0
        }
        strength_score = strength_map.get(signal.metadata.strength, 0.5) * score_weights['strength']

        # 风险分数（风险越高，分数越低）
        risk_score = (1 - signal.risk_score) * score_weights['risk_score']

        # 预期回报分数
        return_score = min(max(signal.expected_return, -1), 1) * score_weights['expected_return']

        # 流动性分数
        liquidity_score = signal.liquidity_score * score_weights['liquidity']

        # 波动率分数（波动率越高，分数越低）
        volatility_score = (1 - min(signal.volatility, 1)) * score_weights['volatility']

        # 总分数
        total_score = confidence_score + strength_score + risk_score + return_score + liquidity_score + volatility_score

        # 应用信号类型权重
        type_weights = self.signal_config.get('signal_type_weights', {
            'BUY': 1.0,
            'SELL': 1.0,
            'HOLD': 0.5,
            'CLOSE': 0.8,
            'REVERSE': 1.2,
            'SCALP': 1.1,
            'SWING': 1.0,
            'POSITION': 0.9,
            'ARBITRAGE': 1.3,
            'HEDGE': 0.7
        })
        type_weight = type_weights.get(signal.signal_type.value, 1.0)

        return total_score * type_weight

    def _record_signals(self, signals: Dict[str, List[TradingSignal]]):
        """记录信号到历史"""
        for symbol, symbol_signals in signals.items():
            for signal in symbol_signals:
                self._record_signal(signal)

                # 添加到信号队列
                self.signal_queue.append(signal)

                # 更新性能统计
                self.performance_stats['signals_generated'] += 1

    def _record_signal(self, signal: TradingSignal):
        """记录单个信号"""
        # 添加到历史记录
        self.signal_history.append(signal)

        # 添加到当前信号字典
        self.generated_signals[signal.id] = signal

        # 保持历史记录长度
        max_history = self.signal_config.get('max_signal_history', 10000)
        if len(self.signal_history) > max_history:
            self.signal_history = self.signal_history[-max_history:]

        # 保持当前信号数量
        max_current = self.signal_config.get('max_current_signals', 1000)
        if len(self.generated_signals) > max_current:
            # 移除最旧的信号
            oldest_ids = sorted(self.generated_signals.keys(),
                                key=lambda id: self.generated_signals[id].timestamp)[
                         :len(self.generated_signals) - max_current]
            for old_id in oldest_ids:
                del self.generated_signals[old_id]

    def _handle_processing_error(self, error: Exception):
        """处理处理错误"""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'component': 'signal_engine'
        }

        # 记录错误
        self.error_log.append(error_info)

        # 错误率统计
        recent_errors = [e for e in self.error_log
                         if datetime.fromisoformat(e['timestamp']) > datetime.now() - timedelta(hours=1)]

        if len(recent_errors) > 10:  # 1小时内超过10个错误
            logger.error("信号引擎错误率过高，可能需要人工干预")

            # 尝试自动恢复
            if self._attempt_auto_recovery():
                logger.info("信号引擎自动恢复成功")
            else:
                logger.error("信号引擎自动恢复失败")

    def _attempt_auto_recovery(self) -> bool:
        """尝试自动恢复"""
        try:
            # 清理缓存
            self._indicator_cache.clear()
            self._model_cache.clear()

            # 重置市场状态
            self._market_state.clear()

            # 重新初始化指标
            self._initialize_indicators()

            # 清理部分历史记录
            if len(self.signal_history) > 5000:
                self.signal_history = self.signal_history[-5000:]

            if len(self.generated_signals) > 500:
                # 只保留最近500个信号
                recent_ids = sorted(self.generated_signals.keys(),
                                    key=lambda id: self.generated_signals[id].timestamp)[-500:]
                self.generated_signals = {id: self.generated_signals[id] for id in recent_ids}

            logger.info("信号引擎自动恢复完成")
            return True

        except Exception as e:
            logger.error(f"信号引擎自动恢复失败: {e}")
            return False

    # 技术指标实现方法
    def _calculate_sma(self, symbol: str, price_data: Dict,
                       volume_data: Dict, params: Dict) -> List[TradingSignal]:
        """计算简单移动平均线信号"""
        signals = []

        try:
            # 获取价格数据
            closes = price_data.get('close', [])
            if len(closes) < params.get('period', 20):
                return signals

            # 计算SMA
            period = params.get('period', 20)
            sma = talib.SMA(np.array(closes), timeperiod=period)

            if len(sma) < 2 or np.isnan(sma[-1]):
                return signals

            # 生成信号
            current_price = closes[-1]
            current_sma = sma[-1]
            prev_sma = sma[-2] if len(sma) >= 2 else current_sma

            # 价格上穿SMA - 买入信号
            if current_price > current_sma and closes[-2] <= prev_sma:
                signal = TradingSignal(
                    id=f"sma_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    price=current_price,
                    timestamp=datetime.now().isoformat(),
                    metadata=SignalMetadata(
                        source=SignalSource.TECHNICAL,
                        confidence=0.6,
                        strength=SignalStrength.MILD,
                        parameters={'period': period, 'type': 'sma_crossover'}
                    ),
                    weight=0.7,
                    risk_score=0.4
                )
                signals.append(signal)

            # 价格下穿SMA - 卖出信号
            elif current_price < current_sma and closes[-2] >= prev_sma:
                signal = TradingSignal(
                    id=f"sma_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=current_price,
                    timestamp=datetime.now().isoformat(),
                    metadata=SignalMetadata(
                        source=SignalSource.TECHNICAL,
                        confidence=0.6,
                        strength=SignalStrength.MILD,
                        parameters={'period': period, 'type': 'sma_crossover'}
                    ),
                    weight=0.7,
                    risk_score=0.4
                )
                signals.append(signal)

        except Exception as e:
            logger.error(f"SMA指标计算失败 {symbol}: {e}")

        return signals

    def _calculate_ema(self, symbol: str, price_data: Dict,
                       volume_data: Dict, params: Dict) -> List[TradingSignal]:
        """计算指数移动平均线信号"""
        signals = []

        try:
            # 获取价格数据
            closes = price_data.get('close', [])
            if len(closes) < params.get('period', 12):
                return signals

            # 计算EMA
            period = params.get('period', 12)
            ema = talib.EMA(np.array(closes), timeperiod=period)

            if len(ema) < 2 or np.isnan(ema[-1]):
                return signals

            # 生成信号
            current_price = closes[-1]
            current_ema = ema[-1]
            prev_ema = ema[-2] if len(ema) >= 2 else current_ema

            # 价格上穿EMA - 买入信号
            if current_price > current_ema and closes[-2] <= prev_ema:
                signal = TradingSignal(
                    id=f"ema_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    price=current_price,
                    timestamp=datetime.now().isoformat(),
                    metadata=SignalMetadata(
                        source=SignalSource.TECHNICAL,
                        confidence=0.65,
                        strength=SignalStrength.MILD,
                        parameters={'period': period, 'type': 'ema_crossover'}
                    ),
                    weight=0.75,
                    risk_score=0.35
                )
                signals.append(signal)

            # 价格下穿EMA - 卖出信号
            elif current_price < current_ema and closes[-2] >= prev_ema:
                signal = TradingSignal(
                    id=f"ema_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=current_price,
                    timestamp=datetime.now().isoformat(),
                    metadata=SignalMetadata(
                        source=SignalSource.TECHNICAL,
                        confidence=0.65,
                        strength=SignalStrength.MILD,
                        parameters={'period': period, 'type': 'ema_crossover'}
                    ),
                    weight=0.75,
                    risk_score=0.35
                )
                signals.append(signal)

        except Exception as e:
            logger.error(f"EMA指标计算失败 {symbol}: {e}")

        return signals

    def _calculate_macd(self, symbol: str, price_data: Dict,
                        volume_data: Dict, params: Dict) -> List[TradingSignal]:
        """计算MACD指标信号"""
        signals = []

        try:
            # 获取价格数据
            closes = price_data.get('close', [])
            if len(closes) < 26:  # MACD需要至少26个数据点
                return signals

            # 计算MACD
            fast_period = params.get('fast_period', 12)
            slow_period = params.get('slow_period', 26)
            signal_period = params.get('signal_period', 9)

            macd, macd_signal, macd_hist = talib.MACD(
                np.array(closes),
                fastperiod=fast_period,
                slowperiod=slow_period,
                signalperiod=signal_period
            )

            if len(macd) < 2 or np.isnan(macd[-1]) or np.isnan(macd_signal[-1]):
                return signals

            # 生成信号
            current_macd = macd[-1]
            current_signal = macd_signal[-1]
            prev_macd = macd[-2] if len(macd) >= 2 else current_macd
            prev_signal = macd_signal[-2] if len(macd_signal) >= 2 else current_signal

            # MACD上穿信号线 - 买入信号
            if current_macd > current_signal and prev_macd <= prev_signal:
                signal = TradingSignal(
                    id=f"macd_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    price=closes[-1],
                    timestamp=datetime.now().isoformat(),
                    metadata=SignalMetadata(
                        source=SignalSource.TECHNICAL,
                        confidence=0.7,
                        strength=SignalStrength.STRONG,
                        parameters={
                            'fast_period': fast_period,
                            'slow_period': slow_period,
                            'signal_period': signal_period,
                            'type': 'macd_crossover'
                        }
                    ),
                    weight=0.8,
                    risk_score=0.3
                )
                signals.append(signal)

            # MACD下穿信号线 - 卖出信号
            elif current_macd < current_signal and prev_macd >= prev_signal:
                signal = TradingSignal(
                    id=f"macd_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=closes[-1],
                    timestamp=datetime.now().isoformat(),
                    metadata=SignalMetadata(
                        source=SignalSource.TECHNICAL,
                        confidence=0.7,
                        strength=SignalStrength.STRONG,
                        parameters={
                            'fast_period': fast_period,
                            'slow_period': slow_period,
                            'signal_period': signal_period,
                            'type': 'macd_crossover'
                        }
                    ),
                    weight=0.8,
                    risk_score=0.3
                )
                signals.append(signal)

        except Exception as e:
            logger.error(f"MACD指标计算失败 {symbol}: {e}")

        return signals

    # 其他技术指标实现...
    def _calculate_rsi(self, symbol: str, price_data: Dict,
                       volume_data: Dict, params: Dict) -> List[TradingSignal]:
        """计算RSI指标信号"""
        signals = []

        try:
            # 获取价格数据
            closes = price_data.get('close', [])
            if len(closes) < params.get('period', 14) + 1:
                return signals

            # 计算RSI
            period = params.get('period', 14)
            rsi = talib.RSI(np.array(closes), timeperiod=period)

            if len(rsi) < 1 or np.isnan(rsi[-1]):
                return signals

            # 生成信号
            current_rsi = rsi[-1]
            overbought = params.get('overbought', 70)
            oversold = params.get('oversold', 30)

            # RSI超卖 - 买入信号
            if current_rsi < oversold:
                signal = TradingSignal(
                    id=f"rsi_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    price=closes[-1],
                    timestamp=datetime.now().isoformat(),
                    metadata=SignalMetadata(
                        source=SignalSource.TECHNICAL,
                        confidence=0.75,
                        strength=SignalStrength.STRONG,
                        parameters={
                            'period': period,
                            'overbought': overbought,
                            'oversold': oversold,
                            'type': 'rsi_oversold'
                        }
                    ),
                    weight=0.85,
                    risk_score=0.25
                )
                signals.append(signal)

            # RSI超买 - 卖出信号
            elif current_rsi > overbought:
                signal = TradingSignal(
                    id=f"rsi_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=closes[-1],
                    timestamp=datetime.now().isoformat(),
                    metadata=SignalMetadata(
                        source=SignalSource.TECHNICAL,
                        confidence=0.75,
                        strength=SignalStrength.STRONG,
                        parameters={
                            'period': period,
                            'overbought': overbought,
                            'oversold': oversold,
                            'type': 'rsi_overbought'
                        }
                    ),
                    weight=0.85,
                    risk_score=0.25
                )
                signals.append(signal)

        except Exception as e:
            logger.error(f"RSI指标计算失败 {symbol}: {e}")

        return signals

    # 量化策略实现...
    def _mean_reversion_strategy(self, symbol: str, price_data: Dict,
                                 market_data: Dict, params: Dict) -> List[TradingSignal]:
        """均值回归策略"""
        signals = []

        try:
            # 获取价格数据
            closes = price_data.get('close', [])
            if len(closes) < params.get('lookback_period', 20):
                return signals

            # 计算均值和标准差
            lookback = params.get('lookback_period', 20)
            recent_prices = closes[-lookback:]
            mean = np.mean(recent_prices)
            std = np.std(recent_prices)

            if std == 0:
                return signals

            # 计算Z-score
            current_price = closes[-1]
            z_score = (current_price - mean) / std

            # 生成信号
            entry_threshold = params.get('entry_threshold', 2.0)
            exit_threshold = params.get('exit_threshold', 0.5)

            # Z-score过高 - 卖出信号（价格会回归均值）
            if z_score > entry_threshold:
                signal = TradingSignal(
                    id=f"mean_reversion_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=current_price,
                    timestamp=datetime.now().isoformat(),
                    metadata=SignalMetadata(
                        source=SignalSource.QUANTITATIVE,
                        confidence=min(z_score / 3, 0.9),  # 置信度基于Z-score
                        strength=SignalStrength.STRONG,
                        parameters={
                            'lookback_period': lookback,
                            'entry_threshold': entry_threshold,
                            'exit_threshold': exit_threshold,
                            'z_score': z_score,
                            'type': 'mean_reversion'
                        }
                    ),
                    weight=0.8,
                    risk_score=0.4,
                    expected_return=mean - current_price  # 预期回归到均值
                )
                signals.append(signal)

            # Z-score过低 - 买入信号（价格会回归均值）
            elif z_score < -entry_threshold:
                signal = TradingSignal(
                    id=f"mean_reversion_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    price=current_price,
                    timestamp=datetime.now().isoformat(),
                    metadata=SignalMetadata(
                        source=SignalSource.QUANTITATIVE,
                        confidence=min(abs(z_score) / 3, 0.9),
                        strength=SignalStrength.STRONG,
                        parameters={
                            'lookback_period': lookback,
                            'entry_threshold': entry_threshold,
                            'exit_threshold': exit_threshold,
                            'z_score': z_score,
                            'type': 'mean_reversion'
                        }
                    ),
                    weight=0.8,
                    risk_score=0.4,
                    expected_return=current_price - mean  # 预期回归到均值
                )
                signals.append(signal)

        except Exception as e:
            logger.error(f"均值回归策略失败 {symbol}: {e}")

        return signals

    def _momentum_strategy(self, symbol: str, price_data: Dict,
                           market_data: Dict, params: Dict) -> List[TradingSignal]:
        """动量策略"""
        signals = []

        try:
            # 获取价格数据
            closes = price_data.get('close', [])
            if len(closes) < params.get('momentum_period', 10) + 1:
                return signals

            # 计算动量
            period = params.get('momentum_period', 10)
            momentum = (closes[-1] / closes[-period] - 1) * 100  # 百分比动量

            # 生成信号
            entry_threshold = params.get('entry_threshold', 5.0)  # 5%
            exit_threshold = params.get('exit_threshold', 2.0)  # 2%

            # 正动量强劲 - 买入信号
            if momentum > entry_threshold:
                signal = TradingSignal(
                    id=f"momentum_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    price=closes[-1],
                    timestamp=datetime.now().isoformat(),
                    metadata=SignalMetadata(
                        source=SignalSource.QUANTITATIVE,
                        confidence=min(momentum / 20, 0.9),  # 置信度基于动量强度
                        strength=SignalStrength.VERY_STRONG,
                        parameters={
                            'momentum_period': period,
                            'entry_threshold': entry_threshold,
                            'exit_threshold': exit_threshold,
                            'momentum': momentum,
                            'type': 'momentum'
                        }
                    ),
                    weight=0.9,
                    risk_score=0.5,  # 动量策略风险较高
                    expected_return=momentum / 2  # 预期回报为动量的一半
                )
                signals.append(signal)

            # 负动量强劲 - 卖出信号
            elif momentum < -entry_threshold:
                signal = TradingSignal(
                    id=f"momentum_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=closes[-1],
                    timestamp=datetime.now().isoformat(),
                    metadata=SignalMetadata(
                        source=SignalSource.QUANTITATIVE,
                        confidence=min(abs(momentum) / 20, 0.9),
                        strength=SignalStrength.VERY_STRONG,
                        parameters={
                            'momentum_period': period,
                            'entry_threshold': entry_threshold,
                            'exit_threshold': exit_threshold,
                            'momentum': momentum,
                            'type': 'momentum'
                        }
                    ),
                    weight=0.9,
                    risk_score=0.5,
                    expected_return=abs(momentum) / 2
                )
                signals.append(signal)

        except Exception as e:
            logger.error(f"动量策略执行失败 {symbol}: {e}")

        return signals

    def _breakout_strategy(self, symbol: str, price_data: Dict,
                           market_data: Dict, params: Dict) -> List[TradingSignal]:
        """突破策略"""
        signals = []

        try:
            # 获取价格数据
            highs = price_data.get('high', [])
            lows = price_data.get('low', [])
            closes = price_data.get('close', [])

            if len(highs) < params.get('lookback_period', 20):
                return signals

            # 计算阻力位和支撑位
            lookback = params.get('lookback_period', 20)
            resistance = max(highs[-lookback:])
            support = min(lows[-lookback:])
            current_price = closes[-1]

            # 生成突破信号
            breakout_threshold = params.get('breakout_threshold', 0.02)  # 2%

            # 上破阻力位 - 买入信号
            if current_price > resistance * (1 + breakout_threshold):
                signal = TradingSignal(
                    id=f"breakout_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    price=current_price,
                    timestamp=datetime.now().isoformat(),
                    metadata=SignalMetadata(
                        source=SignalSource.QUANTITATIVE,
                        confidence=0.7,
                        strength=SignalStrength.STRONG,
                        parameters={
                            'lookback_period': lookback,
                            'resistance': resistance,
                            'breakout_threshold': breakout_threshold,
                            'type': 'breakout'
                        }
                    ),
                    weight=0.8,
                    risk_score=0.4,
                    expected_return=0.05  # 预期5%回报
                )
                signals.append(signal)

            # 下破支撑位 - 卖出信号
            elif current_price < support * (1 - breakout_threshold):
                signal = TradingSignal(
                    id=f"breakout_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=current_price,
                    timestamp=datetime.now().isoformat(),
                    metadata=SignalMetadata(
                        source=SignalSource.QUANTITATIVE,
                        confidence=0.7,
                        strength=SignalStrength.STRONG,
                        parameters={
                            'lookback_period': lookback,
                            'support': support,
                            'breakout_threshold': breakout_threshold,
                            'type': 'breakout'
                        }
                    ),
                    weight=0.8,
                    risk_score=0.4,
                    expected_return=0.05
                )
                signals.append(signal)

        except Exception as e:
            logger.error(f"突破策略执行失败 {symbol}: {e}")

        return signals

    def _arbitrage_strategy(self, symbol: str, price_data: Dict,
                            market_data: Dict, params: Dict) -> List[TradingSignal]:
        """套利策略"""
        signals = []

        try:
            # 需要多个相关品种的数据
            correlated_symbols = params.get('correlated_symbols', [])
            if not correlated_symbols:
                return signals

            # 获取相关品种价格
            main_price = price_data.get('close', [])[-1] if price_data.get('close') else 0
            if main_price == 0:
                return signals

            arbitrage_opportunities = []
            for corr_symbol in correlated_symbols:
                if corr_symbol in market_data.get('prices', {}):
                    corr_price = market_data['prices'][corr_symbol].get('close', [])[-1]
                    if corr_price and corr_price > 0:
                        # 计算价格差异百分比
                        price_ratio = main_price / corr_price
                        historical_ratio = params.get('historical_ratio', {}).get(corr_symbol, 1.0)
                        deviation = abs(price_ratio - historical_ratio) / historical_ratio

                        if deviation > params.get('arbitrage_threshold', 0.05):  # 5%偏差
                            arbitrage_opportunities.append({
                                'symbol': corr_symbol,
                                'deviation': deviation,
                                'ratio': price_ratio,
                                'expected_ratio': historical_ratio
                            })

            # 生成套利信号
            for opportunity in arbitrage_opportunities:
                if opportunity['deviation'] > params.get('min_deviation', 0.03):
                    # 决定套利方向
                    if opportunity['ratio'] > opportunity['expected_ratio']:
                        # 主品种相对高估，卖出主品种/买入相关品种
                        signal = TradingSignal(
                            id=f"arbitrage_{symbol}_{opportunity['symbol']}_{int(time.time())}",
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            price=main_price,
                            timestamp=datetime.now().isoformat(),
                            metadata=SignalMetadata(
                                source=SignalSource.ARBITRAGE,
                                confidence=min(opportunity['deviation'] * 10, 0.9),
                                strength=SignalStrength.STRONG,
                                parameters={
                                    'correlated_symbol': opportunity['symbol'],
                                    'deviation': opportunity['deviation'],
                                    'current_ratio': opportunity['ratio'],
                                    'expected_ratio': opportunity['expected_ratio'],
                                    'type': 'statistical_arbitrage'
                                }
                            ),
                            weight=0.85,
                            risk_score=0.3,  # 套利策略风险较低
                            expected_return=opportunity['deviation'] * 0.5  # 预期回报为偏差的一半
                        )
                        signals.append(signal)

                    else:
                        # 主品种相对低估，买入主品种/卖出相关品种
                        signal = TradingSignal(
                            id=f"arbitrage_{symbol}_{opportunity['symbol']}_{int(time.time())}",
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            price=main_price,
                            timestamp=datetime.now().isoformat(),
                            metadata=SignalMetadata(
                                source=SignalSource.ARBITRAGE,
                                confidence=min(opportunity['deviation'] * 10, 0.9),
                                strength=SignalStrength.STRONG,
                                parameters={
                                    'correlated_symbol': opportunity['symbol'],
                                    'deviation': opportunity['deviation'],
                                    'current_ratio': opportunity['ratio'],
                                    'expected_ratio': opportunity['expected_ratio'],
                                    'type': 'statistical_arbitrage'
                                }
                            ),
                            weight=0.85,
                            risk_score=0.3,
                            expected_return=opportunity['deviation'] * 0.5
                        )
                        signals.append(signal)

        except Exception as e:
            logger.error(f"套利策略执行失败 {symbol}: {e}")

        return signals

    def _statistical_arbitrage(self, symbol: str, price_data: Dict,
                               market_data: Dict, params: Dict) -> List[TradingSignal]:
        """统计套利策略"""
        signals = []

        try:
            # 需要历史价格数据计算统计关系
            lookback_period = params.get('lookback_period', 60)
            if len(price_data.get('close', [])) < lookback_period:
                return signals

            # 获取相关品种数据
            paired_symbol = params.get('paired_symbol')
            if not paired_symbol or paired_symbol not in market_data.get('prices', {}):
                return signals

            paired_prices = market_data['prices'][paired_symbol].get('close', [])
            if len(paired_prices) < lookback_period:
                return signals

            # 计算价格序列的对数
            main_log_prices = np.log(np.array(price_data['close'][-lookback_period:]))
            paired_log_prices = np.log(np.array(paired_prices[-lookback_period:]))

            # 计算协整关系
            score, pvalue, _ = coint(main_log_prices, paired_log_prices)

            if pvalue < params.get('cointegration_threshold', 0.05):  # 协整关系显著
                # 计算价差（对冲比率）
                hedge_ratio = np.polyfit(main_log_prices, paired_log_prices, 1)[0]
                spread = main_log_prices - hedge_ratio * paired_log_prices

                # 计算价差的Z-score
                spread_mean = np.mean(spread)
                spread_std = np.std(spread)
                current_spread = spread[-1]
                z_score = (current_spread - spread_mean) / spread_std

                # 生成信号基于Z-score
                entry_z = params.get('entry_z_score', 2.0)
                exit_z = params.get('exit_z_score', 0.5)

                if z_score > entry_z:
                    # 价差过高，卖出主品种/买入配对品种
                    signal = TradingSignal(
                        id=f"stat_arb_{symbol}_{paired_symbol}_{int(time.time())}",
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        price=price_data['close'][-1],
                        timestamp=datetime.now().isoformat(),
                        metadata=SignalMetadata(
                            source=SignalSource.ARBITRAGE,
                            confidence=min(abs(z_score) / 3, 0.9),
                            strength=SignalStrength.STRONG,
                            parameters={
                                'paired_symbol': paired_symbol,
                                'z_score': z_score,
                                'hedge_ratio': hedge_ratio,
                                'p_value': pvalue,
                                'type': 'statistical_arbitrage'
                            }
                        ),
                        weight=0.9,
                        risk_score=0.2,  # 统计套利风险较低
                        expected_return=abs(z_score) * 0.01
                    )
                    signals.append(signal)

                elif z_score < -entry_z:
                    # 价差过低，买入主品种/卖出配对品种
                    signal = TradingSignal(
                        id=f"stat_arb_{symbol}_{paired_symbol}_{int(time.time())}",
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        price=price_data['close'][-1],
                        timestamp=datetime.now().isoformat(),
                        metadata=SignalMetadata(
                            source=SignalSource.ARBITRAGE,
                            confidence=min(abs(z_score) / 3, 0.9),
                            strength=SignalStrength.STRONG,
                            parameters={
                                'paired_symbol': paired_symbol,
                                'z_score': z_score,
                                'hedge_ratio': hedge_ratio,
                                'p_value': pvalue,
                                'type': 'statistical_arbitrage'
                            }
                        ),
                        weight=0.9,
                        risk_score=0.2,
                        expected_return=abs(z_score) * 0.01
                    )
                    signals.append(signal)

        except Exception as e:
            logger.error(f"统计套利策略执行失败 {symbol}: {e}")

        return signals

    def _volatility_strategy(self, symbol: str, price_data: Dict,
                             market_data: Dict, params: Dict) -> List[TradingSignal]:
        """波动率策略"""
        signals = []

        try:
            closes = price_data.get('close', [])
            if len(closes) < params.get('volatility_period', 20):
                return signals

            # 计算波动率（历史波动率）
            period = params.get('volatility_period', 20)
            returns = np.diff(np.log(closes[-period:]))
            volatility = np.std(returns) * np.sqrt(252)  # 年化波动率

            # 获取隐含波动率（如果有）
            implied_vol = market_data.get('options', {}).get(symbol, {}).get('implied_vol', volatility)

            # 波动率交易策略
            vol_threshold = params.get('volatility_threshold', 0.3)
            mean_reversion_speed = params.get('mean_reversion_speed', 0.1)

            if volatility > vol_threshold and volatility > implied_vol * 1.1:
                # 波动率过高，预期回归 - 卖出波动率（做空Gamma）
                signal = TradingSignal(
                    id=f"volatility_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=closes[-1],
                    timestamp=datetime.now().isoformat(),
                    metadata=SignalMetadata(
                        source=SignalSource.QUANTITATIVE,
                        confidence=0.6,
                        strength=SignalStrength.MILD,
                        parameters={
                            'historical_volatility': volatility,
                            'implied_volatility': implied_vol,
                            'threshold': vol_threshold,
                            'type': 'volatility_mean_reversion'
                        }
                    ),
                    weight=0.7,
                    risk_score=0.6,  # 波动率策略风险较高
                    expected_return=-mean_reversion_speed * (volatility - vol_threshold)
                )
                signals.append(signal)

            elif volatility < vol_threshold * 0.7 and volatility < implied_vol * 0.9:
                # 波动率过低，预期上升 - 买入波动率（做多Gamma）
                signal = TradingSignal(
                    id=f"volatility_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    price=closes[-1],
                    timestamp=datetime.now().isoformat(),
                    metadata=SignalMetadata(
                        source=SignalSource.QUANTITATIVE,
                        confidence=0.6,
                        strength=SignalStrength.MILD,
                        parameters={
                            'historical_volatility': volatility,
                            'implied_volatility': implied_vol,
                            'threshold': vol_threshold,
                            'type': 'volatility_mean_reversion'
                        }
                    ),
                    weight=0.7,
                    risk_score=0.6,
                    expected_return=mean_reversion_speed * (vol_threshold - volatility)
                )
                signals.append(signal)

        except Exception as e:
            logger.error(f"波动率策略执行失败 {symbol}: {e}")

        return signals

    def _correlation_trading(self, symbol: str, price_data: Dict,
                             market_data: Dict, params: Dict) -> List[TradingSignal]:
        """相关性交易策略"""
        signals = []

        try:
            # 获取相关品种列表
            correlated_symbols = params.get('correlated_symbols', [])
            if not correlated_symbols:
                return signals

            # 计算与每个相关品种的滚动相关性
            lookback = params.get('correlation_lookback', 30)
            closes = price_data.get('close', [])
            if len(closes) < lookback:
                return signals

            main_returns = np.diff(np.log(closes[-lookback:]))

            correlation_changes = []
            for corr_symbol in correlated_symbols:
                if corr_symbol in market_data.get('prices', {}):
                    corr_closes = market_data['prices'][corr_symbol].get('close', [])
                    if len(corr_closes) >= lookback:
                        corr_returns = np.diff(np.log(corr_closes[-lookback:]))

                        # 计算滚动相关性
                        if len(main_returns) == len(corr_returns):
                            current_corr = np.corrcoef(main_returns, corr_returns)[0, 1]

                            # 获取历史平均相关性
                            historical_corr = params.get('historical_correlations', {}).get(corr_symbol, 0.8)

                            # 计算相关性变化
                            corr_change = current_corr - historical_corr
                            correlation_changes.append({
                                'symbol': corr_symbol,
                                'current_correlation': current_corr,
                                'historical_correlation': historical_corr,
                                'change': corr_change
                            })

            # 基于相关性变化生成信号
            corr_threshold = params.get('correlation_threshold', 0.2)
            for corr_data in correlation_changes:
                if abs(corr_data['change']) > corr_threshold:
                    if corr_data['change'] > 0:
                        # 相关性增强，买入配对交易
                        signal = TradingSignal(
                            id=f"correlation_{symbol}_{corr_data['symbol']}_{int(time.time())}",
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            price=closes[-1],
                            timestamp=datetime.now().isoformat(),
                            metadata=SignalMetadata(
                                source=SignalSource.QUANTITATIVE,
                                confidence=min(abs(corr_data['change']) * 5, 0.8),
                                strength=SignalStrength.MILD,
                                parameters={
                                    'correlated_symbol': corr_data['symbol'],
                                    'correlation_change': corr_data['change'],
                                    'current_correlation': corr_data['current_correlation'],
                                    'historical_correlation': corr_data['historical_correlation'],
                                    'type': 'correlation_trading'
                                }
                            ),
                            weight=0.6,
                            risk_score=0.5,
                            expected_return=corr_data['change'] * 0.02
                        )
                        signals.append(signal)
                    else:
                        # 相关性减弱，卖出或减少配对交易
                        signal = TradingSignal(
                            id=f"correlation_{symbol}_{corr_data['symbol']}_{int(time.time())}",
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            price=closes[-1],
                            timestamp=datetime.now().isoformat(),
                            metadata=SignalMetadata(
                                source=SignalSource.QUANTITATIVE,
                                confidence=min(abs(corr_data['change']) * 5, 0.8),
                                strength=SignalStrength.MILD,
                                parameters={
                                    'correlated_symbol': corr_data['symbol'],
                                    'correlation_change': corr_data['change'],
                                    'current_correlation': corr_data['current_correlation'],
                                    'historical_correlation': corr_data['historical_correlation'],
                                    'type': 'correlation_trading'
                                }
                            ),
                            weight=0.6,
                            risk_score=0.5,
                            expected_return=abs(corr_data['change']) * 0.02
                        )
                        signals.append(signal)

        except Exception as e:
            logger.error(f"相关性交易策略执行失败 {symbol}: {e}")

        return signals

    def _calculate_weighted_strength(self, signals: List[TradingSignal]) -> SignalStrength:
        """计算加权信号强度"""
        if not signals:
            return SignalStrength.WEAK

        # 计算加权平均置信度
        total_weight = sum(signal.weight for signal in signals)
        if total_weight == 0:
            return SignalStrength.WEAK

        weighted_confidence = sum(signal.metadata.confidence * signal.weight for signal in signals) / total_weight

        # 根据置信度映射到强度等级
        if weighted_confidence >= 0.8:
            return SignalStrength.VERY_STRONG
        elif weighted_confidence >= 0.6:
            return SignalStrength.STRONG
        elif weighted_confidence >= 0.4:
            return SignalStrength.MILD
        else:
            return SignalStrength.WEAK

    def _create_strong_signal(self, signals: List[TradingSignal], signal_type: SignalType) -> TradingSignal:
        """创建强势信号"""
        if not signals:
            return None

        # 选择置信度最高的信号作为基础
        best_signal = max(signals, key=lambda s: s.metadata.confidence)

        # 增强信号强度
        strong_signal = TradingSignal(
            id=f"strong_{best_signal.symbol}_{int(time.time())}",
            symbol=best_signal.symbol,
            signal_type=signal_type,
            price=best_signal.price,
            timestamp=datetime.now().isoformat(),
            metadata=SignalMetadata(
                source=SignalSource.COMPOSITE,
                confidence=min(best_signal.metadata.confidence * 1.2, 0.95),
                strength=SignalStrength.VERY_STRONG,
                parameters={
                    'base_signals': [s.id for s in signals],
                    'composite_type': 'strong_signal'
                }
            ),
            weight=1.0,
            risk_score=best_signal.risk_score * 0.8,  # 强势信号风险较低
            expected_return=best_signal.expected_return * 1.5
        )

        return strong_signal

    def _create_hold_signal(self, signals: List[TradingSignal]) -> TradingSignal:
        """创建持有信号"""
        if not signals:
            return None

        # 使用第一个信号的基本信息
        base_signal = signals[0]

        hold_signal = TradingSignal(
            id=f"hold_{base_signal.symbol}_{int(time.time())}",
            symbol=base_signal.symbol,
            signal_type=SignalType.HOLD,
            price=base_signal.price,
            timestamp=datetime.now().isoformat(),
            metadata=SignalMetadata(
                source=SignalSource.COMPOSITE,
                confidence=0.5,
                strength=SignalStrength.WEAK,
                parameters={
                    'conflicting_signals': [s.id for s in signals],
                    'reason': 'signal_conflict_resolution'
                }
            ),
            weight=0.3,
            risk_score=0.2,  # 持有信号风险最低
            expected_return=0.0
        )

        return hold_signal

    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        try:
            # 计算基本统计
            total_signals = len(self.signal_history)
            executed_signals = len([s for s in self.signal_history if s.status == SignalStatus.EXECUTED])
            rejected_signals = len([s for s in self.signal_history if s.status == SignalStatus.REJECTED])

            success_rate = executed_signals / total_signals if total_signals > 0 else 0

            # 计算平均持有时间（如果有可能）
            avg_hold_time = self._calculate_average_hold_time()

            # 计算风险调整回报
            sharpe_ratio = self._calculate_sharpe_ratio()

            report = {
                'timestamp': datetime.now().isoformat(),
                'performance_summary': {
                    'total_signals_generated': total_signals,
                    'signals_executed': executed_signals,
                    'signals_rejected': rejected_signals,
                    'success_rate': success_rate,
                    'average_hold_time_minutes': avg_hold_time,
                    'sharpe_ratio': sharpe_ratio
                },
                'strategy_performance': self._get_strategy_performance(),
                'risk_metrics': self._get_risk_metrics(),
                'recent_activity': self._get_recent_activity()
            }

            return report

        except Exception as e:
            logger.error(f"性能报告生成失败: {e}")
            return {'error': str(e)}

    def _calculate_average_hold_time(self) -> float:
        """计算平均持有时间"""
        try:
            executed_signals = [s for s in self.signal_history if s.status == SignalStatus.EXECUTED]
            if not executed_signals:
                return 0.0

            hold_times = []
            for signal in executed_signals:
                if hasattr(signal, 'execution_time') and hasattr(signal, 'completion_time'):
                    hold_time = (datetime.fromisoformat(signal.completion_time) -
                                 datetime.fromisoformat(signal.execution_time)).total_seconds() / 60
                    hold_times.append(hold_time)

            return np.mean(hold_times) if hold_times else 0.0

        except Exception as e:
            logger.error(f"平均持有时间计算失败: {e}")
            return 0.0

    def _calculate_sharpe_ratio(self) -> float:
        """计算夏普比率"""
        try:
            # 简化实现 - 实际中应该基于实际回报计算
            executed_signals = [s for s in self.signal_history
                                if s.status == SignalStatus.EXECUTED and hasattr(s, 'actual_return')]

            if len(executed_signals) < 10:
                return 0.0

            returns = [s.actual_return for s in executed_signals if hasattr(s, 'actual_return')]
            if not returns:
                return 0.0

            avg_return = np.mean(returns)
            std_return = np.std(returns)

            if std_return > 0:
                return avg_return / std_return * np.sqrt(252)  # 年化夏普比率
            return 0.0

        except Exception as e:
            logger.error(f"夏普比率计算失败: {e}")
            return 0.0

    def _get_strategy_performance(self) -> Dict[str, Any]:
        """获取各策略性能"""
        strategy_stats = {}

        try:
            for signal in self.signal_history:
                strategy_type = signal.metadata.parameters.get('type', 'unknown')
                if strategy_type not in strategy_stats:
                    strategy_stats[strategy_type] = {
                        'count': 0,
                        'executed': 0,
                        'successful': 0,
                        'total_return': 0.0,
                        'avg_confidence': 0.0
                    }

                stats = strategy_stats[strategy_type]
                stats['count'] += 1
                stats['avg_confidence'] += signal.metadata.confidence

                if signal.status == SignalStatus.EXECUTED:
                    stats['executed'] += 1
                    if hasattr(signal, 'actual_return') and signal.actual_return > 0:
                        stats['successful'] += 1
                    if hasattr(signal, 'actual_return'):
                        stats['total_return'] += signal.actual_return

            # 计算平均值和成功率
            for strategy, stats in strategy_stats.items():
                if stats['count'] > 0:
                    stats['avg_confidence'] /= stats['count']
                    stats['success_rate'] = stats['successful'] / stats['executed'] if stats['executed'] > 0 else 0
                else:
                    stats['success_rate'] = 0

            return strategy_stats

        except Exception as e:
            logger.error(f"策略性能统计失败: {e}")
            return {}

    def _get_risk_metrics(self) -> Dict[str, Any]:
        """获取风险指标"""
        try:
            executed_signals = [s for s in self.signal_history if s.status == SignalStatus.EXECUTED]

            if not executed_signals:
                return {}

            returns = [s.actual_return for s in executed_signals if hasattr(s, 'actual_return')]
            risk_scores = [s.risk_score for s in executed_signals]

            if not returns:
                return {}

            # 计算基本风险指标
            max_drawdown = min(returns) if returns else 0
            avg_risk_score = np.mean(risk_scores) if risk_scores else 0
            volatility = np.std(returns) if returns else 0

            return {
                'max_drawdown': max_drawdown,
                'average_risk_score': avg_risk_score,
                'volatility': volatility,
                'var_95': np.percentile(returns, 5) if len(returns) >= 10 else 0,  # 95% VaR
                'expected_shortfall': np.mean([r for r in returns if r <= np.percentile(returns, 5)]) if len(
                    returns) >= 10 else 0
            }

        except Exception as e:
            logger.error(f"风险指标计算失败: {e}")
            return {}

    def _get_recent_activity(self) -> Dict[str, Any]:
        """获取近期活动"""
        try:
            # 获取最近24小时的信号
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_signals = [s for s in self.signal_history
                              if datetime.fromisoformat(s.timestamp) >= cutoff_time]

            activity = {
                'signals_generated_24h': len(recent_signals),
                'top_strategies': self._get_top_strategies(recent_signals),
                'most_active_symbols': self._get_most_active_symbols(recent_signals),
                'recent_errors': len([e for e in self.error_log
                                      if datetime.fromisoformat(e['timestamp']) >= cutoff_time])
            }

            return activity

        except Exception as e:
            logger.error(f"近期活动统计失败: {e}")
            return {}

    def _get_top_strategies(self, signals: List[TradingSignal]) -> List[Dict[str, Any]]:
        """获取顶级策略"""
        strategy_counts = {}
        for signal in signals:
            strategy_type = signal.metadata.parameters.get('type', 'unknown')
            strategy_counts[strategy_type] = strategy_counts.get(strategy_type, 0) + 1

        top_strategies = sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return [{'strategy': s[0], 'count': s[1]} for s in top_strategies]

    def _get_most_active_symbols(self, signals: List[TradingSignal]) -> List[Dict[str, Any]]:
        """获取最活跃品种"""
        symbol_counts = {}
        for signal in signals:
            symbol_counts[signal.symbol] = symbol_counts.get(signal.symbol, 0) + 1

        top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return [{'symbol': s[0], 'count': s[1]} for s in top_symbols]

    def cleanup(self):
        """清理资源"""
        try:
            # 关闭线程池
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)

            # 清理缓存
            if hasattr(self, '_indicator_cache'):
                self._indicator_cache.clear()

            if hasattr(self, '_model_cache'):
                self._model_cache.clear()

            # 保存历史数据（可选）
            self._save_signal_history()

            logger.info("信号引擎资源清理完成")

        except Exception as e:
            logger.error(f"信号引擎资源清理失败: {e}")

    def _save_signal_history(self):
        """保存信号历史"""
        try:
            # 简化实现 - 实际中应该保存到数据库或文件
            history_file = f"signal_history_{datetime.now().strftime('%Y%m%d')}.json"
            history_data = [signal.to_dict() for signal in self.signal_history[-1000:]]  # 保存最近1000个信号

            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)

            logger.info(f"信号历史已保存: {history_file}")

        except Exception as e:
            logger.warning(f"信号历史保存失败: {e}")

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
def validate_signal(signal: TradingSignal) -> bool:
    """验证信号有效性"""
    try:
        required_fields = ['id', 'symbol', 'signal_type', 'price', 'timestamp']
        for field in required_fields:
            if not getattr(signal, field, None):
                return False

        # 验证价格合理性
        if signal.price <= 0:
            return False

        # 验证时间格式
        try:
            datetime.fromisoformat(signal.timestamp)
        except ValueError:
            return False

        return True

    except Exception:
        return False

def calculate_signal_quality(signal: TradingSignal, market_data: Dict) -> float:
    """计算信号质量评分"""
    try:
        quality_score = 0.0

        # 基于置信度
        quality_score += signal.metadata.confidence * 0.4

        # 基于风险评分（风险越低质量越高）
        quality_score += (1 - signal.risk_score) * 0.3

        # 基于流动性
        quality_score += signal.liquidity_score * 0.2

        # 基于波动率（适度波动率质量较高）
        optimal_volatility = 0.2  # 20%年化波动率
        vol_penalty = min(abs(signal.volatility - optimal_volatility) / optimal_volatility, 1.0)
        quality_score += (1 - vol_penalty) * 0.1

        return min(max(quality_score, 0.0), 1.0)

    except Exception:
        return 0.0

if __name__ == "__main__":
    # 测试代码
    config = {
        'signal_generation': {
            'technical_indicators': {
                'enabled_indicators': ['sma', 'ema', 'macd', 'rsi']
            },
            'quantitative_methods': {
                'enabled_methods': ['mean_reversion', 'momentum', 'breakout']
            },
            'validation_rules': {
                'min_confidence': 0.3,
                'max_risk_score': 0.7
            }
        }
    }

    # 创建信号引擎实例
    engine = SignalEngine(config)

    # 测试市场数据
    test_market_data = {
        'timestamp': datetime.now().isoformat(),
        'symbols': ['AAPL', 'GOOGL'],
        'prices': {
            'AAPL': {
                'open': [150.0, 151.0, 152.0],
                'high': [152.0, 153.0, 154.0],
                'low': [149.0, 150.0, 151.0],
                'close': [151.0, 152.0, 153.0]
            },
            'GOOGL': {
                'open': [2800.0, 2810.0, 2820.0],
                'high': [2820.0, 2830.0, 2840.0],
                'low': [2790.0, 2800.0, 2810.0],
                'close': [2810.0, 2820.0, 2830.0]
            }
        },
        'volumes': {
            'AAPL': {'volume': [1000000, 1100000, 1200000]},
            'GOOGL': {'volume': [500000, 550000, 600000]}
        }
    }

    # 生成信号
    signals = engine.process(test_market_data)
    print(f"生成信号数量: {sum(len(sigs) for sigs in signals.values())}")

    # 获取性能报告
    report = engine.get_performance_report()
    print("性能报告:", json.dumps(report, indent=2))

    # 清理
    engine.cleanup()