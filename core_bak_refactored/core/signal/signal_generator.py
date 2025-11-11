"""
信号生成器 - 业务层
从 core_bak/signal_engine.py 拆分
职责: 基于技术指标和策略生成交易信号
"""

import pandas as pd
from typing import Dict, List, Optional
import logging

from .signal_models import TradingSignal, SignalType, SignalStrength, SignalSource
from ...infrastructure.technical_indicators import TechnicalIndicators

logger = logging.getLogger('DeepSeekQuant.SignalGenerator')


class SignalGenerator:
    """信号生成器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.indicators = TechnicalIndicators()
    
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


