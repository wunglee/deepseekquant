"""
信号生成器 - 业务层
从 core_bak/signal_engine.py 拆分
职责: 基于技术指标和策略生成交易信号
"""

from typing import Dict, List
import logging
from datetime import datetime
from collections import defaultdict

from .signal_models import TradingSignal, SignalType, SignalStrength, SignalSource, SignalMetadata
# from infrastructure.technical_indicators import TechnicalIndicators  # TODO: 待实现

logger = logging.getLogger('DeepSeekQuant.SignalGenerator')


class SignalGenerator:
    """信号生成器"""
    
    def __init__(self, config: Dict):
        self.config = config
        # self.indicators = TechnicalIndicators()  # TODO: 待实现
        
        # 配置项（按专家方案）
        self.technical_config = config.get('technical_config', {})
        self.quantitative_config = config.get('quantitative_config', {})
        self.ml_config = config.get('ml_config', {})
        self.signal_config = config.get('signal_config', {})
    
    def generate_signals(self, market_data: Dict) -> Dict[str, List[TradingSignal]]:
        """生成交易信号"""
        signals = {}
        
        symbols = market_data.get('symbols', [])
        
        for symbol in symbols:
            try:
                symbol_signals = self._generate_symbol_signals(symbol, market_data)
                if symbol_signals:
                    signals[symbol] = symbol_signals
            except Exception as e:
                logger.error(f"信号生成失败 {symbol}: {e}")
        
        return signals
    
    def _generate_symbol_signals(self, symbol: str, market_data: Dict) -> List[TradingSignal]:
        """生成单个品种的交易信号"""
        signals = []
        
        try:
            price_data = market_data.get('prices', {}).get(symbol, {})
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
    
    def _generate_technical_signals(self, symbol: str, price_data: Dict, volume_data: Dict) -> List[TradingSignal]:
        """生成技术指标信号"""
        signals = []
        
        # 简化实现 - 实际需要调用TechnicalIndicators
        enabled_indicators = self.technical_config.get('enabled_indicators', [])
        
        # TODO: 实现具体的技术指标信号生成逻辑
        # 当前返回空列表，避免破坏测试
        
        return signals
    
    def _generate_quantitative_signals(self, symbol: str, price_data: Dict, market_data: Dict) -> List[TradingSignal]:
        """生成量化策略信号"""
        signals = []
        
        # 简化实现
        enabled_methods = self.quantitative_config.get('enabled_methods', [])
        
        # TODO: 实现量化策略信号生成逻辑
        
        return signals
    
    def _generate_ml_signals(self, symbol: str, price_data: Dict, fundamental_data: Dict) -> List[TradingSignal]:
        """生成机器学习信号"""
        signals = []
        
        if not self.ml_config.get('enabled', False):
            return signals
        
        # TODO: 实现机器学习模型预测逻辑
        
        return signals
    
    def _generate_composite_signals(self, symbol: str, existing_signals: List[TradingSignal], market_data: Dict) -> List[TradingSignal]:
        """生成复合信号"""
        composite_signals = []
        
        try:
            # 按信号类型分组
            signal_groups = defaultdict(list)
            for signal in existing_signals:
                key = (signal.signal_type, signal.timeframe)
                signal_groups[key].append(signal)
            
            # 为符合阈值的组创建复合信号
            composite_threshold = self.signal_config.get('composite_threshold', 3)
            for (signal_type, timeframe), signal_list in signal_groups.items():
                if len(signal_list) >= composite_threshold:
                    composite_signal = self._create_composite_signal(
                        symbol, signal_type, timeframe, signal_list, market_data
                    )
                    if composite_signal:
                        composite_signals.append(composite_signal)
        
        except Exception as e:
            logger.error(f"复合信号生成失败 {symbol}: {e}")
        
        return composite_signals
    
    def _create_composite_signal(self, symbol: str, signal_type: SignalType, timeframe: str, signals: List[TradingSignal], market_data: Dict) -> TradingSignal:
        """创建复合信号"""
        try:
            # 计算加权平均
            total_weight = sum(signal.weight for signal in signals)
            if total_weight <= 0:
                return None
            
            weighted_price = sum(signal.price * signal.weight for signal in signals) / total_weight
            weighted_confidence = sum(signal.metadata.confidence * signal.weight for signal in signals) / total_weight
            
            # 使用置信度最高的信号作为基础
            base_signal = max(signals, key=lambda s: s.metadata.confidence)
            
            return TradingSignal(
                id=f"composite_{symbol}_{int(datetime.now().timestamp())}",
                symbol=symbol,
                signal_type=signal_type,
                price=weighted_price,
                timestamp=datetime.now().isoformat(),
                metadata=SignalMetadata(
                    source=SignalSource.COMPOSITE,
                    confidence=weighted_confidence,
                    strength=SignalStrength.STRONG,
                    tags=['composite', 'aggregated']
                ),
                weight=total_weight / len(signals),
                timeframe=timeframe
            )
        
        except Exception as e:
            logger.error(f"复合信号创建失败 {symbol}: {e}")
            return None
    
    def _create_strong_signal(self, signal_type: SignalType, signals: List[TradingSignal]) -> TradingSignal:
        """创建强势信号"""
        if not signals:
            return None
        
        # 选择置信度最高的信号
        best_signal = max(signals, key=lambda s: s.metadata.confidence)
        
        return TradingSignal(
            id=f"strong_{best_signal.symbol}_{int(datetime.now().timestamp())}",
            symbol=best_signal.symbol,
            signal_type=signal_type,
            price=best_signal.price,
            timestamp=datetime.now().isoformat(),
            metadata=SignalMetadata(
                source=SignalSource.COMPOSITE,
                confidence=min(best_signal.metadata.confidence * 1.2, 0.95),
                strength=SignalStrength.VERY_STRONG,
                parameters={'base_signals': [s.id for s in signals]}
            ),
            weight=1.0,
            risk_score=best_signal.risk_score * 0.8,
            expected_return=best_signal.expected_return * 1.5
        )
    
    def _create_hold_signal(self, signals: List[TradingSignal]) -> TradingSignal:
        """创建持有信号"""
        if not signals:
            return None
        
        base_signal = signals[0]
        
        return TradingSignal(
            id=f"hold_{base_signal.symbol}_{int(datetime.now().timestamp())}",
            symbol=base_signal.symbol,
            signal_type=SignalType.HOLD,
            price=base_signal.price,
            timestamp=datetime.now().isoformat(),
            metadata=SignalMetadata(
                source=SignalSource.COMPOSITE,
                confidence=0.5,
                strength=SignalStrength.WEAK,
                parameters={'conflicting_signals': [s.id for s in signals]}
            ),
            weight=0.3,
            risk_score=0.2,
            expected_return=0.0
        )
    
    def _filter_duplicate_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """过滤重复信号"""
        # 简单去重：基于信号ID
        seen_ids = set()
        unique_signals = []
        
        for signal in signals:
            if signal.id not in seen_ids:
                seen_ids.add(signal.id)
                unique_signals.append(signal)
        
        return unique_signals
