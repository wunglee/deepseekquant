"""
市场机制检测器

职责：检测不同市场的特殊机制（熔断、涨跌停、LULD等）
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MarketMechanismDetector(ABC):
    """市场机制检测器基类"""
    
    @abstractmethod
    def detect_anomalies(self, returns: pd.Series, prices: Optional[pd.Series] = None) -> Dict[str, Any]:
        """检测市场机制触发的异常"""
        raise NotImplementedError("子类必须实现此方法")
    
    def adjust_returns(self, returns: pd.Series, anomaly_type: str) -> pd.Series:
        """根据检测到的异常调整收益率"""
        # 默认实现：不调整
        return returns


class ChinaMarketDetector(MarketMechanismDetector):
    """A股市场机制检测器"""
    
    def __init__(self, config: Dict):
        self.limit_thresholds = config.get('limit_thresholds', {
            'main_board': 0.10,
            'gem': 0.20,
            'st': 0.05,
            'kcb': 0.20
        })
        self.detection_threshold = config.get('detection_threshold', 0.95)
    
    def detect_anomalies(self, returns: pd.Series, prices: Optional[pd.Series] = None) -> Dict[str, Any]:
        """检测A股市场异常"""
        anomalies = {}
        
        if returns is None or len(returns) == 0:
            return anomalies
        
        # 涨跌停检测
        for board_type, threshold in self.limit_thresholds.items():
            limit_hit = self._detect_limit_up_down(returns, threshold, board_type)
            if limit_hit:
                anomalies[f'limit_up_down_{board_type}'] = {
                    'type': 'limit_up_down',
                    'board_type': board_type,
                    'threshold': threshold,
                    'severity': 'high',
                    'count': limit_hit['count'],
                    'dates': limit_hit['dates']
                }
        
        return anomalies
    
    def _detect_limit_up_down(self, returns: pd.Series, threshold: float, board_type: str) -> Optional[Dict]:
        """检测涨跌停"""
        if len(returns) == 0:
            return None
        
        abs_returns = np.abs(returns.values)
        detection_level = self.detection_threshold * threshold
        
        limit_hits = abs_returns >= detection_level
        hit_count = np.sum(limit_hits)
        
        if hit_count > 0:
            hit_indices = np.where(limit_hits)[0]
            hit_dates = returns.index[hit_indices].tolist() if hasattr(returns, 'index') else hit_indices.tolist()
            
            return {
                'count': int(hit_count),
                'dates': hit_dates,
                'ratio': float(hit_count / len(returns))
            }
        
        return None


class USMarketDetector(MarketMechanismDetector):
    """美股市场机制检测器"""
    
    def __init__(self, config: Dict):
        self.circuit_breaker_levels = config.get('circuit_breaker_levels', [0.07, 0.13, 0.20])
        self.luld_threshold = config.get('luld_threshold', 0.05)
        self.luld_window = config.get('luld_window', 5)  # 5分钟窗口
        self.detection_threshold = config.get('detection_threshold', 0.95)
    
    def detect_anomalies(self, returns: pd.Series, prices: Optional[pd.Series] = None) -> Dict[str, Any]:
        """检测美股市场异常"""
        anomalies = {}
        
        if returns is None or len(returns) == 0:
            return anomalies
        
        # 熔断机制检测
        circuit_anomaly = self._detect_circuit_breaker(returns)
        if circuit_anomaly:
            anomalies['circuit_breaker'] = circuit_anomaly
        
        # 波动率中断检测 (LULD)
        if prices is not None and len(prices) > 0:
            luld_anomaly = self._detect_luld(returns, prices)
            if luld_anomaly:
                anomalies['luld'] = luld_anomaly
        
        # 盘后交易异常检测
        after_hours_anomaly = self._detect_after_hours_abnormalities(returns)
        if after_hours_anomaly:
            anomalies['after_hours'] = after_hours_anomaly
        
        return anomalies
    
    def _detect_circuit_breaker(self, returns: pd.Series) -> Optional[Dict[str, Any]]:
        """检测熔断机制触发"""
        if len(returns) == 0:
            return None
        
        abs_returns = np.abs(returns.values)
        
        for level in sorted(self.circuit_breaker_levels, reverse=True):
            detection_level = level * self.detection_threshold
            
            if np.any(abs_returns >= detection_level):
                hit_indices = np.where(abs_returns >= detection_level)[0]
                hit_dates = returns.index[hit_indices].tolist() if hasattr(returns, 'index') else hit_indices.tolist()
                
                return {
                    'type': 'circuit_breaker',
                    'level': float(level),
                    'severity': 'high' if level >= 0.13 else 'medium',
                    'count': len(hit_indices),
                    'dates': hit_dates
                }
        
        return None
    
    def _detect_luld(self, returns: pd.Series, prices: pd.Series) -> Optional[Dict[str, Any]]:
        """检测波动率中断 (Limit Up-Limit Down)"""
        if len(prices) < self.luld_window:
            return None
        
        # 计算滚动窗口内的价格变化
        rolling_return = prices.pct_change().rolling(window=self.luld_window).sum()
        
        luld_hits = np.abs(rolling_return) >= self.luld_threshold
        hit_count = luld_hits.sum()
        
        if hit_count > 0:
            hit_indices = np.where(luld_hits)[0]
            hit_dates = prices.index[hit_indices].tolist() if hasattr(prices, 'index') else hit_indices.tolist()
            
            return {
                'type': 'luld',
                'threshold': float(self.luld_threshold),
                'window': self.luld_window,
                'severity': 'medium',
                'count': int(hit_count),
                'dates': hit_dates
            }
        
        return None
    
    def _detect_after_hours_abnormalities(self, returns: pd.Series) -> Optional[Dict[str, Any]]:
        """检测盘后交易异常"""
        if len(returns) == 0:
            return None
        
        # 简化实现：检测异常大的收益率（盘后波动通常较大）
        large_threshold = 0.08  # 超过8%的收益率
        large_returns = returns[np.abs(returns) > large_threshold]
        abnormal_ratio = len(large_returns) / len(returns)
        
        if abnormal_ratio > 0.1:  # 超过10%的收益率异常
            return {
                'type': 'after_hours_abnormality',
                'abnormal_ratio': float(abnormal_ratio),
                'severity': 'low',
                'threshold': large_threshold,
                'count': len(large_returns)
            }
        
        return None


class HongKongMarketDetector(MarketMechanismDetector):
    """港股市场机制检测器"""
    
    def __init__(self, config: Dict):
        self.vcm_threshold = config.get('vcm_threshold', 0.10)  # 港股波动性调节机制
    
    def detect_anomalies(self, returns: pd.Series, prices: Optional[pd.Series] = None) -> Dict[str, Any]:
        """检测港股市场异常"""
        anomalies = {}
        
        if returns is None or len(returns) == 0:
            return anomalies
        
        # 检测VCM（波动性调节机制）触发
        vcm_anomaly = self._detect_vcm(returns)
        if vcm_anomaly:
            anomalies['vcm'] = vcm_anomaly
        
        return anomalies
    
    def _detect_vcm(self, returns: pd.Series) -> Optional[Dict[str, Any]]:
        """检测VCM（Volatility Control Mechanism）"""
        if len(returns) == 0:
            return None
        
        abs_returns = np.abs(returns.values)
        vcm_hits = abs_returns >= self.vcm_threshold * 0.95
        
        hit_count = np.sum(vcm_hits)
        if hit_count > 0:
            hit_indices = np.where(vcm_hits)[0]
            hit_dates = returns.index[hit_indices].tolist() if hasattr(returns, 'index') else hit_indices.tolist()
            
            return {
                'type': 'vcm',
                'threshold': float(self.vcm_threshold),
                'severity': 'medium',
                'count': int(hit_count),
                'dates': hit_dates
            }
        
        return None


class BaseMarketDetector(MarketMechanismDetector):
    """默认市场检测器（无特殊机制）"""
    
    def __init__(self, config: Dict):
        self.extreme_threshold = config.get('extreme_threshold', 0.10)
    
    def detect_anomalies(self, returns: pd.Series, prices: Optional[pd.Series] = None) -> Dict[str, Any]:
        """检测通用市场异常"""
        anomalies = {}
        
        if returns is None or len(returns) == 0:
            return anomalies
        
        # 仅检测极端波动
        extreme_anomaly = self._detect_extreme_movements(returns)
        if extreme_anomaly:
            anomalies['extreme_movement'] = extreme_anomaly
        
        return anomalies
    
    def _detect_extreme_movements(self, returns: pd.Series) -> Optional[Dict[str, Any]]:
        """检测极端价格波动"""
        if len(returns) == 0:
            return None
        
        abs_returns = np.abs(returns.values)
        extreme_hits = abs_returns >= self.extreme_threshold
        
        hit_count = np.sum(extreme_hits)
        if hit_count > 0:
            hit_indices = np.where(extreme_hits)[0]
            hit_dates = returns.index[hit_indices].tolist() if hasattr(returns, 'index') else hit_indices.tolist()
            
            return {
                'type': 'extreme_movement',
                'threshold': float(self.extreme_threshold),
                'severity': 'low',
                'count': int(hit_count),
                'dates': hit_dates
            }
        
        return None
