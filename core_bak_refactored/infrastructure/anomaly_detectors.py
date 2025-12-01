"""
异常检测器模块（纯技术实现） - 基础设施层

职责：
- 提供与业务无关的纯数学/统计异常检测算法
- 实现多种异常检测算法（统计+机器学习）
- 支持配置化启用/禁用

架构原则：
- 不包含任何业务领域概念
- 只接收纯数值数据
- 参数全部显式传入，不使用业务默认值
- 函数命名使用数学/统计术语，而非业务术语
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging

logger = logging.getLogger('DeepSeekQuant.Infrastructure.AnomalyDetectors')


@dataclass
class AnomalyResult:
    """异常检测结果"""
    anomaly_indices: List[int]  # 异常点索引
    anomaly_scores: List[float]  # 异常分数
    method: str  # 检测方法名称
    threshold: float  # 阈值
    metadata: Dict[str, Any]  # 额外元数据


class ZScoreDetector:
    """Z-Score异常检测器（3-sigma规则）"""
    
    def __init__(self, threshold: float = 3.0):
        """
        Args:
            threshold: Z-score阈值，默认3.0（3-sigma）
        """
        self.threshold = threshold
        self._mean: Optional[float] = None
        self._std: Optional[float] = None
    
    @property
    def name(self) -> str:
        return 'z_score'
    
    def fit(self, data: np.ndarray) -> None:
        """计算均值和标准差"""
        if data.ndim > 1:
            data = data.ravel()
        self._mean = np.mean(data)
        self._std = np.std(data)
    
    def detect(self, data: np.ndarray) -> AnomalyResult:
        """检测Z-score超过阈值的点"""
        if data.ndim > 1:
            data = data.ravel()
        
        # 如果未fit，使用当前数据的统计量
        mean = self._mean if self._mean is not None else np.mean(data)
        std = self._std if self._std is not None else np.std(data)
        
        if std < 1e-8:
            # 标准差接近0，无异常
            return AnomalyResult(
                anomaly_indices=[],
                anomaly_scores=[],
                method=self.name,
                threshold=self.threshold,
                metadata={'mean': float(mean), 'std': float(std)}
            )
        
        z_scores = np.abs((data - mean) / std)
        anomaly_mask = z_scores > self.threshold
        anomaly_indices = np.where(anomaly_mask)[0].tolist()
        anomaly_scores = z_scores[anomaly_mask].tolist()
        
        return AnomalyResult(
            anomaly_indices=anomaly_indices,
            anomaly_scores=anomaly_scores,
            method=self.name,
            threshold=self.threshold,
            metadata={'mean': float(mean), 'std': float(std)}
        )


class IQRDetector:
    """IQR（四分位距）异常检测器"""
    
    def __init__(self, multiplier: float = 1.5):
        """
        Args:
            multiplier: IQR倍数，默认1.5（Tukey's fence）
        """
        self.multiplier = multiplier
        self._q1: Optional[float] = None
        self._q3: Optional[float] = None
        self._iqr: Optional[float] = None
    
    @property
    def name(self) -> str:
        return 'iqr'
    
    def fit(self, data: np.ndarray) -> None:
        """计算四分位数"""
        if data.ndim > 1:
            data = data.ravel()
        self._q1 = np.percentile(data, 25)
        self._q3 = np.percentile(data, 75)
        self._iqr = self._q3 - self._q1
    
    def detect(self, data: np.ndarray) -> AnomalyResult:
        """检测超出Tukey's fence的点"""
        if data.ndim > 1:
            data = data.ravel()
        
        # 如果未fit，使用当前数据的统计量
        q1 = self._q1 if self._q1 is not None else np.percentile(data, 25)
        q3 = self._q3 if self._q3 is not None else np.percentile(data, 75)
        iqr = self._iqr if self._iqr is not None else (q3 - q1)
        
        if iqr < 1e-8:
            # IQR接近0，无异常
            return AnomalyResult(
                anomaly_indices=[],
                anomaly_scores=[],
                method=self.name,
                threshold=self.multiplier,
                metadata={'q1': float(q1), 'q3': float(q3), 'iqr': float(iqr)}
            )
        
        lower_bound = q1 - self.multiplier * iqr
        upper_bound = q3 + self.multiplier * iqr
        
        anomaly_mask = (data < lower_bound) | (data > upper_bound)
        anomaly_indices = np.where(anomaly_mask)[0].tolist()
        
        # 计算异常分数（距离边界的IQR倍数）
        anomaly_scores = []
        for idx in anomaly_indices:
            if data[idx] < lower_bound:
                score = (lower_bound - data[idx]) / iqr
            else:
                score = (data[idx] - upper_bound) / iqr
            anomaly_scores.append(float(score))
        
        return AnomalyResult(
            anomaly_indices=anomaly_indices,
            anomaly_scores=anomaly_scores,
            method=self.name,
            threshold=self.multiplier,
            metadata={
                'q1': float(q1),
                'q3': float(q3),
                'iqr': float(iqr),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
        )


class RollingStdDetector:
    """滚动标准差异常检测器"""
    
    def __init__(self, window: int = 20, multiplier: float = 2.0):
        """
        Args:
            window: 滚动窗口大小
            multiplier: 标准差倍数
        """
        self.window = window
        self.multiplier = multiplier
    
    @property
    def name(self) -> str:
        return 'rolling_std'
    
    def fit(self, data: np.ndarray) -> None:
        """无需训练"""
        pass
    
    def detect(self, data: np.ndarray) -> AnomalyResult:
        """检测超出滚动均值±N倍滚动标准差的点"""
        if data.ndim > 1:
            data = data.ravel()
        
        if len(data) < self.window:
            return AnomalyResult(
                anomaly_indices=[],
                anomaly_scores=[],
                method=self.name,
                threshold=self.multiplier,
                metadata={'data_length': len(data), 'required_window': self.window}
            )
        
        # 使用pandas计算滚动统计量
        series = pd.Series(data)
        rolling_mean = series.rolling(window=self.window, center=True).mean()
        rolling_std = series.rolling(window=self.window, center=True).std()
        
        # 填充边界NaN值（使用全局统计量）
        rolling_mean = rolling_mean.fillna(series.mean())
        rolling_std = rolling_std.fillna(series.std())
        
        # 计算偏离度
        deviations = np.abs((data - rolling_mean.values) / (rolling_std.values + 1e-8))
        anomaly_mask = deviations > self.multiplier
        anomaly_indices = np.where(anomaly_mask)[0].tolist()
        anomaly_scores = deviations[anomaly_mask].tolist()
        
        return AnomalyResult(
            anomaly_indices=anomaly_indices,
            anomaly_scores=anomaly_scores,
            method=self.name,
            threshold=self.multiplier,
            metadata={'window': self.window}
        )


class AnomalyDetectionManager:
    """异常检测管理器（纯技术实现）"""
    
    def __init__(self, detectors: List[Any]):
        """
        Args:
            detectors: 异常检测器列表
        """
        self.detectors = detectors
    
    def detect_all(self, data: np.ndarray) -> Dict[str, AnomalyResult]:
        """使用所有检测器检测异常
        
        Args:
            data: 待检测数据
            
        Returns:
            字典，key为检测器名称，value为检测结果
        """
        results = {}
        for detector in self.detectors:
            try:
                result = detector.detect(data)
                results[detector.name] = result
            except Exception as e:
                logger.error(f"检测器{detector.name}执行失败: {e}")
        
        return results
    
    def get_aggregated_anomalies(self, 
                                data: np.ndarray,
                                min_votes: int = 2) -> List[int]:
        """聚合多个检测器结果，返回投票>=min_votes的异常点
        
        Args:
            data: 待检测数据
            min_votes: 最小投票数
            
        Returns:
            异常点索引列表
        """
        all_results = self.detect_all(data)
        
        if not all_results:
            return []
        
        # 统计每个索引的投票数
        vote_counts: Dict[int, int] = {}
        for result in all_results.values():
            for idx in result.anomaly_indices:
                vote_counts[idx] = vote_counts.get(idx, 0) + 1
        
        # 返回投票数>=min_votes的索引
        aggregated = [idx for idx, votes in vote_counts.items() if votes >= min_votes]
        aggregated.sort()
        
        return aggregated