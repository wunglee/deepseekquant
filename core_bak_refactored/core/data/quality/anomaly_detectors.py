"""异常检测器模块（ML方法剥离版）

职责：
- 提供可插拔的ML异常检测接口
- 实现多种异常检测算法（统计+机器学习）
- 支持配置化启用/禁用

设计原则：
- Protocol接口：支持依赖注入
- 可选依赖：ML库仅在启用时加载
- 独立性：不依赖data_fetcher
- 可扩展：易于添加新检测器

已实现的检测器：
- 统计方法：ZScoreDetector, IQRDetector, RollingStdDetector
- ML方法：IsolationForestDetector, LOFDetector

TODO - 未实现的检测器：
- AutoencoderDetector: 虽然data_fetcher.py有配置声明，但实际实现从未完成
  适用于高维特征的复杂异常模式，需要keras/tensorflow
"""

import logging
from typing import List, Dict, Any, Optional

import numpy as np

from core_bak_refactored.infrastructure import ZScoreDetector, IQRDetector, RollingStdDetector, AnomalyDetectionManager, \
    AnomalyResult

logger = logging.getLogger('DeepSeekQuant.AnomalyDetectors')


# ========== ML方法检测器（可选依赖） ==========

class IsolationForestDetector:
    """Isolation Forest异常检测器（需要sklearn）"""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Args:
            contamination: 异常比例估计
            random_state: 随机种子
        """
        self.contamination = contamination
        self.random_state = random_state
        self._model = None
        
        try:
            from sklearn.ensemble import IsolationForest
            self._IsolationForest = IsolationForest
        except ImportError:
            logger.warning("sklearn未安装，IsolationForestDetector不可用")
            self._IsolationForest = None
    
    @property
    def name(self) -> str:
        return 'isolation_forest'
    
    def fit(self, data: np.ndarray) -> None:
        """训练Isolation Forest模型"""
        if self._IsolationForest is None:
            raise ImportError("需要安装sklearn: pip install scikit-learn")
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        self._model = self._IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state
        )
        self._model.fit(data)
        logger.debug(f"IsolationForest已训练 (n_samples={len(data)})")
    
    def detect(self, data: np.ndarray) -> AnomalyResult:
        """检测异常点"""
        if self._IsolationForest is None:
            raise ImportError("需要安装sklearn: pip install scikit-learn")
        
        if self._model is None:
            # 如果未训练，先训练
            self.fit(data)
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # 预测：-1表示异常，1表示正常
        predictions = self._model.predict(data)
        scores = -self._model.score_samples(data)  # 负分数越大越异常
        
        anomaly_mask = predictions == -1
        anomaly_indices = np.where(anomaly_mask)[0].tolist()
        anomaly_scores = scores[anomaly_mask].tolist()
        
        logger.debug(f"IsolationForest检测: 发现{len(anomaly_indices)}个异常点 "
                    f"(contamination={self.contamination})")
        
        return AnomalyResult(
            anomaly_indices=anomaly_indices,
            anomaly_scores=anomaly_scores,
            method=self.name,
            threshold=self.contamination,
            metadata={'n_estimators': getattr(self._model, 'n_estimators', 100)}
        )


class LOFDetector:
    """LOF（局部异常因子）检测器（需要sklearn）"""
    
    def __init__(self, n_neighbors: int = 20, contamination: float = 0.1):
        """
        Args:
            n_neighbors: 邻居数量
            contamination: 异常比例估计
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self._model = None
        
        try:
            from sklearn.neighbors import LocalOutlierFactor
            self._LocalOutlierFactor = LocalOutlierFactor
        except ImportError:
            logger.warning("sklearn未安装，LOFDetector不可用")
            self._LocalOutlierFactor = None
    
    @property
    def name(self) -> str:
        return 'lof'
    
    def fit(self, data: np.ndarray) -> None:
        """LOF是非参数方法，无需预训练"""
        pass
    
    def detect(self, data: np.ndarray) -> AnomalyResult:
        """检测异常点"""
        if self._LocalOutlierFactor is None:
            raise ImportError("需要安装sklearn: pip install scikit-learn")
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # LOF使用fit_predict进行检测
        model = self._LocalOutlierFactor(
            n_neighbors=min(self.n_neighbors, len(data) - 1),
            contamination=self.contamination
        )
        predictions = model.fit_predict(data)
        scores = -model.negative_outlier_factor_  # 负分数越大越异常
        
        anomaly_mask = predictions == -1
        anomaly_indices = np.where(anomaly_mask)[0].tolist()
        anomaly_scores = scores[anomaly_mask].tolist()
        
        logger.debug(f"LOF检测: 发现{len(anomaly_indices)}个异常点 "
                    f"(n_neighbors={self.n_neighbors})")
        
        return AnomalyResult(
            anomaly_indices=anomaly_indices,
            anomaly_scores=anomaly_scores,
            method=self.name,
            threshold=self.contamination,
            metadata={'n_neighbors': self.n_neighbors}
        )


# ========== 异常检测器管理器 ==========

class AnomalyDetectorManager:
    """异常检测器管理器
    
    职责：
    - 管理多个检测器
    - 配置化启用/禁用
    - 结果聚合
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置字典，示例：
                {
                    'z_score': {'enabled': True, 'threshold': 3.0},
                    'iqr': {'enabled': True, 'multiplier': 1.5},
                    'isolation_forest': {'enabled': False, 'contamination': 0.1}
                }
        """
        self.config = config or {}
        self.statistical_detectors = []
        self.ml_detectors: Dict[str, Any] = {}
        self._initialize_detectors()
        self.anomaly_detection_manager = AnomalyDetectionManager(self.statistical_detectors)
    
    def _initialize_detectors(self) -> None:
        """根据配置初始化检测器"""
        # Z-Score
        z_config = self.config.get('z_score', {})
        if z_config.get('enabled', True):
            self.statistical_detectors.append(ZScoreDetector(
                threshold=z_config.get('threshold', 3.0)
            ))
        
        # IQR
        iqr_config = self.config.get('iqr', {})
        if iqr_config.get('enabled', True):
            self.statistical_detectors.append(IQRDetector(
                multiplier=iqr_config.get('multiplier', 1.5)
            ))
        
        # Rolling Std
        rolling_config = self.config.get('rolling_std', {})
        if rolling_config.get('enabled', True):
            self.statistical_detectors.append(RollingStdDetector(
                window=rolling_config.get('window', 20),
                multiplier=rolling_config.get('multiplier', 2.0)
            ))
        
        # Isolation Forest（可选ML方法）
        if_config = self.config.get('isolation_forest', {})
        if if_config.get('enabled', False):
            try:
                self.ml_detectors['isolation_forest'] = IsolationForestDetector(
                    contamination=if_config.get('contamination', 0.1)
                )
            except ImportError as e:
                logger.warning(f"无法初始化IsolationForestDetector: {e}")
        
        # LOF（可选ML方法）
        lof_config = self.config.get('lof', {})
        if lof_config.get('enabled', False):
            try:
                self.ml_detectors['lof'] = LOFDetector(
                    n_neighbors=lof_config.get('n_neighbors', 20),
                    contamination=lof_config.get('contamination', 0.1)
                )
            except ImportError as e:
                logger.warning(f"无法初始化LOFDetector: {e}")
        
        # TODO: Autoencoder异常检测（未实现）
        # 说明：虽然data_fetcher.py中有配置声明，但实际实现从未完成
        # 建议配置格式：
        # autoencoder_config = self.config.get('autoencoder', {})
        # if autoencoder_config.get('enabled', False):
        #     self.ml_detectors['autoencoder'] = AutoencoderDetector(
        #         threshold=autoencoder_config.get('threshold', 0.05),
        #         encoding_dim=autoencoder_config.get('encoding_dim', 32),
        #         epochs=autoencoder_config.get('epochs', 50)
        #     )
        # 参考实现：
        # - 使用keras/tensorflow构建自编码器
        # - 训练阶段学习正常数据模式
        # - 检测阶段计算重构误差，超过阈值为异常
        # - 适用于高维特征的复杂异常模式
        
        logger.info(f"已初始化{len(self.statistical_detectors)}个统计检测器和{len(self.ml_detectors)}个ML检测器")
    
    def detect_all(self, data: np.ndarray) -> Dict[str, AnomalyResult]:
        """使用所有检测器检测异常
        
        Args:
            data: 待检测数据
        
        Returns:
            字典，key为检测器名称，value为检测结果
        """
        # 使用基础设施层的统计检测器
        results = self.anomaly_detection_manager.detect_all(data)
        
        # 使用ML检测器
        for name, detector in self.ml_detectors.items():
            try:
                result = detector.detect(data)
                results[name] = result
            except Exception as e:
                logger.error(f"检测器{name}执行失败: {e}")
        
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
        
        logger.info(f"聚合检测: {len(all_results)}个检测器, "
                   f"发现{len(aggregated)}个异常点 (min_votes={min_votes})")
        
        return aggregated
