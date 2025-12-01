"""ML异常检测器测试"""

import pytest
import numpy as np
import pandas as pd
from core_bak_refactored.core.data.quality.anomaly_detectors import (
    ZScoreDetector,
    IQRDetector,
    RollingStdDetector,
    AnomalyDetectorManager,
    AnomalyResult
)


class TestZScoreDetector:
    """测试Z-Score异常检测器"""
    
    def test_detect_no_anomalies(self):
        """测试无异常情况"""
        detector = ZScoreDetector(threshold=3.0)
        data = np.random.normal(100, 10, 100)
        
        result = detector.detect(data)
        
        assert isinstance(result, AnomalyResult)
        assert result.method == 'z_score'
        assert len(result.anomaly_indices) <= 5  # 正常情况下3-sigma异常极少
    
    def test_detect_with_anomalies(self):
        """测试包含异常点"""
        detector = ZScoreDetector(threshold=3.0)
        data = np.random.normal(100, 10, 100)
        data[50] = 200  # 添加明显异常点
        
        result = detector.detect(data)
        
        assert len(result.anomaly_indices) >= 1
        assert 50 in result.anomaly_indices
    
    def test_fit_and_detect(self):
        """测试预训练模型"""
        detector = ZScoreDetector(threshold=3.0)
        train_data = np.random.normal(100, 10, 100)
        
        detector.fit(train_data)
        
        test_data = np.array([100, 100, 200, 100])  # 200是异常点
        result = detector.detect(test_data)
        
        assert len(result.anomaly_indices) >= 1


class TestIQRDetector:
    """测试IQR异常检测器"""
    
    def test_detect_outliers(self):
        """测试检测离群值"""
        detector = IQRDetector(multiplier=1.5)
        data = np.random.normal(10, 1, 50); data = np.append(data, 100)  # 100是明显离群值
        
        result = detector.detect(data)
        
        assert len(result.anomaly_indices) >= 1
        assert 50 in result.anomaly_indices
        assert result.metadata['iqr'] > 0
    
    def test_no_variation(self):
        """测试无变化数据"""
        detector = IQRDetector(multiplier=1.5)
        data = np.array([100] * 50)
        
        result = detector.detect(data)
        
        assert len(result.anomaly_indices) == 0  # IQR=0，无异常


class TestRollingStdDetector:
    """测试移动标准差检测器"""
    
    def test_detect_with_sufficient_data(self):
        """测试数据点足够"""
        detector = RollingStdDetector(window=10, multiplier=2.0)
        data = np.random.normal(100, 5, 50)
        data[25] = 150  # 添加异常点
        
        result = detector.detect(data)
        
        assert isinstance(result, AnomalyResult)
        # 应该检测到异常点
        assert len(result.anomaly_indices) >= 1
    
    def test_detect_with_insufficient_data(self):
        """测试数据点不足"""
        detector = RollingStdDetector(window=20, multiplier=2.0)
        data = np.array([100, 100, 100])  # 只有3个点
        
        result = detector.detect(data)
        
        assert len(result.anomaly_indices) == 0
        assert 'required_window' in result.metadata


class TestAnomalyDetectorManager:
    """测试异常检测器管理器"""
    
    def test_initialization_default_config(self):
        """测试默认配置初始化"""
        manager = AnomalyDetectorManager()
        
        # 默认启用统计方法
        assert len(manager.statistical_detectors) >= 3
        # 检查是否有Z-Score检测器
        has_zscore = any(hasattr(detector, 'name') and detector.name == 'z_score' for detector in manager.statistical_detectors)
        assert has_zscore
        # 检查是否有IQR检测器
        has_iqr = any(hasattr(detector, 'name') and detector.name == 'iqr' for detector in manager.statistical_detectors)
        assert has_iqr
    
    def test_initialization_custom_config(self):
        """测试自定义配置"""
        config = {
            'z_score': {'enabled': True, 'threshold': 2.5},
            'iqr': {'enabled': False},
            'rolling_std': {'enabled': True, 'window': 15}
        }
        manager = AnomalyDetectorManager(config)
        
        # 检查统计检测器
        stat_detector_names = [detector.name for detector in manager.statistical_detectors]
        assert 'z_score' in stat_detector_names
        assert 'iqr' not in stat_detector_names
        assert 'rolling_std' in stat_detector_names
    
    def test_detect_all(self):
        """测试所有检测器"""
        manager = AnomalyDetectorManager()
        data = np.random.normal(100, 10, 50)
        data[25] = 200  # 添加异常点
        
        results = manager.detect_all(data)
        
        assert len(results) >= 3  # 至少有3个统计方法
        assert all(isinstance(r, AnomalyResult) for r in results.values())
    
    def test_get_aggregated_anomalies(self):
        """测试聚合异常检测"""
        manager = AnomalyDetectorManager()
        data = np.random.normal(100, 10, 50)
        data[25] = 200  # 添加明显异常点
        
        aggregated = manager.get_aggregated_anomalies(data, min_votes=2)
        
        # 多个检测器应该都检测到第25个点
        assert isinstance(aggregated, list)
        assert len(aggregated) >= 1


# 测试DataQualityChecker中ML检测集成
class TestDataQualityCheckerMLIntegration:
    """测试DataQualityChecker的ML检测集成"""
    
    def test_ml_detection_disabled(self):
        """测试ML检测禁用状态"""
        from core_bak_refactored.core.data.quality import DataQualityChecker
        
        checker = DataQualityChecker(enable_ml_detection=False)
        data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100),
            'close': np.random.normal(100, 10, 100)
        })
        
        result = checker.detect_anomalies_ml(data)
        
        assert result['enabled'] == False
    
    def test_ml_detection_enabled(self):
        """测试ML检测启用"""
        from core_bak_refactored.core.data.quality import DataQualityChecker
        
        ml_config = {
            'z_score': {'enabled': True, 'threshold': 3.0},
            'iqr': {'enabled': True, 'multiplier': 1.5}
        }
        checker = DataQualityChecker(enable_ml_detection=True, ml_config=ml_config)
        
        data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100),
            'close': np.random.normal(100, 10, 100)
        })
        data.loc[50, 'close'] = 200  # 添加异常点
        
        result = checker.detect_anomalies_ml(data, target_column='close', min_votes=1)
        
        assert result['enabled'] == True
        assert 'total_points' in result
        assert 'anomaly_count' in result
        assert 'detector_results' in result
    
    def test_ml_detection_insufficient_data(self):
        """测试数据不足"""
        from core_bak_refactored.core.data.quality import DataQualityChecker
        
        checker = DataQualityChecker(enable_ml_detection=True)
        data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=5),
            'close': [100, 100, 100, 100, 100]
        })
        
        result = checker.detect_anomalies_ml(data)
        
        assert result['enabled'] == True
        assert result.get('reason') == 'insufficient_data'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
