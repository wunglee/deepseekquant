"""
高斯过程模型 - 基础设施层
从 core_bak/bayesian_optimizer.py 拆分
职责: 提供通用的高斯过程回归功能
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
import pickle
from typing import Tuple, Optional
import logging

logger = logging.getLogger('DeepSeekQuant.Infrastructure.GaussianProcess')


class GaussianProcessModel:
    """高斯过程回归模型"""
    
    def __init__(self, noise_level: float = 0.1, random_seed: Optional[int] = None):
        """
        初始化高斯过程模型
        
        Args:
            noise_level: 噪声水平
            random_seed: 随机种子
        """
        self.noise_level = noise_level
        self.random_seed = random_seed
        self.gp = self._create_gaussian_process()
        self.is_fitted = False
        
    def _create_gaussian_process(self) -> GaussianProcessRegressor:
        """
        创建高斯过程回归器
        从 core_bak/bayesian_optimizer.py:_create_gaussian_process 提取
        """
        kernel = ConstantKernel(1.0) * Matern(
            length_scale=1.0,
            nu=2.5
        ) + WhiteKernel(noise_level=self.noise_level)

        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self.random_seed
        )
    
    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        训练高斯过程模型
        从 core_bak/bayesian_optimizer.py:_train_gaussian_process 提取
        
        Args:
            X: 训练特征
            y: 训练标签
            
        Returns:
            模型质量评分
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.gp.fit(X, y)
            
            self.is_fitted = True
            quality = self._evaluate_model_quality(X, y)
            return quality
            
        except Exception as e:
            logger.error(f"高斯过程训练失败: {e}")
            raise
    
    def _evaluate_model_quality(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        评估模型质量
        从 core_bak/bayesian_optimizer.py:_evaluate_model_quality 提取
        """
        try:
            if len(y) >= 5:
                scores = cross_val_score(
                    self.gp, X, y,
                    cv=min(5, len(y)),
                    scoring='neg_mean_squared_error'
                )
                return float(np.mean(scores))
            else:
                return 0.0
        except:
            return 0.0
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple:
        """
        预测
        
        Args:
            X: 输入特征
            return_std: 是否返回标准差
            
        Returns:
            预测值（和标准差）
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.gp.predict(X, return_std=return_std)
    
    def serialize(self) -> bytes:
        """
        序列化模型
        从 core_bak/bayesian_optimizer.py:_serialize_gp_model 提取
        """
        try:
            model_data = {
                'kernel': self.gp.kernel_,
                'alpha': self.gp.alpha,
                'X_train': self.gp.X_train_,
                'y_train': self.gp.y_train_,
                'noise_level': self.noise_level
            }
            return pickle.dumps(model_data)
        except Exception as e:
            logger.error(f"模型序列化失败: {e}")
            return b''
    
    def deserialize(self, data: bytes):
        """
        反序列化模型
        从 core_bak/bayesian_optimizer.py:_deserialize_gp_model 提取
        """
        try:
            model_data = pickle.loads(data)
            self.gp.kernel_ = model_data['kernel']
            self.gp.alpha = model_data['alpha']
            self.gp.X_train_ = model_data['X_train']
            self.gp.y_train_ = model_data['y_train']
            self.noise_level = model_data.get('noise_level', 0.1)
            self.is_fitted = True
        except Exception as e:
            logger.error(f"模型反序列化失败: {e}")
            raise
