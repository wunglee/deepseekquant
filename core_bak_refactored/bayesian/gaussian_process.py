"""
高斯过程模块
从 core_bak/bayesian_optimizer.py 拆分
职责: 高斯过程回归器的创建、训练和序列化
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
import pickle
import logging
from typing import Any, Optional

logger = logging.getLogger('DeepSeekQuant.GaussianProcess')


class GaussianProcessModel:
    """高斯过程模型封装"""
    
    def __init__(self, noise_level: float = 0.1, random_seed: Optional[int] = None):
        """
        初始化高斯过程模型
        
        Args:
            noise_level: 噪声水平
            random_seed: 随机种子
        """
        self.noise_level = noise_level
        self.random_seed = random_seed
        self.gp_model = self._create_gaussian_process()
        self.is_trained = False
        
    def _create_gaussian_process(self) -> GaussianProcessRegressor:
        """创建高斯过程回归器"""
        # 核函数配置
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
        
        Args:
            X: 训练特征
            y: 训练标签
            
        Returns:
            模型质量评分
        """
        try:
            # 训练模型
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.gp_model.fit(X, y)
            
            self.is_trained = True
            
            # 评估模型质量
            quality = self._evaluate_model_quality(X, y)
            return quality
            
        except Exception as e:
            logger.error(f"高斯过程训练失败: {e}")
            raise
    
    def _evaluate_model_quality(self, X: np.ndarray, y: np.ndarray) -> float:
        """评估模型质量"""
        try:
            # 使用交叉验证评估模型
            if len(y) >= 5:
                scores = cross_val_score(
                    self.gp_model, X, y,
                    cv=min(5, len(y)),
                    scoring='neg_mean_squared_error'
                )
                return float(np.mean(scores))
            else:
                return 0.0
        except:
            return 0.0
    
    def predict(self, X: np.ndarray, return_std: bool = False):
        """
        预测
        
        Args:
            X: 输入特征
            return_std: 是否返回标准差
            
        Returns:
            预测值（和标准差）
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.gp_model.predict(X, return_std=return_std)
    
    def serialize(self) -> bytes:
        """序列化高斯过程模型"""
        try:
            model_data = {
                'kernel': self.gp_model.kernel_,
                'alpha': self.gp_model.alpha,
                'X_train': self.gp_model.X_train_,
                'y_train': self.gp_model.y_train_,
                'noise_level': self.noise_level
            }
            return pickle.dumps(model_data)
        except Exception as e:
            logger.error(f"模型序列化失败: {e}")
            return b''
    
    def deserialize(self, data: bytes):
        """反序列化高斯过程模型"""
        try:
            model_data = pickle.loads(data)
            
            # 重建模型
            self.gp_model.kernel_ = model_data['kernel']
            self.gp_model.alpha = model_data['alpha']
            self.gp_model.X_train_ = model_data['X_train']
            self.gp_model.y_train_ = model_data['y_train']
            self.noise_level = model_data.get('noise_level', 0.1)
            self.is_trained = True
            
        except Exception as e:
            logger.error(f"模型反序列化失败: {e}")
            raise
