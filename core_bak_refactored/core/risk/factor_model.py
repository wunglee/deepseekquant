"""
因子模型协方差计算 - 业务层
实现Fama-French因子模型和混合协方差估计

专家建议：
- US市场: Fama-French 5因子 + 行业因子 (20个)
- CN市场: PCA统计因子 + 政策因子 (15个)
- 混合模型: alpha*因子协方差 + (1-alpha)*样本协方差
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger('DeepSeekQuant.FactorModel')


@dataclass
class FactorModelConfig:
    """因子模型配置"""
    market: str = 'US'  # 市场标识
    n_factors: int = 20  # 因子数量
    factor_types: List[str] = field(default_factory=lambda: ['market', 'size', 'value', 'profitability', 'investment'])
    shrinkage_alpha: float = 0.7  # 因子模型权重 (0.7*因子 + 0.3*样本)
    min_observations: int = 60  # 最小观测数
    use_industry_factors: bool = True  # 是否使用行业因子
    
    # Fama-French 5因子
    ff5_factors: List[str] = field(default_factory=lambda: [
        'Mkt-RF',  # 市场超额收益
        'SMB',     # Small Minus Big (规模因子)
        'HML',     # High Minus Low (价值因子)
        'RMW',     # Robust Minus Weak (盈利能力因子)
        'CMA'      # Conservative Minus Aggressive (投资因子)
    ])


class FactorModelEstimator:
    """因子模型协方差估计器"""
    
    def __init__(self, config: Optional[FactorModelConfig] = None):
        """
        初始化因子模型估计器
        
        Args:
            config: 因子模型配置
        """
        self.config = config or FactorModelConfig()
        self.factor_returns = None  # 因子收益序列
        self.factor_loadings = None  # 因子载荷矩阵 (N资产 x K因子)
        self.specific_variance = None  # 特质方差向量 (N,)
        self.factor_covariance = None  # 因子协方差矩阵 (K x K)
    
    def estimate_factor_loadings(
        self,
        returns: pd.DataFrame,
        factor_returns: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        估计因子载荷矩阵 (N资产 x K因子)
        
        使用时间序列回归: r_i,t = alpha_i + beta_i' * f_t + epsilon_i,t
        
        Args:
            returns: 资产收益率矩阵 (T x N)
            factor_returns: 因子收益率矩阵 (T x K), None则自动生成
        
        Returns:
            因子载荷矩阵 (N x K)
        """
        try:
            T, N = returns.shape
            
            if T < self.config.min_observations:
                logger.warning(f"观测数{T}不足最小要求{self.config.min_observations}")
                return pd.DataFrame()
            
            # 生成或使用因子收益
            if factor_returns is None:
                factor_returns = self._generate_statistical_factors(returns)
            
            K = factor_returns.shape[1]
            
            # 时间序列回归估计beta
            loadings = np.zeros((N, K))
            specific_vars = np.zeros(N)
            
            for i in range(N):
                y = returns.iloc[:, i].values
                X = factor_returns.values
                
                # 添加截距项
                X_with_const = np.column_stack([np.ones(T), X])
                
                # OLS回归: beta = (X'X)^-1 * X'y
                try:
                    beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                    loadings[i, :] = beta[1:]  # 去除截距
                    
                    # 计算残差方差
                    residuals = y - X_with_const @ beta
                    specific_vars[i] = np.var(residuals, ddof=K+1)
                except np.linalg.LinAlgError:
                    logger.warning(f"资产{i}回归失败，使用默认值")
                    loadings[i, :] = 0
                    specific_vars[i] = np.var(y)
            
            self.factor_loadings = pd.DataFrame(
                loadings,
                index=returns.columns,
                columns=factor_returns.columns
            )
            self.specific_variance = pd.Series(specific_vars, index=returns.columns)
            
            logger.info(f"因子载荷估计完成: {N}资产 x {K}因子")
            return self.factor_loadings
        
        except Exception as e:
            logger.error(f"因子载荷估计失败: {e}")
            return pd.DataFrame()
    
    def estimate_factor_covariance(
        self,
        factor_returns: pd.DataFrame,
        use_shrinkage: bool = True
    ) -> pd.DataFrame:
        """
        估计因子协方差矩阵 (K x K)
        
        Args:
            factor_returns: 因子收益率矩阵 (T x K)
            use_shrinkage: 是否使用收缩估计
        
        Returns:
            因子协方差矩阵 (K x K)
        """
        try:
            # 样本协方差
            sample_cov = factor_returns.cov()
            
            if not use_shrinkage:
                self.factor_covariance = sample_cov
                return sample_cov
            
            # Ledoit-Wolf收缩估计
            K = len(sample_cov)
            
            # 收缩目标：单位矩阵（标准化后）
            avg_var = np.trace(sample_cov) / K
            target = np.eye(K) * avg_var
            
            # 收缩强度估计（简化版）
            T = len(factor_returns)
            shrinkage = min(1.0, 10.0 / T)
            
            # 收缩估计
            shrunk_cov = shrinkage * target + (1 - shrinkage) * sample_cov
            
            self.factor_covariance = shrunk_cov
            logger.info(f"因子协方差估计完成: {K}x{K}, 收缩强度={shrinkage:.3f}")
            return self.factor_covariance
        
        except Exception as e:
            logger.error(f"因子协方差估计失败: {e}")
            return pd.DataFrame()
    
    def compute_covariance_matrix(
        self,
        returns: pd.DataFrame,
        factor_returns: Optional[pd.DataFrame] = None,
        use_hybrid: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        计算因子模型协方差矩阵
        
        Cov = B * F * B' + D
        其中:
        - B: 因子载荷矩阵 (N x K)
        - F: 因子协方差矩阵 (K x K)
        - D: 特质方差对角矩阵 (N x N)
        
        Args:
            returns: 资产收益率矩阵 (T x N)
            factor_returns: 因子收益率矩阵 (T x K), None则自动生成
            use_hybrid: 是否使用混合模型
        
        Returns:
            (协方差矩阵, 元数据)
        """
        try:
            T, N = returns.shape
            
            # 1. 估计因子载荷
            if factor_returns is None:
                factor_returns = self._generate_statistical_factors(returns)
            
            self.factor_returns = factor_returns
            loadings = self.estimate_factor_loadings(returns, factor_returns)
            
            if loadings.empty:
                logger.warning("因子载荷估计失败，回退样本协方差")
                return returns.cov(), {'method': 'sample', 'success': False}
            
            # 2. 估计因子协方差
            factor_cov = self.estimate_factor_covariance(factor_returns, use_shrinkage=True)
            
            # 3. 因子模型协方差: Cov = B * F * B' + D
            B = loadings.values
            F = factor_cov.values
            D = np.diag(self.specific_variance.values)
            
            factor_model_cov = B @ F @ B.T + D
            
            # 4. 混合模型（可选）
            if use_hybrid:
                sample_cov = returns.cov().values
                alpha = self.config.shrinkage_alpha
                hybrid_cov = alpha * factor_model_cov + (1 - alpha) * sample_cov
                
                cov_matrix = pd.DataFrame(
                    hybrid_cov,
                    index=returns.columns,
                    columns=returns.columns
                )
                method = 'hybrid'
            else:
                cov_matrix = pd.DataFrame(
                    factor_model_cov,
                    index=returns.columns,
                    columns=returns.columns
                )
                method = 'factor_model'
            
            # 5. 元数据
            metadata = {
                'method': method,
                'success': True,
                'n_assets': N,
                'n_factors': len(factor_returns.columns),
                'n_observations': T,
                'shrinkage_alpha': self.config.shrinkage_alpha if use_hybrid else 0.0,
                'avg_specific_variance': float(self.specific_variance.mean()),
                'factor_contribution': float(np.trace(B @ F @ B.T) / np.trace(cov_matrix.values))
            }
            
            logger.info(
                f"因子模型协方差计算完成: {method}, "
                f"因子贡献={metadata['factor_contribution']:.1%}"
            )
            
            return cov_matrix, metadata
        
        except Exception as e:
            logger.error(f"因子模型协方差计算失败: {e}")
            return returns.cov(), {'method': 'sample', 'success': False, 'error': str(e)}
    
    def _generate_statistical_factors(
        self,
        returns: pd.DataFrame,
        n_factors: Optional[int] = None
    ) -> pd.DataFrame:
        """
        使用PCA生成统计因子
        
        Args:
            returns: 资产收益率矩阵 (T x N)
            n_factors: 因子数量, None则使用配置
        
        Returns:
            因子收益率矩阵 (T x K)
        """
        try:
            if n_factors is None:
                n_factors = min(self.config.n_factors, returns.shape[1] // 2)
            
            # 标准化收益率
            returns_standardized = (returns - returns.mean()) / returns.std()
            
            # PCA分解
            U, s, Vt = np.linalg.svd(returns_standardized.values, full_matrices=False)
            
            # 提取前K个主成分作为因子收益
            factor_returns = U[:, :n_factors] * s[:n_factors]
            
            # 转换为DataFrame
            factor_df = pd.DataFrame(
                factor_returns,
                index=returns.index,
                columns=[f'PC{i+1}' for i in range(n_factors)]
            )
            
            # 计算解释方差比例
            explained_var_ratio = s[:n_factors]**2 / (s**2).sum()
            total_explained = explained_var_ratio.sum()
            
            logger.info(
                f"PCA统计因子生成: {n_factors}因子, "
                f"解释方差={total_explained:.1%}"
            )
            
            return factor_df
        
        except Exception as e:
            logger.error(f"PCA因子生成失败: {e}")
            # 回退：使用市场因子（等权平均）
            market_factor = returns.mean(axis=1)
            return pd.DataFrame({'Market': market_factor}, index=returns.index)
    
    def get_factor_summary(self) -> Dict[str, Any]:
        """
        获取因子模型摘要统计
        
        Returns:
            因子模型统计信息
        """
        if self.factor_loadings is None:
            return {'error': '因子载荷未估计'}
        
        summary = {
            'n_assets': len(self.factor_loadings),
            'n_factors': len(self.factor_loadings.columns),
            'avg_loading': float(self.factor_loadings.abs().mean().mean()),
            'max_loading': float(self.factor_loadings.abs().max().max()),
            'avg_specific_vol': float(np.sqrt(self.specific_variance.mean())),
        }
        
        if self.factor_covariance is not None:
            summary['factor_volatility'] = float(np.sqrt(np.diag(self.factor_covariance)).mean())
        
        return summary
