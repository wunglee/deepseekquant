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
import hashlib

logger = logging.getLogger('DeepSeekQuant.FactorModel')

# 外部化风险模型参数（默认阈值）
try:
    from common import RISK_MODEL_CONFIG as _RMC
except Exception:
    _RMC = {
        'factor_model': {
            'condition_number_threshold': 1e10,
            'ridge_alpha': 0.1
        }
    }


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
    """因子模型协方差估计器
    
    P0优化 (专家建议):
    - P0-2: 数值稳定性检查 (条件数+Ridge回归)
    - P0-3: 缓存集成 (预计50-70%计算加速)
    """
    
    def __init__(self, config: Optional[FactorModelConfig] = None, cache_service=None):
        """
        初始化因子模型估计器
        
        Args:
            config: 因子模型配置
            cache_service: 缓存服务 (P0-3优化)
        """
        self.config = config or FactorModelConfig()
        self.factor_returns = None  # 因子收益序列
        self.factor_loadings = None  # 因子载荷矩阵 (N资产 x K因子)
        self.specific_variance = None  # 特质方差向量 (N,)
        self.factor_covariance = None  # 因子协方差矩阵 (K x K)
        
        # P0-3: 缓存集成
        if cache_service is None:
            try:
                from core_bak_refactored.infrastructure.cache_service import get_cache_service
                self.cache_service = get_cache_service()
            except Exception:
                self.cache_service = None
        else:
            self.cache_service = cache_service
        if self.cache_service:
            logger.info("因子模型缓存已启用 (P0-3优化)")
        # 内部L1缓存（进程内，TTL）
        try:
            from common import RISK_MODEL_CONFIG as _RMC
            self._l1_ttl_seconds = int(_RMC.get('factor_model', {}).get('cache_ttl_seconds', 3600))
        except Exception:
            self._l1_ttl_seconds = 3600
        self._l1_cache: Dict[str, Tuple[Tuple[pd.DataFrame, Dict[str, Any]], float]] = {}
    
    def _generate_cache_key(self, returns: pd.DataFrame, method: str = 'factor_loadings') -> str:
        """
        生成缓存键 (P0-3优化)
        
        Args:
            returns: 收益率DataFrame
            method: 方法名称
        
        Returns:
            缓存键字符串
        """
        # 生成返回数据的哈希
        data_hash = hashlib.md5(
            returns.values.tobytes() + 
            str(returns.columns.tolist()).encode()
        ).hexdigest()[:16]
        
        # 缓存键格式: factor_model:{method}:{market}:{n_assets}:{hash}
        cache_key = f"factor_model:{method}:{self.config.market}:{len(returns.columns)}:{data_hash}"
        return cache_key
    
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
            
            # 动态阈值（专家建议）
            thresholds = self._get_dynamic_thresholds(getattr(self.config, 'market', 'US'), T)
            condition_threshold = thresholds['condition_number']
            ridge_alpha = thresholds['ridge_alpha']
            logger.debug(
                f"阈值来源: external_config+market+sample_size, condition_threshold={condition_threshold:.2e}, ridge_alpha={ridge_alpha}"
            )
            
            # 时间序列回归估计beta
            loadings = np.zeros((N, K))
            specific_vars = np.zeros(N)
            r_squared_values = np.zeros(N)  # 跟踪R^2
            
            for i in range(N):
                try:
                    y = returns.iloc[:, i].values
                    X = factor_returns.values
                    
                    # 添加截距项
                    X_with_const = np.column_stack([np.ones(T), X])
                    
                    # P0-2: 数值稳定性检查 (专家建议)
                    condition_number = np.linalg.cond(X_with_const)
                    
                    if condition_number > condition_threshold:
                        logger.warning(
                            f"资产{returns.columns[i]}: 因子矩阵条件数过高 {condition_number:.2e}, "
                            f"自动切换到Ridge回归(alpha={ridge_alpha})"
                        )
                        # Ridge降级
                        beta_ridge = self._ridge_regression(y, X, alpha=ridge_alpha)
                        beta_with_const = np.concatenate([[0], beta_ridge])
                        # 评估Ridge影响（与伪OLS比较）
                        try:
                            ols_beta = np.linalg.pinv(X) @ y
                        except Exception:
                            ols_beta = np.zeros(K)
                        impact = self._assess_ridge_impact(y, X, ols_beta=ols_beta, ridge_beta=beta_ridge)
                        if impact.get('max_beta_change', 0.0) > 0.5:
                            logger.warning(
                                f"资产{returns.columns[i]}: Ridge影响较大，max_beta_change={impact['max_beta_change']:.3f}, mse_change_pct={impact.get('mse_change_pct', 0.0):.3f}"
                            )
                    else:
                        # OLS回归: QR分解提高数值稳定性
                        try:
                            Q, R_qr = np.linalg.qr(X_with_const)
                            beta_with_const = np.linalg.solve(R_qr, Q.T @ y)
                        except np.linalg.LinAlgError:
                            # 回退到伪逆
                            logger.warning(f"资产{returns.columns[i]}: QR分解失败, 使用伪逆")
                            beta_with_const = np.linalg.pinv(X_with_const) @ y
                    
                    loadings[i, :] = beta_with_const[1:]
                    
                    # 残差与R^2
                    residuals = y - X_with_const @ beta_with_const
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r_squared_values[i] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    # 无偏估计
                    specific_vars[i] = ss_res / (T - K - 1) if T > K + 1 else np.var(residuals)
                
                except (np.linalg.LinAlgError, ValueError) as e:
                    logger.warning(f"资产{returns.columns[i]}回归失败: {e}，使用默认值")
                    loadings[i, :] = 0
                    specific_vars[i] = np.var(returns.iloc[:, i].values)
                    r_squared_values[i] = 0
            
            self.factor_loadings = pd.DataFrame(
                loadings,
                index=returns.columns,
                columns=factor_returns.columns
            )
            self.specific_variance = pd.Series(specific_vars, index=returns.columns)
            self.r_squared = pd.Series(r_squared_values, index=returns.columns)
            
            avg_r2 = r_squared_values.mean()
            logger.info(
                f"因子载荷估计完成: {N}资产 x {K}因子, "
                f"平均R^2={avg_r2:.2%}"
            )
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
        
        P0-3优化：缓存集成（预计50-70%计算加速）
        
        Args:
            returns: 资产收益率矩阵 (T x N)
            factor_returns: 因子收益率矩阵 (T x K), None则自动生成
            use_hybrid: 是否使用混合模型
        
        Returns:
            (协方差矩阵, 元数据)
        """
        try:
            T, N = returns.shape
            
            # P0-3: 尝试从缓存获取（L1→L2顺序）
            method_tag = 'covariance_hybrid' if use_hybrid else 'covariance_factor'
            cache_key = self._generate_cache_key(returns, method_tag)
            # L1缓存检查
            l1_entry = self._l1_cache.get(cache_key)
            from datetime import datetime
            now_ts = datetime.now().timestamp()
            if l1_entry is not None:
                (cached_cov, cached_meta), ts = l1_entry
                if now_ts - ts <= self._l1_ttl_seconds:
                    logger.info("因子模型协方差: L1缓存命中 (P0-3优化)")
                    # 保持内部状态一致（供摘要统计）
                    if factor_returns is None:
                        factor_returns = self._generate_statistical_factors(returns)
                    self.factor_returns = factor_returns
                    try:
                        self.estimate_factor_loadings(returns, factor_returns)
                    except Exception:
                        pass
                    return (cached_cov, cached_meta)
                else:
                    # 过期删除
                    self._l1_cache.pop(cache_key, None)
            # L2缓存（外部）
            if self.cache_service:
                cached_result = self.cache_service.get(cache_key)
                if cached_result is not None:
                    logger.info("因子模型协方差: L2缓存命中 (P0-3优化)")
                    # 保持内部状态一致（供摘要统计）
                    if factor_returns is None:
                        factor_returns = self._generate_statistical_factors(returns)
                    self.factor_returns = factor_returns
                    try:
                        self.estimate_factor_loadings(returns, factor_returns)
                    except Exception:
                        pass
                    return cached_result

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
                'factor_contribution': float(np.trace(B @ F @ B.T) / np.trace(cov_matrix.values)),
                'thresholds': {
                    'condition_number': float(self._get_dynamic_thresholds(getattr(self.config, 'market', 'US'), T)['condition_number']),
                    'ridge_alpha': float(self._get_dynamic_thresholds(getattr(self.config, 'market', 'US'), T)['ridge_alpha'])
                }
            }
            
            # P1-3: 模型诊断增强
            if hasattr(self, 'r_squared') and self.r_squared is not None:
                diagnostics = self._compute_model_diagnostics(returns, factor_returns, loadings)
                metadata['diagnostics'] = diagnostics
            
            logger.info(
                f"因子模型协方差计算完成: {method}, "
                f"因子贡献={metadata['factor_contribution']:.1%}"
            )
            
            # P0-3: 缓存结果
            result = (cov_matrix, metadata)
            # 设置L2/L1缓存
            if self.cache_service:
                cache_key = self._generate_cache_key(returns, 'covariance')
                self.cache_service.set(cache_key, result, ttl=3600)  # 1小时TTL
                logger.debug("因子模型协方差: L2缓存设置 (P0-3优化)")
            # L1缓存设置
            self._l1_cache[cache_key] = (result, datetime.now().timestamp())
            logger.debug("因子模型协方差: L1缓存设置 (P0-3优化)")
            
            return result

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
    
    def _get_dynamic_thresholds(self, market_type: str, n_observations: int) -> Dict[str, float]:
        """按市场与样本量动态调整数值稳定性阈值（结合外部化配置）"""
        base_defaults = _RMC.get('factor_model', {
            'condition_number_threshold': 1e10,
            'ridge_alpha': 0.1
        })
        # 市场差异映射（在外部化默认值基础上调整）
        base_config = {
            'US': {
                'condition_number': float(base_defaults.get('condition_number_threshold', 1e10)),
                'ridge_alpha': float(base_defaults.get('ridge_alpha', 0.1))
            },
            'CN': {'condition_number': 1e8, 'ridge_alpha': 0.2},
            'HK': {'condition_number': 1e9, 'ridge_alpha': 0.15}
        }
        config = base_config.get(market_type, base_config['US']).copy()
        # 根据观测数调整：数据量少时更保守
        if n_observations < 100:
            config['condition_number'] *= 0.1  # 更严格
            config['ridge_alpha'] *= 2.0      # 更强正则化
        return config

    def _assess_ridge_impact(self, y: np.ndarray, X: np.ndarray,
                             ols_beta: np.ndarray, ridge_beta: np.ndarray) -> Dict[str, float]:
        """评估Ridge回归对结果的影响（专家建议）"""
        try:
            ols_pred = X @ ols_beta
            ridge_pred = X @ ridge_beta
            mse_change_pct = float(np.mean((ridge_pred - ols_pred) ** 2) / (np.var(y) if np.var(y) > 0 else 1.0))
            max_beta_change = float(np.max(np.abs(ridge_beta - ols_beta)))
            return {
                'mse_change_pct': mse_change_pct,
                'max_beta_change': max_beta_change
            }
        except Exception:
            return {'mse_change_pct': 0.0, 'max_beta_change': 0.0}
    def _compute_model_diagnostics(
        self,
        returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        loadings: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        P1-3: 计算模型诊断指标
        
        包括：因子质量、残差分析（异方差性、正态性、自相关）
        
        Args:
            returns: 资产收益率
            factor_returns: 因子收益率
            loadings: 因子载荷
        
        Returns:
            诊断指标字典
        """
        try:
            diagnostics = {}
            
            # 1. 因子质量指标
            if hasattr(self, 'r_squared'):
                diagnostics['factor_quality'] = {
                    'avg_r_squared': float(self.r_squared.mean()),
                    'min_r_squared': float(self.r_squared.min()),
                    'max_r_squared': float(self.r_squared.max()),
                    'poor_fit_assets': int((self.r_squared < 0.3).sum())  # R²<0.3的资产数
                }
            
            # 2. 残差分析
            residuals_all = []
            for i, asset in enumerate(returns.columns):
                y = returns.iloc[:, i].values
                X = factor_returns.values
                
                # 预测值
                beta = loadings.iloc[i].values
                y_pred = X @ beta
                
                # 残差
                residuals = y - y_pred
                residuals_all.append(residuals)
            
            residuals_matrix = np.array(residuals_all)
            
            # 2.1 异方差性检测（Breusch-Pagan简化版）
            # 检查残差方差是否随时间变化
            T = len(residuals_matrix[0])
            mid_point = T // 2
            var_first_half = np.var(residuals_matrix[:, :mid_point], axis=1).mean()
            var_second_half = np.var(residuals_matrix[:, mid_point:], axis=1).mean()
            heteroscedasticity_ratio = var_second_half / var_first_half if var_first_half > 0 else 1.0
            
            # 2.2 正态性检测（偏度和峰度）
            from scipy import stats
            skewness = stats.skew(residuals_matrix, axis=1).mean()
            kurtosis = stats.kurtosis(residuals_matrix, axis=1).mean()
            
            # 2.3 自相关检测（Durbin-Watson近似）
            autocorr_1 = np.corrcoef(
                residuals_matrix[:, :-1].flatten(),
                residuals_matrix[:, 1:].flatten()
            )[0, 1]
            
            diagnostics['residual_analysis'] = {
                'heteroscedasticity_ratio': round(heteroscedasticity_ratio, 3),
                'heteroscedasticity_warning': heteroscedasticity_ratio > 1.5 or heteroscedasticity_ratio < 0.67,
                'skewness': round(skewness, 3),
                'kurtosis': round(kurtosis, 3),
                'normality_warning': abs(skewness) > 1.0 or abs(kurtosis) > 3.0,
                'autocorrelation_lag1': round(autocorr_1, 3),
                'autocorrelation_warning': abs(autocorr_1) > 0.3
            }
            
            # 3. 整体模型健康评分
            warnings_count = sum([
                diagnostics['residual_analysis']['heteroscedasticity_warning'],
                diagnostics['residual_analysis']['normality_warning'],
                diagnostics['residual_analysis']['autocorrelation_warning']
            ])
            
            if warnings_count == 0:
                model_health = 'excellent'
            elif warnings_count == 1:
                model_health = 'good'
            elif warnings_count == 2:
                model_health = 'acceptable'
            else:
                model_health = 'poor'
            
            diagnostics['model_health'] = model_health
            
            logger.info(
                f"P1-3诊断: 平均R²={diagnostics['factor_quality']['avg_r_squared']:.2%}, "
                f"模型健康={model_health}"
            )
            
            return diagnostics
            
        except Exception as e:
            logger.warning(f"模型诊断计算失败: {e}")
            return {'error': str(e)}
    
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
