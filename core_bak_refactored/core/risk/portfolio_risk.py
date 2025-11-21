"""
组合风险分析 - 业务层
从 core_bak/risk_manager.py 拆分
职责: 组合层面的风险分析、风险贡献度

优化: 集成增量计算和并行执行 (阶段1+2)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from .risk_metrics_service import RiskMetricsService
from infrastructure.risk_metrics import StatisticalCalculator

logger = logging.getLogger('DeepSeekQuant.PortfolioRisk')

# 条件导入优化组件
try:
    from infrastructure.parallel_executor import get_parallel_executor
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False
    logger.warning("并行执行器未找到，并行计算将被禁用")

try:
    from .incremental_calculator import IncrementalCovarianceCalculator
    INCREMENTAL_AVAILABLE = True
except ImportError:
    INCREMENTAL_AVAILABLE = False
    logger.warning("增量计算器未找到，增量计算将被禁用")


# ============= 静态辅助函数 (优化并行计算) =============

# 全局缓存配置和分析器（进程级）
_WORKER_ANALYZER_CACHE = {}

def _get_or_create_analyzer(config_dict: Dict[str, Any]) -> 'PortfolioRiskAnalyzer':
    """
    获取或创建分析器（进程级缓存）
    优化：避免每次任务都重建分析器
    
    Args:
        config_dict: 配置字典（必须可序列化）
    
    Returns:
        分析器实例
    """
    import os
    worker_id = os.getpid()
    
    if worker_id not in _WORKER_ANALYZER_CACHE:
        # 首次创建，缓存之
        from core_bak_refactored.core.risk.portfolio_risk import PortfolioRiskAnalyzer
        analyzer = PortfolioRiskAnalyzer(config_dict, enable_parallel=False)
        _WORKER_ANALYZER_CACHE[worker_id] = analyzer
        logger.debug(f"进程Worker {worker_id}: 创建分析器")
    
    return _WORKER_ANALYZER_CACHE[worker_id]

def _calculate_single_portfolio_static(
    item: Tuple[str, Any, Dict[str, Any], Dict[str, Any]]
) -> Tuple[str, Dict[str, Any]]:
    """
    静态函数：计算单个组合风险
    用于并行计算，避免序列化整个PortfolioRiskAnalyzer对象
    
    优化：
    1. 使用进程级缓存分析器，避免重复创建
    2. config_dict传递而非对象，减少序列化开销
    """
    portfolio_id, portfolio_state, market_data, config_dict = item
    try:
        # 使用缓存的分析器
        analyzer = _get_or_create_analyzer(config_dict)
        
        data = {
            'portfolio_state': portfolio_state,
            'market_data': market_data
        }
        result = analyzer.analyze(data, {})
        return (portfolio_id, result)
    except Exception as e:
        logger.error(f"组合{portfolio_id}计算失败: {e}")
        return (portfolio_id, {})


class PortfolioRiskAnalyzer:
    """组合风险分析器 - 集成增量计算和并行优化"""
    
    def __init__(self, config: Dict, enable_parallel: bool = True, enable_incremental: bool = True):
        """
        初始化组合风险分析器
        
        Args:
            config: 配置字典
            enable_parallel: 启用并行计算 (默认True)
            enable_incremental: 启用增量计算 (默认True)
        """
        self.config = config
        self.risk_metrics_service = RiskMetricsService(config)
        
        # 优化组件
        self.enable_parallel = enable_parallel and PARALLEL_AVAILABLE
        self.enable_incremental = enable_incremental and INCREMENTAL_AVAILABLE
        
        if self.enable_parallel:
            self.parallel_executor = get_parallel_executor()
            logger.info("并行计算已启用")
        
        if self.enable_incremental:
            self.incremental_calculator = IncrementalCovarianceCalculator()
            logger.info("增量计算已启用")
    
    def _adjust_for_limit_hits(self, returns: np.ndarray, limit_threshold: float = 0.10) -> np.ndarray:
        """
        调整涨跌停导致的收益率截断（专家建议）
        
        Args:
            returns: 收益率数组
            limit_threshold: 涨跌停阈值（默认10%）
        
        Returns:
            调整后的收益率
        """
        try:
            import pandas as pd
            # 检测涨跌停日（95%阈值）
            detection_threshold = limit_threshold * 0.95
            limit_days = np.abs(returns) >= detection_threshold
            
            if np.sum(limit_days) == 0:
                return returns
            
            # 专家建议: 检测连续涨跌停情况 (第14轮新增)
            consecutive_limit_days = self._detect_consecutive_limit_hits(limit_days)
            if consecutive_limit_days > 0:
                logger.warning(
                    f"检测到{consecutive_limit_days}天连续涨跌停，"
                    f"总计{np.sum(limit_days)}/{ len(returns)}天涨跌停，"
                    f"风险可能被低估 (专家建议: 连续涨跌停误差15-25%)"
                )
            
            # 使用前后均值填充（简化方法）
            adjusted_returns = returns.copy()
            returns_series = pd.Series(adjusted_returns)
            
            # 将涨跌停日标记为NaN
            returns_series[limit_days] = np.nan
            
            # 前向填充
            returns_series = returns_series.fillna(method='ffill')
            # 后向填充（处理开头的NaN）
            returns_series = returns_series.fillna(method='bfill')
            # 如果还有NaN，填充0
            returns_series = returns_series.fillna(0)
            
            logger.debug(f"涨跌停调整: {np.sum(limit_days)}/{ len(returns)}天被调整")
            return returns_series.values
        except Exception as e:
            logger.warning(f"涨跌停调整失败: {e}, 返回原始数据")
            return returns
    
    def _align_returns_with_forward_fill(self, returns_data: Dict[str, np.ndarray], 
                                         min_required_length: int = 30) -> Optional[pd.DataFrame]:
        """
        使用前向填充对齐收益率序列（专家建议 + 第14轮增强）
        
        Args:
            returns_data: 符号 -> 收益率数组的字典
            min_required_length: 最小数据长度要求
        
        Returns:
            对齐后的DataFrame，失败返回None
        """
        try:
            # 1. 识别新上市资产 (专家建议: 上市时间<180天)
            new_listing_symbols = self._identify_new_listings(returns_data, max_days=180)
            
            # 转换为DataFrame，使用最新数据填充早期缺失
            max_len = max(len(r) for r in returns_data.values())
            
            # 对齐所有序列到最大长度
            aligned_data = {}
            for symbol, returns in returns_data.items():
                if len(returns) < max_len:
                    # 用NaN填充前面
                    padded = np.full(max_len, np.nan)
                    padded[-len(returns):] = returns
                    aligned_data[symbol] = padded
                else:
                    aligned_data[symbol] = returns
            
            df = pd.DataFrame(aligned_data)
            
            # 前向填充（用最新可用数据填充历史缺失）
            aligned = df.ffill()
            
            # 如果还有缺失值（开头的NaN），使用后向填充
            aligned = aligned.bfill()
            
            # 删除仍然有NaN的行
            aligned = aligned.dropna()
            
            # 检查最小数据长度要求
            if len(aligned) < min_required_length:
                logger.warning(f"对齐后数据长度不足{min_required_length}，当前{len(aligned)}，尝试插值")
                # 使用线性插值补充缺失值
                aligned = df.interpolate(method='linear').dropna()
                
                if len(aligned) < min_required_length:
                    logger.error(f"插值后仍不足{min_required_length}，对齐失败")
                    return None
            
            # 2. 对新上市资产降权处理 (专家建议: 减半权重)
            if new_listing_symbols:
                for symbol in new_listing_symbols:
                    if symbol in aligned.columns:
                        aligned[symbol] = aligned[symbol] * 0.5
                        logger.info(f"新上市资产{symbol}降权50%，数据长度={len(returns_data[symbol])}")
            
            logger.debug(f"数据对齐完成: {len(returns_data)}个资产, {len(aligned)}个数据点, {len(new_listing_symbols)}个新上市")
            return aligned
        
        except Exception as e:
            logger.error(f"数据对齐失败: {e}")
            return None
    
    def _detect_consecutive_limit_hits(self, limit_days: np.ndarray) -> int:
        """
        检测连续涨跌停天数（专家建议 - 第14轮新增）
        
        Args:
            limit_days: 涨跌停日标记数组
        
        Returns:
            最大连续涨跌停天数
        """
        max_consecutive = 0
        current_consecutive = 0
        
        for is_limit in limit_days:
            if is_limit:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _identify_new_listings(self, returns_data: Dict[str, np.ndarray], max_days: int = 180) -> List[str]:
        """
        识别新上市资产（专家建议 - 第14轮新增）
        
        Args:
            returns_data: 收益率数据
            max_days: 最大天数阈值 (默认180天)
        
        Returns:
            新上市资产符号列表
        """
        new_listings = []
        for symbol, returns in returns_data.items():
            # 使用数据长度近似上市时间
            if len(returns) < max_days:
                new_listings.append(symbol)
                logger.debug(f"识别为新上市资产: {symbol}, 数据长度={len(returns)}天")
        return new_listings
    
    def calculate_portfolio_returns(self, portfolio_state, market_data: Dict[str, Any]) -> pd.Series:
        """计算组合收益序列"""
        try:
            # 获取组合中所有资产
            symbols = list(portfolio_state.allocations.keys())
            if not symbols:
                logger.warning("组合中没有资产")
                return pd.Series()

            # 获取价格数据并确保时间对齐
            price_data = {}
            min_length = float('inf')

            for symbol in symbols:
                if symbol in market_data['prices']:
                    closes = market_data['prices'][symbol].get('close', [])
                    if len(closes) > 0:
                        price_data[symbol] = closes
                        min_length = min(min_length, len(closes))
                    else:
                        logger.warning(f"符号 {symbol} 没有价格数据，跳过")
                        continue
                else:
                    logger.warning(f"市场数据中缺少符号 {symbol}，跳过")
                    continue

            # 若所有资产均不可用，返回空序列
            if not price_data:
                logger.warning("有效价格数据为空，返回空序列")
                return pd.Series()
            if min_length < 2:
                logger.warning("价格数据不足")
                return pd.Series()

            # 截取相同长度的价格序列
            aligned_prices = {}
            for symbol, prices in price_data.items():
                aligned_prices[symbol] = prices[-min_length:]

            # 计算对数收益（使用基础设施层统一方法）
            returns_data = {}
            for symbol, prices in aligned_prices.items():
                log_returns = StatisticalCalculator.calculate_log_returns(np.array(prices))
                returns_data[symbol] = log_returns

            # 创建DataFrame
            returns_df = pd.DataFrame(returns_data)

            # 获取用于计算的符号列
            symbols_used = list(returns_df.columns)
            # 获取权重（仅针对有效符号）
            weights = np.array([portfolio_state.allocations[s].weight for s in symbols_used])
            # 权重重标定：当有效符号的权重和偏离1时进行归一化
            total_weight = float(weights.sum())
            if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
                weights = weights / total_weight
                logger.info(f"有效资产权重重新标定: {total_weight:.3f} -> 1.000")

            # 计算加权组合收益
            portfolio_returns = returns_df.dot(weights)

            # 转换为Series并设置时间索引
            if 'timestamp' in market_data and len(market_data['timestamp']) >= len(portfolio_returns):
                # 使用最后一个时间戳作为索引（假设时间戳是升序排列）
                timestamps = market_data['timestamp'][-len(portfolio_returns):]
                portfolio_returns = pd.Series(portfolio_returns.values, index=timestamps)
            else:
                # 使用数值索引作为备选
                portfolio_returns = pd.Series(portfolio_returns.values)

            logger.debug(f"组合收益计算完成: 数据点={len(portfolio_returns)}, 期间={len(portfolio_returns)}天")
            return portfolio_returns

        except Exception as e:
            logger.error(f"组合收益计算失败: {e}")
            return pd.Series()
    
    def calculate_risk_contributions(self, portfolio_state, correlation_matrix: pd.DataFrame) -> Dict[str, float]:
        """计算风险贡献度（基于相关性矩阵与权重）"""
        risk_contributions = {}
        try:
            if correlation_matrix is None or correlation_matrix.empty:
                return risk_contributions
            
            symbols = list(portfolio_state.allocations.keys())
            weights = np.array([portfolio_state.allocations[symbol].weight for symbol in symbols])
            
            # 确保相关性矩阵维度匹配
            if len(weights) != correlation_matrix.shape[0]:
                logger.warning("权重与相关性矩阵维度不匹配")
                return risk_contributions
            
            # 计算组合方差
            portfolio_variance = weights.T @ correlation_matrix.values @ weights
            
            if portfolio_variance > 0:
                # 计算边际风险贡献
                marginal_risk = (correlation_matrix.values @ weights) / np.sqrt(portfolio_variance)
                
                # 计算风险贡献度
                for i, symbol in enumerate(symbols):
                    risk_contribution = weights[i] * marginal_risk[i]
                    risk_contributions[symbol] = float(risk_contribution)
            
            return risk_contributions
        
        except Exception as e:
            logger.error(f"风险贡献度计算失败: {e}")
            return {}
    
    def calculate_risk_contributions_covariance(self, portfolio_state, covariance_matrix: pd.DataFrame) -> Dict[str, float]:
        """基于协方差矩阵计算风险贡献度（边际风险贡献法）"""
        contributions: Dict[str, float] = {}
        try:
            if covariance_matrix is None or covariance_matrix.empty:
                return contributions
            symbols = list(portfolio_state.allocations.keys())
            weights = np.array([portfolio_state.allocations[symbol].weight for symbol in symbols])
            if len(weights) != covariance_matrix.shape[0]:
                logger.warning("权重与协方差矩阵维度不匹配")
                return contributions
            portfolio_variance = float(weights.T @ covariance_matrix.values @ weights)
            if portfolio_variance <= 0:
                return contributions
            marginal_risk = (covariance_matrix.values @ weights) / np.sqrt(portfolio_variance)
            for i, symbol in enumerate(symbols):
                contributions[symbol] = float(weights[i] * marginal_risk[i])
            return contributions
        except Exception as e:
            logger.error(f"协方差风险贡献度计算失败: {e}")
            return {}
    
    def calculate_factor_risk_attribution(self, portfolio_state, factor_exposures: pd.DataFrame, 
                                          factor_covariance: pd.DataFrame) -> Dict[str, Any]:
        """
        计算因子级风险归因分解
        
        基于 Barra 风险模型：组合方差 = X' * F * X + Δ
        
        Args:
            portfolio_state: 组合状态
            factor_exposures: 因子暴露矩阵 (N资产 x K因子)
            factor_covariance: 因子协方差矩阵 (K x K)
            
        Returns:
            {
                'market_risk': 市场因子风险,
                'industry_risk': 行业因子风险,
                'style_risk': 风格因子风险,
                'specific_risk': 特质风险,
                'total_risk': 总风险,
                'factor_contributions': 各因子贡献明细
            }
        """
        try:
            if factor_exposures is None or factor_exposures.empty:
                logger.warning("因子暴露数据为空")
                return {}
            
            symbols = list(portfolio_state.allocations.keys())
            weights = np.array([portfolio_state.allocations[symbol].weight for symbol in symbols])
            
            # 确保符号匹配
            common_symbols = [s for s in symbols if s in factor_exposures.index]
            if not common_symbols:
                logger.warning("没有匹配的符号")
                return {}
            
            # 调整权重与暴露
            matched_weights = np.array([portfolio_state.allocations[s].weight for s in common_symbols])
            matched_exposures = factor_exposures.loc[common_symbols].values
            
            # 组合因子暴露：X_p = w' * X
            portfolio_factor_exposure = matched_weights @ matched_exposures
            
            # 因子风险贡献：X_p' * F * X_p
            factor_variance = float(portfolio_factor_exposure @ factor_covariance.values @ portfolio_factor_exposure.T)
            
            # 分解到各个因子
            factor_contributions = {}
            for i, factor_name in enumerate(factor_covariance.columns):
                # 单个因子的边际贡献
                marginal_contribution = 2 * portfolio_factor_exposure[i] * (factor_covariance.iloc[i, :].values @ portfolio_factor_exposure)
                factor_contributions[factor_name] = float(marginal_contribution)
            
            # 按因子类型聚合（假设因子命名约定）
            market_risk = sum(v for k, v in factor_contributions.items() if k.startswith('market'))
            industry_risk = sum(v for k, v in factor_contributions.items() if k.startswith('industry'))
            style_risk = sum(v for k, v in factor_contributions.items() if k.startswith('style'))
            
            # 特质风险（简化：假设为10%总风险）
            specific_risk = factor_variance * 0.1
            
            total_risk = factor_variance + specific_risk
            
            return {
                'market_risk': float(market_risk),
                'industry_risk': float(industry_risk),
                'style_risk': float(style_risk),
                'specific_risk': float(specific_risk),
                'total_risk': float(total_risk),
                'factor_contributions': factor_contributions
            }
            
        except Exception as e:
            logger.error(f"因子风险归因失败: {e}")
            return {}
    
    def analyze(self, data: Dict[str, Any], risk_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        综合分析组合风险（P1增强：7维度分析）
        
        根据专家指导，返回完整的7维度风险分析：
        1. total_risk - 组合总风险（波动率）
        2. volatility - 组合波动率（年化）
        3. var_95 - 95% VaR
        4. cvar_95 - 95% CVaR
        5. sharpe_ratio - 夏普比率
        6. max_drawdown - 最大回撤
        7. risk_contributions - 各资产风险贡献
        """
        # 初始化结果（专家推荐的7维度结构）
        result = {
            'total_risk': 0.0,              # 组合总风险（使用波动率，具有可加性）
            'volatility': 0.0,              # 组合波动率（年化）
            'var_95': 0.0,                  # 95% VaR
            'cvar_95': 0.0,                 # 95% CVaR
            'sharpe_ratio': 0.0,            # 夏普比率
            'max_drawdown': 0.0,            # 最大回撤
            'risk_contributions': {},       # 各资产风险贡献
            # 额外保留的原有字段
            'portfolio_returns': pd.Series(),
            'concentration_risk': 0.0
        }
        
        try:
            portfolio_state = data.get('portfolio_state')
            market_data = data.get('market_data')
            
            if not portfolio_state:
                logger.warning("组合状态为空")
                return result
            
            # 1. 计算组合收益序列
            portfolio_returns = pd.Series()
            if market_data:
                portfolio_returns = self.calculate_portfolio_returns(portfolio_state, market_data)
                result['portfolio_returns'] = portfolio_returns
            
            # 2. 计算波动率（总风险）
            if len(portfolio_returns) > 1:
                daily_volatility = float(portfolio_returns.std())
                annual_volatility = daily_volatility * np.sqrt(self.config.get('trading_days_per_year', 252))
                result['volatility'] = annual_volatility
                result['total_risk'] = annual_volatility  # 专家指导：总风险=波动率
            
            # 3. 计算VaR和CVaR（使用RiskMetricsService）
            if len(portfolio_returns) > 1:
                # 转换为Series类型（RiskMetricsService需要Series）
                if not isinstance(portfolio_returns, pd.Series):
                    portfolio_returns = pd.Series(portfolio_returns)
                
                var_95 = self.risk_metrics_service.calculate_value_at_risk(
                    portfolio_returns, 
                    confidence_level=0.95
                )
                cvar_95 = self.risk_metrics_service.calculate_expected_shortfall(
                    portfolio_returns, 
                    confidence_level=0.95
                )
                result['var_95'] = abs(var_95)  # VaR通常为负值，取绝对值表示损失
                result['cvar_95'] = abs(cvar_95)
            
            # 4. 计算夏普比率（使用增强版，考虑市场风险溢价）
            if len(portfolio_returns) > 1:
                # 使用增强版夏普比率（国际化支持）
                enhanced_result = self.risk_metrics_service.calculate_sharpe_ratio_enhanced(
                    portfolio_returns,
                    risk_free_rate=None,  # 使用动态无风险利率
                    include_market_premium=True,  # 包含市场溢价
                    adjust_for_anomalies=True,  # 调整市场异常
                    prices=None
                )
                result['sharpe_ratio'] = enhanced_result['enhanced_sharpe']
            
            # 5. 计算最大回撤
            if len(portfolio_returns) > 1:
                max_dd = self.risk_metrics_service.calculate_max_drawdown(portfolio_returns)
                result['max_drawdown'] = abs(max_dd)  # 取绝对值表示回撤幅度
            
            # 6. 计算风险贡献度（智能选择：协方差>稳健矩阵自动生成>相关性矩阵）
            cov_matrix = data.get('covariance_matrix')
            if cov_matrix is not None:
                # 优先使用提供的协方差矩阵
                result['risk_contributions'] = self.calculate_risk_contributions_covariance(
                    portfolio_state, cov_matrix
                )
            else:
                # 未提供协方差矩阵，检查是否有相关性矩阵
                corr_matrix = data.get('correlation_matrix')
                if corr_matrix is not None:
                    result['risk_contributions'] = self.calculate_risk_contributions(
                        portfolio_state, corr_matrix
                    )
                else:
                    # 未提供任何矩阵，自动从收益序列生成稳健矩阵
                    if len(portfolio_returns) > 1 and market_data:
                        logger.info("未提供协方差/相关性矩阵，自动生成稳健矩阵用于风险贡献计算")
                        # 构造多资产收益DataFrame用于矩阵生成
                        symbols = list(portfolio_state.allocations.keys())
                        returns_data = {}
                        for symbol in symbols:
                            if symbol in market_data.get('prices', {}):
                                closes = market_data['prices'][symbol].get('close', [])
                                if len(closes) >= 2:
                                    log_returns = StatisticalCalculator.calculate_log_returns(np.array(closes))
                                    # 专家建议：A股市场需要涨跌停调整
                                    if self.config.get('market_type') == 'CN':
                                        log_returns = self._adjust_for_limit_hits(log_returns)
                                    returns_data[symbol] = log_returns
                        
                        if len(returns_data) > 0:
                            # 专家建议：改进数据对齐策略，使用前向填充+插值
                            returns_df = self._align_returns_with_forward_fill(returns_data)
                            
                            if returns_df is not None and len(returns_df) > 0:
                                # 生成收缩协方差矩阵
                                auto_cov = self.risk_metrics_service.compute_shrunk_covariance(returns_df)
                                result['risk_contributions'] = self.calculate_risk_contributions_covariance(
                                    portfolio_state, auto_cov
                                )
                                result['_auto_generated_covariance'] = True  # 标记为自动生成
                            else:
                                logger.warning("数据对齐后长度不足，无法生成稳健矩阵")
                        else:
                            logger.warning("无足够收益数据生成稳健矩阵，风险贡献为空")
            
            # 7. 计算集中度风险（HHI）
            weights_list = [alloc.weight for alloc in portfolio_state.allocations.values()]
            if weights_list:
                hhi = sum(w ** 2 for w in weights_list)
                result['concentration_risk'] = float(min(hhi, 1.0))
            
            logger.debug(f"组合风险分析完成: 波动率={result['volatility']:.4f}, VaR={result['var_95']:.4f}, 夏普={result['sharpe_ratio']:.2f}")
            return result
        
        except Exception as e:
            logger.error(f"组合风险分析失败: {e}")
            return result
    
    # ============= 并行计算优化方法 (阶段2新增) =============
    
    def batch_calculate_portfolio_risk(
        self,
        portfolios: List[Tuple[str, Any, Dict[str, Any]]],
        use_parallel: Optional[bool] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        批量计算多个组合的风险指标（并行优化）
        
        优化 (P0-1 专家建议):
        1. 共享配置数据，减少序列化开销
        2. 进程级缓存分析器，避免重复创建
        
        Args:
            portfolios: List[(portfolio_id, portfolio_state, market_data)]
            use_parallel: 是否使用并行，None使用全局配置
        
        Returns:
            {portfolio_id: risk_result}
        """
        if use_parallel is None:
            use_parallel = self.enable_parallel
        
        # 判断是否并行
        n_portfolios = len(portfolios)
        if use_parallel and n_portfolios >= 10 and hasattr(self, 'parallel_executor'):
            logger.info(f"并行计算{n_portfolios}个组合风险 (P0优化: 共享配置)")
            
            # P0优化: 共享配置数据，减少序列化开销
            config_dict = self._prepare_shared_config()
            
            # 为每个任务添加配置
            portfolios_with_config = [
                (pid, pstate, mdata, config_dict)
                for pid, pstate, mdata in portfolios
            ]
            
            # 使用静态方法避免self序列化
            results_list = self.parallel_executor.map_cpu_intensive(
                _calculate_single_portfolio_static,
                portfolios_with_config
            )
        else:
            logger.info(f"串行计算{n_portfolios}个组合风险")
            results_list = [
                self._calculate_single_portfolio(item) 
                for item in portfolios
            ]
        
        # 转换为字典
        results_dict = {pid: result for pid, result in results_list if result}
        
        logger.info(
            f"批量风险计算完成: {len(results_dict)}/{n_portfolios}成功"
        )
        return results_dict
    
    def _prepare_shared_config(self) -> Dict[str, Any]:
        """
        准备共享配置数据（P0优化）
        
        将config转换为可序列化的字典，避免传递复杂对象
        
        Returns:
            配置字典
        """
        # 如果config已经是字典，直接返回
        if isinstance(self.config, dict):
            return self.config
        
        # 如果是对象，转换为字典
        if hasattr(self.config, '__dict__'):
            return self.config.__dict__
        
        # 默认返回原值
        return self.config
    
    def _calculate_single_portfolio(
        self, 
        item: Tuple[str, Any, Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any]]:
        """计算单个组合风险（实例方法）"""
        portfolio_id, portfolio_state, market_data = item
        try:
            data = {
                'portfolio_state': portfolio_state,
                'market_data': market_data
            }
            result = self.analyze(data, {})
            return (portfolio_id, result)
        except Exception as e:
            logger.error(f"组合{portfolio_id}计算失败: {e}")
            return (portfolio_id, {})
    
    def batch_calculate_risk_contributions(
        self,
        portfolios_with_cov: List[Tuple[str, Any, pd.DataFrame]],
        use_parallel: Optional[bool] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        批量计算风险贡献度（并行优化）
        
        Args:
            portfolios_with_cov: List[(portfolio_id, portfolio_state, covariance_matrix)]
            use_parallel: 是否使用并行
        
        Returns:
            {portfolio_id: {symbol: contribution}}
        """
        if use_parallel is None:
            use_parallel = self.enable_parallel
        
        def calculate_single_contribution(item):
            portfolio_id, portfolio_state, cov_matrix = item
            try:
                contributions = self.calculate_risk_contributions_covariance(
                    portfolio_state, cov_matrix
                )
                return (portfolio_id, contributions)
            except Exception as e:
                logger.error(f"组合{portfolio_id}风险贡献度计算失败: {e}")
                return (portfolio_id, {})
        
        n_items = len(portfolios_with_cov)
        if use_parallel and n_items >= 10 and hasattr(self, 'parallel_executor'):
            logger.info(f"并行计算{n_items}个风险贡献度")
            results_list = self.parallel_executor.map_cpu_intensive(
                calculate_single_contribution,
                portfolios_with_cov
            )
        else:
            results_list = [calculate_single_contribution(item) for item in portfolios_with_cov]
        
        results_dict = {pid: result for pid, result in results_list if result}
        return results_dict
    
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """
        获取优化组件的性能指标
        
        Returns:
            {
                'parallel_metrics': 并行计算指标,
                'incremental_metrics': 增量计算指标
            }
        """
        metrics = {
            'parallel_enabled': self.enable_parallel,
            'incremental_enabled': self.enable_incremental
        }
        
        if self.enable_parallel and hasattr(self, 'parallel_executor'):
            metrics['parallel_metrics'] = self.parallel_executor.get_metrics()
        
        if self.enable_incremental and hasattr(self, 'incremental_calculator'):
            metrics['incremental_metrics'] = {
                'consecutive_updates': self.incremental_calculator.consecutive_updates,
                'cumulative_error': self.incremental_calculator.cumulative_error
            }
        
        return metrics


