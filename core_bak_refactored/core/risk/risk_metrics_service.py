"""
风险指标业务服务
职责：将数学计算映射为金融风险概念，包含业务逻辑
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import logging
from scipy import stats

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from infrastructure.risk_metrics import StatisticalCalculator
from core.risk.international_config import MarketConfigManager
from core.risk.international_enhancements import InternationalEnhancements

logger = logging.getLogger('DeepSeekQuant.RiskMetricsService')


class RiskMetricsService(InternationalEnhancements):
    """风险指标业务服务 - 负责数学到业务的映射，支持国际化"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.calculator = StatisticalCalculator()
        
        # 国际化增强：市场类型和配置
        self.market_type = config.get('market_type', 'CN')  # 默认A股市场
        self.config_manager = MarketConfigManager()
        
        # 获取市场特定配置
        self.market_configs = config.get('market_configs', {})
        if self.market_type not in self.market_configs:
            # 如果缺少配置，使用配置管理器生成默认配置
            default_config = self.config_manager.generate_config_template(self.market_type)
            self.market_configs = default_config['market_configs']
        
        self.current_market_config = self.market_configs.get(
            self.market_type,
            self.market_configs.get('CN', {})  # 回退到CN配置
        )
        
        # 市场异常检测历史记录
        self.anomaly_history = []
        
        # 业务参数（可配置，优先使用市场配置）
        self.trading_days_per_year = self.current_market_config.get(
            'trading_days',
            config.get('trading_days_per_year', 252)
        )
        
        # P0修复：分层置信度配置
        self.confidence_levels = config.get('confidence_levels', {
            'daily_monitoring': 0.95,      # 日常监控
            'risk_limit': 0.99,            # 风险限额
            'regulatory_reporting': 0.99   # 监管报告
        })
        self.default_confidence_level = config.get('default_confidence_level', 
                                                   self.confidence_levels['daily_monitoring'])
        
        # P0修复：支持动态无风险利率（优先使用市场配置）
        self.risk_free_rate = self.current_market_config.get(
            'risk_free_rate',
            config.get('risk_free_rate', 0.03)
        )
        self.dynamic_risk_free_rate = config.get('dynamic_risk_free_rate', None)  # 优先使用
        
        # P0修复：涨跌停配置（仅CN市场适用）
        if self.market_type == 'CN':
            self.limit_thresholds = self.current_market_config.get('limit_thresholds', {
                'main_board': 0.10,     # 主板±10%
                'gem': 0.20,            # 创业板±20%
                'st': 0.05,             # ST股±5%
                'kcb': 0.20             # 科创板±20%
            })
            self.default_limit_threshold = config.get('default_limit_threshold', 0.10)
        else:
            self.limit_thresholds = {}
            self.default_limit_threshold = None
        
        # US市场熔断配置
        if self.market_type == 'US':
            self.circuit_breaker_levels = self.current_market_config.get(
                'circuit_breaker_levels',
                [0.07, 0.13, 0.20]
            )
            self.luld_threshold = self.current_market_config.get('luld_threshold', 0.05)
            self.luld_window = self.current_market_config.get('luld_window', 5)
        else:
            self.circuit_breaker_levels = []
            self.luld_threshold = None
            self.luld_window = None
        
        logger.info(
            f"风险指标服务初始化完成 - 市场: {self.market_type}, "
            f"交易日: {self.trading_days_per_year}, "
            f"无风险利率: {self.risk_free_rate:.4f}"
        )
    
    def _detect_market_anomalies(self, returns: pd.Series, prices: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        检测市场异常（内嵌检测逻辑）
        
        业务层逻辑：根据不同市场类型检测特定的市场机制
        
        Args:
            returns: 收益率序列
            prices: 价格序列（可选，用于LULD检测）
            
        Returns:
            异常字典
        """
        anomalies = {}
        
        if returns is None or len(returns) == 0:
            return anomalies
        
        # CN市场：涨跌停检测
        if self.market_type == 'CN' and self.limit_thresholds:
            for board_type, threshold in self.limit_thresholds.items():
                limit_hit = self._detect_cn_limit_up_down(returns, threshold, board_type)
                if limit_hit:
                    anomalies[f'limit_up_down_{board_type}'] = {
                        'type': 'limit_up_down',
                        'board_type': board_type,
                        'threshold': threshold,
                        'severity': 'high',
                        'count': limit_hit['count'],
                        'dates': limit_hit.get('dates', [])
                    }
        
        # US市场：熔断、LULD检测
        elif self.market_type == 'US':
            # 熔断机制
            circuit_anomaly = self._detect_us_circuit_breaker(returns)
            if circuit_anomaly:
                anomalies['circuit_breaker'] = circuit_anomaly
            
            # LULD检测
            if prices is not None and len(prices) > 0:
                luld_anomaly = self._detect_us_luld(returns, prices)
                if luld_anomaly:
                    anomalies['luld'] = luld_anomaly
        
        return anomalies
    
    def _detect_cn_limit_up_down(self, returns: pd.Series, threshold: float, board_type: str) -> Optional[Dict]:
        """检测CN市场涨跌停"""
        detection_threshold = 0.95  # 95%阈值
        abs_returns = np.abs(returns.values)
        detection_level = detection_threshold * threshold
        
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
    
    def _detect_us_circuit_breaker(self, returns: pd.Series) -> Optional[Dict[str, Any]]:
        """检测US市场熔断机制"""
        if not self.circuit_breaker_levels or len(returns) == 0:
            return None
        
        abs_returns = np.abs(returns.values)
        detection_threshold = 0.95
        
        for level in sorted(self.circuit_breaker_levels, reverse=True):
            detection_level = level * detection_threshold
            
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
    
    def _detect_us_luld(self, returns: pd.Series, prices: pd.Series) -> Optional[Dict[str, Any]]:
        """检测US市场LULD（波动率中断）"""
        if self.luld_threshold is None or self.luld_window is None:
            return None
        
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
    
    def get_risk_free_rate(self, date=None, benchmark='CGB') -> float:
        """
        获取无风险利率
        
        P0修复：支持动态获取无风险利率
        
        Args:
            date: 日期（预留，用于未来从数据源获取）
            benchmark: 基准类型（CGB=国债, SHIBOR=银行间拆借）
            
        Returns:
            年化无风险利率
        """
        # 优先使用动态配置的利率
        if self.dynamic_risk_free_rate is not None:
            return self.dynamic_risk_free_rate
        
        # 后备：使用静态配置
        # TODO: 未来可从数据源动态获取（如Wind, Bloomberg）
        return self.risk_free_rate
    
    def _has_limit_hit(self, returns: pd.Series, threshold: float = 0.95, 
                       board_type: str = 'main_board') -> bool:
        """
        检测涨跌停
        
        P0修复：A股涨跌停检测机制
        
        Args:
            returns: 收益率序列
            threshold: 检测阈值（默认95%）
            board_type: 板块类型 ('main_board', 'gem', 'st', 'kcb')
            
        Returns:
            是否检测到涨跌停
        """
        if returns is None or len(returns) == 0:
            return False
        
        # 获取该板块的涨跌停限制
        limit = self.limit_thresholds.get(board_type, self.default_limit_threshold)
        if limit is None:
            return False  # 无涨跌停限制的市场
        
        # 检测收益率是否接近涨跌停阈值
        abs_returns = np.abs(returns.values)
        limit_hit = np.any(abs_returns >= threshold * limit)
        
        return bool(limit_hit)

    def calculate_volatility(self, 
                           returns: pd.Series, 
                           window: Optional[int] = None,
                           annualize: bool = True) -> float:
        """
        计算波动率【业务层】
        
        业务含义：衡量资产价格波动的剧烈程度
        应用场景：风险预算、仓位管理、风险控制
        
        Args:
            returns: 收益率序列
            window: 滚动窗口大小，None表示全样本
            annualize: 是否年化
        """
        try:
            # 调用基础设施层
            std = self.calculator.calculate_standard_deviation(returns.values, window)
            
            # 业务逻辑：年化处理
            if annualize:
                std = std * np.sqrt(self.trading_days_per_year)
            
            return float(std)
            
        except Exception as e:
            logger.error(f"波动率计算失败: {e}")
            return 0.0

    def calculate_value_at_risk(self, 
                              returns: pd.Series, 
                              confidence_level: Optional[float] = None,
                              method: str = 'historical',
                              adjust_limit: bool = True,
                              board_type: str = 'main_board') -> float:
        """
        计算风险价值（VaR）【业务层】
        
        业务含义：在给定置信度下可能的最大损失
        应用场景：风险限额、监管报告、风险监控
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平，默认0.95
            method: 计算方法 'historical'或'parametric'
            adjust_limit: 是否启用涨跌停检测（P0修复）
            board_type: 板块类型，用于涨跌停检测
        """
        try:
            # P0修复：涨跌停检测
            if adjust_limit and self._has_limit_hit(returns, board_type=board_type):
                logger.warning(
                    f"检测到涨跌停数据（板块类型: {board_type}），"
                    f"收益率分布可能被截断，当前VaR可能低估真实风险。"
                    f"建议：在风险报告中标注此情况，并考虑使用更保守的风险限额。"
                )
            
            confidence = confidence_level or self.default_confidence_level
            quantile_level = 1 - confidence
            
            if method == 'historical':
                # 历史模拟法
                var = self.calculator.calculate_quantile(returns.values, quantile_level)
                return float(abs(var))  # 转换为正数表示损失
                
            elif method == 'parametric':
                # 参数法（正态分布假设）
                mean_return = float(returns.mean())
                std_return = self.calculate_volatility(returns, annualize=False)
                
                z_score = stats.norm.ppf(quantile_level)
                var = mean_return + z_score * std_return
                return float(abs(var))
                
            else:
                raise ValueError(f"不支持的VaR计算方法: {method}")
                
        except Exception as e:
            logger.error(f"VaR计算失败: {e}")
            return 0.1

    def calculate_expected_shortfall(self, 
                                   returns: pd.Series, 
                                   confidence_level: Optional[float] = None) -> float:
        """
        计算预期短缺（CVaR/ES）【业务层】
        
        业务含义：超过VaR的损失的期望值
        应用场景：尾部风险管理、压力测试
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
        """
        try:
            confidence = confidence_level or self.default_confidence_level
            var = self.calculate_value_at_risk(returns, confidence)
            
            # 计算超过VaR的损失的平均值
            losses_beyond_var = returns[returns <= -var]
            if len(losses_beyond_var) > 0:
                cvar = float(abs(losses_beyond_var.mean()))
            else:
                # P0修复：使用参数法CVaR作为保守估计
                logger.warning(f"尾部数据不足，使用参数法CVaR估计")
                cvar = self._calculate_parametric_cvar(returns, confidence)
                
            return cvar
            
        except Exception as e:
            logger.error(f"CVaR计算失败: {e}")
            return 0.15
    
    def _calculate_parametric_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """
        参数法CVaR（正态分布假设）
        
        公式：CVaR = μ + σ * (φ(z_α) / α)
        其中：φ为正态分布PDF，z_α为VaR对应的z分数
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
            
        Returns:
            参数法CVaR（损失金额，正数）
        """
        mu = float(returns.mean())
        sigma = float(returns.std())
        alpha = 1 - confidence_level
        
        # 正态分布CVaR公式
        z_alpha = stats.norm.ppf(alpha)
        cvar = mu + sigma * (stats.norm.pdf(z_alpha) / alpha)
        
        return float(abs(cvar))

    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        计算最大回撤【业务层】
        
        业务含义：从峰值到谷底的最大跌幅
        应用场景：风险评估、策略选择、风险控制
        
        Args:
            returns: 收益率序列
        """
        try:
            # 计算累积收益
            cumulative = np.cumprod(1 + returns.values) - 1
            
            # 调用基础设施层
            drawdown = self.calculator.calculate_cumulative_peak_deviation(cumulative)
            max_dd = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
            return float(max_dd)
            
        except Exception as e:
            logger.error(f"最大回撤计算失败: {e}")
            return 0.0

    def calculate_sharpe_ratio(self, 
                             returns: pd.Series, 
                             risk_free_rate: Optional[float] = None) -> float:
        """
        计算夏普比率【业务层】
        
        业务含义：每单位风险的超额收益
        应用场景：策略绩效评估、风险调整收益比较
        
        Args:
            returns: 收益率序列
            risk_free_rate: 年化无风险利率（None则使用动态获取）
        """
        try:
            # P0修复：使用动态无风险利率
            rf_rate = risk_free_rate if risk_free_rate is not None else self.get_risk_free_rate()
            daily_rf = rf_rate / self.trading_days_per_year
            
            # 调用基础设施层
            sharpe = self.calculator.calculate_mean_std_ratio(returns.values, daily_rf)
            
            # 业务逻辑：年化
            sharpe_annual = sharpe * np.sqrt(self.trading_days_per_year)
            return float(sharpe_annual)
            
        except Exception as e:
            logger.error(f"夏普比率计算失败: {e}")
            return 0.0

    def calculate_sortino_ratio(self, 
                              returns: pd.Series, 
                              risk_free_rate: Optional[float] = None) -> float:
        """
        计算索提诺比率【业务层】
        
        业务含义：每单位下行风险的超额收益
        应用场景：更准确的风险调整收益评估
        
        Args:
            returns: 收益率序列
            risk_free_rate: 年化无风险利率（None则使用动态获取）
        """
        try:
            # P0修复：使用动态无风险利率
            rf_rate = risk_free_rate if risk_free_rate is not None else self.get_risk_free_rate()
            daily_rf = rf_rate / self.trading_days_per_year
            
            excess_returns = returns.values - daily_rf
            mean_excess = float(np.mean(excess_returns))
            
            # 调用基础设施层计算下行标准差
            downside_std = self.calculator.calculate_downside_deviation(returns.values, daily_rf)
            
            if downside_std == 0:
                return 0.0
            
            # 业务逻辑：年化
            sortino = mean_excess / downside_std
            sortino_annual = sortino * np.sqrt(self.trading_days_per_year)
            return float(sortino_annual)
            
        except Exception as e:
            logger.error(f"索提诺比率计算失败: {e}")
            return 0.0

    def calculate_beta(self, 
                     asset_returns: pd.Series, 
                     market_returns: pd.Series) -> float:
        """
        计算贝塔系数【业务层】
        
        业务含义：资产相对于市场的系统性风险
        应用场景：资产定价、风险分解、组合构建
        
        Args:
            asset_returns: 资产收益率序列
            market_returns: 市场收益率序列
        """
        try:
            # 调用基础设施层
            beta = self.calculator.calculate_covariance_variance_ratio(
                asset_returns.values, market_returns.values
            )
            return float(beta) if not np.isnan(beta) else 1.0
            
        except Exception as e:
            logger.error(f"贝塔计算失败: {e}")
            return 1.0

    def calculate_alpha(self, 
                      asset_returns: pd.Series, 
                      market_returns: pd.Series,
                      beta: Optional[float] = None) -> float:
        """
        计算阿尔法【业务层】
        
        业务含义：超越市场的超额收益
        应用场景：主动管理能力评估、策略绩效
        
        Args:
            asset_returns: 资产收益率序列
            market_returns: 市场收益率序列
            beta: 贝塔系数，None则自动计算
        """
        try:
            if beta is None:
                beta = self.calculate_beta(asset_returns, market_returns)
                
            # 调用基础设施层计算残差
            residuals = self.calculator.calculate_residual(
                asset_returns.values, market_returns.values, beta
            )
            
            alpha_daily = float(np.mean(residuals))
            
            # 业务逻辑：年化
            alpha_annual = alpha_daily * self.trading_days_per_year
            return float(alpha_annual)
            
        except Exception as e:
            logger.error(f"阿尔法计算失败: {e}")
            return 0.0

    def calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """
        计算卡尔玛比率【业务层】
        
        业务含义：年化收益与最大回撤的比率
        应用场景：回撤敏感的策略评估
        
        Args:
            returns: 收益率序列
        """
        try:
            # 计算年化收益
            annual_return = float(returns.mean()) * self.trading_days_per_year
            
            # 计算最大回撤
            max_dd = self.calculate_max_drawdown(returns)
            
            if max_dd > 0:
                return float(annual_return / max_dd)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"卡尔玛比率计算失败: {e}")
            return 0.0

    def calculate_all_metrics(self, 
                            returns: pd.Series,
                            market_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        计算所有风险指标【业务层】
        
        应用场景：综合风险评估报告
        
        Args:
            returns: 资产收益率序列
            market_returns: 市场收益率序列（可选）
        """
        try:
            metrics = {
                'volatility': self.calculate_volatility(returns),
                'var_95': self.calculate_value_at_risk(returns, 0.95),
                'cvar_95': self.calculate_expected_shortfall(returns, 0.95),
                'max_drawdown': self.calculate_max_drawdown(returns),
                'sharpe_ratio': self.calculate_sharpe_ratio(returns),
                'sortino_ratio': self.calculate_sortino_ratio(returns),
                'calmar_ratio': self.calculate_calmar_ratio(returns),
            }
            
            # 如果提供了市场收益，计算beta和alpha
            if market_returns is not None and len(market_returns) > 0:
                metrics.update({
                    'beta': self.calculate_beta(returns, market_returns),
                    'alpha': self.calculate_alpha(returns, market_returns),
                })
                
            return metrics
            
        except Exception as e:
            logger.error(f"综合风险指标计算失败: {e}")
            return {}
