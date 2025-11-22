"""
风险计算器 - 业务层
从 core_bak/risk_manager.py 拆分
职责: 协调器 - 统一风险计算入口，委托给业务服务层
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, TypedDict, Protocol
import logging
import time
import warnings

from .risk_metrics_service import RiskMetricsService
from .risk_models import RiskMetric
from ..share.market_config import MarketConfigManager
from ..share.exchange_rates import CurrencyConverter, ExchangeRateAdapter

# 导入数据预处理器
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from core_bak_refactored.infrastructure.data_preprocessor import RiskDataPreprocessor

logger = logging.getLogger('DeepSeekQuant.RiskCalculator')

class MarketPriceData(TypedDict):
    close: List[float]
    high: List[float]
    low: List[float]
    volume: List[float]
    currency: Optional[str]  # TODO：补充了currency字段，待确认，来源：docs/answer.md

class MarketData(TypedDict):
    prices: Dict[str, MarketPriceData]
    risk_free_rate: Optional[float]
    market_returns: Optional[List[float]]

class PortfolioAllocation(TypedDict):
    weight: float

class PortfolioState(Protocol):
    allocations: Dict[str, PortfolioAllocation]


class RiskCalculator:
    """
    风险计算器 - 纯协调器
    
    职责：
    - 提供统一的风险计算入口
    - 委托给 RiskMetricsService 进行实际计算
    - 使用 RiskDataPreprocessor 处理数据提取
    
    设计原则：
    - 不实现具体算法，仅负责委托
    - 不直接处理数据，委托给预处理器
    """
    
    def __init__(self, config: Dict):
        # 国际化：市场配置管理器
        self.config_manager = MarketConfigManager()
        
        # 验证配置完整性
        config_errors = self.config_manager.validate_market_config(config)
        if config_errors:
            logger.warning(f"配置验证发现问题: {config_errors}")
        
        # 识别市场类型
        self.market_type = config.get('market_type', 'CN')
        
        # 确保配置完整性（自动补全缺失配置）
        if 'market_configs' not in config or self.market_type not in config.get('market_configs', {}):
            logger.warning(f"缺少{self.market_type}市场配置，使用默认配置")
            default_config = self.config_manager.generate_config_template(self.market_type)
            config['market_configs'] = default_config['market_configs']
        
        self.config = config
        self.risk_metrics_service = RiskMetricsService(config)
        self.preprocessor = RiskDataPreprocessor()
        # 可选：外部实时汇率适配器（由业务层注入）
        self.exchange_rate_adapter: Optional[ExchangeRateAdapter] = None
        self._currency_converter = CurrencyConverter()

        # TODO：补充了货币一致性检查初始化，待确认，来源：docs/answer.md
        # 基准货币与严格检查开关
        market_info = self.config_manager.get_market_info(self.market_type)
        self.base_currency = market_info.get('currency', 'CNY')
        market_configs = self.config.get('market_configs', {})
        current_market_cfg = market_configs.get(self.market_type, {})
        if 'base_currency' in current_market_cfg:
            self.base_currency = current_market_cfg['base_currency']
        default_strict = self._get_default_strict_mode(self.market_type)
        self.strict_currency_check = bool(self.config.get('strict_currency_check', default_strict))

        logger.info(
            f"风险计算器初始化完成 - 市场: {self.market_type}, "
            f"配置验证: {'有警告' if config_errors else '通过'}, 基准货币: {self.base_currency}"
        )
    
    def _get_min_data_points(self) -> int:
        """读取配置中的最小数据点阈值，默认63（约3个月交易日）"""
        try:
            return int(self.config.get('min_data_points', 63))
        except Exception:
            return 63
    
    def calculate_volatility(self, returns: pd.Series, window: Optional[int] = None, annualize: bool = True) -> float:
        """委托给 RiskMetricsService"""
        return self.risk_metrics_service.calculate_volatility(returns, window, annualize)

    def calculate_correlation_matrix(self, asset_returns: pd.DataFrame) -> pd.DataFrame:
        """相关性矩阵"""
        if asset_returns is None or asset_returns.empty:
            return pd.DataFrame()
        return asset_returns.corr().fillna(0.0)

    def calculate_var_historical(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """委托给 RiskMetricsService"""
        return self.risk_metrics_service.calculate_value_at_risk(returns, confidence_level, 'historical')

    def calculate_var_parametric(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """委托给 RiskMetricsService"""
        return self.risk_metrics_service.calculate_value_at_risk(returns, confidence_level, 'parametric')


    def calculate_var_monte_carlo(self, portfolio_state: 'PortfolioState', market_data: 'MarketData', confidence_level: float) -> float:
        """
        蒙特卡洛法VaR（简化实现）
        
        注：此方法待移至 RiskMetricsService，当前保留兼容性
        """
        logger.warning("蒙特卡洛 VaR 计算待优化，当前使用简化实现")
        try:
            start_time = time.time()
            n_simulations = int(self.config.get('monte_carlo_sims', 1000))
            if n_simulations < 1000:
                n_simulations = 1000
            symbols = list(portfolio_state.allocations.keys())
            returns_data = {}
            for symbol in symbols:
                prices = market_data['prices'][symbol].get('close', [])
                min_points = self._get_min_data_points()
                if len(prices) >= min_points:
                    # 使用预处理器计算收益
                    returns_data[symbol] = self.preprocessor.extract_returns_from_prices(np.array(prices))
            if not returns_data:
                logger.warning(
                    f"calculate_var_monte_carlo: 价格数据不足, 市场{self.market_type}, 返回NaN"
                )
                return float('nan')
            min_len = min(len(v) for v in returns_data.values())
            aligned = np.column_stack([v[-min_len:] for v in returns_data.values()])
            mean_vec = aligned.mean(axis=0)
            cov_mat = np.cov(aligned.T)
            # 从配置获取随机种子，支持可重现性
            random_seed = self.config.get('monte_carlo_seed', 42)
            np.random.seed(random_seed)
            sims = np.random.multivariate_normal(mean_vec, cov_mat, n_simulations)
            weights = np.array([alloc.get('weight', 0.0) for alloc in portfolio_state.allocations.values()])
            portfolio_sims = sims @ weights
            var = np.percentile(portfolio_sims, (1 - confidence_level) * 100)
            elapsed = time.time() - start_time
            logger.info(
                f"calculate_var_monte_carlo: 完成, 市场{self.market_type}, "
                f"耗时{elapsed:.3f}s, 模拟{n_simulations}次"
            )
            return float(var)
        except Exception as e:
            logger.error(f"calculate_var_monte_carlo: 计算异常, 市场{self.market_type}: {e}")
            return float('nan')


    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """委托给 RiskMetricsService"""
        return self.risk_metrics_service.calculate_max_drawdown(returns)
    
    # TODO：补充了货币一致性检查方法，待确认，来源：docs/answer.md
    def _runtime_currency_check(self, data: Dict[str, Any]) -> List[str]:
        """运行时货币一致性检查（仅日志，不阻断）"""
        warnings_list: List[str] = []
        market_data = data.get('market_data', {})
        prices = market_data.get('prices', {}) or {}

        detected_currencies = set()
        missing_currency_symbols: List[str] = []

        for symbol, price_data in prices.items():
            currency = price_data.get('currency')
            if currency:
                detected_currencies.add(currency)
            else:
                missing_currency_symbols.append(symbol)

        if missing_currency_symbols:
            warnings_list.append(f"{len(missing_currency_symbols)}个标的缺少货币信息")

        if len(detected_currencies) > 1:
            warnings_list.append(f"多币种检测: {detected_currencies}")

        if detected_currencies and getattr(self, 'base_currency', None) and self.base_currency not in detected_currencies:
            warnings_list.append(f"基准货币{self.base_currency}不在检测货币中")

        portfolio = data.get('portfolio', {})
        portfolio_currency = portfolio.get('base_currency')
        if portfolio_currency and getattr(self, 'base_currency', None) and portfolio_currency != self.base_currency:
            warnings_list.append(f"组合货币{portfolio_currency}≠基准货币{self.base_currency}")

        return warnings_list

    def _handle_currency_warnings(self, warnings: List[str]) -> None:
        """分级处理货币警告（严格模式可抛错）"""
        if not warnings:
            return
        # 分类警告（根据市场类型调整严重性）
        classified = self._classify_currency_warnings(warnings)
        for msg in classified.get('info', []):
            logger.info(f"货币信息: {msg}")
        for msg in classified.get('warning', []):
            logger.warning(f"货币警告: {msg}")
        for msg in classified.get('error', []):
            logger.error(f"货币错误: {msg}")
        # 严格模式下如存在错误则抛出异常
        if classified.get('error') and getattr(self, 'strict_currency_check', False):
            raise ValueError(f"货币单位检查失败: {classified.get('error')}")

    def _check_risk_parameters_currency(self, data: Dict[str, Any]) -> List[str]:
        """检查风险参数（无风险利率/市场收益）的货币一致性"""
        warnings_list: List[str] = []
        market_data = data.get('market_data', {})
        risk_free_info = market_data.get('risk_free_rate_info', {})
        if risk_free_info.get('currency') and risk_free_info['currency'] != self.base_currency:
            warnings_list.append(
                f"无风险利率货币{risk_free_info['currency']}≠基准货币{self.base_currency}"
            )
        market_returns_info = market_data.get('market_returns_info', {})
        if market_returns_info.get('currency') and market_returns_info['currency'] != self.base_currency:
            warnings_list.append(
                f"市场收益货币{market_returns_info['currency']}≠基准货币{self.base_currency}"
            )
        return warnings_list

    def _classify_currency_warnings(self, warnings: List[str]) -> Dict[str, List[str]]:
        """更精细的警告分类，按市场类型调整严重性"""
        info: List[str] = []
        warn: List[str] = []
        err: List[str] = []
        for w in warnings:
            if '多币种' in w:
                # 美股多币种常见，降级为信息
                if self.market_type == 'US':
                    info.append(w)
                else:
                    warn.append(w)
            elif ('缺少' in w):
                warn.append(w)
            elif ('不在' in w) or ('≠' in w):
                # 组合不一致在美股更严格，提升为错误
                if self.market_type == 'US':
                    err.append(w)
                else:
                    warn.append(w)
            else:
                warn.append(w)
        return {'info': info, 'warning': warn, 'error': err}

    def _get_default_strict_mode(self, market_type: str) -> bool:
        """根据市场类型获取默认严格模式（基于专家answer.md建议：US/HK/SG/JP严格）"""
        market_strict_defaults = {
            'US': True,
            'HK': True,
            'SG': True,  # 新加坡默认严格模式
            'JP': True,  # 日本调整为严格模式（日元为重要国际货币）
            'CN': False,
            'EU': False,
        }
        return market_strict_defaults.get(market_type, False)

    def _assess_data_source_quality(self, prices: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """评估数据源货币信息完整性，且C/D级时记录告警提示自动处理"""
        total_symbols = len(prices)
        symbols_with_currency = sum(1 for p in prices.values() if p.get('currency'))
        currency_coverage = symbols_with_currency / total_symbols if total_symbols > 0 else 0.0
        # 调整评级：A(>=95%)、B(>=80%)、C(>=50%)、D(<50%)
        if currency_coverage >= 0.95:
            rating = 'A'
        elif currency_coverage >= 0.80:
            rating = 'B'
        elif currency_coverage >= 0.50:
            rating = 'C'
        else:
            rating = 'D'
        # 如果为C/D级，记录告警（暂不自动触发清洗，留待后续增强）
        if rating in ['C', 'D']:
            missing_count = total_symbols - symbols_with_currency
            logger.warning(
                f"数据源质量{rating}级：货币覆盖{currency_coverage:.2%}，"
                f"{missing_count}个标的缺失货币信息 - 建议触发数据清洗"
            )
        return {
            'total_symbols': total_symbols,
            'currency_coverage': currency_coverage,
            'quality_rating': rating,
        }

    def attach_exchange_rate_adapter(self, adapter: 'ExchangeRateAdapter') -> None:
        """注入外部实时汇率适配器（不改变现有指标计算）"""
        self.exchange_rate_adapter = adapter
        logger.info(f"已注入外部汇率适配器: {adapter.__class__.__name__}")

    def _unify_currency_for_portfolio(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """在存在多币种或不一致时，统一转换到基准货币（仅生成摘要，不影响计算）"""
        if not self.exchange_rate_adapter:
            return None
        market_data = data.get('market_data', {})
        prices = market_data.get('prices', {}) or {}
        # 仅当检测到多币种或组合≠基准货币时才尝试转换
        warnings_list = self._runtime_currency_check(data)
        if not warnings_list:
            return None
        try:
            rates = self.exchange_rate_adapter.get_rates(self.market_type)
            # 构造最小组合估值结构（从prices推断）
            portfolio = {'allocations': {}}
            for symbol, price_data in prices.items():
                # 使用最后一个close近似估值（仅摘要用途）
                closes = price_data.get('close') or []
                last_value = float(closes[-1]) if closes else 0.0
                src_cur = price_data.get('currency', self.base_currency)
                portfolio['allocations'][symbol] = {'currency': src_cur, 'value': last_value}
            summary = self._currency_converter.convert_portfolio_currency(
                portfolio, target_currency=self.base_currency, rates=rates
            )
            logger.info(
                f"统一货币摘要: 目标{self.base_currency}, 合计{summary['total_converted_value']:.4f}")
            return summary
        except Exception as e:
            logger.warning(f"统一货币失败（不影响计算）: {e}")
            return None

    def _us_compliance_logging(self, currency_warnings: List[str]) -> None:
        """美股合规性日志记录（仅US市场），增强结构化日志"""
        if self.market_type != 'US' or not currency_warnings:
            return
        import uuid
        from datetime import datetime as dt
        compliance_events = []
        for warning in currency_warnings:
            if ('多币种' in warning) or ('≠' in warning):
                compliance_events.append({
                    'event_id': str(uuid.uuid4()),
                    'event_type': 'CURRENCY_INCONSISTENCY',
                    'message': warning,
                    'timestamp': dt.utcnow().isoformat() + 'Z',  # ISO8601
                    'market': self.market_type,
                    'severity': 'MEDIUM' if '多币种' in warning else 'HIGH',
                    'automated_action': 'LOG_ONLY'  # 暂不阻断交易
                })
        if compliance_events:
            logger.warning(
                f"[US_COMPLIANCE_EVENT] 货币一致性问题检测 - 市场: {self.market_type}, "
                f"警告数量: {len(compliance_events)}, 详情: {'; '.join(currency_warnings)}",
                extra={'compliance_events': compliance_events}
            )

    def calculate_all_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        计算所有风险指标
        
        职责：
        - 委托 RiskDataPreprocessor 提取数据
        - 委托 RiskMetricsService 计算指标
        """
        try:
            # TODO：补充了货币一致性运行时检查，强化数据源质量评估，来源：docs/answer.md
            currency_warnings = self._runtime_currency_check(data)
            currency_warnings += self._check_risk_parameters_currency(data)
            self._handle_currency_warnings(currency_warnings)
            # 美股合规日志
            self._us_compliance_logging(currency_warnings)
            # 数据源质量评估（调整为A/B/C/D分级）
            data_quality: Dict[str, Any] = {}
            market_data_prices = data.get('market_data', {}).get('prices', {}) or {}
            if market_data_prices:
                data_quality = self._assess_data_source_quality(market_data_prices)
                logger.info(f"数据源质量评级: {data_quality['quality_rating']}, "
                            f"货币覆盖率: {data_quality['currency_coverage']:.2%}")
            # 在存在多币种/不一致时，尝试统一货币（仅摘要，不影响指标计算）
            self._unify_currency_for_portfolio(data)

            # 数据提取委托给预处理器
            returns = self.preprocessor.extract_returns_from_dict(data)
            market_returns = self.preprocessor.extract_market_returns_from_dict(data)
            # 智能缓存失效触发（波动率或重大事件）
            try:
                from core_bak_refactored.infrastructure.cache_service import get_smart_invalidation_manager
                manager = get_smart_invalidation_manager()
                returns_std = float(np.std(returns.values)) if returns is not None and len(returns) > 0 else 0.0
                base_threshold = float(self.config.get('volatility_spike_threshold', self.config.get('market_configs', {}).get(self.market_type, {}).get('volatility_spike_threshold', 0.05)))
                vol_tier = 'NORMAL'
                if returns_std > base_threshold * 2.0:
                    vol_tier = 'EXTREME'
                elif returns_std > base_threshold * 1.5:
                    vol_tier = 'HIGH'
                elif returns_std > base_threshold:
                    vol_tier = 'MEDIUM'
                market_status = 'EXTREME' if (vol_tier == 'EXTREME' or bool(data.get('liquidity_stressed', False))) else ('VOLATILE' if vol_tier in ('HIGH','MEDIUM') else 'NORMAL')
                context = {
                    'time_window': str(int(time.time())),
                    'param_version': str(self.config.get('param_version', 'v1')),
                    'market_data_updated': bool(data.get('market_data_updated', False)),
                    'volatility': returns_std,
                    'market_type': self.market_type,
                    'portfolio_size': len(data.get('portfolio_state', {}).get('allocations', {})) if isinstance(data.get('portfolio_state'), dict) else (len(getattr(data.get('portfolio_state'), 'allocations', {})) if data.get('portfolio_state') else 0),
                    'data_quality_rating': data_quality.get('quality_rating'),
                    'volatility_tier': vol_tier,
                    'market_status': market_status,
                    'circuit_breaker_triggered': bool(data.get('circuit_breaker_triggered', False)),
                    'extreme_correlation_breakdown': bool(data.get('extreme_correlation_breakdown', False)),
                    'limit_hit_ratio': float(data.get('limit_hit_ratio', 0.0)),
                    'major_market_event': bool(data.get('major_market_event', False))
                }
                threshold = float(self.config.get('volatility_spike_threshold', self.config.get('market_configs', {}).get(self.market_type, {}).get('volatility_spike_threshold', 0.05)))
                limit_threshold = float(self.config.get('limit_hit_ratio_threshold', self.config.get('market_configs', {}).get(self.market_type, {}).get('limit_hit_ratio_threshold', 0.3)))
                event_trigger = bool(data.get('circuit_breaker_triggered', False)) or bool(data.get('extreme_correlation_breakdown', False)) or (float(data.get('limit_hit_ratio', 0.0)) > limit_threshold) or bool(data.get('major_market_event', False))
                weights = self.config.get('market_configs', {}).get(self.market_type, {}).get('event_weights', {})
                trigger_score = (returns_std / threshold if threshold > 0 else 0.0)
                trigger_score += (weights.get('circuit_breaker', 0.0) if bool(data.get('circuit_breaker_triggered', False)) else 0.0)
                trigger_score += (weights.get('extreme_correlation', 0.0) if bool(data.get('extreme_correlation_breakdown', False)) else 0.0)
                trigger_score += (weights.get('limit_hits', 0.0) if float(data.get('limit_hit_ratio', 0.0)) > limit_threshold else 0.0)
                trigger_score += (weights.get('major_event', 0.0) if bool(data.get('major_market_event', False)) else 0.0)
                context['trigger_score'] = float(trigger_score)
                # 影响范围估算
                try:
                    if isinstance(data.get('portfolio_state'), dict):
                        context['affected_symbols_count'] = len(data.get('portfolio_state', {}).get('allocations', {}))
                    else:
                        context['affected_symbols_count'] = len(getattr(data.get('portfolio_state'), 'allocations', {})) if data.get('portfolio_state') else 0
                except Exception:
                    context['affected_symbols_count'] = 0
                if returns_std > threshold or event_trigger:
                    manager.check_and_invalidate(context)
            except Exception:
                pass
            start_time = time.time()
            
            # 验证数据有效性
            min_points = self._get_min_data_points()
            if not self.preprocessor.validate_returns_data(returns, min_length=min_points):
                logger.warning(
                f"calculate_all_metrics: 收益数据不足, 市场{self.market_type}, "
                f"至少需要{min_points}个数据点"
            )
                return {}
            
            # 计算委托给服务层
            metrics = self.risk_metrics_service.calculate_all_metrics(returns, market_returns)
            elapsed = time.time() - start_time
            logger.info(
                f"calculate_all_metrics: 完成, 市场{self.market_type}, "
                f"耗时{elapsed:.3f}s, 指标{len(metrics)}个"
            )
            return metrics
            
        except Exception as e:
            logger.error(f"风险指标计算失败, 市场{self.market_type}: {e}")
            return {}
    
    def simulate_correlation_breakdown(self, scenario, portfolio_state, market_data):
        """迁移到 StressTester，暂不在此实现"""
        warnings.warn(
            "simulate_correlation_breakdown 已废弃，请使用 StressTester.simulate_correlation_breakdown",
            DeprecationWarning,
            stacklevel=2
        )
        raise NotImplementedError("Use StressTester.simulate_correlation_breakdown")




