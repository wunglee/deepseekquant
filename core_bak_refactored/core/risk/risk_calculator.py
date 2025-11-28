"""
风险计算器 - 业务层
从 core_bak/risk_manager.py 拆分
职责: 协调器 - 统一风险计算入口，委托给业务服务层

完成特性摘要（迁移自TODO.md）：
- 货币一致性检查：参数货币检查、警告分类（info/warning/error）、默认严格模式（US/HK/SG/JP）
- 数据源质量评估：currency覆盖度评级；多币种检测与日志
- 合规日志：US市场SEC/FINRA合规事件记录
- 汇率适配器：`attach_exchange_rate_adapter` 与 `_unify_currency_for_portfolio` 集成
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
        self.current_market_cfg = current_market_cfg
        default_strict = self._get_default_strict_mode(self.market_type)
        self.strict_currency_check = bool(self.config.get('strict_currency_check', default_strict))

        logger.info(
            f"风险计算器初始化完成 - 市场: {self.market_type}, "
            f"配置验证: {'有警告' if config_errors else '通过'}, 基准货币: {self.base_currency}"
        )
        self._validate_required_fields()
    
    def _get_min_data_points(self) -> int:
        """读取配置中的最小数据点阈值（支持分市场覆盖），默认63（约3个月交易日）"""
        try:
            # 优先使用分市场配置
            market_cfg = (getattr(self, 'current_market_cfg', None)
                          or self.config.get('market_configs', {}).get(self.market_type, {}))
            if isinstance(market_cfg, dict) and 'min_data_points' in market_cfg:
                return int(market_cfg.get('min_data_points', 63))
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
        蒙特卡洛法VaR（委托给服务层）
        
        @deprecated: 迁移至 RiskMetricsService.calculate_var_monte_carlo
        此处保留兼容性调用，建议直接使用服务层方法
        """
        import warnings
        warnings.warn(
            "RiskCalculator.calculate_var_monte_carlo 已迁移至 RiskMetricsService，请直接调用服务层方法",
            DeprecationWarning,
            stacklevel=2
        )
        logger.info(f"委托蒙特卡洛VaR至服务层，市场{self.market_type}")
        
        # 委托给服务层
        return self.risk_metrics_service.calculate_var_monte_carlo(
            portfolio_state=portfolio_state,
            market_data=market_data,
            confidence_level=confidence_level
        )


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

    def _determine_dynamic_strict_mode(self, data: Dict[str, Any]) -> Optional[bool]:
        """
        动态严格模式决策器（仅按配置阈值判断；无配置则不覆盖）
        
        Args:
            data: 包含 portfolio 和 market_data 的数据字典
            
        Returns:
            True/False（需覆盖静态严格模式），或 None（无配置/数据不足，保持静态模式）
        """
        try:
            dynamic_cfg = self.config.get('dynamic_currency_strict_mode', {})
            if not isinstance(dynamic_cfg, dict) or not bool(dynamic_cfg.get('enabled', False)):
                return None
            portfolio = data.get('portfolio', {}) or {}
            allocations = portfolio.get('allocations', {}) or {}
            market_data = data.get('market_data', {}) or {}
            # 计算各子评分（均需配置与数据支持）
            mc_score = self._calculate_multi_currency_score(allocations, market_data, dynamic_cfg)
            cb_score = self._calculate_cross_border_score(portfolio, dynamic_cfg)
            reg_score = self._calculate_regulatory_overlay_score(allocations, dynamic_cfg)
            # 仅当配置中要求的维度都有分数时才计算综合评分
            scores = {}
            if mc_score is not None:
                scores['multi_currency'] = mc_score
            if cb_score is not None:
                scores['cross_border'] = cb_score
            if reg_score is not None:
                scores['regulatory'] = reg_score
            if not scores:
                return None
            comp_score = self._calculate_comprehensive_score(scores, dynamic_cfg)
            if comp_score is None:
                return None
            threshold = float(dynamic_cfg.get('comprehensive_trigger_score')) if dynamic_cfg.get('comprehensive_trigger_score') is not None else None
            if threshold is None:
                return None
            return bool(comp_score >= threshold)
        except Exception:
            return None

    def _calculate_multi_currency_score(self, allocations: Dict[str, Any], market_data: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[float]:
        """
        多币种占比分数：使用组合权重与价格货币，需提供阈值multi_currency_ratio_threshold
        
        Args:
            allocations: 组合配置，symbol -> {'weight': float}
            market_data: 市场数据，包含 prices
            cfg: 动态严格模式配置
            
        Returns:
            非基准货币权重比例（0-1），或 None（配置缺失）
        """
        try:
            thr = cfg.get('multi_currency_ratio_threshold')
            if thr is None:
                return None
            prices = (market_data.get('prices') or {})
            total_w = 0.0
            non_base_w = 0.0
            for symbol, alloc in allocations.items():
                w = float(alloc.get('weight', 0.0))
                total_w += w
                cur = (prices.get(symbol, {}) or {}).get('currency')
                if cur and cur != self.base_currency:
                    non_base_w += w
            ratio = (non_base_w / total_w) if total_w > 0 else 0.0
            return float(ratio)
        except Exception:
            return None

    def _calculate_cross_border_score(self, portfolio: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[float]:
        """
        跨境敞口分数：优先使用 portfolio['cross_border_exposure']；否则返回None
        
        Args:
            portfolio: 组合数据，需包含 cross_border_exposure 字段
            cfg: 动态严格模式配置，需包含 cross_border_exposure_threshold
            
        Returns:
            跨境敞口比例（0-1），或 None（数据缺失）
        """
        try:
            per_market_thr = (cfg.get('cross_border_exposure_threshold') or {})
            if not isinstance(per_market_thr, dict):
                return None
            exposure = portfolio.get('cross_border_exposure')
            if exposure is None:
                return None
            return float(exposure)
        except Exception:
            return None

    def _calculate_regulatory_overlay_score(self, allocations: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[float]:
        """
        监管叠加评分：需提供regulatory_overlay_rules；无明确规则或数据则返回None
        
        Args:
            allocations: 组合配置
            cfg: 动态严格模式配置，需包含 regulatory_overlay_rules
            
        Returns:
            监管叠加分数（0-1），或 None（规则/数据缺失）
        """
        try:
            rules = (cfg.get('regulatory_overlay_rules') or {}).get(self.market_type)
            if not isinstance(rules, list) or not rules:
                return None
            # 简化：无可靠度量数据时不计算，返回None，保持配置驱动
            return None
        except Exception:
            return None

    def _calculate_comprehensive_score(self, scores: Dict[str, float], cfg: Dict[str, Any]) -> Optional[float]:
        """
        综合评分：仅当cfg提供component_weights且所需维度分数都存在时计算，否则返回None
        
        Args:
            scores: 子评分字典，key 为维度名称（multi_currency/cross_border/regulatory）
            cfg: 动态严格模式配置，需包含 component_weights
            
        Returns:
            加权综合评分（0-1），或 None（配置缺失或维度不足）
        """
        try:
            comp_w = cfg.get('component_weights')
            if not isinstance(comp_w, dict):
                return None
            total = 0.0
            weight_sum = 0.0
            for k, w in comp_w.items():
                s = scores.get(k)
                if s is None:
                    # 配置要求的维度缺失分数，返回None
                    return None
                total += float(s) * float(w)
                weight_sum += float(w)
            if weight_sum <= 0.0:
                return None
            return float(total / weight_sum)
        except Exception:
            return None

    def _calculate_currency_coverage(self, prices: Dict[str, Dict[str, Any]]) -> tuple[int, int, float]:
        """
        计算货币字段覆盖率（公共辅助方法）
        
        Returns:
            tuple: (总标的数, 有货币字段的标的数, 覆盖率)
        """
        total_symbols = len(prices)
        symbols_with_currency = sum(1 for p in prices.values() if p.get('currency'))
        currency_coverage = symbols_with_currency / total_symbols if total_symbols > 0 else 0.0
        return total_symbols, symbols_with_currency, currency_coverage

    def _assess_data_source_quality(self, prices: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """评估数据源货币信息完整性，且C/D级时记录告警提示自动处理"""
        total_symbols, symbols_with_currency, currency_coverage = self._calculate_currency_coverage(prices)
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

    def _assess_data_quality_multi(self, market_data: Dict[str, Any], dq_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        多维度数据质量评估（配置驱动；数据不足时返回None或仅提供部分维度）
        
        Args:
            market_data: 市场数据字典，包含 prices 等字段
            dq_cfg: 数据质量评估配置，需包含 enabled/base_weights/grade_thresholds
            
        Returns:
            包含 overall_score/dimension_scores/quality_grade 的字典，或 None（配置未启用/不完整）
        """
        try:
            if not isinstance(dq_cfg, dict) or not bool(dq_cfg.get('enabled', False)):
                return None
            base_weights = dq_cfg.get('base_weights') or dq_cfg.get('weights')
            grade_thresholds = dq_cfg.get('grade_thresholds') or {}
            if not isinstance(base_weights, dict) or not isinstance(grade_thresholds, dict):
                return None
            prices = (market_data.get('prices') or {})
            # 仅实现 completeness：基于currency_coverage
            _, _, currency_coverage = self._calculate_currency_coverage(prices)
            # 维度得分（0-100）
            dimension_scores: Dict[str, float] = {
                'completeness': float(currency_coverage * 100.0)
            }
            # 仅当所有需要的维度都有分数时计算overall；否则返回partial并按coverage给出grade
            required_dims = list(base_weights.keys())
            if all(dim in dimension_scores for dim in required_dims):
                total = 0.0
                ws = 0.0
                for dim, w in base_weights.items():
                    total += float(dimension_scores.get(dim, 0.0)) * float(w)
                    ws += float(w)
                overall_score = float(total / ws) if ws > 0 else 0.0
                grade = self._convert_score_to_grade(overall_score, grade_thresholds)
                return {
                    'overall_score': overall_score,
                    'dimension_scores': dimension_scores,
                    'quality_grade': grade
                }
            else:
                # partial 输出：仅基于coverage给出grade
                overall_score = float(dimension_scores['completeness'])
                grade = self._convert_score_to_grade(overall_score, grade_thresholds)
                return {
                    'overall_score': overall_score,
                    'dimension_scores': dimension_scores,
                    'quality_grade': grade
                }
        except Exception:
            return None

    def _convert_score_to_grade(self, score: float, thresholds: Dict[str, Any]) -> str:
        """
        根据配置阈值将分数转换为等级（A/B/C/D）
        
        Args:
            score: 得分（0-100）
            thresholds: 阈值字典，包含 A/B/C 的边界值
            
        Returns:
            质量等级字符串（A/B/C/D）
        """
        try:
            a = float(thresholds.get('A', 90))
            b = float(thresholds.get('B', 75))
            c = float(thresholds.get('C', 60))
            if score >= a:
                return 'A'
            if score >= b:
                return 'B'
            if score >= c:
                return 'C'
            return 'D'
        except Exception:
            return 'D'

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

    def _us_compliance_logging(self, currency_warnings: List[str], data_quality: Optional[Dict[str, Any]] = None) -> None:
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
                extra={'compliance_events': compliance_events, 'data_quality': (data_quality or {})}
            )

    def calculate_all_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        计算所有风险指标
        
        职责：
        - 委托 RiskDataPreprocessor 提取数据
        - 委托 RiskMetricsService 计算指标
        """
        try:
            audit_events = []
            t_currency_start = time.time()
            # TODO：补充了货币一致性运行时检查，强化数据源质量评估，来源：docs/answer.md
            currency_warnings = self._runtime_currency_check(data)
            currency_warnings += self._check_risk_parameters_currency(data)
            dynamic_val = self._determine_dynamic_strict_mode(data)
            if dynamic_val is not None:
                prev = bool(getattr(self, 'strict_currency_check', False))
                self.strict_currency_check = bool(dynamic_val)
                logger.info(f"动态严格模式覆盖: {prev} -> {self.strict_currency_check}")
            self._handle_currency_warnings(currency_warnings)
            # 美股合规日志
            self._us_compliance_logging(currency_warnings, data_quality)
            audit_events.append({'step': 'currency_checks', 'duration': time.time() - t_currency_start, 'status': 'success'})
            # 数据源质量评估（调整为A/B/C/D分级）
            data_quality: Dict[str, Any] = {}
            market_data_prices = data.get('market_data', {}).get('prices', {}) or {}
            if market_data_prices:
                t_dataq_start = time.time()
                data_quality = self._assess_data_source_quality(market_data_prices)
                dq_cfg = self.config.get('data_quality_assessment', {})
                if isinstance(dq_cfg, dict) and bool(dq_cfg.get('enabled', False)):
                    dq_multi = self._assess_data_quality_multi({'prices': market_data_prices}, dq_cfg)
                    if dq_multi:
                        data_quality.update({'multi': dq_multi})
                logger.info(f"数据源质量评级: {data_quality['quality_rating']}, "
                            f"货币覆盖率: {data_quality['currency_coverage']:.2%}")
                audit_events.append({'step': 'data_quality_assessment', 'duration': time.time() - t_dataq_start, 'status': 'success'})
            # 在存在多币种/不一致时，尝试统一货币（仅摘要，不影响指标计算）
            t_unify_start = time.time()
            self._unify_currency_for_portfolio(data)
            audit_events.append({'step': 'currency_unify_summary', 'duration': time.time() - t_unify_start, 'status': 'success'})

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
                f"耗时{elapsed:.3f}s, 指标{len(metrics)}个",
                extra={'audit_events': audit_events, 'min_points': min_points}
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

    def _validate_required_fields(self) -> None:
        """验证必要配置项的可用性（仅记录告警，不阻断）。来源：docs/answer.md 高优先级改进建议。
        - 动态严格模式：enabled=True 时要求 component_weights/comprehensive_trigger_score；
        - 数据质量评估：enabled=True 时要求 base_weights/grade_thresholds。
        注：组合字段（如 portfolio.cross_border_exposure）在运行时校验，不在初始化阻断。
        """
        try:
            issues = []
            dyn_cfg = self.config.get('dynamic_currency_strict_mode', {}) or {}
            if isinstance(dyn_cfg, dict) and bool(dyn_cfg.get('enabled', False)):
                if dyn_cfg.get('component_weights') is None:
                    issues.append('动态严格模式缺少配置项: component_weights')
                if dyn_cfg.get('comprehensive_trigger_score') is None:
                    issues.append('动态严格模式缺少配置项: comprehensive_trigger_score')
            dq_cfg = self.config.get('data_quality_assessment', {}) or {}
            if isinstance(dq_cfg, dict) and bool(dq_cfg.get('enabled', False)):
                base_weights = dq_cfg.get('base_weights') or dq_cfg.get('weights')
                grade_thresholds = dq_cfg.get('grade_thresholds')
                if not isinstance(base_weights, dict):
                    issues.append('数据质量评估缺少配置项: base_weights/weights')
                if not isinstance(grade_thresholds, dict):
                    issues.append('数据质量评估缺少配置项: grade_thresholds')
            if issues:
                logger.warning(f"配置必要项校验发现问题: {'; '.join(issues)}")
        except Exception:
            # 仅记录，不阻断初始化
            pass

    def _get_market_specific_config(self, config_key: str, default_config: Dict[str, Any]) -> Dict[str, Any]:
        """获取市场特定配置，回退到默认配置。来源：docs/answer.md 建议配置结构。
        Args:
            config_key: 顶层配置键，例如 'data_quality_assessment' 或 'dynamic_currency_strict_mode'
            default_config: 当未提供市场特异配置时使用的默认配置
        Returns:
            dict: 当前市场的特定配置或默认配置
        """
        try:
            top = self.config.get(config_key, {}) or {}
            market_specific = top.get('market_specific', {}) or {}
            if isinstance(market_specific, dict) and self.market_type in market_specific:
                return market_specific.get(self.market_type, default_config) or default_config
            return default_config
        except Exception:
            return default_config


