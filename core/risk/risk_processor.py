from typing import Any, Dict
from core.base_processor import BaseProcessor
from common import RiskAssessment, RiskLevel

class RiskProcessor(BaseProcessor):
    def _initialize_core(self) -> bool:
        return True

    def _process_core(self, *args, **kwargs) -> Any:
        metric = kwargs.get('metric', 'var')
        signal = kwargs.get('signal', {})
        limits: Dict[str, float] = kwargs.get('limits', {})
        price = signal.get('price', 0.0)
        quantity = signal.get('quantity', 0.0)
        max_position_value = limits.get('max_position_value', float('inf'))
        vol_threshold = limits.get('volatility_threshold', float('inf'))
        concentration_threshold = limits.get('concentration_threshold', 0.5)  # 50%
        prices_history = kwargs.get('prices', [])
        
        position_value = price * quantity
        approved = True
        reason = "Within limits"
        warnings = []
        risk_score = 0.1
        
        if position_value > max_position_value:
            approved = False
            reason = f"Position value {position_value:.2f} exceeds limit {max_position_value:.2f}"
            warnings.append('POSITION_LIMIT_EXCEEDED')
            risk_score = max(risk_score, 0.8)
        
        # 波动率阈值检查
        if prices_history and vol_threshold < float('inf'):
            closes = [float(p) for p in prices_history]
            from core.data.data_fetcher import DataFetcher
            vol = DataFetcher().compute_volatility(closes)
            if vol > vol_threshold:
                approved = False
                warnings.append('VOLATILITY_EXCEEDED')
                reason = f"Volatility {vol:.4f} exceeds threshold {vol_threshold:.4f}"
                risk_score = max(risk_score, 0.7)
        
        # 集中度检查（示例：单一标的权重过高）
        target_weight = kwargs.get('target_weight')
        if target_weight is not None and target_weight > concentration_threshold:
            approved = False
            warnings.append('CONCENTRATION_EXCEEDED')
            reason = f"Target weight {target_weight:.2f} exceeds threshold {concentration_threshold:.2f}"
            risk_score = max(risk_score, 0.6)
        
        # 组合层面的集中度与HHI检查
        weights = kwargs.get('weights', {})
        if isinstance(weights, dict) and weights:
            try:
                max_w = max(float(w) for w in weights.values())
                if max_w > concentration_threshold:
                    approved = False
                    if 'CONCENTRATION_EXCEEDED' not in warnings:
                        warnings.append('CONCENTRATION_EXCEEDED')
                    reason = f"Max weight {max_w:.2f} exceeds threshold {concentration_threshold:.2f}"
                    risk_score = max(risk_score, 0.65)
                hhi = sum(float(w) ** 2 for w in weights.values())
                hhi_thr = float(limits.get('hhi_threshold', 0.4))
                if hhi > hhi_thr:
                    approved = False
                    warnings.append('PORTFOLIO_HHI_EXCEEDED')
                    reason = f"HHI {hhi:.2f} exceeds threshold {hhi_thr:.2f}"
                    risk_score = max(risk_score, 0.7)
            except Exception:
                warnings.append('WEIGHTS_PARSE_ERROR')
        
        # 相关性检查（基于历史收盘价的平均相关）
        histories = kwargs.get('histories', {})
        corr_thr = float(limits.get('correlation_threshold', float('inf')))
        if isinstance(histories, dict) and histories and corr_thr < float('inf'):
            try:
                from core.data.data_fetcher import DataFetcher
                df = DataFetcher()
                syms = list(histories.keys())
                pair_corr = []
                for i in range(len(syms)):
                    for j in range(i+1, len(syms)):
                        a = [float(x) for x in histories[syms[i]]]
                        b = [float(x) for x in histories[syms[j]]]
                        corr = df.compute_pairwise_correlation(a, b)
                        pair_corr.append(corr)
                avg_corr = sum(pair_corr) / len(pair_corr) if pair_corr else 0.0
                if avg_corr > corr_thr:
                    approved = False
                    warnings.append('PORTFOLIO_CORRELATION_HIGH')
                    reason = f"Avg corr {avg_corr:.2f} exceeds threshold {corr_thr:.2f}"
                    risk_score = max(risk_score, 0.7)
            except Exception:
                warnings.append('CORRELATION_CHECK_ERROR')
        
        # 最大回撤阈值检查（逐标的，若任一超过则警告）
        mdd_thr = float(limits.get('max_drawdown_threshold', float('inf')))
        if isinstance(histories, dict) and histories and mdd_thr < float('inf'):
            try:
                from core.data.data_fetcher import DataFetcher
                df = DataFetcher()
                for sym, closes in histories.items():
                    closes_f = [float(x) for x in closes]
                    mdd = df.compute_max_drawdown(closes_f)
                    if mdd > mdd_thr:
                        approved = False
                        warnings.append('MAX_DRAWDOWN_EXCEEDED')
                        reason = f"Max DD {mdd:.2f} exceeds threshold {mdd_thr:.2f}"
                        risk_score = max(risk_score, 0.75)
                        break
            except Exception:
                warnings.append('DRAWDOWN_CHECK_ERROR')
        
        assessment = RiskAssessment(
            approved=approved,
            reason=reason,
            risk_level=RiskLevel.HIGH if not approved else RiskLevel.LOW,
            warnings=warnings,
            max_position_size=max_position_value,
            suggested_allocation=kwargs.get('suggested_allocation', 0.05),
            risk_score=risk_score
        )
        return {'status': 'success', 'metric': metric, 'assessment': assessment.to_dict()}

    def _cleanup_core(self):
        pass
