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
