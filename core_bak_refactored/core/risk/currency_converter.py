from typing import Dict, Any, Protocol


class ExchangeRateAdapter(Protocol):
    """统一的外部实时汇率适配器接口（业务层）。
    说明：此接口的具体实现可替换、可配置、当前阶段可使用模拟实现。
    返回值要求支持嵌套或扁平键：
      - 嵌套：{"USD": {"CNY": 7.1}}
      - 扁平：{"USD->CNY": 7.1}
    """

    def get_rates(self, market_type: str) -> Dict[str, Any]:
        """获取指定市场的实时汇率字典。"""
        ...


class MockExchangeRateAdapter:
    """开发阶段的模拟适配器实现。"""

    def __init__(self, mock_rates: Dict[str, Any] | None = None) -> None:
        self._mock_rates = mock_rates or {
            'USD': {'CNY': 7.1, 'HKD': 7.8},
            'HKD': {'USD': 0.128205},
            'USD->EUR': 0.92,
        }

    def get_rates(self, market_type: str) -> Dict[str, Any]:
        # 简化：当前忽略market_type，返回全局模拟汇率
        return dict(self._mock_rates)


class CurrencyConverter:
    """风险业务层的汇率转换服务（仅负责计算，不关心来源）。"""

    def __init__(self) -> None:
        pass

    def _get_rate(self, src: str, tgt: str, rates: Dict[str, Any]) -> float:
        if src == tgt:
            return 1.0
        if isinstance(rates, dict):
            src_map = rates.get(src)
            if isinstance(src_map, dict) and tgt in src_map:
                return float(src_map[tgt])
            flat_key = f"{src}->{tgt}"
            if flat_key in rates:
                return float(rates[flat_key])
        return 1.0

    def convert_portfolio_currency(self, portfolio: Dict[str, Any], target_currency: str, rates: Dict[str, Any]) -> Dict[str, Any]:
        allocations = portfolio.get('allocations', {})
        details: Dict[str, Any] = {}
        total_converted = 0.0
        for symbol, info in allocations.items():
            value = float(info.get('value', 0.0))
            src_currency = info.get('currency', target_currency)
            rate = self._get_rate(src_currency, target_currency, rates)
            converted_value = value * rate
            details[symbol] = {
                'converted_value': converted_value,
                'source_currency': src_currency,
                'rate': rate,
            }
            total_converted += converted_value
        return {
            'target_currency': target_currency,
            'total_converted_value': total_converted,
            'details': details,
        }

    def calculate_currency_exposure(self, portfolio: Dict[str, Any]) -> Dict[str, float]:
        allocations = portfolio.get('allocations', {})
        exposures: Dict[str, float] = {}
        for _, info in allocations.items():
            curr = info.get('currency', 'UNKNOWN')
            value = float(info.get('value', 0.0))
            exposures[curr] = exposures.get(curr, 0.0) + value
        return exposures
