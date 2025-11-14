"""
CurrencyConverter - 汇率转换服务（基础设施层）

职责：
- 提供统一的汇率转换能力（最小可用版本，MVP）
- 计算投资组合的货币敞口

约束：
- 不依赖外部网络数据源；通过配置提供汇率或回退汇率
- 接口契约稳定，便于风险模块调用与未来扩展
"""
from typing import Dict, Any


class CurrencyConverter:
    """汇率转换服务（MVP）"""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        config = config or {}
        # 期望嵌套字典：{"USD": {"CNY": 7.1, "HKD": 7.8}, ...}
        self.rate_sources: Dict[str, Dict[str, float]] = config.get("exchange_rate_sources", {})
        # 回退支持嵌套或扁平键：{"USD": {"CNY": 7.1}} 或 {"USD->CNY": 7.1}
        self.fallback_rates: Dict[str, Any] = config.get("fallback_exchange_rates", {})

    def _get_rate(self, src: str, tgt: str) -> float:
        if src == tgt:
            return 1.0
        # 优先使用主数据源
        if isinstance(self.rate_sources, dict):
            src_map = self.rate_sources.get(src)
            if isinstance(src_map, dict) and tgt in src_map:
                return float(src_map[tgt])
        # 回退数据源：嵌套或扁平键
        if isinstance(self.fallback_rates, dict):
            src_map = self.fallback_rates.get(src)
            if isinstance(src_map, dict) and tgt in src_map:
                return float(src_map[tgt])
            flat_key = f"{src}->{tgt}"
            if flat_key in self.fallback_rates:
                return float(self.fallback_rates[flat_key])
        # 未命中则返回1.0（MVP策略：不中断调用）
        return 1.0

    def convert_portfolio_currency(self, portfolio: Dict[str, Any], target_currency: str) -> Dict[str, Any]:
        """将组合估值统一转换为目标货币。
        期望组合结构：{"allocations": {symbol: {"currency": str, "value": float}}}
        返回：{"target_currency": str, "total_converted_value": float, "details": {symbol: {...}}}
        """
        allocations = portfolio.get("allocations", {})
        details: Dict[str, Any] = {}
        total_converted = 0.0
        for symbol, info in allocations.items():
            value = float(info.get("value", 0.0))
            src_currency = info.get("currency", target_currency)
            rate = self._get_rate(src_currency, target_currency)
            converted_value = value * rate
            details[symbol] = {
                "converted_value": converted_value,
                "source_currency": src_currency,
                "rate": rate,
            }
            total_converted += converted_value
        return {
            "target_currency": target_currency,
            "total_converted_value": total_converted,
            "details": details,
        }

    def calculate_currency_exposure(self, portfolio: Dict[str, Any]) -> Dict[str, float]:
        """按货币汇总组合敞口（未转换前的币种分布）。"""
        allocations = portfolio.get("allocations", {})
        exposures: Dict[str, float] = {}
        for _, info in allocations.items():
            curr = info.get("currency", "UNKNOWN")
            value = float(info.get("value", 0.0))
            exposures[curr] = exposures.get(curr, 0.0) + value
        return exposures
