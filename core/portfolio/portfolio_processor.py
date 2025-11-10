from typing import Any, Dict, List
from dataclasses import dataclass

from core.base_processor import BaseProcessor
from infrastructure.interfaces import InfrastructureProvider
from common import AllocationMethod, PortfolioObjective

@dataclass
class Position:
    symbol: str
    quantity: float
    price: float

class PortfolioProcessor(BaseProcessor):
    def _initialize_core(self) -> bool:
        self.logger = InfrastructureProvider.get('logging').get_logger('DeepSeekQuant.Portfolio')
        return True

    def _process_core(self, *args, **kwargs) -> Any:
        positions: List[Dict[str, Any]] = kwargs.get('positions', [])
        method: AllocationMethod = kwargs.get('method', AllocationMethod.EQUAL_WEIGHT)
        objective: PortfolioObjective = kwargs.get('objective', PortfolioObjective.MAXIMIZE_SHARPE)

        # 简化：按等权或按市值占比归一化
        weights: Dict[str, float] = {}
        if not positions:
            return {"status": "error", "message": "缺少持仓列表"}

        if method == AllocationMethod.EQUAL_WEIGHT:
            w = 1.0 / len(positions)
            for p in positions:
                weights[p['symbol']] = round(w, 6)
        else:
            total_mv = sum(p['quantity'] * p['price'] for p in positions)
            if total_mv <= 0:
                return {"status": "error", "message": "总市值为0"}
            for p in positions:
                mv = p['quantity'] * p['price']
                weights[p['symbol']] = round(mv / total_mv, 6)

        # 约束：权重上下限与归一化
        min_w = float(kwargs.get('min_weight', 0.0))
        max_w = float(kwargs.get('max_weight', 1.0))
        if min_w > 0.0 or max_w < 1.0:
            for k in list(weights.keys()):
                weights[k] = max(min_w, min(max_w, weights[k]))
            s = sum(weights.values())
            if s > 0:
                for k in list(weights.keys()):
                    weights[k] = round(weights[k] / s, 6)

        # 生成再平衡指令（最小实现）
        current_positions: Dict[str, float] = kwargs.get('current_positions', {})
        rebalance_instructions: List[Dict[str, Any]] = []
        turnover = 0.0
        total_capital = sum(p['quantity'] * p['price'] for p in positions)
        for p in positions:
            symbol = p['symbol']
            target_mv = weights[symbol] * total_capital
            current_mv = (current_positions.get(symbol, 0.0)) * p['price']
            delta_mv = target_mv - current_mv
            qty_change = delta_mv / p['price'] if p['price'] > 0 else 0.0
            # 最小交易量阈值（绝对值）
            min_trade_qty = float(kwargs.get('min_trade_qty', 0.0))
            if abs(qty_change) < min_trade_qty:
                qty_change = 0.0
            turnover += abs(delta_mv)
            rebalance_instructions.append({
                'symbol': symbol,
                'target_weight': weights[symbol],
                'quantity_change': round(qty_change, 6)
            })
        turnover_rate = round(turnover / total_capital, 6) if total_capital > 0 else 0.0

        self.logger.info(f"组合分配完成: method={method.value}, objective={objective.value}, turnover={turnover_rate}")
        return {"status": "success", "weights": weights, "objective": objective.value, "rebalance": rebalance_instructions, "turnover_rate": turnover_rate}

    def _cleanup_core(self):
        pass
