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

        self.logger.info(f"组合分配完成: method={method.value}, objective={objective.value}")
        return {"status": "success", "weights": weights, "objective": objective.value}

    def _cleanup_core(self):
        pass
