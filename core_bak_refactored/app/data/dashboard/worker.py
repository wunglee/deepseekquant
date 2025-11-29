from typing import Any

from .aggregator import aggregate


class UpdateWorker:
    """仪表板更新工作器（职责单一：单次刷新）"""

    def run_once(self, monitor: Any) -> dict:
        return aggregate(monitor)
