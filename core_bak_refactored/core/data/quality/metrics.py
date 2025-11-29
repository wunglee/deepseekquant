from typing import Any, Dict

from core_bak_refactored.core.data.data_fetcher import DataFetcher


def get_quality_metrics(fetcher: DataFetcher) -> Dict[str, Any]:
    """读取数据质量指标（职责单一：聚合读取）"""
    return fetcher.get_data_quality_metrics()
