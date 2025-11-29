from core_bak_refactored.core.data.quality.metrics import get_quality_metrics
from core_bak_refactored.core.data.data_fetcher import DataFetcher
import pytest


@pytest.mark.asyncio
async def test_quality_metrics_reading():
    fetcher = DataFetcher({'primary': 'yahoo', 'cache_enabled': False})
    qm = get_quality_metrics(fetcher)
    assert isinstance(qm, dict) and 'overall_quality' in qm
