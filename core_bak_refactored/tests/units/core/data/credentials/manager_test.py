from core_bak_refactored.core.data.credentials.manager import ApiCredentialsManager
from core_bak_refactored.core.data.data_fetcher import DataFetcher
import pytest


@pytest.mark.asyncio
async def test_credentials_manager_get():
    config = {'primary': 'yahoo', 'cache_enabled': False, 'sources': {'alpha_vantage': {'enabled': True, 'api_key': 'k'}}}
    fetcher = DataFetcher(config)
    mgr = ApiCredentialsManager(fetcher)
    creds = mgr.get('alpha_vantage')
    assert isinstance(creds, dict)
