"""FundamentalDataService 单元测试"""
import pytest
import asyncio
from core_bak_refactored.core.data.providers.fundamental_data_service import FundamentalDataService


@pytest.fixture(scope="function")
def event_loop():
    """创建测试事件循环"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.mark.asyncio
async def test_fundamental_data_service_error_handling():
    """测试基本面数据服务的错误处理"""
    service = FundamentalDataService()
    
    try:
        # 测试不存在的symbol
        result = await service.get_fundamental_data('INVALID_SYMBOL_12345')
        
        assert isinstance(result, dict)
        assert 'error' in result  # 应返回错误信息
    finally:
        # 确保清理异步资源
        if hasattr(service, 'session') and service.session:
            await service.session.close()
        # 等待事件循环处理剩余任务
        await asyncio.sleep(0.1)
