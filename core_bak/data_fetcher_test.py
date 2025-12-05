"""data_fetcher.py的单元测试

测试专家完整版DataFetcher及相关核心类的基础功能
使用依赖注入方式Mock数据源
"""
import pytest
from datetime import datetime, timedelta
from core_bak.data_fetcher import (
    DataFetcher,
    DataValidator,
    DataQualityMonitorBasic, MarketData, DataSourceType
)
from core_bak_refactored.tests.fixtures.core.data.mock_historical_data_provider import MockHistoricalDataProvider


@pytest.mark.asyncio
async def test_data_fetcher_init_with_mock_source():
    """测试DataFetcher使用Mock数据源初始化"""
    # 创建Mock数据源
    mock_provider = MockHistoricalDataProvider()
    
    # 准备Mock数据源适配器（将MockHistoricalDataProvider适配为DataFetcher接口）
    async def mock_fetch_data(symbol, period, interval, data_type, adjustments):
        """Mock数据源适配器"""
        # 使用MockHistoricalDataProvider生成数据
        df = mock_provider.get_index_prices(
            index_id=symbol,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        # 转换为MarketData列表
        result = []
        for _, row in df.iterrows():
            result.append(MarketData(
                symbol=symbol,
                timestamp=row['date'],
                open=row['close'] * 0.99,
                high=row['close'] * 1.01,
                low=row['close'] * 0.98,
                close=row['close'],
                volume=row['volume']
            ))
        return result
    
    # 通过依赖注入方式注入Mock数据源
    custom_sources = {
        DataSourceType.IEX_CLOUD.value: mock_fetch_data,  # 替代IEX_CLOUD
        'mock': mock_fetch_data  # 额外的mock源
    }
    
    config = {
        'cache_enabled': False,
        'primary': DataSourceType.YAHOO_FINANCE.value
    }
    
    # 使用依赖注入创建DataFetcher
    df = DataFetcher(config=config, custom_sources=custom_sources)
    
    # 验证数据源已注册
    assert DataSourceType.IEX_CLOUD.value in df.data_sources
    assert 'mock' in df.data_sources
    assert df.config == config
    assert df.cache_enabled == False
    
    await df.cleanup()  # 清理资源


@pytest.mark.asyncio
async def test_data_fetcher_fetch_with_injected_mock():
    """测试使用注入的Mock数据源获取数据"""
    # 创建Mock数据源
    mock_provider = MockHistoricalDataProvider()
    
    # Mock数据源适配器
    async def mock_fetch_data(symbol, period, interval, data_type, adjustments):
        df = mock_provider.get_index_prices(
            index_id=symbol,
            start_date='2024-01-01',
            end_date='2024-01-10'
        )
        result = []
        for _, row in df.iterrows():
            result.append(MarketData(
                symbol=symbol,
                timestamp=row['date'],
                open=row['close'] * 0.99,
                high=row['close'] * 1.01,
                low=row['close'] * 0.98,
                close=row['close'],
                volume=row['volume']
            ))
        return result
    
    # 注入Mock数据源并设为主数据源
    custom_sources = {
        'test_mock': mock_fetch_data
    }
    
    config = {
        'cache_enabled': False,
        'primary': 'test_mock'  # 使用注入的mock作为主数据源
    }
    
    df = DataFetcher(config=config, custom_sources=custom_sources)
    
    # Mock _fetch_symbol_data方法
    async def mock_fetch_symbol_data(symbol, period, interval, data_type, adjustments):
        return await custom_sources['test_mock'](symbol, period, interval, data_type, adjustments)
    
    df._fetch_symbol_data = mock_fetch_symbol_data
    
    # 执行测试
    result = await df.get_historical_data(
        symbols=['AAPL'],
        period='1mo',
        interval='1d'
    )
    
    # 验证结果
    assert 'AAPL' in result
    assert isinstance(result['AAPL'], list)
    assert len(result['AAPL']) > 0
    assert result['AAPL'][0].symbol == 'AAPL'
    assert result['AAPL'][0].close > 0
    
    await df.cleanup()  # 清理资源


class TestDataValidator:
    """测试DataValidator数据验证器"""
    
    def test_validator_init(self):
        """测试DataValidator初始化"""
        config = {}
        validator = DataValidator(config)
        
        assert validator.config == config
        assert validator.validation_rules is not None
        assert 'price_validation' in validator.validation_rules
        assert 'volume_validation' in validator.validation_rules
    
    def test_validate_empty_data(self):
        """测试验证空数据"""
        validator = DataValidator({})
        result = validator.validate_market_data([])
        
        assert result['valid'] == False
        assert '空数据' in str(result['errors'])
    
    def test_validate_valid_data(self):
        """测试验证正常数据"""
        validator = DataValidator({})
        
        # 创建正常数据
        data = [
            MarketData(
                symbol='AAPL',
                timestamp=datetime(2024, 1, 1),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000
            ),
            MarketData(
                symbol='AAPL',
                timestamp=datetime(2024, 1, 2),
                open=103.0,
                high=106.0,
                low=102.0,
                close=105.0,
                volume=1100000
            )
        ]
        
        result = validator.validate_market_data(data)
        
        # 验证结果结构
        assert 'valid' in result
        assert 'errors' in result
        assert 'warnings' in result
        assert isinstance(result['errors'], list)
    
    def test_validate_invalid_price_range(self):
        """测试验证价格超出范围"""
        validator = DataValidator({})
        
        # 创建价格异常数据（close为负数）
        data = [
            MarketData(
                symbol='AAPL',
                timestamp=datetime(2024, 1, 1),
                open=100.0,
                high=105.0,
                low=99.0,
                close=-10.0,  # 无效价格
                volume=1000000
            )
        ]
        
        result = validator.validate_market_data(data)
        assert result['valid'] == False
        assert len(result['errors']) > 0
    
    def test_validate_price_consistency(self):
        """测试价格一致性验证"""
        validator = DataValidator({})
        
        # 创建价格不一致数据 (close > high)
        data = [
            MarketData(
                symbol='AAPL',
                timestamp=datetime(2024, 1, 1),
                open=100.0,
                high=105.0,
                low=99.0,
                close=110.0,  # close > high, 不一致
                volume=1000000
            )
        ]
        
        result = validator.validate_market_data(data)
        
        # 应该检测到价格不一致错误
        assert result['valid'] == False
        assert any('price_consistency' in str(err.get('type', '')) for err in result['errors'])


class TestDataQualityMonitorBasic:
    """测试DataQualityMonitorBasic基础质量监控器"""
    
    def test_monitor_basic_init(self):
        """测试DataQualityMonitorBasic初始化"""
        config = {}
        monitor = DataQualityMonitorBasic(config)
        
        assert monitor.config == config
        assert hasattr(monitor, 'quality_history')
        assert hasattr(monitor, 'anomaly_detector')
        assert hasattr(monitor, 'data_validator')
    
    def test_monitor_basic_data_quality(self):
        """测试基础数据质量监控"""
        config = {}
        monitor = DataQualityMonitorBasic(config)
        
        # 创建Mock数据（包含metadata字段）
        mock_data = [
            MarketData(
                symbol='AAPL',
                timestamp=datetime(2024, 1, 1),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
                metadata={'data_source': 'mock'}  # 添加metadata
            ),
            MarketData(
                symbol='AAPL',
                timestamp=datetime(2024, 1, 2),
                open=103.0,
                high=106.0,
                low=102.0,
                close=105.0,
                volume=1100000,
                metadata={'data_source': 'mock'}  # 添加metadata
            )
        ]
        
        # 执行质量监控
        result = monitor.monitor_data_quality(mock_data)
        
        # 验证结果结构
        assert isinstance(result, dict)
        assert 'overall_score' in result
        assert 'dimension_scores' in result
        assert 'anomalies_detected' in result
        assert 'validation_errors' in result
        assert isinstance(result['overall_score'], float)
        assert 0.0 <= result['overall_score'] <= 1.0
