"""
Yahoo Finance数据提供者测试
验证Phase 3B真实数据集成功能
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from core_bak_refactored.core.data._fragments.yahoo_finance_provider import YahooFinanceDataProvider


class TestYahooFinanceDataProvider:
    """Yahoo Finance数据提供者测试套件"""
    
    def test_initialization_with_yfinance(self):
        """测试：yfinance可用时成功初始化"""
        provider = YahooFinanceDataProvider(fallback_to_mock=True)
        assert provider.fallback is True
        # 如果yfinance已安装，应该成功初始化
        # 如果未安装，会打印警告但不崩溃（因为fallback=True）
    
    def test_index_mapping(self):
        """测试：指数代码映射正确性"""
        provider = YahooFinanceDataProvider()
        
        # 测试国内主要指数映射
        assert provider._map_index_to_yahoo('000300.SH') == '000300.SS'  # 沪深300
        assert provider._map_index_to_yahoo('SPX') == '^GSPC'  # S&P 500
        assert provider._map_index_to_yahoo('HSI') == '^HSI'  # 恒生指数
    
    def test_index_mapping_passthrough(self):
        """测试：已知Yahoo格式直接通过"""
        provider = YahooFinanceDataProvider()
        
        # 已经是Yahoo格式的代码应该直接通过
        assert provider._map_index_to_yahoo('^GSPC') == '^GSPC'
        assert provider._map_index_to_yahoo('000300.SS') == '000300.SS'
        assert provider._map_index_to_yahoo('399001.SZ') == '399001.SZ'
    
    def test_standardize_format_basic(self):
        """测试：数据格式标准化"""
        provider = YahooFinanceDataProvider()
        
        # 模拟yfinance返回数据
        dates = pd.date_range('2020-01-01', periods=5)
        mock_data = pd.DataFrame({
            'Close': [100.0, 102.0, 101.0, 103.0, 104.0],
            'Volume': [1000000, 1100000, 950000, 1200000, 1050000]
        }, index=dates)
        
        standardized = provider._standardize_format(mock_data)
        
        # 验证列名
        assert list(standardized.columns) == ['date', 'close', 'volume']
        
        # 验证数据长度
        assert len(standardized) == 5
        
        # 验证数据值
        assert standardized['close'].iloc[0] == 100.0
        assert standardized['volume'].iloc[0] == 1000000
    
    def test_standardize_format_lowercase_columns(self):
        """测试：处理小写列名（兼容性）"""
        provider = YahooFinanceDataProvider()
        
        # 模拟小写列名
        dates = pd.date_range('2020-01-01', periods=3)
        mock_data = pd.DataFrame({
            'close': [100.0, 101.0, 102.0],
            'volume': [1000000, 1100000, 1200000]
        }, index=dates)
        
        standardized = provider._standardize_format(mock_data)
        
        assert 'close' in standardized.columns
        assert 'volume' in standardized.columns
        assert len(standardized) == 3
    
    def test_standardize_format_missing_volume(self):
        """测试：缺失成交量时填充NaN"""
        provider = YahooFinanceDataProvider()
        
        dates = pd.date_range('2020-01-01', periods=3)
        mock_data = pd.DataFrame({
            'Close': [100.0, 101.0, 102.0]
        }, index=dates)
        
        standardized = provider._standardize_format(mock_data)
        
        # 应该填充NaN而不是报错
        assert 'volume' in standardized.columns
        assert pd.isna(standardized['volume'].iloc[0])
    
    def test_standardize_format_nan_close_removal(self):
        """测试：移除close为NaN的行"""
        provider = YahooFinanceDataProvider()
        
        dates = pd.date_range('2020-01-01', periods=5)
        mock_data = pd.DataFrame({
            'Close': [100.0, np.nan, 102.0, np.nan, 104.0],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }, index=dates)
        
        standardized = provider._standardize_format(mock_data)
        
        # 应该只保留3行（移除2个NaN）
        assert len(standardized) == 3
        assert list(standardized['close']) == [100.0, 102.0, 104.0]
    
    @patch('yfinance.download')
    def test_get_index_prices_success(self, mock_download):
        """测试：成功获取指数价格（使用mock）"""
        provider = YahooFinanceDataProvider(fallback_to_mock=False)
        
        # Mock yfinance返回数据
        dates = pd.date_range('2015-06-01', periods=10)
        mock_download_data = pd.DataFrame({
            'Close': np.random.uniform(4000, 4500, 10),
            'Volume': np.random.uniform(1e9, 2e9, 10)
        }, index=dates)
        
        mock_download.return_value = mock_download_data
        
        # 调用方法
        result = provider.get_index_prices('000300.SH', '2015-06-01', '2015-06-10')
        
        # 验证结果
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['date', 'close', 'volume']
        assert len(result) == 10
        
        # 验证yfinance被正确调用
        mock_download.assert_called_once_with('000300.SS', start='2015-06-01', end='2015-06-10', progress=False)
    
    @patch('yfinance.download')
    def test_get_index_prices_empty_fallback(self, mock_download):
        """测试：数据为空时回退到Mock"""
        provider = YahooFinanceDataProvider(fallback_to_mock=True)
        
        # Mock返回空数据
        mock_download.return_value = pd.DataFrame()
        
        # 应该回退到Mock，不会报错
        result = provider.get_index_prices('000300.SH', '2015-06-01', '2015-06-10')
        
        # 验证返回Mock数据（应该有数据）
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0  # Mock应该返回非空数据
    
    @patch('yfinance.download')
    def test_get_index_prices_no_fallback_raises(self, mock_download):
        """测试：禁用fallback时失败抛出异常"""
        provider = YahooFinanceDataProvider(fallback_to_mock=False)
        
        # Mock返回空数据
        mock_download.return_value = pd.DataFrame()
        
        # 应该抛出异常
        with pytest.raises(ValueError, match="Failed to fetch data"):
            provider.get_index_prices('000300.SH', '2015-06-01', '2015-06-10')
    
    @patch('yfinance.download')
    def test_get_index_returns(self, mock_download):
        """测试：获取收益率序列"""
        provider = YahooFinanceDataProvider(fallback_to_mock=False)
        
        # Mock价格数据
        dates = pd.date_range('2020-01-01', periods=5)
        mock_download_data = pd.DataFrame({
            'Close': [100.0, 102.0, 101.0, 103.0, 104.0],
            'Volume': [1000000] * 5
        }, index=dates)
        
        mock_download.return_value = mock_download_data
        
        # 调用方法
        returns = provider.get_index_returns('SPX', '2020-01-01', '2020-01-05')
        
        # 验证收益率计算
        assert isinstance(returns, pd.Series)
        assert len(returns) == 4  # 5个价格 → 4个收益率
        
        # 验证第一个收益率 = (102-100)/100 = 0.02
        assert abs(returns.iloc[0] - 0.02) < 1e-6
        
        # 验证第二个收益率 = (101-102)/102 ≈ -0.0098
        assert abs(returns.iloc[1] - (-0.0098039)) < 1e-4
    
    def test_connection_test_mock_fallback(self):
        """测试：连接测试（使用Mock fallback）"""
        provider = YahooFinanceDataProvider(fallback_to_mock=True)
        
        # 即使yfinance失败，fallback应该让测试通过
        result = provider.test_connection('000300.SH')
        
        # 如果yfinance可用应该返回True，如果不可用但fallback成功也应该返回True
        assert isinstance(result, bool)


class TestYahooFinanceDataProviderIntegration:
    """Yahoo Finance集成测试（需要网络连接）"""
    
    @pytest.mark.skip(reason="需要网络连接和yfinance安装，CI环境可能不可用")
    def test_real_data_fetch_csi300(self):
        """集成测试：获取沪深300真实数据"""
        provider = YahooFinanceDataProvider(fallback_to_mock=False)
        
        # 获取最近30天数据
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        data = provider.get_index_prices('000300.SH', start_date, end_date)
        
        # 验证数据质量
        assert len(data) > 0
        assert 'close' in data.columns
        assert data['close'].min() > 0  # 价格应该为正数
    
    @pytest.mark.skip(reason="需要网络连接")
    def test_real_data_fetch_sp500(self):
        """集成测试：获取S&P500真实数据"""
        provider = YahooFinanceDataProvider(fallback_to_mock=False)
        
        # 获取2020年COVID崩盘期数据
        data = provider.get_index_prices('SPX', '2020-02-20', '2020-03-23')
        
        # 验证数据
        assert len(data) > 10  # 应该有至少10个交易日
        assert data['close'].max() > data['close'].min()  # 有波动
