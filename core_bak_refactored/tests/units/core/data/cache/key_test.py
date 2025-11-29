import hashlib
import pytest
from core_bak_refactored.core.data.cache.key import generate_key


class TestGenerateKey:
    """测试缓存键生成功能（扩展版）。"""

    def test_generate_cache_key_basic(self):
        """测试基本的缓存键生成。"""
        key = generate_key(['AAPL', 'MSFT'], '1y', '1d', 'ohlcv', True)
        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hash length

    def test_generate_cache_key_deterministic(self):
        """测试相同输入生成相同的键。"""
        key1 = generate_key(['AAPL', 'MSFT'], '1y', '1d', 'ohlcv', True)
        key2 = generate_key(['AAPL', 'MSFT'], '1y', '1d', 'ohlcv', True)
        assert key1 == key2

    def test_generate_cache_key_symbols_order_agnostic(self):
        """测试符号顺序不影响结果。"""
        key1 = generate_key(['AAPL', 'MSFT'], '1y', '1d', 'ohlcv', True)
        key2 = generate_key(['MSFT', 'AAPL'], '1y', '1d', 'ohlcv', True)
        assert key1 == key2

    # === 新增测试：扩展覆盖 ===

    def test_generate_key_parameters_affect_result(self):
        """测试不同参数生成不同的键。"""
        base_key = generate_key(['AAPL'], '1y', '1d', 'ohlcv', True)
        
        assert generate_key(['AAPL'], '6mo', '1d', 'ohlcv', True) != base_key
        assert generate_key(['AAPL'], '1y', '1h', 'ohlcv', True) != base_key
        assert generate_key(['AAPL'], '1y', '1d', 'dividends', True) != base_key
        assert generate_key(['AAPL'], '1y', '1d', 'ohlcv', False) != base_key

    def test_generate_key_empty_symbols(self):
        """测试空符号列表的处理。"""
        key = generate_key([], '1y', '1d', 'ohlcv', True)
        assert isinstance(key, str)
        assert len(key) == 32

    def test_generate_key_single_vs_multiple_symbols(self):
        """测试单个和多个符号生成不同的键。"""
        key_single = generate_key(['AAPL'], '1y', '1d', 'ohlcv', True)
        key_multiple = generate_key(['AAPL', 'MSFT'], '1y', '1d', 'ohlcv', True)
        assert key_single != key_multiple

    def test_generate_key_special_characters_in_symbols(self):
        """测试符号中包含特殊字符（如指数 ^VIX）。"""
        key = generate_key(['^VIX', '^GSPC'], '1d', '1d', 'ohlcv', False)
        assert isinstance(key, str)
        assert len(key) == 32

    def test_generate_key_reproducibility(self):
        """测试跨会话的重现性（MD5哈希的确定性）。"""
        key = generate_key(['AAPL', 'MSFT'], '1y', '1d', 'ohlcv', True)
        
        # 验证与预期的MD5哈希一致
        expected_data = "AAPL_MSFT_1y_1d_ohlcv_True"
        expected_key = hashlib.md5(expected_data.encode()).hexdigest()
        
        assert key == expected_key, f"Hash implementation changed! Expected: {expected_key}, Got: {key}"
