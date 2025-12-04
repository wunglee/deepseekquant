"""
市场枚举定义（共享模块）

职责：定义标准化的市场代码枚举
用途：替换项目中所有字符串常量，提供类型安全和自动补全
"""

from enum import Enum
from typing import Any


class MarketCode(str, Enum):
    """
    市场代码枚举
    
    继承自str使其可直接用于字符串比较和字典键
    覆盖market_config.py中的所有市场
    """
    CN = 'CN'  # 中国A股
    US = 'US'  # 美国股市
    HK = 'HK'  # 香港股市
    JP = 'JP'  # 日本股市
    EU = 'EU'  # 欧洲股市
    SG = 'SG'  # 新加坡股市
    UNKNOWN = 'UNKNOWN'  # 未识别/默认市场
    
    @classmethod
    def parse(cls, code: Any) -> 'MarketCode':
        """集中解析市场代码（字符串/枚举），失败回退为 UNKNOWN
        
        Args:
            code: 市场代码（字符串或枚举）
        
        Returns:
            MarketCode: 解析后的枚举，无法识别时返回 UNKNOWN
        
        Examples:
            >>> MarketCode.parse('CN')
            <MarketCode.CN: 'CN'>
            >>> MarketCode.parse(MarketCode.US)
            <MarketCode.US: 'US'>
            >>> MarketCode.parse('invalid')
            <MarketCode.UNKNOWN: 'UNKNOWN'>
        """
        if isinstance(code, cls):
            return code
        if isinstance(code, str):
            code_upper = code.upper()
            if cls.is_valid(code_upper):
                return cls(code_upper)
        return cls.UNKNOWN
    
    @classmethod
    def get_all_codes(cls) -> list:
        """获取所有市场代码"""
        return [market.value for market in cls]
    
    @classmethod
    def is_valid(cls, code: str) -> bool:
        """验证市场代码是否有效"""
        return code in cls.get_all_codes()
    
    def __str__(self) -> str:
        """支持直接字符串转换"""
        return self.value


class DataSource(str, Enum):
    """
    数据源枚举
    
    统一管理所有支持的数据源
    """
    YAHOO = 'yahoo'              # Yahoo Finance（全球）
    JOINQUANT = 'joinquant'      # 聚宽（A股优先）
    WIND = 'wind'                # Wind金融终端（港股、A股）
    TUSHARE = 'tushare'          # Tushare（A股、港股）
    ALPHA_VANTAGE = 'alpha_vantage'  # Alpha Vantage（美股）
    IEX = 'iex'                  # IEX Cloud（美股）
    MOCK = 'mock'                # 模拟数据源（测试）
    
    @classmethod
    def get_all_sources(cls) -> list:
        """获取所有数据源"""
        return [source.value for source in cls]
    
    @classmethod
    def is_valid(cls, source: str) -> bool:
        """验证数据源是否有效"""
        return source in cls.get_all_sources()
    
    def __str__(self) -> str:
        """支持直接字符串转换"""
        return self.value


# 数据源区域优先级映射（基于专家answer.md第2轮5.1节）
# 已迁移至 core_bak_refactored/config/regional_data_source.yml
# 直接从配置管理器加载
from core_bak_refactored.core.share.config_manager import ConfigManager
config_manager = ConfigManager()
REGIONAL_DATA_SOURCE_PRIORITY = config_manager.get('regional_data_source', {})
