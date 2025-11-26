"""
市场枚举定义（共享模块）

职责：定义标准化的市场代码枚举
用途：替换项目中所有字符串常量，提供类型安全和自动补全
"""

from enum import Enum


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
REGIONAL_DATA_SOURCE_PRIORITY = {
    MarketCode.CN: [
        DataSource.JOINQUANT,    # A股主数据源
        DataSource.TUSHARE,      # A股备用数据源
        DataSource.WIND,         # A股第三选择
        DataSource.YAHOO,        # 全球回退
        DataSource.MOCK          # 兜底
    ],
    MarketCode.US: [
        DataSource.YAHOO,        # 美股主数据源
        DataSource.ALPHA_VANTAGE,  # 美股备用
        DataSource.IEX,          # 美股第三选择
        DataSource.MOCK          # 兜底
    ],
    MarketCode.HK: [
        DataSource.WIND,         # 港股主数据源
        DataSource.TUSHARE,      # 港股备用（部分支持）
        DataSource.YAHOO,        # 全球回退
        DataSource.JOINQUANT,    # 第四选择
        DataSource.MOCK          # 兜底
    ],
    MarketCode.JP: [
        DataSource.YAHOO,        # 日股主数据源
        DataSource.MOCK          # 兜底
    ],
    MarketCode.EU: [
        DataSource.YAHOO,        # 欧股主数据源
        DataSource.MOCK          # 兜底
    ],
    MarketCode.SG: [
        DataSource.YAHOO,        # 新加坡主数据源
        DataSource.MOCK          # 兜底
    ],
    'default': [
        DataSource.YAHOO,        # 默认优先Yahoo
        DataSource.MOCK          # 兜底
    ]
}
