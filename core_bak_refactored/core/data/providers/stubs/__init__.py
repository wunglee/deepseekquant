"""
数据源 Stub 适配器

状态：临时实现，待真实API集成

包含：
- JoinQuant stub (A股优先数据源)
- Wind stub (港股优先数据源)
- Tushare stub (A股/港股备用数据源)
"""

from .joinquant_stub import JoinQuantStub
from .wind_stub import WindStub
from .tushare_stub import TushareStub

__all__ = ['JoinQuantStub', 'WindStub', 'TushareStub']
