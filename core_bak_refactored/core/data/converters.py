"""数据结构转换器

职责：
- MarketData <-> DataFrame 互转
- 支持数据质量检查的数据结构适配

设计原则：
- 单一职责：仅负责数据格式转换
- 向后兼容：保留所有字段信息
- 高性能：使用向量化操作
"""

from typing import List, Optional
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger('DeepSeekQuant.Converters')


def market_data_to_dataframe(data: List, symbol_filter: Optional[str] = None) -> pd.DataFrame:
    """将MarketData列表转换为DataFrame
    
    Args:
        data: MarketData对象列表
        symbol_filter: 可选的标的过滤（仅保留指定symbol的数据）
    
    Returns:
        包含OHLCV等字段的DataFrame
        
    示例:
        >>> from core_bak_refactored.core.data.types import MarketData
        >>> market_data_list = [...]  # List[MarketData]
        >>> df = market_data_to_dataframe(market_data_list)
        >>> df.columns
        Index(['date', 'open', 'high', 'low', 'close', 'volume', 
               'adj_close', 'turnover', 'vwap', 'symbol'], dtype='object')
    """
    if not data:
        return pd.DataFrame()
    
    records = []
    for d in data:
        # 过滤标的
        if symbol_filter and getattr(d, 'symbol', None) != symbol_filter:
            continue
        
        record = {
            'date': getattr(d, 'timestamp', None) or getattr(d, 'date', None),
            'open': getattr(d, 'open', None),
            'high': getattr(d, 'high', None),
            'low': getattr(d, 'low', None),
            'close': getattr(d, 'close', None),
            'volume': getattr(d, 'volume', None),
            'adj_close': getattr(d, 'adj_close', None),
            'turnover': getattr(d, 'turnover', None),
            'vwap': getattr(d, 'vwap', None),
            'symbol': getattr(d, 'symbol', 'unknown')
        }
        records.append(record)
    
    if not records:
        logger.warning(f"转换后数据为空（原始{len(data)}条，过滤器={symbol_filter}）")
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    # 类型转换
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # 数值字段转换
    numeric_fields = ['open', 'high', 'low', 'close', 'volume', 
                     'adj_close', 'turnover', 'vwap']
    for field in numeric_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors='coerce')
    
    logger.debug(f"MarketData -> DataFrame: {len(data)}条 -> {len(df)}行")
    
    return df


def dataframe_to_market_data(df: pd.DataFrame, 
                             market_data_class,
                             symbol: str = 'unknown',
                             data_source: str = 'converted') -> List:
    """将DataFrame转换为MarketData列表（反向转换，如需要）
    
    Args:
        df: 数据DataFrame
        market_data_class: MarketData类（需传入，避免循环依赖）
        symbol: 默认标的代码
        data_source: 数据源标识
    
    Returns:
        MarketData对象列表
        
    注意:
        - 此方法较少使用，主要用于兼容旧接口
        - 推荐直接使用DataFrame进行质量检查
    """
    if df.empty:
        return []
    
    data_list = []
    
    for _, row in df.iterrows():
        market_data = market_data_class(
            symbol=row.get('symbol', symbol),
            timestamp=row.get('date', datetime.now()),
            open=float(row.get('open', 0.0)) if pd.notna(row.get('open')) else 0.0,
            high=float(row.get('high', 0.0)) if pd.notna(row.get('high')) else 0.0,
            low=float(row.get('low', 0.0)) if pd.notna(row.get('low')) else 0.0,
            close=float(row.get('close', 0.0)) if pd.notna(row.get('close')) else 0.0,
            volume=float(row.get('volume', 0.0)) if pd.notna(row.get('volume')) else 0.0,
            metadata={
                'data_source': data_source,
                'converted_from': 'dataframe'
            }
        )
        
        # 可选字段
        if 'adj_close' in row and pd.notna(row['adj_close']):
            market_data.adj_close = float(row['adj_close'])
        if 'turnover' in row and pd.notna(row['turnover']):
            market_data.turnover = float(row['turnover'])
        if 'vwap' in row and pd.notna(row['vwap']):
            market_data.vwap = float(row['vwap'])
        
        data_list.append(market_data)
    
    logger.debug(f"DataFrame -> MarketData: {len(df)}行 -> {len(data_list)}条")
    
    return data_list


def aggregate_multi_symbol_data(data: List, 
                                group_by_symbol: bool = True) -> pd.DataFrame:
    """聚合多标的数据为宽格式DataFrame（用于跨标的分析）
    
    Args:
        data: MarketData对象列表（可能包含多个symbol）
        group_by_symbol: 是否按symbol分组（True=宽格式，False=长格式）
    
    Returns:
        聚合后的DataFrame
        
    示例:
        长格式（group_by_symbol=False）:
            date       | symbol    | close | volume
            2024-01-01 | AAPL      | 150.0 | 1000000
            2024-01-01 | MSFT      | 300.0 | 800000
        
        宽格式（group_by_symbol=True）:
            date       | close_AAPL | close_MSFT | volume_AAPL | volume_MSFT
            2024-01-01 | 150.0      | 300.0      | 1000000     | 800000
    """
    if not data:
        return pd.DataFrame()
    
    df = market_data_to_dataframe(data)
    
    if not group_by_symbol or 'symbol' not in df.columns:
        return df
    
    # 转换为宽格式（pivot）
    pivot_df = df.pivot_table(
        index='date',
        columns='symbol',
        values=['open', 'high', 'low', 'close', 'volume'],
        aggfunc='first'  # 假设每天每标的只有一条数据
    )
    
    # 扁平化列名
    pivot_df.columns = [f"{col[0]}_{col[1]}" for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()
    
    logger.debug(f"聚合多标的数据: {len(data)}条 -> {len(pivot_df)}行 x {len(pivot_df.columns)}列")
    
    return pivot_df
