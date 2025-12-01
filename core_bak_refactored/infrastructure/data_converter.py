"""
数据转换工具 - 基础设施层

职责：提供与业务无关的纯数据结构转换函数
- 列表到DataFrame转换
- DataFrame到列表转换
- 多标的数据聚合

架构原则：
- 不包含任何业务领域概念
- 只接收纯数据结构
- 参数全部显式传入
- 函数命名使用通用术语
"""

from typing import List, Any, Optional
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger('DeepSeekQuant.Infrastructure.DataConverter')


class DataConverter:
    """数据转换工具类（纯工具），不包含业务术语"""
    
    @staticmethod
    def list_to_dataframe(data: List[Any], 
                         field_mapping: Optional[dict] = None,
                         symbol_filter: Optional[str] = None) -> pd.DataFrame:
        """
        将对象列表转换为DataFrame
        
        Args:
            data: 对象列表
            field_mapping: 字段映射字典（属性名到列名的映射）
            symbol_filter: 可选的标的过滤
            
        Returns:
            DataFrame
        """
        if not data:
            return pd.DataFrame()
        
        records = []
        for d in data:
            # 过滤标的
            if symbol_filter and getattr(d, 'symbol', None) != symbol_filter:
                continue
            
            # 构建记录
            record = {}
            if field_mapping:
                # 使用字段映射
                for attr_name, col_name in field_mapping.items():
                    record[col_name] = getattr(d, attr_name, None)
            else:
                # 自动提取常见字段
                common_fields = ['timestamp', 'date', 'open', 'high', 'low', 'close', 
                               'volume', 'adj_close', 'turnover', 'vwap', 'symbol']
                for field in common_fields:
                    value = getattr(d, field, None)
                    if value is not None:
                        record[field] = value
            
            if record:
                records.append(record)
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        
        # 类型转换
        if 'date' in df.columns or 'timestamp' in df.columns:
            date_col = 'date' if 'date' in df.columns else 'timestamp'
            df[date_col] = pd.to_datetime(df[date_col])
        
        # 数值字段转换
        numeric_fields = ['open', 'high', 'low', 'close', 'volume', 
                         'adj_close', 'turnover', 'vwap']
        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce')
        
        return df
    
    @staticmethod
    def dataframe_to_list(df: pd.DataFrame, 
                         object_class: type,
                         field_mapping: Optional[dict] = None) -> List[Any]:
        """
        将DataFrame转换为对象列表
        
        Args:
            df: DataFrame
            object_class: 目标对象类
            field_mapping: 字段映射字典（列名到属性名的映射）
            
        Returns:
            对象列表
        """
        if df.empty:
            return []
        
        objects = []
        
        for _, row in df.iterrows():
            # 构建构造函数参数
            kwargs = {}
            
            if field_mapping:
                # 使用字段映射
                for col_name, attr_name in field_mapping.items():
                    if col_name in row:
                        kwargs[attr_name] = row[col_name]
            else:
                # 直接使用列名作为属性名
                for col_name, value in row.items():
                    kwargs[col_name] = value
            
            try:
                obj = object_class(**kwargs)
                objects.append(obj)
            except Exception as e:
                logger.warning(f"创建对象失败: {e}")
                continue
        
        return objects
    
    @staticmethod
    def aggregate_multi_source_data(data_list: List[List[Any]], 
                                  group_by_field: str = 'symbol') -> pd.DataFrame:
        """
        聚合多源数据为宽格式DataFrame
        
        Args:
            data_list: 数据源列表
            group_by_field: 分组字段名
            
        Returns:
            聚合后的DataFrame
        """
        if not data_list:
            return pd.DataFrame()
        
        # 合并所有数据
        all_data = []
        for data in data_list:
            all_data.extend(data)
        
        if not all_data:
            return pd.DataFrame()
        
        # 转换为DataFrame
        df = DataConverter.list_to_dataframe(all_data)
        
        if group_by_field not in df.columns:
            return df
        
        # 转换为宽格式（pivot）
        try:
            pivot_df = df.pivot_table(
                index='date' if 'date' in df.columns else df.columns[0],
                columns=group_by_field,
                values=['open', 'high', 'low', 'close', 'volume'],
                aggfunc='first'
            )
            
            # 扁平化列名
            pivot_df.columns = [f"{col[0]}_{col[1]}" for col in pivot_df.columns]
            pivot_df = pivot_df.reset_index()
            
            return pivot_df
        except Exception as e:
            logger.warning(f"数据聚合失败: {e}")
            return df