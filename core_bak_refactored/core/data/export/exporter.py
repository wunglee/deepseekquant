"""
数据导出器

职责：
1. 导出数据到不同格式（CSV、JSON、Parquet等）
2. 支持流式导出大数据集
3. 压缩和批量导出
4. 数据格式化和编码处理
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import csv
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataExporter:
    """数据导出器，支持多种导出格式。"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化数据导出器。
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_to_csv(
        self,
        data: List[Dict],
        filename: str,
        encoding: str = 'utf-8'
    ) -> bool:
        """
        导出到CSV文件。
        
        Args:
            data: 数据列表
            filename: 文件名
            encoding: 编码格式
        
        Returns:
            是否成功
        """
        if not data:
            logger.warning("数据为空，跳过导出")
            return False
        
        try:
            filepath = self.output_dir / filename
            
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False, encoding=encoding)
            
            logger.info(f"成功导出CSV: {filepath}, {len(data)}条记录")
            return True
            
        except Exception as e:
            logger.error(f"导出CSV失败: {e}")
            return False
    
    def export_to_json(
        self,
        data: List[Dict],
        filename: str,
        indent: Optional[int] = 2,
        encoding: str = 'utf-8'
    ) -> bool:
        """
        导出到JSON文件。
        
        Args:
            data: 数据列表
            filename: 文件名
            indent: 缩进空格数
            encoding: 编码格式
        
        Returns:
            是否成功
        """
        if not data:
            logger.warning("数据为空，跳过导出")
            return False
        
        try:
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding=encoding) as f:
                json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
            
            logger.info(f"成功导出JSON: {filepath}, {len(data)}条记录")
            return True
            
        except Exception as e:
            logger.error(f"导出JSON失败: {e}")
            return False
    
    def export_to_parquet(
        self,
        data: List[Dict],
        filename: str,
        compression: str = 'snappy'
    ) -> bool:
        """
        导出到Parquet文件。
        
        Args:
            data: 数据列表
            filename: 文件名
            compression: 压缩方式
        
        Returns:
            是否成功
        """
        if not data:
            logger.warning("数据为空，跳过导出")
            return False
        
        try:
            filepath = self.output_dir / filename
            
            df = pd.DataFrame(data)
            df.to_parquet(filepath, compression=compression, index=False)
            
            logger.info(f"成功导出Parquet: {filepath}, {len(data)}条记录")
            return True
            
        except Exception as e:
            logger.error(f"导出Parquet失败: {e}")
            return False
    
    def export_to_excel(
        self,
        data: List[Dict],
        filename: str,
        sheet_name: str = 'Sheet1'
    ) -> bool:
        """
        导出到Excel文件。
        
        Args:
            data: 数据列表
            filename: 文件名
            sheet_name: 工作表名称
        
        Returns:
            是否成功
        """
        if not data:
            logger.warning("数据为空，跳过导出")
            return False
        
        try:
            filepath = self.output_dir / filename
            
            df = pd.DataFrame(data)
            df.to_excel(filepath, sheet_name=sheet_name, index=False)
            
            logger.info(f"成功导出Excel: {filepath}, {len(data)}条记录")
            return True
            
        except Exception as e:
            logger.error(f"导出Excel失败: {e}")
            return False
    
    def export_batch_to_csv(
        self,
        data_dict: Dict[str, List[Dict]],
        prefix: str = 'export'
    ) -> int:
        """
        批量导出到CSV文件。
        
        Args:
            data_dict: 名称到数据的映射
            prefix: 文件名前缀
        
        Returns:
            成功导出的文件数
        """
        success_count = 0
        
        for name, data in data_dict.items():
            filename = f"{prefix}_{name}.csv"
            if self.export_to_csv(data, filename):
                success_count += 1
        
        logger.info(f"批量导出完成: {success_count}/{len(data_dict)}")
        return success_count
    
    def stream_export_to_csv(
        self,
        data_generator,
        filename: str,
        chunk_size: int = 1000
    ) -> bool:
        """
        流式导出大数据集到CSV。
        
        Args:
            data_generator: 数据生成器
            filename: 文件名
            chunk_size: 每次写入的记录数
        
        Returns:
            是否成功
        """
        try:
            filepath = self.output_dir / filename
            first_chunk = True
            total_records = 0
            
            for chunk in data_generator:
                if not chunk:
                    continue
                
                df = pd.DataFrame(chunk)
                
                # 首次写入包含表头
                df.to_csv(
                    filepath,
                    mode='w' if first_chunk else 'a',
                    header=first_chunk,
                    index=False
                )
                
                first_chunk = False
                total_records += len(chunk)
            
            logger.info(f"流式导出完成: {filepath}, {total_records}条记录")
            return True
            
        except Exception as e:
            logger.error(f"流式导出失败: {e}")
            return False
    
    def export_with_metadata(
        self,
        data: List[Dict],
        filename: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        导出数据及元数据到JSON。
        
        Args:
            data: 数据列表
            filename: 文件名
            metadata: 元数据字典
        
        Returns:
            是否成功
        """
        try:
            filepath = self.output_dir / filename
            
            export_data = {
                'metadata': metadata,
                'data': data
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"成功导出数据和元数据: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"导出失败: {e}")
            return False
    
    def export_compressed(
        self,
        data: List[Dict],
        filename: str,
        format: str = 'json',
        compression: str = 'gzip'
    ) -> bool:
        """
        压缩导出数据。
        
        Args:
            data: 数据列表
            filename: 文件名（不含扩展名）
            format: 格式（json/csv）
            compression: 压缩方式（gzip/bz2/xz）
        
        Returns:
            是否成功
        """
        if not data:
            logger.warning("数据为空，跳过导出")
            return False
        
        try:
            df = pd.DataFrame(data)
            
            # 根据格式和压缩方式确定扩展名
            ext_map = {
                'gzip': '.gz',
                'bz2': '.bz2',
                'xz': '.xz'
            }
            
            ext = ext_map.get(compression, '')
            full_filename = f"{filename}.{format}{ext}"
            filepath = self.output_dir / full_filename
            
            if format == 'json':
                df.to_json(filepath, orient='records', compression=compression)
            elif format == 'csv':
                df.to_csv(filepath, index=False, compression=compression)
            else:
                logger.error(f"不支持的格式: {format}")
                return False
            
            logger.info(f"成功导出压缩文件: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"压缩导出失败: {e}")
            return False
    
    def get_export_summary(self) -> Dict[str, Any]:
        """
        获取导出统计摘要。
        
        Returns:
            统计摘要字典
        """
        try:
            files = list(self.output_dir.glob('*'))
            
            return {
                'output_dir': str(self.output_dir),
                'total_files': len(files),
                'file_types': {
                    'csv': len(list(self.output_dir.glob('*.csv'))),
                    'json': len(list(self.output_dir.glob('*.json'))),
                    'parquet': len(list(self.output_dir.glob('*.parquet'))),
                    'excel': len(list(self.output_dir.glob('*.xlsx')))
                }
            }
        except Exception as e:
            logger.error(f"获取导出摘要失败: {e}")
            return {}
