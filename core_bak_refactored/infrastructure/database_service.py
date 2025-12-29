"""
数据库服务模块 - Database Service

职责:
- 统一管理数据库连接和配置
- 提供数据缓存和增量更新功能
- 集成 MarketDataRepository
- 实现数据持久化和查询优化

从 infrastructure/database.py 扩展而来
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

from core_bak_refactored.core.share.config_manager import ConfigManager
from core_bak_refactored.infrastructure.database import (
    get_database,
    MarketDataRepository,
    SQLiteDatabase
)

logger = logging.getLogger('DeepSeekQuant.Infrastructure.DatabaseService')


class DatabaseService:
    """
    数据库服务
    
    提供统一的数据库访问接口，支持:
    - 配置驱动的数据库初始化
    - 数据缓存和增量更新
    - 性能监控和优化
    """
    
    _instance: Optional['DatabaseService'] = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化数据库服务"""
        if self._initialized:
            return
        
        self.config_manager = ConfigManager()
        self.db_config = self._load_database_config()
        
        # 初始化数据库
        self.database = self._init_database()
        self.repository = MarketDataRepository(self.database)
        
        # 缓存配置
        self.cache_enabled = self.db_config.get('cache_strategy', {}).get('enabled', True)
        self.incremental_enabled = self.db_config.get('cache_strategy', {}).get(
            'incremental_update', {}
        ).get('enabled', True)
        
        # 性能监控
        self.monitoring_enabled = self.db_config.get('monitoring', {}).get('enabled', True)
        self.slow_query_threshold = self.db_config.get('monitoring', {}).get(
            'slow_query_threshold', 1000
        )
        
        self._initialized = True
        logger.info("数据库服务初始化完成")
    
    def _load_database_config(self) -> Dict[str, Any]:
        """加载数据库配置"""
        try:
            db_config = self.config_manager.get_config('database')
            if not db_config:
                logger.warning("未找到数据库配置,使用默认配置")
                return self._get_default_config()
            return db_config
        except Exception as e:
            logger.error(f"加载数据库配置失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认数据库配置"""
        return {
            'database_type': 'sqlite',
            'sqlite': {
                'database_path': 'data/market_data.db',
                'check_same_thread': False,
                'timeout': 30.0
            },
            'cache_strategy': {
                'enabled': True,
                'incremental_update': {
                    'enabled': True,
                    'warmup_days': 30,
                    'max_backfill_days': 365
                }
            },
            'monitoring': {
                'enabled': True,
                'slow_query_threshold': 1000
            }
        }
    
    def _init_database(self) -> SQLiteDatabase:
        """初始化数据库连接"""
        db_type = self.db_config.get('database_type', 'sqlite')
        
        if db_type == 'sqlite':
            sqlite_config = self.db_config.get('sqlite', {})
            db_path = sqlite_config.get('database_path', 'data/market_data.db')
            
            # 转换为绝对路径
            if not os.path.isabs(db_path):
                # 获取项目根目录（当前文件的父父目录）
                current_file = os.path.abspath(__file__)
                infrastructure_dir = os.path.dirname(current_file)
                core_bak_refactored_dir = os.path.dirname(infrastructure_dir)
                workspace_root = os.path.dirname(core_bak_refactored_dir)
                db_path = os.path.join(workspace_root, db_path)
            
            # 确保目录存在
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            
            database = get_database('sqlite', database_path=db_path)
            database.connect()
            
            # 应用 SQLite 性能优化
            self._optimize_sqlite(database, sqlite_config)
            
            logger.info(f"SQLite 数据库已连接: {db_path}")
            return database
        
        else:
            raise ValueError(f"不支持的数据库类型: {db_type}")
    
    def _optimize_sqlite(self, database: SQLiteDatabase, config: Dict[str, Any]) -> None:
        """应用 SQLite 性能优化配置"""
        try:
            # WAL 模式
            journal_mode = config.get('journal_mode', 'WAL')
            database.execute(f"PRAGMA journal_mode={journal_mode}")
            
            # 同步模式
            synchronous = config.get('synchronous', 'NORMAL')
            database.execute(f"PRAGMA synchronous={synchronous}")
            
            # 缓存大小
            cache_size = config.get('cache_size', -64000)
            database.execute(f"PRAGMA cache_size={cache_size}")
            
            # 临时存储
            temp_store = config.get('temp_store', 'MEMORY')
            database.execute(f"PRAGMA temp_store={temp_store}")
            
            database.commit()
            logger.info("SQLite 性能优化配置已应用")
        
        except Exception as e:
            logger.warning(f"应用 SQLite 优化失败: {e}")
    
    def get_cached_data(
        self,
        index_id: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        source: str = None
    ) -> Optional[pd.DataFrame]:
        """
        从缓存获取数据
        
        Args:
            index_id: 指数代码
            start_date: 开始日期 (pd.Timestamp)
            end_date: 结束日期 (pd.Timestamp)
            source: 数据源
        
        Returns:
            DataFrame 或 None（如果缓存未命中）
        """
        if not self.cache_enabled:
            return None
        
        try:
            start_time = pd.Timestamp.now()
            
            # 查询数据
            df = self.repository.query_prices(index_id, start_date, end_date)
            
            # 性能监控
            if self.monitoring_enabled:
                elapsed = (pd.Timestamp.now() - start_time).total_seconds() * 1000
                if elapsed > self.slow_query_threshold:
                    logger.warning(
                        f"慢查询: {index_id} {start_date}~{end_date} "
                        f"耗时 {elapsed:.0f}ms"
                    )
            
            if df.empty:
                logger.debug(f"缓存未命中: {index_id}")
                return None
            
            logger.info(
                f"缓存命中: {index_id} 返回 {len(df)} 条数据 "
                f"({df['date'].min()} ~ {df['date'].max()})"
            )
            return df
        
        except Exception as e:
            logger.error(f"从缓存获取数据失败: {e}")
            return None
    
    def cache_data(
        self,
        index_id: str,
        data: pd.DataFrame,
        source: str
    ) -> bool:
        """
        缓存数据到数据库
        
        Args:
            index_id: 指数代码
            data: 价格数据
            source: 数据源
        
        Returns:
            是否成功
        """
        if not self.cache_enabled or data.empty:
            return False
        
        try:
            start_time = pd.Timestamp.now()
            
            # 插入数据
            row_count = self.repository.insert_prices(index_id, data, source)
            
            # 性能监控
            if self.monitoring_enabled:
                elapsed = (pd.Timestamp.now() - start_time).total_seconds() * 1000
                logger.info(
                    f"数据已缓存: {index_id} 插入 {row_count} 条 "
                    f"耗时 {elapsed:.0f}ms"
                )
            
            return row_count > 0
        
        except Exception as e:
            logger.error(f"缓存数据失败: {e}")
            return False
    
    def get_incremental_update_params(
        self,
        index_id: str,
        requested_count: int
    ) -> Dict[str, Any]:
        """
        计算增量更新参数
        
        Args:
            index_id: 指数代码
            requested_count: 请求的数据条数
        
        Returns:
            包含 start_date, end_date, need_fetch 等信息的字典
        """
        if not self.incremental_enabled:
            # 不启用增量更新,返回完整请求
            return {
                'need_fetch': True,
                'start_date': None,
                'end_date': None,
                'count': requested_count,
                'reason': '增量更新未启用'
            }
        
        try:
            # 检查本地数据
            latest_date = self.repository.get_latest_date(index_id)
            
            if not latest_date:
                # 没有本地数据,需要全量获取
                config = self.db_config.get('cache_strategy', {}).get('incremental_update', {})
                max_backfill = config.get('max_backfill_days', 365)
                
                return {
                    'need_fetch': True,
                    'start_date': None,
                    'end_date': None,
                    'count': min(requested_count, max_backfill),
                    'reason': '本地无数据,需要初始化'
                }
            
            # 有本地数据,检查是否需要更新
            latest_dt = pd.to_datetime(latest_date)
            today = pd.Timestamp.now().normalize()
            days_old = (today - latest_dt).days
            
            config = self.db_config.get('cache_strategy', {}).get('incremental_update', {})
            update_interval_days = config.get('update_interval', 86400) / 86400
            
            if days_old <= 1:
                # 数据很新,不需要更新
                return {
                    'need_fetch': False,
                    'start_date': None,
                    'end_date': None,
                    'count': 0,
                    'reason': f'本地数据已是最新 (最新日期: {latest_date})',
                    'use_cache': True
                }
            
            # 需要增量更新
            start_date = (latest_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            
            return {
                'need_fetch': True,
                'start_date': start_date,
                'end_date': None,  # 到今天
                'count': days_old + 5,  # 多获取几天以防遗漏
                'reason': f'需要增量更新 {days_old} 天的数据',
                'is_incremental': True
            }
        
        except Exception as e:
            logger.error(f"计算增量更新参数失败: {e}")
            return {
                'need_fetch': True,
                'start_date': None,
                'end_date': None,
                'count': requested_count,
                'reason': f'计算失败: {e}'
            }
    
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        try:
            # 表行数
            result = self.database.fetch_one(
                "SELECT COUNT(*) as count FROM index_prices"
            )
            total_rows = result['count'] if result else 0
            
            # 索引数量
            result = self.database.fetch_one(
                "SELECT COUNT(DISTINCT index_id) as count FROM index_prices"
            )
            index_count = result['count'] if result else 0
            
            # 数据日期范围
            result = self.database.fetch_one(
                "SELECT MIN(date) as min_date, MAX(date) as max_date FROM index_prices"
            )
            date_range = {
                'start': result['min_date'] if result and result['min_date'] else None,
                'end': result['max_date'] if result and result['max_date'] else None
            }
            
            # 数据库文件大小
            db_path = self.database.database_path
            file_size = 0
            if db_path != ':memory:' and os.path.exists(db_path):
                file_size = os.path.getsize(db_path) / 1024 / 1024  # MB
            
            return {
                'total_rows': total_rows,
                'index_count': index_count,
                'date_range': date_range,
                'file_size_mb': round(file_size, 2),
                'database_path': db_path
            }
        
        except Exception as e:
            logger.error(f"获取数据库统计信息失败: {e}")
            return {}
    
    def close(self) -> None:
        """关闭数据库连接"""
        if self.database:
            self.database.close()
            logger.info("数据库连接已关闭")


# 全局数据库服务实例
_db_service: Optional[DatabaseService] = None


def get_database_service() -> DatabaseService:
    """获取数据库服务实例（单例）"""
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService()
    return _db_service
