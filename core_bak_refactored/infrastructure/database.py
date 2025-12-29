"""
Database Layer - 数据库抽象层

职责：
- 提供统一的数据库访问接口
- 支持多种数据库实现（SQLite内存/文件、PostgreSQL、MySQL等）
- 管理数据库连接生命周期
- 提供事务管理

设计原则：
- 抽象优于具体：通过Protocol定义接口
- 配置驱动：支持运行时切换数据库类型
- 资源安全：自动管理连接和事务
"""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol, List, Dict, Any, Optional, Union

import pandas as pd

logger = logging.getLogger('DeepSeekQuant.Infrastructure.Database')


class DatabaseProtocol(Protocol):
    """数据库接口协议"""
    
    def connect(self) -> None:
        """建立数据库连接"""
        ...
    
    def close(self) -> None:
        """关闭数据库连接"""
        ...
    
    def execute(self, query: str, params: tuple = ()) -> Any:
        """执行SQL语句"""
        ...
    
    def fetch_all(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """查询所有结果"""
        ...
    
    def fetch_one(self, query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        """查询单条结果"""
        ...
    
    def insert_many(self, table: str, data: List[Dict[str, Any]]) -> int:
        """批量插入数据"""
        ...
    
    def commit(self) -> None:
        """提交事务"""
        ...
    
    def rollback(self) -> None:
        """回滚事务"""
        ...


class SQLiteDatabase:
    """
    SQLite数据库实现
    
    特点：
    - 轻量级，无需独立服务器
    - 支持内存模式（:memory:）和文件模式
    - 适合开发、测试和小规模生产环境
    
    使用示例：
        # 内存数据库
        db = SQLiteDatabase(':memory:')
        
        # 文件数据库
        db = SQLiteDatabase('data/market_data.db')
    """
    
    def __init__(self, database_path: Union[str, Path] = ':memory:'):
        """
        初始化SQLite数据库
        
        Args:
            database_path: 数据库路径
                          - ':memory:' 内存数据库（重启丢失）
                          - 文件路径 持久化数据库
        """
        self.database_path = str(database_path)
        self.connection: Optional[sqlite3.Connection] = None
        self.is_connected = False
        
        # 自动创建目录
        if database_path != ':memory:':
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    
    def connect(self) -> None:
        """建立数据库连接"""
        if self.is_connected:
            logger.warning("数据库已连接，跳过重复连接")
            return
        
        try:
            self.connection = sqlite3.connect(
                self.database_path,
                check_same_thread=False,  # 允许多线程访问
                timeout=30.0  # 超时30秒
            )
            
            # 启用外键约束
            self.connection.execute("PRAGMA foreign_keys = ON")
            
            # 设置返回格式为字典
            self.connection.row_factory = sqlite3.Row
            
            self.is_connected = True
            logger.info(f"SQLite数据库已连接: {self.database_path}")
            
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise
    
    def close(self) -> None:
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            self.is_connected = False
            logger.info("数据库连接已关闭")
    
    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """
        执行SQL语句
        
        Args:
            query: SQL语句
            params: 参数元组
        
        Returns:
            Cursor对象
        """
        if not self.is_connected:
            self.connect()
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            return cursor
        except Exception as e:
            logger.error(f"SQL执行失败: {query[:100]}... Error: {e}")
            raise
    
    def fetch_all(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """查询所有结果"""
        cursor = self.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def fetch_one(self, query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        """查询单条结果"""
        cursor = self.execute(query, params)
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def insert_many(self, table: str, data: List[Dict[str, Any]]) -> int:
        """
        批量插入数据
        
        Args:
            table: 表名
            data: 数据列表
        
        Returns:
            插入的行数
        """
        if not data:
            return 0
        
        columns = list(data[0].keys())
        placeholders = ','.join(['?' for _ in columns])
        column_names = ','.join(columns)
        
        query = f"INSERT OR REPLACE INTO {table} ({column_names}) VALUES ({placeholders})"
        
        values = [tuple(row[col] for col in columns) for row in data]
        
        cursor = self.connection.cursor()
        cursor.executemany(query, values)
        
        return cursor.rowcount
    
    def commit(self) -> None:
        """提交事务"""
        if self.connection:
            self.connection.commit()
    
    def rollback(self) -> None:
        """回滚事务"""
        if self.connection:
            self.connection.rollback()
    
    @contextmanager
    def transaction(self):
        """
        事务上下文管理器
        
        使用示例：
            with db.transaction():
                db.execute("INSERT ...")
                db.execute("UPDATE ...")
        """
        try:
            yield self
            self.commit()
        except Exception as e:
            self.rollback()
            logger.error(f"事务回滚: {e}")
            raise
    
    def table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        result = self.fetch_one(query, (table_name,))
        return result is not None
    
    def create_table_from_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """
        从DataFrame创建表
        
        Args:
            table_name: 表名
            df: DataFrame数据
        """
        if not self.is_connected:
            self.connect()
        
        # 使用pandas的to_sql方法
        df.to_sql(
            table_name,
            self.connection,
            if_exists='replace',
            index=False
        )
        logger.info(f"表 {table_name} 已创建，共 {len(df)} 行")
    
    def __enter__(self):
        """支持with语句"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持with语句"""
        if exc_type:
            self.rollback()
        else:
            self.commit()
        self.close()


class MarketDataRepository:
    """
    市场数据仓库
    
    职责：
    - 管理历史价格数据的存储和查询
    - 提供增量更新机制
    - 数据去重和质量保证
    
    设计原则：
    - 单一职责：只负责数据持久化，不处理业务逻辑
    - 接口稳定：对外暴露统一的CRUD接口
    """
    
    def __init__(self, database: DatabaseProtocol):
        """
        初始化市场数据仓库
        
        Args:
            database: 数据库实现（支持协议的任何实现）
        """
        self.db = database
        self._ensure_tables()
    
    def _ensure_tables(self) -> None:
        """确保数据表存在"""
        if not self.db.is_connected:
            self.db.connect()
        
        # 创建指数价格表
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS index_prices (
                index_id TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL NOT NULL,
                volume REAL,
                source TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (index_id, date)
            )
        """)
        
        # 创建索引
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_index_date 
            ON index_prices(index_id, date)
        """)
        
        # 创建数据源同步记录表
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS sync_records (
                index_id TEXT NOT NULL,
                source TEXT NOT NULL,
                last_sync_date TEXT,
                sync_count INTEGER DEFAULT 0,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (index_id, source)
            )
        """)
        
        self.db.commit()
        logger.info("数据表已初始化")
    
    def get_latest_date(self, index_id: str) -> Optional[pd.Timestamp]:
        """
        获取指定指数的最新数据日期
        
        Args:
            index_id: 指数代码
        
        Returns:
            最新日期（pd.Timestamp）或None
        """
        query = """
            SELECT MAX(date) as latest_date 
            FROM index_prices 
            WHERE index_id = ?
        """
        result = self.db.fetch_one(query, (index_id,))
        if result and result['latest_date']:
            return pd.to_datetime(result['latest_date'])
        return None
    
    def get_date_range(self, index_id: str) -> Optional[Dict[str, pd.Timestamp]]:
        """
        获取指定指数的数据日期范围
        
        Returns:
            {'start': pd.Timestamp, 'end': pd.Timestamp} 或 None
        """
        query = """
            SELECT MIN(date) as start_date, MAX(date) as end_date
            FROM index_prices
            WHERE index_id = ?
        """
        result = self.db.fetch_one(query, (index_id,))
        
        if result and result['start_date']:
            return {
                'start': pd.to_datetime(result['start_date']),
                'end': pd.to_datetime(result['end_date'])
            }
        return None
    
    def query_prices(
        self,
        index_id: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        查询价格数据
        
        Args:
            index_id: 指数代码
            start_date: 开始日期（pd.Timestamp）
            end_date: 结束日期（pd.Timestamp）
        
        Returns:
            DataFrame with columns: ['date', 'close', 'volume', ...]
        """
        # 转换为字符串用于SQL查询
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        query = """
            SELECT date, open, high, low, close, volume, source
            FROM index_prices
            WHERE index_id = ? 
              AND date >= ? 
              AND date <= ?
            ORDER BY date ASC
        """
        
        rows = self.db.fetch_all(query, (index_id, start_date_str, end_date_str))
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def insert_prices(
        self,
        index_id: str,
        data: pd.DataFrame,
        source: str
    ) -> int:
        """
        插入/更新价格数据
        
        Args:
            index_id: 指数代码
            data: 价格数据（必须包含 date, close列）
            source: 数据来源
        
        Returns:
            插入的行数
        """
        if data.empty:
            return 0
        
        # 数据准备
        data = data.copy()
        data['index_id'] = index_id
        data['source'] = source
        
        # 日期格式化
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
        
        # 只保留数据库表中存在的列（过滤掉计算列如 returns、is_limit等）
        db_columns = ['index_id', 'date', 'open', 'high', 'low', 'close', 'volume', 'source']
        available_columns = [col for col in db_columns if col in data.columns]
        data_to_insert = data[available_columns]
        
        # 转换为字典列表
        records = data_to_insert.to_dict('records')
        
        # 批量插入
        row_count = self.db.insert_many('index_prices', records)
        self.db.commit()
        
        # 更新同步记录
        latest_date = data['date'].max()
        self._update_sync_record(index_id, source, latest_date, row_count)
        
        logger.info(f"插入 {row_count} 条数据: {index_id} from {source}")
        return row_count
    
    def _update_sync_record(
        self,
        index_id: str,
        source: str,
        last_date: pd.Timestamp,
        count: int
    ) -> None:
        """更新同步记录"""
        query = """
            INSERT OR REPLACE INTO sync_records 
            (index_id, source, last_sync_date, sync_count, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """
        self.db.execute(
            query,
            (index_id, source, last_date.strftime('%Y-%m-%d'), count, pd.Timestamp.now().isoformat())
        )
        self.db.commit()
    
    def get_missing_dates(
        self,
        index_id: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> List[pd.Timestamp]:
        """
        获取缺失的日期
        
        Args:
            index_id: 指数代码
            start_date: 开始日期（pd.Timestamp）
            end_date: 结束日期（pd.Timestamp）
        
        Returns:
            缺失的日期列表（pd.Timestamp）
        """
        # 查询已有数据
        existing = self.query_prices(index_id, start_date, end_date)
        
        if existing.empty:
            # 全部缺失
            date_range = pd.date_range(start_date, end_date, freq='D')
            return list(date_range)
        
        # 计算缺失日期
        existing_dates = set(existing['date'].dt.date)
        all_dates = pd.date_range(start_date, end_date, freq='D')
        
        missing = [d for d in all_dates if d.date() not in existing_dates]
        return missing


def get_database(database_type: str = 'sqlite', **kwargs) -> DatabaseProtocol:
    """
    数据库工厂函数
    
    Args:
        database_type: 数据库类型 ('sqlite', 'postgresql', 'mysql')
        **kwargs: 数据库配置参数
    
    Returns:
        数据库实例
    
    使用示例：
        # 内存数据库
        db = get_database('sqlite')
        
        # 文件数据库
        db = get_database('sqlite', database_path='data/market.db')
        
        # PostgreSQL（未来扩展）
        db = get_database('postgresql', host='localhost', ...)
    """
    if database_type == 'sqlite':
        db_path = kwargs.get('database_path', ':memory:')
        return SQLiteDatabase(db_path)
    
    # TODO: 支持其他数据库类型
    # elif database_type == 'postgresql':
    #     return PostgreSQLDatabase(**kwargs)
    # elif database_type == 'mysql':
    #     return MySQLDatabase(**kwargs)
    
    else:
        raise ValueError(f"不支持的数据库类型: {database_type}")
