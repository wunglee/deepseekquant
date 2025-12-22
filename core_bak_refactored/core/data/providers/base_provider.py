"""  
Provider 基类 - 数据提供者封装

核心职责:
- 对外提供统一的数据接口
- 封装缓存管理器，自动处理缓存读写
- 子类实现具体的 API 调用逻辑

使用示例:
    provider = AKShareDataProvider()
    price_data = provider.get_index_prices(
        index_id='000300.SH',
        start_date='2025-01-01',
        end_date='2025-01-31',
        current_time=datetime.now()
    )
"""
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
import pandas as pd

from core_bak_refactored.core.data.providers.protocols import PriceData, HistoricalDataProvider
from core_bak_refactored.core.share.config_manager import ConfigManager
from core_bak_refactored.core.share.market import MarketUtils
from core_bak_refactored.core.share.market.market_enums import TradingPhase

logger = logging.getLogger('DeepSeekQuant.DataProviders')


class BaseDataProvider(ABC, HistoricalDataProvider):
    """
    数据提供者基类（封装缓存管理）
    
    核心职责:
    1. 对外提供统一的数据接口
    2. 封装缓存管理器，自动处理缓存读写
    3. 子类实现具体的 API 调用逻辑
    
    子类必须实现:
    - _fetch_from_external_api(symbol, start_date, end_date, period) -> PriceData
    """
    
    def __init__(self):
        """初始化数据提供者（从工厂获取缓存管理器）"""
        # 使用工厂方法创建缓存管理器（自动加载配置）
        from core_bak_refactored.infrastructure.cache import create_cache_manager
        self._cache_manager = create_cache_manager()
        self.config_manager = ConfigManager()
    def _get_with_cache(self,
                        index_id: str,
                        start_date: pd.Timestamp,
                        end_date: pd.Timestamp,
                        current_time: pd.Timestamp,
                        period: str = 'daily'):
        """
        带缓存的数据获取（核心方法）
        
        Args:
            index_id: 指数代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            current_time: 当前时间
            period: 数据粒度 ('daily'/'weekly'/'monthly'，传给API，默认 daily)
        
        Returns:
            PriceData 对象
        """
        logger.debug(f"📋 带缓存查询: {index_id}, {start_date} ~ {end_date}, period={period}")
        
        # 使用缓存管理器获取数据（period 是数据的本质属性，必须传给缓存层）
        result_df = self._cache_manager.get_data(
            symbol=index_id,
            from_date=start_date,
            to_date=end_date,
            period=period,  # 数据粒度/K线类型，必须作为缓存键的一部分
            db_fetch_func=None,  # ThreeLayerCacheManager 内部处理数据库缓存
            api_fetch_func=lambda s, e, period: self._fetch_from_api(index_id, s, e, period)
        )
        
        # 转换为 PriceData
        if result_df is not None and not result_df.empty:
            price_data = PriceData.from_dataframe(result_df, index_id)
            logger.info(f"✅ 返回数据: {len(result_df)} 条")
        else:
            # 返回空 PriceData
            price_data = PriceData(
                records=[],
                symbol=index_id,
                start_date=start_date,
                end_date=end_date,
                count=0
            )
            logger.warning(f"⚠️ 所有缓存都无数据: {index_id} {start_date}~{end_date}")
        
        # 设置 needs_realtime_kline 标记
        if price_data and price_data.count > 0:
            self.set_needs_realtime_kline(price_data, current_time)
            logger.debug(f"✅ needs_realtime_kline已设置为: {price_data.needs_realtime_kline}")
        
        return price_data
    
    def _fetch_from_api(self, index_id: str, start_date:pd.Timestamp, end_date:pd.Timestamp, period: str) -> Optional[pd.DataFrame]:
        """
        从 API 获取数据（为缓存管理器提供回调）
        
        Args:
            index_id: 指数代码
            start_date: 开始日期
            end_date: 结束日期
            period: 周期
        
        Returns:
            DataFrame 或 None
        """
        try:
            # 调用子类实现的 API 获取方法（子类负责周期转换）
            result = self._fetch_from_external_api(index_id, start_date, end_date, period)
            
            # 处理不同类型的返回值（支持 mock 测试）
            if isinstance(result, pd.DataFrame):
                # Mock 返回的 DataFrame
                return result if not result.empty else None
            elif result and hasattr(result, 'count') and result.count > 0:
                # PriceData 对象，转换为 DataFrame
                df = result.to_dataframe()
                return df if not df.empty else None
            else:
                return None
        except Exception as e:
            logger.error(f"❌ API查询失败: {index_id} {start_date}~{end_date}, error={e}")
        
        return None
    
    def set_needs_realtime_kline(self, price_data: PriceData, current_time: pd.Timestamp):
        """设置 needs_realtime_kline 标记
        
        根据当前交易时段判断是否需要获取实时K线：
        - 盘前/盘中/午盘：需要获取实时K线（True）
        - 盘后：不需要（False，当天K柱已在历史数据中）
        
        Args:
            price_data: 价格数据对象
            current_time: 当前时间
        """
        market_code = MarketUtils.infer_market_from_symbol(price_data.symbol)
        trading_phase = MarketUtils.determine_trading_phase(market_code, current_time)
        
        # 🔧 直接修改 price_data 对象的属性
        price_data.needs_realtime_kline = trading_phase in [
            TradingPhase.BEFORE_OPEN,
            TradingPhase.TRADING,
            TradingPhase.NOON_BREAK
        ]
    
    # ========================================================================
    # 数据获取接口（对外提供，自动使用缓存）
    # ========================================================================
    
    def get_index_prices(self, index_id: str, start_date: pd.Timestamp, end_date: pd.Timestamp, current_time: pd.Timestamp, period: str = 'daily'):
        """
        获取指数价格数据（对外接口，自动使用缓存）
        
        Args:
            index_id: 指数代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            current_time: 操作时间
            period: 数据粒度 ('daily'/'weekly'/'monthly'，默认 daily)
        
        Returns:
            PriceData: 价格数据对象
        """
        return self._get_with_cache(index_id, start_date, end_date, current_time, period)
    
    def get_stock_prices(self, stock_id: str, start_date: pd.Timestamp, end_date: pd.Timestamp, current_time: pd.Timestamp, period: str = 'daily'):
        """
        获取股票价格数据（对外接口，自动使用缓存）
        
        Args:
            stock_id: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            current_time: 操作时间
            period: 数据粒度 ('daily'/'weekly'/'monthly'，默认 daily)
        
        Returns:
            PriceData: 价格数据对象
        """
        return self._get_with_cache(stock_id, start_date, end_date, current_time, period)
    
    # ========================================================================
    # 内部接口（子类必须实现）
    # ========================================================================
    
    @abstractmethod
    def _fetch_from_external_api(self, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp, period: str = 'daily'):
        """
        从外部API获取数据（抽象方法，子类必须实现）
        
        Args:
            symbol: 股票/指数代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            period: 数据粒度 ('daily'/'weekly'/'monthly'，告诉API返回什么粒度，默认 daily)
        
        Returns:
            PriceData: 价格数据对象
        
        Raises:
            Exception: API调用失败时抛出异常
        """
        pass
    
    def _convert_period(self, price_data: 'PriceData', period: str) -> 'PriceData':
        """周期转换（日线→周线/月线）
        
        🎯 通用工具方法：供子类复用
        💚 强类型: 输入/输出都是 PriceData
        📝 注: 内部使用 pandas.resample()，但仅作为实现细节，对外仍是强类型
        
        使用场景：
        - 子类 API 不支持直接查询周线/月线时（如 AKShare、Finnhub）
        - 子类获取日线数据后，在 _fetch_from_external_api 中调用此方法转换
        
        Args:
            price_data: 日线数据（PriceData对象）
            period: 目标周期 ('weekly' 或 'monthly')
        
        Returns:
            转换后的 PriceData 对象
        """
        from core_bak_refactored.core.share.market.data_types import OHLCVRecord
        import pandas as pd
        
        # 如果已经是目标周期，直接返回
        if period == 'daily':
            return price_data
        
        # 临时转换为 DataFrame 进行周期重采样（这是 pandas 的优势）
        df = price_data.to_dataframe()
        df['date'] = pd.to_datetime(df['date'])
        df_copy = df.set_index('date')
        
        if period == 'weekly':
            df_copy = df_copy.resample('W').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        elif period == 'monthly':
            # 🔧 使用 'ME' 而不是 'M' 避免 FutureWarning
            df_copy = df_copy.resample('ME').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        else:
            # 不支持的周期，直接返回原数据
            logger.warning(f"不支持的周期类型: {period}，返回原始数据")
            return price_data
        
        df_copy = df_copy.reset_index()
        
        # 转换回 PriceData 强类型
        records = [
            OHLCVRecord(
                date=pd.Timestamp(row['date']),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume'])
            )
            for _, row in df_copy.iterrows()
        ]
        
        return PriceData(
            records=records,
            symbol=price_data.symbol,
            start_date=records[0].date if records else price_data.start_date,
            end_date=records[-1].date if records else price_data.end_date,
            count=len(records)
        )
    
    def get_test_symbol(self) -> str:
        """
        获取测试符号（子类可重写）
        
        Returns:
            str: 测试用的股票/指数代码
        """
        return '^GSPC'  # 默认测试符号：标普500
    
    def initialize(self, **kwargs):
        """
        初始化方法（可选实现）
        
        子类可以重写此方法来进行额外的初始化工作，
        例如根据传入的参数初始化客户端连接等。
        
        Args:
            **kwargs: 初始化参数
        """
        pass
    
    # ========================================================================
    # 配置管理接口（具体方法，基类统一实现）
    # ========================================================================
    
    @staticmethod
    @classmethod
    def _get_config_path(cls, filename: str) -> Path:
        """
        获取配置文件路径
        
        Args:
            filename: 配置文件名
            
        Returns:
            Path: 配置文件完整路径
            
        Note:
            使用 ConfigManager.get_config_path() 统一获取配置路径
        """
        from core_bak_refactored.core.share.config_manager import ConfigManager
        config_manager = ConfigManager()
        # 使用 ConfigManager 的封装方法获取配置路径
        config_path_str = config_manager.get_config_path(filename.replace('.yml', ''))
        return Path(config_path_str)
    
    @classmethod
    def test_provider(cls, provider_id: str, credential:str) -> Dict[str, Any]:
        """
        测试数据源连接
        
        Args:
            provider_id: 数据源ID
            credential: 临时凭证（用于测试）
            
        Returns:
            测试结果字典
        """
        logger.info(f"Testing connection for provider: {provider_id}")
        
        try:
            # 获取数据源配置
            config_manager = ConfigManager()
            data_config = config_manager.get_provider_config()
            providers = data_config.providers
            
            provider_config = next((p for p in providers if p.get('id') == provider_id or p.get('name') == provider_id), None)
            
            if not provider_config:
                return {
                    'status': 'error',
                    'test_result': 'failed',
                    'available': False,
                    'message': f'数据源不存在: {provider_id}',
                    'error_code': 'PROVIDER_NOT_FOUND'
                }
            
            # 动态创建适配器实例
            adapter_module = provider_config.get('adapter_module')
            adapter_class = provider_config.get('adapter_class')
            
            if not adapter_module or not adapter_class:
                return {
                    'status': 'error',
                    'test_result': 'failed',
                    'available': False,
                    'message': f'{provider_id} 适配器未实现',
                    'error_code': 'ADAPTER_NOT_IMPLEMENTED'
                }
            
            try:
                # 动态导入类
                module = __import__(adapter_module, fromlist=[adapter_class])
                provider_class = getattr(module, adapter_class)
                
                # 创建临时实例（各Provider从配置读取proxy，无需传参）
                test_instance = provider_class()
                
                # 如果Provider实现了initialize方法，调用它来初始化客户端
                if hasattr(test_instance, 'initialize'):
                    if credential:
                        test_instance.initialize(credential=credential)
                    else:
                        test_instance.initialize()
                
                # 使用适配器自定义的测试符号
                if hasattr(test_instance, 'get_test_symbol'):
                    test_symbol = test_instance.get_test_symbol()
                else:
                    test_symbol = '^GSPC'
                
                from datetime import datetime, timedelta
                end_date = pd.Timestamp.now()
                start_date = pd.Timestamp.now() - pd.Timedelta(days=30)
                
                start_time = time.time()
                
                # 执行测试查询
                test_data = test_instance.get_index_prices(test_symbol, start_date, end_date, datetime.now())
                
                latency_ms = int((time.time() - start_time) * 1000)
                
                # 处理PriceData对象
                if hasattr(test_data, 'to_dataframe'):
                    test_data_df = test_data.to_dataframe()
                    is_empty = test_data_df.empty
                    data_count = len(test_data_df)
                else:
                    is_empty = test_data.empty if test_data is not None else True
                    data_count = len(test_data) if test_data is not None else 0
                
                if test_data is None or is_empty:
                    # 测试失败：连接成功但返回空数据
                    is_available = False
                    message = f'{provider_id} 连接成功，但返回空数据'
                    logger.warning(f"{provider_id} 测试警告: {message}")
                    
                    result = {
                        'status': 'error',
                        'test_result': 'failed',
                        'available': is_available,
                        'message': message,
                        'details': {
                            'test_symbol': test_symbol,
                            'date_range': f'{start_date} to {end_date}',
                            'latency_ms': latency_ms
                        }
                    }
                else:
                    # 测试成功
                    is_available = True
                    message = f'{provider_id} 连接测试通过'
                    logger.info(f"{provider_id} 测试成功: {data_count} 条数据, {latency_ms}ms")
                    
                    result = {
                        'status': 'success',
                        'test_result': 'passed',
                        'available': is_available,
                        'message': message,
                        'details': {
                            'test_symbol': test_symbol,
                            'data_count': data_count,
                            'date_range': f'{start_date} to {end_date}',
                            'latency_ms': latency_ms
                        },
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # 测试成功后，保存凭证到文件
                    if credential:
                        cls.save_credentials(provider_id, credential)
                        logger.info(f"{provider_id} 凭证已保存")
                    
                    # 💚 保存测试状态到配置文件（关键修复）
                    cls.save_test_status(provider_id, 'passed')
                    logger.info(f"{provider_id} 测试状态已保存: passed")
                
                return result
                
            except Exception as test_error:
                logger.error(f"测试 {provider_id} 连接失败: {test_error}")
                return {
                    'status': 'error',
                    'test_result': 'failed',
                    'available': False,
                    'message': f'{provider_id} 连接测试失败: {str(test_error)}',
                    'error_code': 'CONNECTION_TEST_FAILED'
                }
                
        except Exception as e:
            logger.error(f"测试连接失败: {e}")
            return {
                'status': 'error',
                'test_result': 'failed',
                'available': False,
                'message': str(e),
                'error_code': 'TEST_CONNECTION_FAILED'
            }
    
    @classmethod
    def save_credentials(
        cls,
        provider_id: str,
        credential: str,
    ) -> bool:
        """
        保存数据源凭证
        
        Args:
            provider_id: 数据源ID
            credential: 凭证数据
            
        Returns:
            bool: 是否成功
            
        Note:
            凭证保存后不再重置测试状态，状态由下次系统启动时的实时测试决定
        """
        try:
            credentials_yml_path = cls._get_config_path('credentials.yml')
            
            # 读取现有凭证
            if credentials_yml_path.exists():
                with open(credentials_yml_path, 'r', encoding='utf-8') as f:
                    credentials_data = yaml.safe_load(f) or {}
            else:
                credentials_data = {}
            
            # 更新凭证
            credentials_data[provider_id] = credential
            
            # 写入凭证文件
            credentials_yml_path.parent.mkdir(parents=True, exist_ok=True)
            with open(credentials_yml_path, 'w', encoding='utf-8') as f:
                yaml.dump(credentials_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            
            logger.info(f"保存 {provider_id} 凭证成功: {credential}")
            return True
            
        except Exception as e:
            logger.error(f"保存凭证失败: {e}")
            return False
    
    @classmethod
    def delete_credentials(
        cls,
        provider_id: str,
    ) -> bool:
        """
        删除数据源凭证
        
        Args:
            provider_id: 数据源ID
            
        Returns:
            bool: 是否成功
        
        Examples:
            >>> BaseDataProvider.delete_credentials('yahoo')
            True
            >>> BaseDataProvider.delete_credentials('nonexistent_provider')
            True  # 即使凭证不存在也返回 True
        
        Note:
            - 如果凭证文件不存在，返回 True（视为已删除）
            - 如果 provider_id 不存在，也返回 True（视为已删除）
            - 只有当文件操作失败时才返回 False
        """
        try:
            credentials_yml_path = cls._get_config_path('credentials.yml')
            
            # 如果凭证文件不存在，视为已删除
            if not credentials_yml_path.exists():
                logger.info(f"凭证文件不存在，视为已删除: {credentials_yml_path}")
                return True
            
            # 读取现有凭证
            with open(credentials_yml_path, 'r', encoding='utf-8') as f:
                credentials_data = yaml.safe_load(f) or {}
            
            # 如果 provider_id 不存在，视为已删除
            if provider_id not in credentials_data:
                logger.info(f"{provider_id} 凭证不存在，视为已删除")
                return True
            
            # 删除凭证
            del credentials_data[provider_id]
            
            # 写入文件
            with open(credentials_yml_path, 'w', encoding='utf-8') as f:
                yaml.dump(credentials_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            
            logger.info(f"删除 {provider_id} 凭证成功")
            return True
            
        except Exception as e:
            logger.error(f"删除凭证失败: {e}")
            return False


    @classmethod
    def save_test_status(
        cls,
        provider_id: str,
        status: str,
    ) -> bool:
        """
        保存数据源测试状态到配置文件
        
        Args:
            provider_id: 数据源ID
            status: 测试状态 ('passed' | 'failed' | 'untested')
            
        Returns:
            bool: 是否成功
            
        Examples:
            >>> BaseDataProvider.save_test_status('yahoo', 'passed')
            True
            >>> BaseDataProvider.save_test_status('akshare', 'failed')
            True
        
        Note:
            - 状态保存到 data_provider.yml 中对应 provider 的 status 字段
            - 直接写入文件,确保持久化
        """
        try:
            from datetime import datetime
            from core_bak_refactored.core.share.config_manager import ConfigManager
            import yaml
            import os

            config_manager = ConfigManager()
            
            # 获取 data_provider.yml 的路径
            data_yml_path = config_manager.get_config_path('data')
            
            # 读取现有配置
            if os.path.exists(data_yml_path):
                with open(data_yml_path, 'r', encoding='utf-8') as f:
                    data_config = yaml.safe_load(f) or {}
            else:
                logger.error(f"配置文件不存在: {data_yml_path}")
                return False
            
            # 查找并更新 provider 状态
            providers = data_config.get('providers', [])
            provider_found = False
            
            for provider in providers:
                if provider.get('id') == provider_id:
                    provider['status'] = status
                    provider['last_test'] = datetime.now().isoformat()
                    provider_found = True
                    logger.info(f"更新 provider 状态: {provider_id} -> {status}")
                    break
            
            if not provider_found:
                logger.warning(f"Provider {provider_id} 不存在于配置文件中")
                return False
            
            # 💚 关键修复: 写入文件,确保持久化
            with open(data_yml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            
            logger.info(f"{provider_id} 测试状态已保存: {status}")
            
            # 重新加载配置(更新内存)
            config_manager._load_config()
            
            return True
            
        except Exception as e:
            logger.error(f"保存测试状态失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False












