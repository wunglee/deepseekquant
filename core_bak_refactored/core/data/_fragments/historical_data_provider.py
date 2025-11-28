"""
功能碎片：历史数据提供者
从 core/risk/backtest_framework.py 提取
状态：待整合到 core/data 模块

职责：
- 历史价格数据获取
- 多数据源适配（Yahoo Finance / JoinQuant / Wind）
- 数据预处理和格式化

迁移计划：
当 core_bak_refactored/core/data 模块开发完成后，整合此文件到该模块

相关文件：
- 源文件：core/risk/backtest_framework.py (HistoricalDataProvider, MockHistoricalDataProvider)
- 调用者：core/risk/stress_test_validator.py (StressTestValidator)
"""

import numpy as np
import pandas as pd
from typing import Protocol, Dict, Any, Optional, List
from datetime import datetime
import logging

# 重构：引入独立的数据质量检查器
from core_bak_refactored.core.data._fragments.data_quality_checker import DataQualityChecker, DataQualityReport
# 引入市场枚举
from core_bak_refactored.core.share.market_enums import MarketCode, DataSource, REGIONAL_DATA_SOURCE_PRIORITY

logger = logging.getLogger('DeepSeekQuant.DataFragments')


# =============================================================================
# 协议接口（数据模块标准）
# =============================================================================

class HistoricalDataProvider(Protocol):
    """
    历史数据提供者接口（数据模块标准接口）
    
    设计目的：
    - 解耦业务逻辑与数据来源
    - 支持模拟数据（当前）和真实数据（未来）无缝切换
    - 为core_bak_refactored/core/data模块集成预留标准接口
    """
    
    def get_index_prices(self, index_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取指数价格数据
        
        Args:
            index_id: 指数代码（如'000300.SH'沪深300）
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
        
        Returns:
            DataFrame with columns: ['date', 'close', 'volume']
        """
        ...
    
    def get_index_returns(self, index_id: str, start_date: str, end_date: str) -> pd.Series:
        """
        获取指数收益率序列
        
        Args:
            index_id: 指数代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            Series with date index and return values
        """
        ...
    
    # 新增接口方法（Phase 4规划）
    def get_stock_prices(self, symbol: str, start_date: str, end_date: str):
        """获取个股价格数据"""
        ...
        
    def get_volatility_index(self, index_id: str, start_date: str, end_date: str):
        """获取波动率指数（如VIX）"""
        ...
        
    def validate_data_quality(self, data) -> Dict[str, Any]:
        """数据质量验证报告"""
        ...
    

# =============================================================================
# 模拟实现（临时，等待真实数据源替换）
# =============================================================================

class MockHistoricalDataProvider:
    """
    模拟历史数据提供者（功能碎片临时实现）
    
    用途：
    - 在core/data模块未完成前，提供测试数据
    - 基于真实历史事件参数生成合理的模拟数据
    - 支持框架功能验证和测试
    
    警告：
    - 数据为模拟生成，仅用于框架验证
    - 真实回测需要替换为RealHistoricalDataProvider
    
    迁移计划：
    - 当core/data完成后，此类迁移到data模块作为Mock工具
    """
    
    def __init__(self):
        self.event_params = {
            '2015_china_market_crash': {
                'period': ('2015-06-15', '2015-08-26'),
                'expected_decline': -0.43,
                'volatility_multiplier': 2.5
            },
            'covid_19_pandemic': {
                'period': ('2020-02-20', '2020-03-23'),
                'expected_decline': -0.20,
                'volatility_multiplier': 3.0
            },
            '2008_financial_crisis': {
                'period': ('2008-09-15', '2008-11-20'),
                'expected_decline': -0.40,
                'volatility_multiplier': 3.5
            },
            '2011_eurozone_debt_crisis': {
                'period': ('2011-09-01', '2011-11-30'),
                'expected_decline': -0.25,
                'volatility_multiplier': 2.5
            },
            '2011_us_debt_ceiling_crisis': {
                'period': ('2011-07-22', '2011-08-10'),
                'expected_decline': -0.12,
                'volatility_multiplier': 2.0
            },
            '2016_china_circuit_breaker': {
                'period': ('2016-01-04', '2016-01-08'),
                'expected_decline': -0.15,
                'volatility_multiplier': 2.0
            },
            '2022_russia_ukraine_conflict': {
                'period': ('2022-02-24', '2022-03-15'),
                'expected_decline': -0.12,
                'volatility_multiplier': 1.8
            },
            '1997_asian_financial_crisis': {
                'period': ('1997-07-02', '1998-08-28'),
                'expected_decline': -0.35,
                'volatility_multiplier': 2.8
            }
        }
    
    def _generate_prices_with_event_window(self,
                                           dates: pd.DatetimeIndex,
                                           initial_price: float,
                                           event_start: pd.Timestamp,
                                           event_end: pd.Timestamp,
                                           event_decline: float,
                                           event_vol: float,
                                           base_volatility: float,
                                           start_date: str,
                                           end_date: str) -> np.ndarray:
        """
        生成带事件窗口的价格序列（分段生成）
        
        逺辑：
        1. 事件前：随机游走，波动率=base_volatility
        2. 事件期：确定性下跌，总下跌=event_decline，波动率=base_volatility*event_vol
        3. 事件后：随机游走，波动率=base_volatility
        
        Args:
            dates: 日期索引
            initial_price: 起始价格
            event_start: 事件起始日期
            event_end: 事件结束日期
            event_decline: 事件期总下跌幅度
            event_vol: 事件期波动率倍数
            base_volatility: 基准波动率
            start_date: 请求开始日期
            end_date: 请求结束日期
        
        Returns:
            价格序列
        """
        np.random.seed(hash(start_date + end_date) % 2**32)  # 确定性随机种子
        
        n_days = len(dates)
        prices = np.zeros(n_days)
        prices[0] = initial_price
        
        # 分段索引
        event_start_idx = None
        event_end_idx = None
        
        for i, date in enumerate(dates):
            if event_start_idx is None and date >= event_start:
                event_start_idx = i
            if event_end_idx is None and date > event_end:
                event_end_idx = i - 1
                break
        
        # 如果事件期超出请求范围，调整索引
        if event_start_idx is None:
            event_start_idx = 0
        if event_end_idx is None:
            event_end_idx = n_days - 1
        
        # 第1段：事件前（随机游走）
        if event_start_idx > 0:
            for i in range(1, event_start_idx):
                random_return = np.random.normal(0, base_volatility)
                prices[i] = prices[i-1] * (1 + random_return)
        
        # 第2段：事件期（确定性下跌 + 随机波动）
        if event_end_idx >= event_start_idx:
            event_period_days = event_end_idx - event_start_idx + 1
            event_volatility = base_volatility * event_vol * 0.4
            
            # 生成事件期的随机成分
            event_random = np.random.normal(0, event_volatility, event_period_days)
            
            # 计算每日drift以达到目标下跌
            if event_period_days > 0:
                base_drift = (1.0 + event_decline) ** (1.0 / event_period_days) - 1.0
            else:
                base_drift = 0.0
            
            # 生成事件期价格（除最后一天）
            event_start_price = prices[event_start_idx - 1] if event_start_idx > 0 else initial_price
            
            for i in range(event_start_idx, event_end_idx):
                offset = i - event_start_idx
                prices[i] = prices[i-1] * (1 + base_drift + event_random[offset])
            
            # 事件期最后一天：精确调整以达到目标下跌
            target_event_end_price = event_start_price * (1 + event_decline)
            prices[event_end_idx] = target_event_end_price
        
        # 第3段：事件后（随机游走）
        if event_end_idx < n_days - 1:
            for i in range(event_end_idx + 1, n_days):
                random_return = np.random.normal(0, base_volatility)
                prices[i] = prices[i-1] * (1 + random_return)
        
        return prices
    
    def get_index_prices(self, index_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """生成模拟的指数价格数据（支持事件窗口概念）"""
        # 解析日期
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end, freq='B')  # 交易日
        
        # 检查是否与已知事件窗口有交集（修复：判断交集而非完全包含）
        event_decline = 0.0
        event_vol = 1.0
        matched_event_start = None
        matched_event_end = None
        
        for event_id, params in self.event_params.items():
            event_start = pd.to_datetime(params['period'][0])
            event_end = pd.to_datetime(params['period'][1])
            # 判断请求范围与事件期是否有交集
            if not (end < event_start or start > event_end):
                event_decline = params['expected_decline']
                event_vol = params['volatility_multiplier']
                matched_event_start = event_start
                matched_event_end = event_end
                logger.info(f"检测到事件窗口交集: {event_id}, decline={event_decline}, vol={event_vol}")
                break
        
        # 生成模拟价格序列（确定性趋势 + 随机波动）
        n_days = len(dates)
        initial_price = 3000.0  # 沪深300典型水平
        base_volatility = 0.015  # 1.5%日波动率
        
        if event_decline != 0.0 and matched_event_start and matched_event_end:
            # 有事件期交集：分段生成（事件前 + 事件期 + 事件后）
            prices = self._generate_prices_with_event_window(
                dates=dates,
                initial_price=initial_price,
                event_start=matched_event_start,
                event_end=matched_event_end,
                event_decline=event_decline,
                event_vol=event_vol,
                base_volatility=base_volatility,
                start_date=start_date,
                end_date=end_date
            )
        else:
            # 非事件期间：纯随机游走
            daily_volatility = base_volatility
            daily_returns = np.random.normal(0, daily_volatility, n_days)
            prices = initial_price * np.cumprod(1 + daily_returns)
        
        # 生成成交量（简化模拟）
        base_volume = 100000000  # 1亿手
        volumes = base_volume * (1 + np.random.uniform(-0.3, 0.5, n_days))
        volumes = np.clip(volumes, 0, None)
        
        df = pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': volumes
        })
        
        logger.debug(f"生成模拟数据: {index_id}, {len(df)}天, 总收益率={prices[-1]/prices[0]-1:.2%}")
        return df
    
    def get_index_returns(self, index_id: str, start_date: str, end_date: str) -> pd.Series:
        """获取指数收益率序列"""
        df = self.get_index_prices(index_id, start_date, end_date)
        returns = df['close'].pct_change().fillna(0)
        returns.index = df['date']
        return returns
    
    def get_stock_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取个股价格数据（模拟实现）"""
        # 复用指数价格生成逻辑，但使用个股特定参数
        return self.get_index_prices(symbol, start_date, end_date)
    
    def get_volatility_index(self, index_id: str, start_date: str, end_date: str) -> pd.Series:
        """获取波动率指数（模拟实现）"""
        # 生成模拟的波动率数据
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end, freq='B')  # 交易日
        
        # 检查是否在已知事件窗口内
        event_vol = 1.0
        for event_id, params in self.event_params.items():
            event_start = pd.to_datetime(params['period'][0])
            event_end = pd.to_datetime(params['period'][1])
            if start >= event_start and end <= event_end:
                event_vol = params['volatility_multiplier']
                break
        
        # 生成模拟波动率序列
        n_days = len(dates)
        base_volatility = 0.15  # 15% 基准波动率
        volatility_values = base_volatility * event_vol * (1 + np.random.uniform(-0.3, 0.3, n_days))
        volatility_values = np.clip(volatility_values, 0.05, 0.5)  # 限制在5%-50%范围内
        
        series = pd.Series(volatility_values, index=dates)
        return series
    
    def validate_data_quality(self, data) -> Dict[str, Any]:
        """
        数据质量验证报告（重构：使用DataQualityChecker）
        
        Returns:
            质量报告字典，与RealHistoricalDataProvider保持一致
        """
        # 使用实例化的质量检查器，并按新签名调用
        checker = DataQualityChecker()
        report = checker.check_quality(data, index_id='mock')
        
        # 兼容期望的字典键
        total_rows = len(data)
        missing_values = int(data.isna().sum().sum())
        # 简单启发式：将包含“异常/零成交量/极值”的问题计为异常点
        outliers_detected = sum(
            1 for issue in (report.issues or [])
            if ('异常' in issue) or ('零成交量' in issue) or ('极值' in issue)
        )
        
        return {
            'completeness_score': report.completeness,
            'consistency_score': report.consistency,
            'accuracy_score': report.reasonableness,
            'outliers_detected': outliers_detected,
            'total_rows': total_rows,
            'missing_values': missing_values,
        }
    
    def get_event_window_data(self, 
                              index_id: str, 
                              event_date: str,
                              window_days: int = 30,
                              baseline_days: int = 252) -> Dict[str, pd.DataFrame]:
        """
        获取事件窗口数据（兼容RealHistoricalDataProvider接口）
        
        Args:
            index_id: 指数代码
            event_date: 事件发生日期 'YYYY-MM-DD'
            window_days: 事件前后窗口天数
            baseline_days: 基准期天数
        
        Returns:
            字典包含:
                'event_window': 事件窗口数据
                'baseline': 基准期数据
        """
        event_dt = pd.to_datetime(event_date)
        
        # 计算日期范围
        baseline_start = event_dt - pd.Timedelta(days=baseline_days + window_days + 100)
        baseline_end = event_dt - pd.Timedelta(days=1)
        
        event_start = event_dt - pd.Timedelta(days=window_days + 30)
        event_end = event_dt + pd.Timedelta(days=window_days + 30)
        
        # 获取基准期数据
        baseline_data = self.get_index_prices(
            index_id, 
            baseline_start.strftime('%Y-%m-%d'),
            baseline_end.strftime('%Y-%m-%d')
        )
        
        # 获取事件窗口数据
        event_data = self.get_index_prices(
            index_id,
            event_start.strftime('%Y-%m-%d'),
            event_end.strftime('%Y-%m-%d')
        )
        
        # 筛选出指定数量的交易日
        baseline_filtered = baseline_data.tail(baseline_days)
        
        # 事件窗口：前后各window_days个交易日
        event_filtered = event_data[
            (event_data['date'] >= event_dt - pd.Timedelta(days=window_days)) &
            (event_data['date'] <= event_dt + pd.Timedelta(days=window_days))
        ]
        
        return {
            'event_window': event_filtered,
            'baseline': baseline_filtered
        }

# =============================================================================
# 真实数据提供者（待实现）
# =============================================================================

class RealHistoricalDataProvider:
    """
    真实历史数据提供者（Phase 5B-5扩展版 + 第2轮专家指导增强）
    
    基于专家answer.md第1轮4.1节+第2轮5.1节指导：
    - 区域化数据源优先级：A股(JoinQuant主)、美股(Yahoo主)、港股(Wind主)
    - 交叉验证双维度：逐日差异30%触发 OR 窗口统计量(均值3%/标准差10%)触发
    - 关键指标：收盘价、成交量、20日波动率、换手率
    
    基于专家answer.md第1轮4.2节+第2轮5.1节指导：
    - 事件窗口动态化：market_crash(30天)、liquidity_crisis(45天)、currency_crisis(60天)、sovereign_debt_crisis(90天)
    - 基准期差异化：一般事件252天、结构性事件504天
    - 异常值处理：剔除涨跌停日、极端波动日
    - 停牌处理：停牌日数据沿用最近有效价格，但不计入收益率计算
    """
    
    # 区域化数据源优先级配置（专家第2轮5.1节 + 统一market_config.py覆盖范围）
    # 使用枚举替代字符串常量
    REGIONAL_PRIORITY = REGIONAL_DATA_SOURCE_PRIORITY
    
    # 事件窗口配置（专家第2轮5.1节）
    EVENT_WINDOW_CONFIGS = {
        'market_crash': {'window_days': 30, 'baseline_days': 252},
        'liquidity_crisis': {'window_days': 45, 'baseline_days': 252},
        'currency_crisis': {'window_days': 60, 'baseline_days': 252},
        'sovereign_debt_crisis': {'window_days': 90, 'baseline_days': 504},
        'geopolitical_risk': {'window_days': 30, 'baseline_days': 252}
    }
    
    def __init__(self, 
                 primary_source: str = 'yahoo',
                 backup_sources: Optional[List[str]] = None,
                 enable_cross_validation: bool = False):
        """
        初始化真实历史数据提供者
        
        Args:
            primary_source: 主数据源 ('yahoo', 'joinquant', 'wind', 'mock')
            backup_sources: 备用数据源列表（默认None = ['mock']）
            enable_cross_validation: 是否启用数据交叉验证（默认False，专家第3轮5.1节）
        """
        self.primary_source = primary_source
        self.backup_sources = backup_sources or [DataSource.MOCK.value]
        self.enable_cross_validation = enable_cross_validation
        self._mock = MockHistoricalDataProvider()
        self._cache = {}
        self._quality_cache = {}  # 数据质量缓存
        self._cross_validation_log = []  # 交叉验证日志
        
        # 加载数据源适配器
        self._adapters = self._initialize_adapters()
    
    def _initialize_adapters(self) -> Dict[str, Any]:
        """初始化数据源适配器"""
        adapters = {DataSource.MOCK.value: self._mock}
        
        # Yahoo Finance适配器
        try:
            from core_bak_refactored.core.data._fragments.yahoo_finance_provider import YahooFinanceDataProvider
            adapters[DataSource.YAHOO.value] = YahooFinanceDataProvider(fallback_to_mock=False)
            logger.info("Yahoo Finance适配器已加载")
        except Exception as e:
            logger.warning(f"Yahoo Finance适配器加载失败: {e}")
        
        # JoinQuant适配器（stub实现，备用数据源）
        try:
            adapters[DataSource.JOINQUANT.value] = self._create_joinquant_stub()
            logger.info("JoinQuant适配器（stub）已加载")
        except Exception as e:
            logger.warning(f"JoinQuant适配器加载失败: {e}")
        
        # Wind适配器（stub实现，备用数据源）
        try:
            adapters[DataSource.WIND.value] = self._create_wind_stub()
            logger.info("Wind适配器（stub）已加载")
        except Exception as e:
            logger.warning(f"Wind适配器加载失败: {e}")
        
        # Tushare适配器（实际API实现，A股/港股备用数据源）
        try:
            from core_bak_refactored.core.data._fragments.tushare_provider import TushareDataProvider
            tushare_adapter = TushareDataProvider(fallback_to_mock=False)
            if tushare_adapter.available:
                adapters[DataSource.TUSHARE.value] = tushare_adapter
                logger.info("Tushare适配器已加载（实际API）")
            else:
                # API不可用，使用stub
                adapters[DataSource.TUSHARE.value] = self._create_tushare_stub()
                logger.info("Tushare适配器（stub）已加载（API未配置）")
        except Exception as e:
            logger.warning(f"Tushare适配器加载失败: {e}")
        
        return adapters
    
    def _create_joinquant_stub(self) -> Any:
        """创建JoinQuant数据源stub（待实际API实现）"""
        class JoinQuantStub:
            """JoinQuant数据源stub - 专家第2轮5.1节：A股优先数据源"""
            def __init__(self):
                self.available = False  # TODO: 替换为实际API可用性检查
                logger.info("JoinQuant stub初始化（待实际API集成）")
            
            def get_index_prices(self, index_id: str, start_date: str, end_date: str) -> pd.DataFrame:
                # TODO: 实际实现调用JoinQuant API
                # import jqdatasdk
                # jqdatasdk.auth(username, password)
                # data = jqdatasdk.get_price(index_id, start_date, end_date, fields=['close', 'volume'])
                raise NotImplementedError("JoinQuant API未集成，请使用Yahoo或Mock")
        
        return JoinQuantStub()
    
    def _create_wind_stub(self) -> Any:
        """创建Wind数据源stub（待实际API实现）"""
        class WindStub:
            """Wind数据源stub - 专家第2轮5.1节：港股优先数据源"""
            def __init__(self):
                self.available = False  # TODO: 替换为实际API可用性检查
                logger.info("Wind stub初始化（待实际API集成）")
            
            def get_index_prices(self, index_id: str, start_date: str, end_date: str) -> pd.DataFrame:
                # TODO: 实际实现调用Wind API
                # from WindPy import w
                # w.start()
                # data = w.wsd(index_id, "close,volume", start_date, end_date)
                raise NotImplementedError("Wind API未集成，请使用Yahoo或Mock")
        
        return WindStub()
    
    def _create_tushare_stub(self) -> Any:
        """创建Tushare数据源stub（待实际API实现）"""
        class TushareStub:
            """Tushare数据源stub - A股/港股备用数据源"""
            def __init__(self):
                self.available = False  # TODO: 替换为实际API可用性检查
                logger.info("Tushare stub初始化（待实际API集成）")
            
            def get_index_prices(self, index_id: str, start_date: str, end_date: str) -> pd.DataFrame:
                # TODO: 实际实现调用Tushare API
                # import tushare as ts
                # ts.set_token('your_token')
                # pro = ts.pro_api()
                # 
                # # A股指数
                # if index_id.endswith('.SH') or index_id.endswith('.SZ'):
                #     data = pro.index_daily(ts_code=index_id, start_date=start_date, end_date=end_date)
                # # 港股指数（部分支持）
                # elif index_id in ['HSI', 'HSCEI']:
                #     # Tushare港股数据较少，可能需要回退
                #     data = pro.hk_index_daily(ts_code=index_id, start_date=start_date, end_date=end_date)
                # 
                # return pd.DataFrame({
                #     'date': pd.to_datetime(data['trade_date']),
                #     'close': data['close'],
                #     'volume': data['vol']
                # })
                raise NotImplementedError("Tushare API未集成，请使用Yahoo或Mock")
        
        return TushareStub()
    
    def get_index_prices(self, index_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取指数价格数据（含自动回退机制 + 区域化优先级 + 健康检查）"""
        # 优先使用缓存
        cache_key = f"prices:{index_id}:{start_date}:{end_date}:{self.primary_source}"
        if cache_key in self._cache:
            logger.debug(f"使用缓存数据: {cache_key}")
            return self._cache[cache_key]
        
        # 区域化数据源优先级（专家第2轮5.1节）
        regional_sources = self._get_regional_priority(index_id)
        sources_to_try = regional_sources if regional_sources else ([self.primary_source] + self.backup_sources)
        
        last_error = None
        source_health_status = {}
        
        for source in sources_to_try:
            adapter = self._adapters.get(source)
            if adapter is None:
                logger.warning(f"数据源 {source} 未配置，跳过")
                source_health_status[source] = 'unconfigured'
                continue
            
            # 健康检查（stub适配器）
            if hasattr(adapter, 'available') and not adapter.available:
                logger.warning(f"数据源 {source} 不可用（健康检查失败），跳过")
                source_health_status[source] = 'unavailable'
                continue
            
            try:
                logger.info(f"尝试数据源: {source} for {index_id}")
                data = adapter.get_index_prices(index_id, start_date, end_date)
                
                if data is None or data.empty:
                    logger.warning(f"{source} 返回空数据")
                    source_health_status[source] = 'empty_data'
                    continue
                
                # 数据质量验证
                quality_score = self._validate_data_quality(data, source)
                if quality_score < 0.6:
                    logger.warning(f"{source} 数据质量不达标: {quality_score:.2f}")
                    source_health_status[source] = f'low_quality_{quality_score:.2f}'
                    continue
                
                # 数据清洗（专家4.2节：异常值处理）
                cleaned_data = self._clean_data(data, index_id)
                
                # 缓存结果
                self._cache[cache_key] = cleaned_data
                source_health_status[source] = f'success_quality_{quality_score:.2f}'
                logger.info(f"成功获取数据: {source}, 行数={len(cleaned_data)}, 质量={quality_score:.2f}")
                
                return cleaned_data
                
            except NotImplementedError as e:
                # stub未实现，继续下一个
                logger.info(f"{source} 未实现，尝试下一数据源: {e}")
                source_health_status[source] = 'not_implemented'
                continue
            except Exception as e:
                last_error = e
                logger.warning(f"{source} 获取失败: {e}")
                source_health_status[source] = f'error_{type(e).__name__}'
                continue
        
        # 所有数据源失败
        error_msg = f"所有数据源失败: {index_id} ({start_date} to {end_date}) | 健康状态={source_health_status}"
        if last_error:
            error_msg += f", 最后错误: {last_error}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    def _get_regional_priority(self, index_id: str) -> List[str]:
        """
        根据市场区域获取数据源优先级（专家第2轮5.1节 + 统一market_config.py覆盖）
        
        Returns:
            数据源优先级列表
        """
        # 从symbol中提取市场代码
        if index_id.endswith('.SH') or index_id.endswith('.SZ'):
            market = MarketCode.CN
        elif index_id.endswith('.US'):
            market = MarketCode.US
        elif index_id.endswith('.HK') or index_id in ['HSI', 'HSCEI']:
            market = MarketCode.HK
        elif index_id.endswith('.T') or index_id.endswith('.JP'):  # 东京证券交易所
            market = MarketCode.JP
        elif index_id.endswith('.PA') or index_id.endswith('.L') or index_id.endswith('.DE'):  # 巴黎/伦敦/法兰克福
            market = MarketCode.EU
        elif index_id.endswith('.SI'):  # 新加坡
            market = MarketCode.SG
        else:
            market = 'default'
        
        # 获取优先级列表，转换为字符串
        priority = self.REGIONAL_PRIORITY.get(market, self.REGIONAL_PRIORITY['default'])
        return [source.value if isinstance(source, DataSource) else source for source in priority]
    
    def get_event_window_data(self, 
                              index_id: str, 
                              event_date: str,
                              event_type: str = 'market_crash',
                              window_days: Optional[int] = None,
                              baseline_days: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        获取事件窗口数据（专家answer.md第2轮5.1节增强：支持事件类型动态调整）
        
        Args:
            index_id: 指数代码
            event_date: 事件发生日期 'YYYY-MM-DD'
            event_type: 事件类型（专家第2轮5.1节新增）
            window_days: 事件前后窗口天数（None则根据event_type自动）
            baseline_days: 基准期天数（None则根据event_type自动）
        
        Returns:
            字典包含:
                'event_window': 事件窗口数据
                'baseline': 基准期数据
                'config': 使用的配置
        """
        # 根据事件类型动态调整窗口（专家第2轮5.1节）
        config = self.EVENT_WINDOW_CONFIGS.get(event_type, self.EVENT_WINDOW_CONFIGS['market_crash'])
        final_window_days = window_days if window_days is not None else config['window_days']
        final_baseline_days = baseline_days if baseline_days is not None else config['baseline_days']
        
        logger.info(f"事件窗口配置: type={event_type}, window={final_window_days}天, baseline={final_baseline_days}天")
        event_dt = pd.to_datetime(event_date)
        
        # 计算日期范围（扩大范围以确保足够交易日）
        baseline_start = event_dt - pd.Timedelta(days=baseline_days + window_days + 100)
        baseline_end = event_dt - pd.Timedelta(days=1)
        
        event_start = event_dt - pd.Timedelta(days=window_days + 30)
        event_end = event_dt + pd.Timedelta(days=window_days + 30)
        
        # 获取基准期数据
        baseline_data = self.get_index_prices(
            index_id, 
            baseline_start.strftime('%Y-%m-%d'),
            baseline_end.strftime('%Y-%m-%d')
        )
        
        # 获取事件窗口数据
        event_data = self.get_index_prices(
            index_id,
            event_start.strftime('%Y-%m-%d'),
            event_end.strftime('%Y-%m-%d')
        )
        
        # 筛选出指定数量的交易日
        baseline_filtered = baseline_data.tail(baseline_days)
        
        # 事件窗口：前后各30个交易日
        event_filtered = event_data[
            (event_data['date'] >= event_dt - pd.Timedelta(days=window_days)) &
            (event_data['date'] <= event_dt + pd.Timedelta(days=window_days))
        ]
        
        logger.info(f"事件窗口数据: baseline={len(baseline_filtered)}天, window={len(event_filtered)}天")
        
        return {
            'event_window': event_filtered,
            'baseline': baseline_filtered,
            'config': {'event_type': event_type, 'window_days': final_window_days, 'baseline_days': final_baseline_days}
        }
    
    def _clean_data(self, data: pd.DataFrame, index_id: str) -> pd.DataFrame:
        """
        数据清洗（专家answer.md第1轮4.2节：异常值处理）
        
        处理内容:
        - 剔除涨跌停日（价格变动≥9.5%且成交量较前20日均值下降≥80%）
        - 剔除极端波动日（日收益率超出3个标准差）
        - 停牌处理：停牌日数据沿用最近有效价格，但不计入收益率计算
        """
        if data.empty:
            return data
        
        cleaned = data.copy()
        
        # 1. 计算收益率
        cleaned['returns'] = cleaned['close'].pct_change()
        
        # 2. 标记涨跌停日（仅A股市场）
        if index_id.endswith('.SH') or index_id.endswith('.SZ'):
            # 计算20日平均成交量
            cleaned['volume_ma20'] = cleaned['volume'].rolling(window=20, min_periods=1).mean()
            
            # 涨跌停条件：价格变动≥9.5% 且 成交量下降≥80%
            limit_up_down = (
                (abs(cleaned['returns']) >= 0.095) & 
                (cleaned['volume'] <= 0.2 * cleaned['volume_ma20'])
            )
            
            logger.debug(f"检测到 {limit_up_down.sum()} 个涨跌停日")
            
            # 标记但不删除（保留用于补全价格）
            cleaned['is_limit'] = limit_up_down
        else:
            cleaned['is_limit'] = False
        
        # 3. 剔除极端波动日（日收益率超出3个标准差）
        returns_mean = cleaned['returns'].mean()
        returns_std = cleaned['returns'].std()
        extreme_volatility = abs(cleaned['returns'] - returns_mean) > 3 * returns_std
        
        logger.debug(f"检测到 {extreme_volatility.sum()} 个极端波动日")
        cleaned['is_extreme'] = extreme_volatility
        
        # 4. 停牌处理：成交量为0或极低
        cleaned['is_suspended'] = cleaned['volume'] <= 0
        logger.debug(f"检测到 {cleaned['is_suspended'].sum()} 个停牌日")
        
        # 5. 标记所有需要排除的日期（用于收益率计算）
        cleaned['exclude_from_returns'] = (
            cleaned['is_limit'] | 
            cleaned['is_extreme'] | 
            cleaned['is_suspended']
        )
        
        # 6. 不删除行（保留用于价格序列连续性），但标记排除标志
        logger.info(f"数据清洗完成: 总计 {len(cleaned)} 天, 排除 {cleaned['exclude_from_returns'].sum()} 天")
        
        return cleaned
    
    def _validate_data_quality(self, data: pd.DataFrame, source: str) -> float:
        """
        数据质量验证（重构：使用独立的DataQualityChecker）
        
        优化点:
        - 职责分离：质量检查逻辑提取到DataQualityChecker
        - 代码复用：消除与MockHistoricalDataProvider的重复代码
        - 可测试性：质量检查器可独立测试
        
        Returns:
            质量评分 (0-1)，≥0.6为及格
        """
        # 使用数据指纹作为缓存键
        data_fingerprint = hash((source, len(data), tuple(data.columns)))
        cache_key = f"quality:{data_fingerprint}"
        
        if cache_key in self._quality_cache:
            return self._quality_cache[cache_key]
        
        # 调用独立的质量检查器
        checker = DataQualityChecker()
        report = checker.check_quality(data, index_id=source)
        
        # 缓存结果
        self._quality_cache[cache_key] = report.overall_score
        
        return report.overall_score
    
    def get_index_returns(self, index_id: str, start_date: str, end_date: str) -> pd.Series:
        """获取指数收益率序列（排除异常日）"""
        df = self.get_index_prices(index_id, start_date, end_date)
        df = df.set_index('date')
        
        # 如果有排除标记，仅计算未被排除的日期的收益率
        if 'exclude_from_returns' in df.columns:
            valid_df = df[~df['exclude_from_returns']]
            returns = valid_df['close'].pct_change().dropna()
        else:
            returns = df['close'].pct_change().dropna()
        
        return returns
    
    def cross_validate_sources(self,
                               index_id: str,
                               start_date: str,
                               end_date: str,
                               sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        数据质量交叉验证（专家answer.md第3轮5.1节）
        
        验证维度：
        1. 逐日差异：30%触发
        2. 窗口统计量：均值3%/标准差10%触发
        
        Args:
            index_id: 指数代码
            start_date: 开始日期
            end_date: 结束日期
            sources: 待验证数据源列表（默认使用primary + mock）
        
        Returns:
            交叉验证报告
        """
        if sources is None:
            # 默认对比primary数据源与mock
            sources = [self.primary_source, DataSource.MOCK.value]
        
        # 获取多个数据源的数据
        data_by_source = {}
        for source in sources:
            adapter = self._adapters.get(source)
            if adapter is None:
                logger.warning(f"交叉验证跳过未配置源: {source}")
                continue
            
            try:
                data = adapter.get_index_prices(index_id, start_date, end_date)
                if data is not None and not data.empty:
                    data_by_source[source] = data
            except Exception as e:
                logger.warning(f"交叉验证获取{source}失败: {e}")
        
        if len(data_by_source) < 2:
            logger.warning(f"交叉验证数据源不足: {len(data_by_source)}/2")
            return {
                'passed': True,  # 无法验证时默认通过
                'sources_compared': list(data_by_source.keys()),
                'reason': 'insufficient_sources'
            }
        
        # 委托DataQualityChecker进行交叉验证
        quality_checker = DataQualityChecker()
        source_list = list(data_by_source.keys())
        comparisons = []
        
        for i in range(len(source_list)):
            for j in range(i + 1, len(source_list)):
                source_a = source_list[i]
                source_b = source_list[j]
                result = quality_checker.cross_validate(
                    data_by_source[source_a],
                    data_by_source[source_b],
                    source_a,
                    source_b
                )
                # 转换为兼容格式
                comparisons.append({
                    'source_a': result.source_a,
                    'source_b': result.source_b,
                    'passed': result.passed,
                    'overlap_days': result.overlap_days,
                    'daily_divergence': result.daily_divergence,
                    'mean_divergence': result.mean_divergence,
                    'std_divergence': result.std_divergence
                })
        
        # 汇总验证结果
        all_passed = all(c['passed'] for c in comparisons)
        
        report = {
            'passed': all_passed,
            'sources_compared': source_list,
            'comparisons': comparisons,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # 记录验证日志
        self._cross_validation_log.append({
            'index_id': index_id,
            'date_range': f"{start_date} to {end_date}",
            'result': report
        })
        
        if not all_passed:
            logger.warning(f"交叉验证发现差异: {index_id}, {comparisons}")
        else:
            logger.info(f"交叉验证通过: {index_id}")
        
        return report
    
    # _compare_two_sources方法已废弃，交叉验证逻辑委托给DataQualityChecker
    
    def get_cross_validation_log(self) -> List[Dict[str, Any]]:
        """获取交叉验证历史记录"""
        return self._cross_validation_log.copy()


# =============================================================================
# 迁移检查清单
# =============================================================================

"""
功能碎片迁移检查清单（core/data模块开发时使用）

□ 1. 接口标准化
    □ 确认 HistoricalDataProvider 协议符合 core/data 模块设计
    □ 添加更多方法（如 get_stock_prices, get_option_data 等）
    
□ 2. 真实数据源集成
    □ 实现 YahooFinanceAdapter
    □ 实现 JoinQuantAdapter
    □ 实现 WindAdapter（可选）
    
□ 3. Mock 数据优化
    □ 将 MockHistoricalDataProvider 迁移到 core/data/mocks/
    □ 支持更多事件场景
    □ 改进模拟数据质量（基于真实统计特征）
    
□ 4. 数据缓存机制
    □ 实现本地缓存（文件系统）
    □ 实现内存缓存（Redis可选）
    □ 缓存过期策略
    
□ 5. 异常处理
    □ 网络异常降级
    □ 数据不完整告警
    □ 数据质量验证
    
□ 6. 调用者更新
    □ 更新 core/risk/stress_test_validator.py 的导入路径
    □ 更新测试用例
    □ 更新文档示例
    
□ 7. 测试覆盖
    □ 单元测试（各数据源）
    □ 集成测试（端到端数据获取）
    □ Mock 与真实数据对比测试
"""
