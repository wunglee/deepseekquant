"""
市场工具类（领域层共享）

职责：
- 提供市场识别、推断等基础功能
- 支持从 symbol/index_id 推断市场类型
- 可被所有层（应用层、领域层）复用
"""

import logging
from typing import Optional

import pandas as pd

from core_bak_refactored.core.share.market.data_types import OHLCVRecord
from core_bak_refactored.core.share.market.market_enums import MarketCode, TradingPhase

logger = logging.getLogger(__name__)


class MarketUtils:
    """市场工具类
    
    提供市场相关的通用工具方法
    """
    
    @staticmethod
    def infer_market_from_symbol(symbol: str) -> MarketCode:
        """从股票/指数代码推断市场类型
        
        Args:
            symbol: 股票/指数代码（如 '000300.SH', '^GSPC', 'HSI'）
        
        Returns:
            MarketCode: 推断出的市场代码枚举
        
        Examples:
            >>> MarketUtils.infer_market_from_symbol('000300.SH')
            <MarketCode.CN: 'CN'>
            >>> MarketUtils.infer_market_from_symbol('^GSPC')
            <MarketCode.US: 'US'>
            >>> MarketUtils.infer_market_from_symbol('HSI')
            <MarketCode.HK: 'HK'>
            >>> MarketUtils.infer_market_from_symbol('0700.HK')
            <MarketCode.HK: 'HK'>
            >>> MarketUtils.infer_market_from_symbol('N225')
            <MarketCode.JP: 'JP'>
        
        规则：
            - A股市场：.SH（上海）、.SZ（深圳）、.CN
            - 港股市场：.HK、.HKG、HSI（恒生指数）
            - 美股市场：^开头（如 ^GSPC、^DJI、^IXIC）
            - 日本市场：N225（日经指数）
            - 欧洲市场：.EU
            - 新加坡：.SG
            - 默认：MarketCode.CN
        """
        if not symbol:
            return MarketCode.CN
        
        symbol_upper = symbol.upper()
        
        # A股市场（上海/深圳）
        if any(symbol_upper.endswith(suffix) for suffix in ['.SH', '.SZ', '.CN']):
            return MarketCode.CN
        
        # 港股市场
        if any(symbol_upper.endswith(suffix) for suffix in ['.HK', '.HKG']) or symbol_upper == 'HSI':
            return MarketCode.HK
        
        # 美股市场（^GSPC, ^DJI, ^IXIC 等）
        if symbol_upper.startswith('^'):
            return MarketCode.US
        
        # 日本市场
        if symbol_upper == 'N225':
            return MarketCode.JP
        
        # 欧洲市场
        if symbol_upper.endswith('.EU'):
            return MarketCode.EU
        
        # 新加坡市场
        if symbol_upper.endswith('.SG'):
            return MarketCode.SG
        
        # 美股市场（.US 后缀）
        if symbol_upper.endswith('.US'):
            return MarketCode.US
        
        # 默认为 A股市场
        return MarketCode.CN
    
    @staticmethod
    def infer_market_from_metadata(metadata: dict) -> Optional[MarketCode]:
        """从元数据中提取市场类型
        
        Args:
            metadata: 元数据字典（可能包含 'market_type' 或 'market' 字段）
        
        Returns:
            MarketCode: 提取出的市场代码枚举，如果无法提取则返回 None
        
        Examples:
            >>> MarketUtils.infer_market_from_metadata({'market_type': 'CN'})
            <MarketCode.CN: 'CN'>
            >>> MarketUtils.infer_market_from_metadata({'market': MarketCode.US})
            <MarketCode.US: 'US'>
            >>> MarketUtils.infer_market_from_metadata({'other': 'data'})
            None
        """
        if not metadata:
            return None
        
        # 尝试从 market_type 字段提取
        market_type = metadata.get('market_type')
        if market_type:
            if isinstance(market_type, MarketCode):
                return market_type
            if isinstance(market_type, str) and MarketCode.is_valid(market_type.upper()):
                return MarketCode(market_type.upper())
        
        # 尝试从 market 字段提取
        market = metadata.get('market')
        if market:
            if isinstance(market, MarketCode):
                return market
            if isinstance(market, str) and MarketCode.is_valid(market.upper()):
                return MarketCode(market.upper())
        
        return None
    
    @staticmethod
    def detect_market_with_fallback(symbol: str = None, metadata: dict = None) -> MarketCode:
        """综合检测市场类型（优先元数据，其次 symbol 启发式）
        
        Args:
            symbol: 股票/指数代码
            metadata: 元数据字典
        
        Returns:
            MarketCode: 检测出的市场代码枚举
        
        Examples:
            >>> # 优先使用元数据
            >>> MarketUtils.detect_market_with_fallback(
            ...     symbol='000300.SH',
            ...     metadata={'market_type': 'US'}
            ... )
            <MarketCode.US: 'US'>
            
            >>> # 元数据缺失时使用 symbol
            >>> MarketUtils.detect_market_with_fallback(symbol='000300.SH')
            <MarketCode.CN: 'CN'>
            
            >>> # 都缺失时返回默认值
            >>> MarketUtils.detect_market_with_fallback()
            <MarketCode.CN: 'CN'>
        """
        # 1. 优先使用元数据
        if metadata:
            market = MarketUtils.infer_market_from_metadata(metadata)
            if market:
                return market
        
        # 2. 其次使用 symbol 启发式推断
        if symbol:
            return MarketUtils.infer_market_from_symbol(symbol)
        
        # 3. 默认为 A股市场
        return MarketCode.CN
    
    @staticmethod
    def is_trading_day_aware(market: MarketCode) -> bool:
        """判断市场是否需要交易日判断
        
        Args:
            market: 市场代码
        
        Returns:
            bool: 是否需要交易日判断
        
        Note:
            某些市场（如虚拟货币）是 7x24 交易，不需要交易日判断
        """
        # 所有实体股票市场都需要交易日判断
        return market in [
            MarketCode.CN,
            MarketCode.US,
            MarketCode.HK,
            MarketCode.JP,
            MarketCode.EU,
            MarketCode.SG
        ]
    
    @staticmethod
    def standardize_format(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
        """
        标准化指数数据格式（处理不同API的列名差异）
        
        Args:
            df: 原始DataFrame
            symbol: 证券代码（用于处理MultiIndex列名）
        
        Returns:
            标准化的DataFrame with columns: ['date', 'open', 'high', 'low', 'close', 'volume']
            
        数据标准：
        - date: pd.Timestamp 类型，交易日期时间
        - open: float，开盘价
        - high: float，最高价
        - low: float，最低价
        - close: float，收盘价
        - volume: float，成交量
        """
        # 🔧 处理不同 API 返回的列名差异
        # A股: date, open, close, high, low, volume, amount
        # 港股/美股: 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额

        # 处理MultiIndex列结构（Yahoo Finance特有）
        if isinstance(df.columns, pd.MultiIndex):
            # 对于MultiIndex，我们需要提取正确的列
            # 通常格式是 [('Close', 'AAPL'), ('High', 'AAPL'), ...]
            close_col = ('Close', symbol) if ('Close', symbol) in df.columns else 'Close'
            high_col = ('High', symbol) if ('High', symbol) in df.columns else 'High'
            low_col = ('Low', symbol) if ('Low', symbol) in df.columns else 'Low'
            open_col = ('Open', symbol) if ('Open', symbol) in df.columns else 'Open'
            volume_col = ('Volume', symbol) if ('Volume', symbol) in df.columns else 'Volume'
            date_col = df.index.name if df.index.name else 'Date'
        else:
            # 普通列名
            date_col = None
            close_col = None
            high_col = None
            low_col = None
            open_col = None
            volume_col = None

        # 尝试识别日期列
        if not date_col:
            for col in ['日期', 'date', 'Date', 'DATE']:
                if col in df.columns:
                    date_col = col
                    break

        # 尝试识别开盘价列
        if not open_col:
            for col in ['开盘', 'open', 'Open', 'OPEN']:
                if col in df.columns:
                    open_col = col
                    break

        # 尝试识别最高价列
        if not high_col:
            for col in ['最高', 'high', 'High', 'HIGH']:
                if col in df.columns:
                    high_col = col
                    break

        # 尝试识别最低价列
        if not low_col:
            for col in ['最低', 'low', 'Low', 'LOW']:
                if col in df.columns:
                    low_col = col
                    break

        # 尝试识别收盘价列
        if not close_col:
            for col in ['收盘', 'close', 'Close', 'CLOSE', '收盘价']:
                if col in df.columns:
                    close_col = col
                    break

        # 尝试识别成交量列
        if not volume_col:
            for col in ['成交量', 'volume', 'Volume', 'VOLUME']:
                if col in df.columns:
                    volume_col = col
                    break

        if not date_col or not close_col:
            raise ValueError(f"Cannot find date or close columns in DataFrame. Columns: {df.columns.tolist()}")

        # 如果缺少OHLC数据，使用收盘价填充
        if isinstance(df.index, pd.DatetimeIndex):
            dates = df.index
        else:
            dates = pd.to_datetime(df[date_col])
            
        standardized = pd.DataFrame({
            'date': dates,
            'open': df[open_col].astype(float) if open_col in df.columns else df[close_col].astype(float),
            'high': df[high_col].astype(float) if high_col in df.columns else df[close_col].astype(float),
            'low': df[low_col].astype(float) if low_col in df.columns else df[close_col].astype(float),
            'close': df[close_col].astype(float),
            'volume': df[volume_col].astype(float) if volume_col in df.columns else 0.0
        })

        # 按日期排序
        standardized = standardized.sort_values('date').reset_index(drop=True)

        # 数据清洗：移除NaN和异常值
        original_len = len(standardized)
        standardized = standardized.dropna(subset=['close'])
        if len(standardized) < original_len:
            logger.warning(f"Removed {original_len - len(standardized)} rows with missing close prices")

        return standardized
    
    @staticmethod
    def standardize_format_to_price_data(df: pd.DataFrame, symbol: str = "") -> 'PriceData':
        """
        标准化数据格式并转换为PriceData对象
        
        Args:
            df: 原始DataFrame
            symbol: 证券代码
        
        Returns:
            PriceData: 标准化后的数据对象
        """
        # 局部导入避免循环依赖
        from core_bak_refactored.core.data.providers.protocols import PriceData
        
        if df is None or df.empty:
            # 空数据返回空的PriceData
            return PriceData(
                symbol=symbol, 
                records=[], 
                start_date=pd.Timestamp.now(),
                end_date=pd.Timestamp.now(),
                count=0
            )
        
        # 标准化格式
        standardized_df = MarketUtils.standardize_format(df, symbol)
        
        # 转换为OHLCVRecord列表
        records = []
        for _, row in standardized_df.iterrows():
            try:
                record = OHLCVRecord(
                    date=pd.to_datetime(row['date']),
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row['volume'])
                )
                records.append(record)
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid row for {symbol}: {e}")
                continue
        
        # 计算start_date和end_date
        start_date = records[0].date if records else pd.Timestamp.now()
        end_date = records[-1].date if records else pd.Timestamp.now()
        
        return PriceData(
            symbol=symbol, 
            records=records,
            start_date=start_date,
            end_date=end_date,
            count=len(records)
        )
    
    @staticmethod
    def is_index(symbol: str) -> bool:
        """
        判断是否是指数（简化版本，用于模拟测试）
        
        Args:
            symbol: 证券代码
        
        Returns:
            bool: True表示是指数（不可交易），False表示是个股（可交易）
        
        注意：
        - 这是临时的简化判断逻辑，仅用于模拟数据测试
        - 生产环境应该从数据源API获取证券类型信息
        - TODO: 未来接入真实数据源时，应调用数据源的证券信息接口
        
        模拟测试规则（方便测试两种情况）：
        - 指数（不可交易）：000开头（如000001.SH, 000300.SH）或399开头（如399001.SZ）
        - 个股（可交易）：其他代码（如600000.SH, 000001.SZ, 300001.SZ）
        """
        if not symbol:
            return False

        # 提取代码部分（去掉市场后缀）
        code = symbol.split('.')[0]
        market = symbol.split('.')[1] if '.' in symbol else ''

        # 🔧 模拟测试：简化判断，仅用于测试两种情况
        # 指数判断规则：
        # - 上海指数：000xxx.SH (如 000001.SH上证指数, 000300.SH沪深300)
        # - 深圳指数：399xxx.SZ (如 399001.SZ深证成指, 399006.SZ创业板指)
        # 注意：000001.SZ 是平安银行（个股），不是指数
        if len(code) == 6:
            # 上海市场：000开头是指数
            if market == 'SH' and code.startswith('000'):
                return True  # 上海指数
            # 深圳市场：399开头是指数，000开头是个股
            if market == 'SZ' and code.startswith('399'):
                return True  # 深圳指数

        # 其他都视为个股，可交易
        return False
    
    @staticmethod
    def get_last_trade_date(market: MarketCode, trade_date: pd.Timestamp, current_time: pd.Timestamp = None) -> pd.Timestamp:
        """
        获取最后一个交易日（可能是今天，也可能是前一交易日）

        逻辑：
        1. 如果当前时间在盘中或盘后（当天已经开盘过），返回当天
        2. 如果是盘前或周末/节假日，返回前一个交易日

        Args:
            market: 市场代码枚举
            trade_date: 当前请求的日期 (YYYY-MM-DD)
            current_time: 当前时间（用于测试，默认使用系统时间）

        Returns:
            最后一个交易日 (YYYY-MM-DD)

        示例：
        - 周一 10:00 -> 返回当天（当天已开盘）
        - 周一 08:00 -> 返回上周五（当天未开盘）
        - 周六 20:00 -> 返回周五（周末）
        - 周一 16:00 -> 返回当天（当天已收盘）

        TODO: 集成交易日历API处理节假日
        """
        from core_bak_refactored.core.share.config_manager import ConfigManager

        request_date = trade_date
        
        # 使用传入的时间或当前系统时间
        now = current_time if current_time is not None else pd.Timestamp.now()
        now_time = now.time()
        request_weekday = request_date.weekday()

        # 判断请求日期是否为周末
        if request_weekday >= 5:  # 5=周六, 6=周日
            # 周末 -> 返回上周五
            days_to_subtract = request_weekday - 4  # 周六减2天，周日减3天
            last_trade_date = request_date - pd.Timedelta(days=days_to_subtract)
            return last_trade_date

        # 工作日：判断当前时间是否已经开盘
        # 从配置读取市场开盘时间
        config_manager = ConfigManager()
        trading_hours = config_manager.get_trading_hours(market.value)
        
        # 默认A股开盘时间
        market_open_str = trading_hours.get('open', '09:30') if trading_hours else '09:30'
        open_parts = market_open_str.split(':')
        from datetime import time as dt_time
        market_open_time = dt_time(int(open_parts[0]), int(open_parts[1]))

        if now_time >= market_open_time:
            # 当前时间已经开盘或已收盘 -> 返回请求日期
            return trade_date
        else:
            # 当前时间还未开盘（盘前时段） -> 返回前一个交易日
            if request_weekday == 0:  # 周一盘前 -> 返回上周五
                last_trade_date = request_date - pd.Timedelta(days=3)
            else:  # 其他工作日盘前 -> 返回昨天
                last_trade_date = request_date - pd.Timedelta(days=1)
            return last_trade_date
    
    @staticmethod
    def determine_trading_phase(market: MarketCode, now: pd.Timestamp) -> TradingPhase:
        """
        根据市场和当前时间判断交易状态
        
        Args:
            market: 市场代码枚举 (MarketCode.CN, MarketCode.HK, MarketCode.US, etc.)
            now: 当前时间 (datetime)
        
        Returns:
            TradingPhase:
            - 'before_open': 集合竞价时段（开盘前30分钟，如A股 9:00-9:30）
            - 'trading': 交易中（包括午间休市）
            - 'after_close': 收盘后到次日集合竞价前（包括周末、晚上、次日早上）
        """
        from datetime import time as dt_time
        from core_bak_refactored.core.share.market.market_enums import TradingPhase
        from core_bak_refactored.core.share.config_manager import ConfigManager
        import logging
        
        logger = logging.getLogger('DeepSeekQuant.MarketUtils')

        # 🔧 确保market是MarketCode枚举
        if not isinstance(market, MarketCode):
            logger.warning(f"market应为MarketCode枚举，当前类型: {type(market)}，尝试解析")
            market = MarketCode.parse(market)

        current_time = now.time()
        weekday = now.weekday()

        # 🔧 从ConfigManager读取交易时段
        config_manager = ConfigManager()
        trading_hours = config_manager.get_trading_hours(market.value)
        
        # 默认配置（A股）
        if not trading_hours:
            trading_hours = {
                'open': '09:30',
                'close': '15:00',
                'lunch_start': '11:30',
                'lunch_end': '13:00',
                'has_lunch_break': True
            }
        
        # 解析时间
        def parse_time(time_str: str) -> dt_time:
            parts = time_str.split(':')
            return dt_time(int(parts[0]), int(parts[1]))
        
        open_time = parse_time(trading_hours['open'])
        close_time = parse_time(trading_hours['close'])
        
        # 获取午休时间（如果有）
        has_lunch_break = trading_hours.get('has_lunch_break', False)
        if has_lunch_break:
            noon_break_start = parse_time(trading_hours.get('lunch_start', '11:30'))
            noon_break_end = parse_time(trading_hours.get('lunch_end', '13:00'))
        else:
            # 无午休市场（如美股），设置为None
            noon_break_start = None
            noon_break_end = None

        # 🔧 关键修复：区分集合竞价时段
        # 集合竞价时段 = 开盘前30分钟（如A股 9:00-9:30）
        # 正确计算开盘前30分钟
        if open_time.minute >= 30:
            call_auction_start = dt_time(open_time.hour, open_time.minute - 30)
        else:
            call_auction_start = dt_time(open_time.hour - 1, 60 + open_time.minute - 30)

        # 周末不交易
        if weekday >= 5:
            return TradingPhase.AFTER_CLOSE  # 周末视为收盘后
        
        if call_auction_start <= current_time < open_time:
            # 集合竞价时段：清空分时图，轮询盘口
            return TradingPhase.BEFORE_OPEN
        
        # 有午休的市场（如A股、港股、日股）
        if has_lunch_break and noon_break_start and noon_break_end:
            if open_time <= current_time < noon_break_start:
                # 上午交易时段（09:30-11:30）
                return TradingPhase.TRADING
            elif noon_break_start <= current_time < noon_break_end:
                # 午盘休市时段（11:30-13:00）：返回上午数据 + 盘口
                return TradingPhase.NOON_BREAK
            elif noon_break_end <= current_time <= close_time:
                # 下午交易时段（13:00-15:00）
                return TradingPhase.TRADING
            else:
                # 收盘后到次日集合竞价前
                return TradingPhase.AFTER_CLOSE
        else:
            # 无午休的市场（如美股）
            if open_time <= current_time <= close_time:
                # 整个交易日都是交易时段
                return TradingPhase.TRADING
            else:
                # 收盘后到次日集合竞价前
                return TradingPhase.AFTER_CLOSE