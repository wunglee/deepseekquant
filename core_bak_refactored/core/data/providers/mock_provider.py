"""
模拟数据提供者（与真实数据规则一致）

职责：
1. 生成模拟的分时数据（fallback方案）
2. 提供实时K线数据计算

用途：当真实数据源不可用时，提供模拟数据用于开发和测试

设计原则（与 AKShareDataProvider 保持一致）：
1. 盘前时段（before_open）：返回空数据，但个股有盘口
2. 交易时段（trading）：返回实时数据 + 盘口 + 成交明细
3. 午休时段（noon_break）：返回上午数据 + 盘口
4. 盘后时段（after_close）：返回全天数据，无盘口
5. 指数：任何时段都没有盘口和成交明细

技术特性：
- 使用 TickRange 替代 batches 参数
- 基于时间范围生成连续的05秒粒度tick数据
- 支持价格连续性（通过 last_price 参数）
- 模拟真实市场的价格波动特性（趋势 + 随机波动 + 突发波动 + 均值回归）
- 提供实时K线数据计算（带缓存优化）
"""

import random
import logging
import pandas as pd
from typing import Optional, Tuple, List
from datetime import datetime, timedelta

from core_bak_refactored.core.data.providers.base_provider import BaseDataProvider
from core_bak_refactored.core.share.market.market_enums import TradingPhase
from core_bak_refactored.core.data.providers.protocols import (
    IntradayData, IntradayTickRecord, OrderBookLevel, TradeDetailRecord, TickRange, HistoricalDataProvider
)

logger = logging.getLogger(__name__)


class MockDataProvider(BaseDataProvider):
    """模拟数据提供者（分时数据 + 实时K线 + 历史K线）"""

    # 股票名称映射
    NAME_MAP = {
        '000001.SH': '上证指数',
        '000300.SH': '沪深300',
        '399001.SZ': '深证成指',
        '399006.SZ': '创业板指',
        '^GSPC': 'S&P 500',
        'AAPL': 'Apple Inc.'
    }

    def __init__(self):
        """初始化生成器"""
        super().__init__()

    def _fetch_from_external_api(self, symbol: str, start_date: str, end_date: str):
        """
        生成模拟历史K线数据（参考AKShareProvider.get_prices实现）

        注意：历史K线都是已完成的交易日数据，不涉及交易时段判断

        Args:
            symbol: 证券代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            PriceData: 包含OHLCV数据的结构化对象
        """
        import pandas as pd
        from core_bak_refactored.core.data.providers.protocols import PriceData
        from datetime import datetime, timedelta

        logger.info(f"📊 生成模拟历史K线: {symbol}, {start_date} ~ {end_date}")

        # 转换日期
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # 生成交易日期序列（跳过周末）
        dates = []
        current_dt = start_dt
        while current_dt <= end_dt:
            # 跳过周末（0=周一, 6=周日）
            if current_dt.weekday() < 5:
                dates.append(current_dt)
            current_dt += timedelta(days=1)

        # 生成基准价格（使用固定种子保证一致性）
        random.seed(symbol)
        base_price = 3000 + random.random() * 300

        # 生成OHLCV数据
        data_rows = []
        current_price = base_price

        for i, date in enumerate(dates):
            date_str = date.strftime('%Y-%m-%d')

            # 使用日期作为种子，保证每天的数据可重复
            random.seed(symbol + date_str)

            # 生成当天的OHLCV
            daily_change = (random.random() - 0.5) * current_price * 0.03  # ±3%波动
            open_price = current_price + (random.random() - 0.5) * current_price * 0.01
            close_price = current_price + daily_change
            high_price = max(open_price, close_price) + random.random() * current_price * 0.01
            low_price = min(open_price, close_price) - random.random() * current_price * 0.01
            volume = int(1000000 + random.random() * 500000)

            data_rows.append({
                'date': date_str,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume
            })

            # 更新下一天的基准价格
            current_price = close_price

        # 转换为DataFrame
        df = pd.DataFrame(data_rows)

        logger.info(f"✅ 生成完成: {len(df)}条K线数据")

        price_data = PriceData.from_dataframe(df, symbol)
        return price_data

    def generate(self, symbol: str, trade_date: str, tick_range: Optional[TickRange] = None,
                 trading_phase: TradingPhase = TradingPhase.TRADING, last_price: Optional[float] = None,
                 is_index: bool = False) -> IntradayData:
        """
        生成模拟分时数据（与真实数据规则一致）
        
        规则（与 AKShareDataProvider.get_intraday_data 一致）：
        1. 盘前时段（before_open）：返回空数据，但个股有盘口
        2. 交易时段（trading）：返回实时数据 + 盘口 + 成交明细
        3. 午休时段（noon_break）：返回上午数据 + 盘口
        4. 盘后时段（after_close）：返回全天数据，无盘口
        5. 指数：任何时段都没有盘口和成交明细
        
        Args:
            symbol: 证券代码
            trade_date: 交易日期 (YYYY-MM-DD)
            tick_range: Tick数据时间范围，如果None则根据交易时段自动计算
            trading_phase: 交易时段（TradingPhase枚举）
            last_price: 上次请求的最终价格，用于保证价格连续性
            is_index: 是否为指数
        
        Returns:
            IntradayData: 分时数据对象
        """
        # 确保使用枚举类型
        phase = TradingPhase.parse(trading_phase) if isinstance(trading_phase, str) else trading_phase
        logger.info(f"📊 生成模拟数据 - 交易时段: {phase}, 日期: {trade_date}, 是否指数: {is_index}")

        # 获取股票名称
        name = self.NAME_MAP.get(symbol, symbol)

        # 生成基准价格（使用固定种子保证一致性）
        random.seed(symbol + trade_date)
        base_price = 3000 + random.random() * 300
        yesterday_close = base_price
        logger.info(f"💰 生成基准价: {base_price:.2f}")

        # 计算起始价格（优先使用 last_price 保证连续性）
        start_price = last_price if last_price is not None else base_price

        # 🔧 根据交易时段生成数据（与真实数据一致）
        ticks = []
        current_price = start_price
        fetch_order_book = False  # 是否获取盘口

        if phase == TradingPhase.BEFORE_OPEN:
            # 🔧 盘前时段：返回空数据，但个股有盘口
            logger.info("🕒 盘前时段，返回空数据（个股有盘口）")
            ticks = []  # 空数据
            fetch_order_book = not is_index  # 个股有盘口，指数没有

        elif phase == TradingPhase.TRADING:
            # 🔧 交易时段：返回实时数据 + 盘口 + 成交明细
            logger.info("📊 交易时段，返回实时数据 + 盘口 + 成交明细")
            # 如果未提供 tick_range，根据交易时段自动创建
            if tick_range is None:
                tick_range = TickRange.from_trading_phase(phase, trade_date)
                logger.info(f"📅 自动创建 TickRange: {tick_range.start_time} ~ {tick_range.end_time}")

            # 生成分时数据
            ticks, current_price = self._build_ticks_from_range(
                symbol=symbol,
                tick_range=tick_range,
                start_price=start_price
            )
            fetch_order_book = not is_index  # 个股有盘口，指数没有

        elif phase == TradingPhase.NOON_BREAK:
            # 🔧 午休时段：返回上午数据 + 盘口
            logger.info("🌞 午休时段，返回上午数据 + 盘口")
            # 构建上午时间范围（09:30-11:30）
            morning_tick_range = TickRange(
                start_time=pd.Timestamp(f"{trade_date} 09:30:00"),
                end_time=pd.Timestamp(f"{trade_date} 11:30:00"),
                period_seconds=5
            )
            # 生成上午的分时数据
            ticks, current_price = self._build_ticks_from_range(
                symbol=symbol,
                tick_range=morning_tick_range,
                start_price=start_price
            )
            fetch_order_book = not is_index  # 个股有盘口，指数没有

        elif phase == TradingPhase.AFTER_CLOSE:
            # 🔧 盘后时段：返回全天数据，无盘口
            logger.info("🌙 盘后时段，返回全天数据（无盘口）")
            # 构建全天时间范围（09:30-15:00）
            full_day_tick_range = TickRange(
                start_time=pd.Timestamp(f"{trade_date} 09:30:00"),
                end_time=pd.Timestamp(f"{trade_date} 15:00:00"),
                period_seconds=5
            )
            # 生成全天的分时数据
            ticks, current_price = self._build_ticks_from_range(
                symbol=symbol,
                tick_range=full_day_tick_range,
                start_price=start_price
            )
            fetch_order_book = False  # 盘后无盘口

        # 计算涨跌
        change = current_price - yesterday_close
        change_percent = (change / yesterday_close * 100) if yesterday_close > 0 else 0

        # 🔧 计算 should_poll 字段（盘前或盘中需要轮询）
        should_poll = phase in [TradingPhase.BEFORE_OPEN, TradingPhase.TRADING]

        # 🔧 根据规则生成盘口和成交明细
        order_book_bids = []
        order_book_asks = []
        trade_records = []
        order_book_message = ''
        trade_records_message = ''

        if is_index:
            # 指数：任何时段都没有盘口和成交明细
            order_book_message = '指数不可交易'
            trade_records_message = '指数无成交明细'
        else:
            # 个股：根据 fetch_order_book 决定是否生成盘口
            if fetch_order_book:
                order_book_bids, order_book_asks = self._generate_order_book(current_price)
                # 只有交易时段才有成交明细
                if phase == TradingPhase.TRADING and len(ticks) > 0:
                    trade_records = self._generate_trade_details(current_price,
                                                                 ticks[-20:] if len(ticks) >= 20 else ticks)
                elif phase == TradingPhase.NOON_BREAK and len(ticks) > 0:
                    # 午休时段可以显示上午最后的成交明细
                    trade_records = self._generate_trade_details(current_price,
                                                                 ticks[-20:] if len(ticks) >= 20 else ticks)

        logger.info(
            f"✅ 生成完成: {len(ticks)}个tick, {len(order_book_bids)}个买盘, {len(order_book_asks)}个卖盘, {len(trade_records)}条成交")

        return IntradayData(
            symbol=symbol,
            name=name,
            current_price=round(current_price, 2),
            yesterday_close=round(yesterday_close, 2),
            change=round(change, 2),
            change_percent=round(change_percent, 2),
            ticks=ticks,
            order_book_bids=order_book_bids,
            order_book_asks=order_book_asks,
            trade_records=trade_records,
            trade_date=trade_date,
            order_book_message=order_book_message,
            trade_records_message=trade_records_message,
            is_index=is_index,
            should_poll=should_poll  # 🔧 设置 should_poll 字段
        )

    def _build_ticks_from_range(self, symbol: str, tick_range: TickRange,
                                start_price: float) -> Tuple[List[IntradayTickRecord], float]:
        """
        根据时间范围构建分时tick数据
        
        Args:
            symbol: 证券代码
            tick_range: 时间范围
            start_price: 起始价格
        
        Returns:
            (ticks, final_price): 分时tick列表和最终价格
        """
        ticks = []
        total_volume = 0
        total_amount = 0
        current_price = start_price

        # 遍历时间范围内的每个tick点
        current_time = tick_range.start_time
        tick_index = 0

        while current_time <= tick_range.end_time:
            # 跳过午休时段 (12:00-13:00)
            if 12 <= current_time.hour < 13:
                current_time += pd.Timedelta(seconds=tick_range.period_seconds)
                continue

            # 使用时间戳作为随机种子，确保每个时间点的波动是固定的（可重复）
            # 但不同时间点之间是随机的
            time_seed = int(current_time.timestamp())
            random.seed(symbol + str(time_seed))

            # 🔧 价格波动：更真实的市场模拟
            # 1. 主趋势：轻微的随机漂移（避免单向趋势）
            trend = (random.random() - 0.5) * 0.3

            # 2. 随机波动：每个tick的随机变化
            random_change = (random.random() - 0.5) * 0.8

            # 3. 突发波动：偶尔的大幅波动
            spike = 0
            if random.random() > 0.92:  # 8%概率
                spike = (random.random() - 0.5) * 1.5

            # 4. 均值回归：价格离起始价太远时，增加回归压力
            price_diff = current_price - start_price
            mean_reversion = -price_diff * 0.01  # 1%的回归力度

            # 综合波动
            price_change = trend + random_change + spike + mean_reversion
            current_price += price_change

            volume = random.randint(500, 2000)

            total_volume += volume
            total_amount += current_price * volume
            avg_price = total_amount / total_volume if total_volume > 0 else current_price

            ticks.append(IntradayTickRecord(
                time=current_time.strftime('%H:%M:%S'),
                price=round(current_price, 2),
                volume=volume,
                avg_price=round(avg_price, 2)
            ))

            current_time += pd.Timedelta(seconds=tick_range.period_seconds)
            tick_index += 1

        return ticks, current_price

    def _generate_order_book(self, current_price: float) -> Tuple[List[OrderBookLevel], List[OrderBookLevel]]:
        """
        生成模拟盘口数据（每次调用都会变化）
        
        Args:
            current_price: 当前价格
        
        Returns:
            (order_book_bids, order_book_asks): 买盘和卖盘列表
        """
        # 🔧 使用当前时间作为种子，让盘口每次都不同
        import time
        random.seed(int(time.time() * 1000))  # 毫秒级别的种子

        order_book_bids = []
        order_book_asks = []

        for i in range(1, 11):
            order_book_bids.append(OrderBookLevel(
                price=round(current_price - i * 0.01, 2),
                volume=random.randint(1000, 10000)
            ))
            order_book_asks.append(OrderBookLevel(
                price=round(current_price + i * 0.01, 2),
                volume=random.randint(1000, 10000)
            ))

        return order_book_bids, order_book_asks

    def _generate_trade_details(self, current_price: float,
                                recent_ticks: List[IntradayTickRecord]) -> List[TradeDetailRecord]:
        """
        生成模拟成交明细（逐笔成交记录，每次调用都会变化）
        
        Args:
            current_price: 当前价格
            recent_ticks: 最近的tick数据，用于生成更真实的成交明细
        
        Returns:
            成交明细列表
        """
        # 🔧 使用当前时间作为种子，让成交明细每次都不同
        import time
        random.seed(int(time.time() * 1000))  # 毫秒级别的种子

        trade_records = []

        # 基于最近的tick数据生成成交明细
        if recent_ticks and len(recent_ticks) > 0:
            for tick in reversed(recent_ticks[-20:]):  # 最多20条
                trade_records.append(TradeDetailRecord(
                    time=tick.time,
                    price=tick.price,
                    volume=random.randint(100, 500),  # 单笔成交量
                    direction=random.choice(['buy', 'sell'])
                ))
        else:
            # 如果没有tick数据，生成随机的成交明细
            now = datetime.now()
            for i in range(20):
                tick_time = now - timedelta(seconds=i * 5)
                trade_records.append(TradeDetailRecord(
                    time=tick_time.strftime('%H:%M:%S'),
                    price=round(current_price + (random.random() - 0.5) * 0.2, 2),
                    volume=random.randint(100, 2000),
                    direction=random.choice(['buy', 'sell'])
                ))

        return trade_records

    def get_realtime_kline(self, symbol: str, trade_date: str, trading_phase: TradingPhase,
                           is_index: bool, cached: Optional[dict] = None) -> dict:
        """
        获取实时K线数据（领域层方法）
        
        注意：MockProvider用于前端开发测试，需要前端显式传入trading_phase进行模拟控制
        
        职责：
        1. 生成模拟分时数据
        2. 根据分时数据计算OHLCV
        3. 使用缓存优化（开盘价、最高价、最低价）
        4. 盘前时段返回集合竞价价格
        5. 🔧 盘中时段：生成动态变化的实时数据（每次调用都不同）
        
        Args:
            symbol: 证券代码
            trade_date: 交易日期 (YYYY-MM-DD)
            trading_phase: 交易时段（由前端控制，用于模拟）
            is_index: 是否为指数
            cached: 缓存的K线数据（包含 open/high/low）
        
        Returns:
            {
                'date': str,
                'open': float,
                'high': float,
                'low': float,
                'close': float,
                'volume': int,
                'trading_phase': str,  # 交易时段：BEFORE_OPEN, TRADING, AFTER_CLOSE等
                'should_poll': bool  # 服务器根据 trading_phase 决定，前端只依赖此字段控制行为
            }
        """
        import time
        from datetime import datetime

        # 🔧 是否应该启动轮询（盘前或盘中）
        should_poll = trading_phase in [TradingPhase.BEFORE_OPEN, TradingPhase.TRADING]

        # 🔧 盘中时段：生成动态变化的实时数据
        if trading_phase == TradingPhase.TRADING:
            # 使用当前时间戳作为随机种子，让每次调用都生成不同的数据
            current_timestamp = int(time.time() * 1000)  # 毫秒级别
            random.seed(symbol + trade_date + str(current_timestamp))

            # 生成基准价格
            base_price = 3000 + random.random() * 300
            yesterday_close = base_price

            # 使用cached中的open价，如果没有则生成
            if cached and 'open' in cached:
                open_price = cached['open']
                # 从开盘价开始波动
                current_price = open_price + (random.random() - 0.5) * open_price * 0.02  # ±2%波动
            else:
                # 首次调用，生成开盘价
                open_price = base_price * (1 + (random.random() - 0.5) * 0.01)
                current_price = open_price + (random.random() - 0.5) * open_price * 0.01

            # 使用cached中的high/low，并更新
            if cached and 'high' in cached and 'low' in cached:
                high_price = max(cached['high'], current_price)
                low_price = min(cached['low'], current_price)
            else:
                high_price = max(open_price, current_price)
                low_price = min(open_price, current_price)

            # 生成成交量（累积增加）
            if cached and 'volume' in cached:
                base_volume = cached['volume']
                volume = base_volume + random.randint(10000, 50000)
            else:
                volume = random.randint(100000, 500000)

            kline_data = {
                'date': trade_date,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(current_price, 2),  # 当前价（动态变化）
                'volume': volume,
                'trading_phase': trading_phase.name,
                'should_poll': should_poll
            }

            logger.info(
                f"📊 盘中实时K线: open={open_price:.2f}, close={current_price:.2f}, high={high_price:.2f}, low={low_price:.2f}, volume={volume}")
            return kline_data
        # 其他时段（盘前/盘后）：使用原有逻辑
        intraday_data = self.generate(
            symbol=symbol,
            trade_date=trade_date,
            tick_range=None,
            trading_phase=trading_phase,
            last_price=None,
            is_index=is_index
        )

        # 🔧 构建K线数据（缓存优化）
        if intraday_data.ticks and len(intraday_data.ticks) > 0:
            prices = [tick.price for tick in intraday_data.ticks]
            volumes = [tick.volume for tick in intraday_data.ticks]

            # 如果有缓存，复用开盘价
            if cached and 'open' in cached:
                open_price = cached['open']
            else:
                open_price = prices[0]

            kline_data = {
                'date': trade_date,
                'open': open_price,
                'high': max(prices),
                'low': min(prices),
                'close': prices[-1],  # 当前价
                'volume': sum(volumes),
                'trading_phase': trading_phase.name,
                'should_poll': should_poll  # 🔧 服务器根据 trading_phase 决定
            }
        else:
            # 🔧 盘前时段：使用集合竞价价格（非空数据）
            # 盘前有价格波动，不是固定的昨收价
            if trading_phase == TradingPhase.BEFORE_OPEN:
                # 模拟集合竞价的价格波动（在昨收价±1%范围内）
                import time
                from datetime import datetime
                random.seed(int(datetime.now().timestamp() * 1000))
                auction_price = intraday_data.yesterday_close * (1 + random.uniform(-0.01, 0.01))

                kline_data = {
                    'date': trade_date,
                    'open': auction_price,
                    'high': auction_price,
                    'low': auction_price,
                    'close': auction_price,  # 集合竞价价格
                    'volume': 0,  # 盘前无成交量
                    'trading_phase': trading_phase.name,
                    'should_poll': should_poll  # 🔧 服务器根据 trading_phase 决定
                }
            else:
                # 其他时段：使用昨收价
                kline_data = {
                    'date': trade_date,
                    'open': intraday_data.yesterday_close,
                    'high': intraday_data.yesterday_close,
                    'low': intraday_data.yesterday_close,
                    'close': intraday_data.yesterday_close,
                    'volume': 0,
                    'trading_phase': trading_phase.name,
                    'should_poll': should_poll  # 🔧 服务器根据 trading_phase 决定
                }

        return kline_data
