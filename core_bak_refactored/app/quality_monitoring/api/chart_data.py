"""图表数据组装模块

[应用层 - API组件] 图表数据导出功能
状态: ✅ 新增 - 合并数据流API
创建时间: 2025-12-06

职责：
- 组装K线+技术指标+事件的完整图表数据
- 作为应用层胶水层，连接领域层指标服务和数据提供者
- 仅包含数据格式转换和组装逻辑，不包含业务计算

架构原则：
- 依赖领域层的 IndicatorService（技术指标计算）
- 依赖数据提供者接口（数据获取）
- 符合单一职责原则（SRP）：只负责数据组装
- 符合开闭原则（OCP）：扩展新指标无需修改现有代码
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from core_bak_refactored.core.data.providers.protocols import PriceData, OHLCVRecord

logger = logging.getLogger('DeepSeekQuant.App.API.ChartData')


class ChartDataAssembler:
    """图表数据组装器
    
    职责：
    1. 获取K线数据（使用领域层标准 PriceData 类型）
    2. 调用领域层服务计算技术指标
    3. 检测市场事件
    4. 组装完整的图表数据结构
    
    依赖倒置原则（DIP）：
    - 依赖抽象的数据提供者接口，不依赖具体实现
    - 依赖抽象的指标服务接口，不依赖具体实现
    - 使用领域层标准类型 PriceData，避免非强类型 DataFrame
    """
    
    def __init__(self, data_provider: Any, indicator_service: Any) -> None:
        """初始化图表数据组装器
        
        Args:
            data_provider: 数据提供者（实现 get_index_prices 接口）
            indicator_service: 技术指标服务（IndicatorService 实例）
        """
        self._data_provider = data_provider
        self._indicator_service = indicator_service
    
    def assemble_chart_data(self,
                           index_id: str,
                           period: str = 'daily',
                           count: int = 120,
                           before: Optional[str] = None,
                           indicators: Optional[str] = 'all') -> Dict[str, Any]:
        """组装完整的图表数据（全程使用强类型 PriceData）
        
        Args:
            index_id: 股票/指数代码
            period: 周期（daily/weekly/monthly）
            count: 数据条数
            before: 获取此日期之前的数据（YYYY-MM-DD）
            indicators: 需要的指标（逗号分隔或 'all'）
        
        Returns:
            {
                'kline': [...],  # K线数据（包含MA）
                'indicators': {...},  # 技术指标数据
                'events': [...]  # 事件数据
            }
        
        Raises:
            ValueError: 参数无效
            RuntimeError: 数据获取或计算失败
        """
        try:
            logger.info(f"开始组装图表数据: index_id={index_id}, period={period}, count={count}, before={before}")
            
            # 1. 获取K线数据（🔧 额外获取30条用于指标预热，返回 PriceData）
            logger.info("步骤1: 获取K线数据...")
            warmup_count = 30  # 预热数据条数（足够MACD/RSI/KDJ计算）
            price_data_full = self._fetch_kline_data(index_id, period, count + warmup_count, before)
            logger.info(f"K线数据获取成功，共 {price_data_full.count} 条（包含{warmup_count}条预热数据）")
            
            # 2. 计算技术指标（使用完整 PriceData）
            logger.info(f"步骤2: 计算技术指标 ({indicators})...")
            kline_with_ma_full, indicators_data_full = self._calculate_indicators(price_data_full, indicators)
            logger.info(f"技术指标计算成功: {list(indicators_data_full.keys())}")
            
            # 🔧 关键优化：裁剪掉预热数据，只返回请求的条数
            kline_with_ma = kline_with_ma_full[-count:] if len(kline_with_ma_full) > count else kline_with_ma_full
            indicators_data = {
                key: value[-count:] if len(value) > count else value
                for key, value in indicators_data_full.items()
            }
            logger.info(f"裁剪后的数据: kline={len(kline_with_ma)} 条, 指标每个约{len(next(iter(indicators_data.values()), []))} 条")
            
            # 3. 检测市场事件（只在请求的范围内，使用裁剪后的 PriceData）
            logger.info("步骤3: 检测市场事件...")
            price_data_requested = self._slice_price_data(price_data_full, -count)
            events = self._detect_events(price_data_requested)
            logger.info(f"事件检测成功，共 {len(events)} 个事件")
            
            # 4. 组装返回数据
            result = {
                'kline': kline_with_ma,
                'indicators': indicators_data,
                'events': events
            }
            logger.info(f"图表数据组装完成: kline={len(kline_with_ma)} 条, indicators={len(indicators_data)} 个, events={len(events)} 个")
            return result
        
        except Exception as e:
            logger.error(f"组装图表数据失败: {e}", exc_info=True)
            logger.error(f"  - index_id: {index_id}")
            logger.error(f"  - period: {period}")
            logger.error(f"  - count: {count}")
            logger.error(f"  - before: {before}")
            logger.error(f"  - indicators: {indicators}")
            logger.error(f"  - 错误类型: {type(e).__name__}")
            logger.error(f"  - 错误详情: {str(e)}")
            raise RuntimeError(f"图表数据组装失败: {str(e)}") from e
    
    def _fetch_kline_data(self,
                         index_id: str,
                         period: str,
                         count: int,
                         before: Optional[str]) -> PriceData:
        """获取K线数据（使用领域层标准 PriceData 类型）
        
        Args:
            index_id: 股票/指数代码
            period: 周期
            count: 数据条数
            before: 截止日期
        
        Returns:
            PriceData: 强类型价格数据对象
        """
        from datetime import datetime, timedelta
        
        # 计算日期范围
        multiplier = {'daily': 1, 'weekly': 7, 'monthly': 30}.get(period, 1)
        days_needed = count * multiplier * 2  # 预留冗余
        
        if before:
            end_date = datetime.strptime(before, '%Y-%m-%d')
        else:
            end_date = datetime.now()
        
        start_date = end_date - timedelta(days=days_needed)
        
        # 调用数据提供者（返回领域层标准 PriceData 类型）
        price_data = self._data_provider.get_index_prices(
            index_id,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        # 验证数据
        if price_data is None or price_data.count == 0:
            raise ValueError(f"无数据：{index_id}")
        
        logger.info(f"获取到 {price_data.count} 条数据，symbol={price_data.symbol}, 时间范围: {price_data.start_date} to {price_data.end_date}")
        
        # 周期转换（如需要）
        if period != 'daily':
            price_data = self._convert_period(price_data, period, count)
        
        return price_data
    
    def _slice_price_data(self, price_data: PriceData, slice_count: int) -> PriceData:
        """裁剪 PriceData（保持强类型）
        
        Args:
            price_data: 原始价格数据
            slice_count: 裁剪条数（负数表示从尾部取）
        
        Returns:
            裁剪后的 PriceData 对象
        """
        sliced_records = price_data.records[slice_count:] if slice_count < 0 else price_data.records[:slice_count]
        
        return PriceData(
            records=sliced_records,
            symbol=price_data.symbol,
            start_date=sliced_records[0].date if sliced_records else price_data.start_date,
            end_date=sliced_records[-1].date if sliced_records else price_data.end_date,
            count=len(sliced_records)
        )
    
    def _convert_period(self,
                       price_data: PriceData,
                       period: str,
                       count: int) -> PriceData:
        """周期转换（日线→周线/月线）（使用强类型 PriceData）
        
        Args:
            price_data: 日线数据（PriceData对象）
            period: 目标周期
            count: 目标条数
        
        Returns:
            转换后的 PriceData 对象
        """
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
            df_copy = df_copy.resample('M').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        
        df_copy = df_copy.reset_index()
        df_copy = df_copy.tail(count)
        
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
    
    def _calculate_indicators(self,
                             price_data: PriceData,
                             indicators: Optional[str]) -> tuple:
        """计算技术指标（使用强类型 PriceData）
        
        Args:
            price_data: K线数据（PriceData对象）
            indicators: 需要的指标（'all' 或逗号分隔）
        
        Returns:
            (kline_with_ma, indicators_data)
            - kline_with_ma: 包含MA的K线数据列表
            - indicators_data: 各指标数据字典
        """
        # 解析需要的指标
        if indicators == 'all':
            requested_indicators = ['vol', 'macd', 'rsi', 'kdj', 'obv']
        else:
            requested_indicators = [ind.strip().lower() for ind in (indicators or '').split(',') if ind.strip()]
        
        # 准备返回数据
        kline_data = []
        indicators_data = {}
        
        # 提取价格序列用于MA计算
        close_prices = pd.Series([r.close for r in price_data.records])
        ma5 = close_prices.rolling(window=5).mean()
        ma10 = close_prices.rolling(window=10).mean()
        ma20 = close_prices.rolling(window=20).mean()
        
        # 组装K线数据（包含MA）
        for i, record in enumerate(price_data.records):
            kline_record = {
                'date': record.date.strftime('%Y-%m-%d') if hasattr(record.date, 'strftime') else str(record.date),
                'open': self._safe_float(record.open),
                'high': self._safe_float(record.high),
                'low': self._safe_float(record.low),
                'close': self._safe_float(record.close),
                'volume': self._safe_float(record.volume),
                'ma5': self._safe_float(ma5.iloc[i]),
                'ma10': self._safe_float(ma10.iloc[i]),
                'ma20': self._safe_float(ma20.iloc[i])
            }
            kline_data.append(kline_record)
        
        # 计算技术指标（调用领域层服务）
        if 'vol' in requested_indicators:
            indicators_data['vol'] = self._calculate_vol(price_data)
        
        if 'macd' in requested_indicators:
            indicators_data['macd'] = self._calculate_macd(price_data)
        
        if 'rsi' in requested_indicators:
            indicators_data['rsi'] = self._calculate_rsi(price_data)
        
        if 'kdj' in requested_indicators:
            indicators_data['kdj'] = self._calculate_kdj(price_data)
        
        if 'obv' in requested_indicators:
            indicators_data['obv'] = self._calculate_obv(price_data)
        
        return kline_data, indicators_data
    
    def _calculate_vol(self, price_data: PriceData) -> List[Dict]:
        """计算成交量指标（使用强类型 PriceData）"""
        return [
            {
                'date': record.date.strftime('%Y-%m-%d') if hasattr(record.date, 'strftime') else str(record.date),
                'value': self._safe_float(record.volume)
            }
            for record in price_data.records
        ]
    
    def _calculate_macd(self, price_data: PriceData) -> List[Dict]:
        """计算MACD指标（调用领域层服务，使用强类型 PriceData）
        
        注意：
        - 使用Wilder EMA平滑法（pandas标准实现）
        - 前26个周期的值可能为NaN（需要足够数据才能计算）
        - 前端会自动处理null值，不显示对应点
        """
        try:
            logger.info(f"🔧 开始计算MACD，数据行数: {price_data.count}")
            
            # 提取价格序列（强类型 -> Series）
            close_prices = pd.Series([r.close for r in price_data.records])
            logger.info(f"   - close_prices 类型: {type(close_prices)}")
            logger.info(f"   - close_prices 前5个值: {close_prices.head().tolist()}")
            
            # 调用领域层服务计算MACD
            macd, signal, hist = self._indicator_service.calculate_macd(close_prices)
            
            logger.info(f"✅ MACD计算成功，返回类型: macd={type(macd)}, signal={type(signal)}, hist={type(hist)}")
            
            results = []
            for i, record in enumerate(price_data.records):
                # ⚠️ 关键：保留NaN转为null，让前端正确处理数据连续性
                results.append({
                    'date': record.date.strftime('%Y-%m-%d') if hasattr(record.date, 'strftime') else str(record.date),
                    'macd': self._safe_float(macd.iloc[i] if hasattr(macd, 'iloc') else macd[i]),
                    'signal': self._safe_float(signal.iloc[i] if hasattr(signal, 'iloc') else signal[i]),
                    'histogram': self._safe_float(hist.iloc[i] if hasattr(hist, 'iloc') else hist[i])
                })
            
            logger.info(f"MACD计算成功：{len(results)}条数据，前{self._count_leading_nulls(results, 'macd')}条为null（正常）")
            logger.debug(f"MACD前3条数据: {results[:3]}")
            return results
        except Exception as e:
            logger.error(f"⚠️ MACD计算失败，返回空数据: {e}", exc_info=True)
            logger.error(f"   - price_data.count: {price_data.count}")
            logger.error(f"   - price_data.symbol: {price_data.symbol}")
            return []
    
    def _calculate_rsi(self, price_data: PriceData) -> List[Dict]:
        """计算RSI指标（调用领域层服务，使用强类型 PriceData）
        
        注意：
        - 使用Wilder平滑法（alpha=1/14）
        - 前14个周期的值可能为NaN
        - 返回值范围0-100
        """
        try:
            # 提取价格序列
            close_prices = pd.Series([r.close for r in price_data.records])
            rsi = self._indicator_service.calculate_rsi(close_prices)
            
            results = [
                {
                    'date': record.date.strftime('%Y-%m-%d') if hasattr(record.date, 'strftime') else str(record.date),
                    'value': self._safe_float(rsi.iloc[i] if hasattr(rsi, 'iloc') else rsi[i])
                }
                for i, record in enumerate(price_data.records)
            ]
            
            logger.debug(f"RSI计算完成：{len(results)}条数据，前{self._count_leading_nulls(results, 'value')}条为null（正常）")
            return results
        except Exception as e:
            logger.warning(f"RSI计算失败，返回空数据: {e}")
            return []
    
    def _calculate_kdj(self, price_data: PriceData) -> List[Dict]:
        """计算KDJ指标（调用领域层服务，使用强类型 PriceData）
        
        注意：
        - K线：随机指标（周期内价格相对位置）
        - D线：K线的3周期SMA平滑
        - J线：3*K - 2*D（灵敏指标）
        - 前9个周期的值可能为NaN
        """
        try:
            # 提取价格序列
            high_prices = pd.Series([r.high for r in price_data.records])
            low_prices = pd.Series([r.low for r in price_data.records])
            close_prices = pd.Series([r.close for r in price_data.records])
            
            k, d = self._indicator_service.calculate_kdj(
                high_prices,
                low_prices,
                close_prices
            )
            
            # 计算 J 值：J = 3*K - 2*D
            j = 3 * k - 2 * d
            
            results = [
                {
                    'date': record.date.strftime('%Y-%m-%d') if hasattr(record.date, 'strftime') else str(record.date),
                    'k': self._safe_float(k.iloc[i] if hasattr(k, 'iloc') else k[i]),
                    'd': self._safe_float(d.iloc[i] if hasattr(d, 'iloc') else d[i]),
                    'j': self._safe_float(j.iloc[i] if hasattr(j, 'iloc') else j[i])
                }
                for i, record in enumerate(price_data.records)
            ]
            
            logger.debug(f"KDJ计算完成：{len(results)}条数据，前{self._count_leading_nulls(results, 'k')}条为null（正常）")
            return results
        except Exception as e:
            logger.warning(f"KDJ计算失败，返回空数据: {e}")
            return []
    
    def _calculate_obv(self, price_data: PriceData) -> List[Dict]:
        """计算OBV指标（调用领域层服务，使用强类型 PriceData）
        
        注意：
        - 能量潮指标，累计方向性成交量
        - 价格上涨日累加成交量，下跌日累减
        - 第一个值为0（起始点）
        """
        try:
            # 提取价格和成交量序列
            close_prices = pd.Series([r.close for r in price_data.records])
            volumes = pd.Series([r.volume for r in price_data.records])
            
            obv = self._indicator_service.calculate_obv(close_prices, volumes)
            
            results = [
                {
                    'date': record.date.strftime('%Y-%m-%d') if hasattr(record.date, 'strftime') else str(record.date),
                    'value': self._safe_float(obv.iloc[i] if hasattr(obv, 'iloc') else obv[i])
                }
                for i, record in enumerate(price_data.records)
            ]
            
            logger.debug(f"OBV计算完成：{len(results)}条数据")
            return results
        except Exception as e:
            logger.warning(f"OBV计算失败，返回空数据: {e}")
            return []
    
    def _detect_events(self, df: pd.DataFrame) -> List[Dict]:
        """检测市场事件（暴涨暴跌）
        
        Args:
            df: K线数据
        
        Returns:
            事件列表
        """
        events = []
        
        try:
            df_copy = df.copy()
            df_copy['pct_change'] = df_copy['close'].pct_change() * 100
            
            for i in range(len(df_copy)):
                chg = df_copy.iloc[i]['pct_change']
                if pd.isna(chg):
                    continue
                
                chg = float(chg)
                date = df_copy.iloc[i]['date']
                close = float(df_copy.iloc[i]['close'])
                
                if chg <= -5.0:
                    severity = 'critical' if chg < -7 else 'high'
                    events.append({
                        'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                        'type': 'market_crash',
                        'title': f'暴跌 {abs(chg):.2f}%',
                        'decline_pct': chg,
                        'price': close,
                        'impact': 'negative',
                        'severity': severity
                    })
                elif chg >= 5.0:
                    events.append({
                        'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                        'type': 'rally',
                        'title': f'暴涨 {chg:.2f}%',
                        'rise_pct': chg,
                        'price': close,
                        'impact': 'positive',
                        'severity': 'high'
                    })
        
        except Exception as e:
            logger.warning(f"事件检测失败: {e}")
        
        return events
    
    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """安全转换为float，处理NaN"""
        if pd.isna(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def _count_leading_nulls(data: List[Dict], key: str) -> int:
        """统计列表开头有多少个null值"""
        count = 0
        for item in data:
            if item.get(key) is None:
                count += 1
            else:
                break
        return count
