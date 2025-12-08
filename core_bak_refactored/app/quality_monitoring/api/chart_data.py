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

logger = logging.getLogger('DeepSeekQuant.App.API.ChartData')


class ChartDataAssembler:
    """图表数据组装器
    
    职责：
    1. 获取K线数据
    2. 调用领域层服务计算技术指标
    3. 检测市场事件
    4. 组装完整的图表数据结构
    
    依赖倒置原则（DIP）：
    - 依赖抽象的数据提供者接口，不依赖具体实现
    - 依赖抽象的指标服务接口，不依赖具体实现
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
        """组装完整的图表数据
        
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
            
            # 1. 获取K线数据（🔧 额外获取30条用于指标预热）
            logger.info("步骤1: 获取K线数据...")
            warmup_count = 30  # 预热数据条数（足够MACD/RSI/KDJ计算）
            df_full = self._fetch_kline_data(index_id, period, count + warmup_count, before)
            logger.info(f"K线数据获取成功，共 {len(df_full)} 条（包含{warmup_count}条预热数据）")
            
            # 2. 计算技术指标（使用完整数据）
            logger.info(f"步骤2: 计算技术指标 ({indicators})...")
            kline_with_ma_full, indicators_data_full = self._calculate_indicators(df_full, indicators)
            logger.info(f"技术指标计算成功: {list(indicators_data_full.keys())}")
            
            # 🔧 关键优化：裁剪掉预热数据，只返回请求的条数
            kline_with_ma = kline_with_ma_full[-count:] if len(kline_with_ma_full) > count else kline_with_ma_full
            indicators_data = {
                key: value[-count:] if len(value) > count else value
                for key, value in indicators_data_full.items()
            }
            logger.info(f"裁剪后的数据: kline={len(kline_with_ma)} 条, 指标每个约{len(next(iter(indicators_data.values()), []))} 条")
            
            # 3. 检测市场事件（只在请求的范围内）
            logger.info("步骤3: 检测市场事件...")
            df_requested = df_full.tail(count)  # 只在请求的范围检测事件
            events = self._detect_events(df_requested)
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
                         before: Optional[str]) -> pd.DataFrame:
        """获取K线数据
        
        Args:
            index_id: 股票/指数代码
            period: 周期
            count: 数据条数
            before: 截止日期
        
        Returns:
            包含 date, open, high, low, close, volume 的 DataFrame
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
        
        # 调用数据提供者
        df = self._data_provider.get_index_prices(
            index_id,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if df is None or df.empty:
            raise ValueError(f"无数据：{index_id}")
        
        # 🔧 标准化列名（小写）
        df.columns = df.columns.str.lower()
        logger.info(f"数据列名: {df.columns.tolist()}")
        
        # 确保日期列
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime(df.index)
        else:
            df['date'] = pd.to_datetime(df['date'])
        
        # 周期转换
        df = self._convert_period(df, period, count)
        
        return df
    
    def _convert_period(self,
                       df: pd.DataFrame,
                       period: str,
                       count: int) -> pd.DataFrame:
        """周期转换（日线→周线/月线）
        
        Args:
            df: 日线数据
            period: 目标周期
            count: 目标条数
        
        Returns:
            转换后的数据
        """
        df_copy = df.copy()
        df_copy = df_copy.set_index('date')
        
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
        
        return df_copy
    
    def _calculate_indicators(self,
                             df: pd.DataFrame,
                             indicators: Optional[str]) -> tuple:
        """计算技术指标
        
        Args:
            df: K线数据
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
        
        # 计算MA并嵌入K线数据
        ma5 = df['close'].rolling(window=5).mean()
        ma10 = df['close'].rolling(window=10).mean()
        ma20 = df['close'].rolling(window=20).mean()
        
        # 组装K线数据（包含MA）
        for _, row in df.iterrows():
            idx = df.index.get_loc(row.name) if hasattr(row, 'name') else _
            record = {
                'date': row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                'open': self._safe_float(row['open']),
                'high': self._safe_float(row['high']),
                'low': self._safe_float(row['low']),
                'close': self._safe_float(row['close']),
                'volume': self._safe_float(row['volume']),
                'ma5': self._safe_float(ma5.iloc[idx]),
                'ma10': self._safe_float(ma10.iloc[idx]),
                'ma20': self._safe_float(ma20.iloc[idx])
            }
            kline_data.append(record)
        
        # 计算技术指标（调用领域层服务）
        if 'vol' in requested_indicators:
            indicators_data['vol'] = self._calculate_vol(df)
        
        if 'macd' in requested_indicators:
            indicators_data['macd'] = self._calculate_macd(df)
        
        if 'rsi' in requested_indicators:
            indicators_data['rsi'] = self._calculate_rsi(df)
        
        if 'kdj' in requested_indicators:
            indicators_data['kdj'] = self._calculate_kdj(df)
        
        if 'obv' in requested_indicators:
            indicators_data['obv'] = self._calculate_obv(df)
        
        return kline_data, indicators_data
    
    def _calculate_vol(self, df: pd.DataFrame) -> List[Dict]:
        """计算成交量指标"""
        return [
            {
                'date': row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                'value': self._safe_float(row['volume'])
            }
            for _, row in df.iterrows()
        ]
    
    def _calculate_macd(self, df: pd.DataFrame) -> List[Dict]:
        """计算MACD指标（调用领域层服务）
        
        注意：
        - 使用Wilder EMA平滑法（pandas标准实现）
        - 前26个周期的值可能为NaN（需要足够数据才能计算）
        - 前端会自动处理null值，不显示对应点
        """
        try:
            logger.info(f"🔧 开始计算MACD，数据行数: {len(df)}")
            logger.info(f"   - df.columns: {df.columns.tolist()}")
            logger.info(f"   - df['close'] 类型: {type(df['close'])}")
            logger.info(f"   - df['close'] 前5个值: {df['close'].head().tolist() if hasattr(df['close'], 'head') else df['close'][:5]}")
            
            # 调用领域层服务计算MACD
            macd, signal, hist = self._indicator_service.calculate_macd(df['close'])
            
            logger.info(f"✅ MACD计算成功，返回类型: macd={type(macd)}, signal={type(signal)}, hist={type(hist)}")
            
            results = []
            for i in range(len(df)):
                # ⚠️ 关键：保留NaN转为null，让前端正确处理数据连续性
                results.append({
                    'date': df.iloc[i]['date'].strftime('%Y-%m-%d') if hasattr(df.iloc[i]['date'], 'strftime') else str(df.iloc[i]['date']),
                    'macd': self._safe_float(macd.iloc[i] if hasattr(macd, 'iloc') else macd[i]),
                    'signal': self._safe_float(signal.iloc[i] if hasattr(signal, 'iloc') else signal[i]),
                    'histogram': self._safe_float(hist.iloc[i] if hasattr(hist, 'iloc') else hist[i])
                })
            
            logger.info(f"MACD计算成功：{len(results)}条数据，前{self._count_leading_nulls(results, 'macd')}条为null（正常）")
            logger.debug(f"MACD前3条数据: {results[:3]}")
            return results
        except Exception as e:
            logger.error(f"⚠️ MACD计算失败，返回空数据: {e}", exc_info=True)
            logger.error(f"   - df.shape: {df.shape if hasattr(df, 'shape') else 'N/A'}")
            logger.error(f"   - df.columns: {df.columns.tolist() if hasattr(df, 'columns') else 'N/A'}")
            logger.error(f"   - 'close' in df.columns: {'close' in df.columns if hasattr(df, 'columns') else 'N/A'}")
            return []
    
    def _calculate_rsi(self, df: pd.DataFrame) -> List[Dict]:
        """计算RSI指标（调用领域层服务）
        
        注意：
        - 使用Wilder平滑法（alpha=1/14）
        - 前14个周期的值可能为NaN
        - 返回值范围0-100
        """
        try:
            rsi = self._indicator_service.calculate_rsi(df['close'])
            
            results = [
                {
                    'date': df.iloc[i]['date'].strftime('%Y-%m-%d') if hasattr(df.iloc[i]['date'], 'strftime') else str(df.iloc[i]['date']),
                    'value': self._safe_float(rsi.iloc[i] if hasattr(rsi, 'iloc') else rsi[i])
                }
                for i in range(len(df))
            ]
            
            logger.debug(f"RSI计算完成：{len(results)}条数据，前{self._count_leading_nulls(results, 'value')}条为null（正常）")
            return results
        except Exception as e:
            logger.warning(f"RSI计算失败，返回空数据: {e}")
            return []
    
    def _calculate_kdj(self, df: pd.DataFrame) -> List[Dict]:
        """计算KDJ指标（调用领域层服务）
        
        注意：
        - K线：随机指标（周期内价格相对位置）
        - D线：K线的3周期SMA平滑
        - J线：3*K - 2*D（灵敏指标）
        - 前9个周期的值可能为NaN
        """
        try:
            k, d = self._indicator_service.calculate_kdj(
                df['high'],
                df['low'],
                df['close']
            )
            
            # 计算 J 值：J = 3*K - 2*D
            j = 3 * k - 2 * d
            
            results = [
                {
                    'date': df.iloc[i]['date'].strftime('%Y-%m-%d') if hasattr(df.iloc[i]['date'], 'strftime') else str(df.iloc[i]['date']),
                    'k': self._safe_float(k.iloc[i] if hasattr(k, 'iloc') else k[i]),
                    'd': self._safe_float(d.iloc[i] if hasattr(d, 'iloc') else d[i]),
                    'j': self._safe_float(j.iloc[i] if hasattr(j, 'iloc') else j[i])
                }
                for i in range(len(df))
            ]
            
            logger.debug(f"KDJ计算完成：{len(results)}条数据，前{self._count_leading_nulls(results, 'k')}条为null（正常）")
            return results
        except Exception as e:
            logger.warning(f"KDJ计算失败，返回空数据: {e}")
            return []
    
    def _calculate_obv(self, df: pd.DataFrame) -> List[Dict]:
        """计算OBV指标（调用领域层服务）
        
        注意：
        - 能量潮指标，累计方向性成交量
        - 价格上涨日累加成交量，下跌日累减
        - 第一个值为0（起始点）
        """
        try:
            obv = self._indicator_service.calculate_obv(df['close'], df['volume'])
            
            results = [
                {
                    'date': df.iloc[i]['date'].strftime('%Y-%m-%d') if hasattr(df.iloc[i]['date'], 'strftime') else str(df.iloc[i]['date']),
                    'value': self._safe_float(obv.iloc[i] if hasattr(obv, 'iloc') else obv[i])
                }
                for i in range(len(df))
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
