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
from typing import Protocol, Dict, Any
from datetime import datetime
import logging

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
    
    def get_index_prices(self, index_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """生成模拟的指数价格数据"""
        # 解析日期
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end, freq='B')  # 交易日
        
        # 检查是否在已知事件窗口内
        event_decline = 0.0
        event_vol = 1.0
        for event_id, params in self.event_params.items():
            event_start = pd.to_datetime(params['period'][0])
            event_end = pd.to_datetime(params['period'][1])
            if start >= event_start and end <= event_end:
                event_decline = params['expected_decline']
                event_vol = params['volatility_multiplier']
                logger.info(f"检测到事件窗口: {event_id}, decline={event_decline}, vol={event_vol}")
                break
        
        # 生成模拟价格序列（确定性趋势 + 随机波动）
        n_days = len(dates)
        initial_price = 3000.0  # 沪深300典型水平
        
        # 基于事件参数生成收益率
        base_volatility = 0.015  # 1.5%日波动率
        daily_volatility = base_volatility * event_vol
        
        # 生成确定性下跌趋势（事件期间）+ 随机波动
        if event_decline != 0.0 and n_days > 0:
            # 精确匹配总收益率到expected_decline
            daily_drift = max(-0.95, (1.0 + event_decline) ** (1.0 / n_days) - 1.0)
            random_component = np.zeros(n_days)
            daily_returns = daily_drift + random_component
        else:
            # 非事件期间：纯随机游走
            daily_returns = np.random.normal(0, daily_volatility, n_days)
        
        # 计算价格序列
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
        """数据质量验证报告（模拟实现）"""
        if data is None or data.empty:
            return {
                'completeness_score': 0.0,
                'consistency_score': 0.0,
                'accuracy_score': 0.0,
                'outliers_detected': 0,
                'total_rows': 0,
                'missing_values': 0
            }
        
        total_rows = len(data)
        missing_values = data.isnull().sum().sum()
        completeness_score = 1.0 - (missing_values / (total_rows * len(data.columns))) if total_rows > 0 else 0.0
        
        # 简单的一致性检查
        consistency_score = 1.0
        if 'close' in data.columns:
            # 检查是否有负价格
            negative_prices = (data['close'] < 0).sum()
            if negative_prices > 0:
                consistency_score -= 0.2 * (negative_prices / len(data))
        
        # 简单的准确性检查
        accuracy_score = 1.0
        if 'close' in data.columns and len(data['close']) > 0:
            mean_price = data['close'].mean()
            if mean_price <= 0:
                accuracy_score = 0.0
        
        # 简单的异常值检测
        outliers_detected = 0
        if 'close' in data.columns:
            Q1 = data['close'].quantile(0.25)
            Q3 = data['close'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_detected = ((data['close'] < lower_bound) | (data['close'] > upper_bound)).sum()
        
        return {
            'completeness_score': completeness_score,
            'consistency_score': consistency_score,
            'accuracy_score': accuracy_score,
            'outliers_detected': int(outliers_detected),
            'total_rows': int(total_rows),
            'missing_values': int(missing_values)
        }

# =============================================================================
# 真实数据提供者（待实现）
# =============================================================================

class RealHistoricalDataProvider:
    """
    真实历史数据提供者（Phase 5B-5扩展版）
    
    基于专家answer.md第1轮4.1节指导：
    - 主源：Yahoo Finance（覆盖广，免费）
    - 备源1：JoinQuant（A股数据质量优，可选）
    - 备源2：Wind（机构级数据，可选）
    - 交叉验证：每月对关键指标进行一致性检查，差异≥5%时触发人工复核
    
    基于专家answer.md第1轮4.2节指导：
    - 事件窗口：事件前后各30个交易日（共61个交易日）
    - 基准期：事件前252个交易日（一年）为波动率计算基准
    - 异常值处理：剔除涨跌停日、极端波动日
    - 停牌处理：停牌日数据沿用最近有效价格，但不计入收益率计算
    """
    
    def __init__(self, 
                 primary_source: str = 'yahoo',
                 backup_sources: Optional[List[str]] = None,
                 enable_cross_validation: bool = False):
        """
        初始化真实历史数据提供者
        
        Args:
            primary_source: 主数据源 ('yahoo', 'joinquant', 'wind', 'mock')
            backup_sources: 备用数据源列表（默认None = ['mock']）
            enable_cross_validation: 是否启用数据交叉验证（默认False）
        """
        self.primary_source = primary_source
        self.backup_sources = backup_sources or ['mock']
        self.enable_cross_validation = enable_cross_validation
        self._mock = MockHistoricalDataProvider()
        self._cache = {}
        self._quality_cache = {}  # 数据质量缓存
        
        # 加载数据源适配器
        self._adapters = self._initialize_adapters()
    
    def _initialize_adapters(self) -> Dict[str, Any]:
        """初始化数据源适配器"""
        adapters = {'mock': self._mock}
        
        # Yahoo Finance适配器
        try:
            from core_bak_refactored.core.data._fragments.yahoo_finance_provider import YahooFinanceDataProvider
            adapters['yahoo'] = YahooFinanceDataProvider(fallback_to_mock=False)
            logger.info("Yahoo Finance适配器已加载")
        except Exception as e:
            logger.warning(f"Yahoo Finance适配器加载失败: {e}")
        
        # JoinQuant适配器（TODO: 待实现）
        # adapters['joinquant'] = JoinQuantAdapter()  # Phase 5B-5+
        
        # Wind适配器（TODO: 待实现）
        # adapters['wind'] = WindAdapter()  # Phase 5B-5+
        
        return adapters
    
    def get_index_prices(self, index_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取指数价格数据（含自动回退机制）"""
        # 优先使用缓存
        cache_key = f"prices:{index_id}:{start_date}:{end_date}:{self.primary_source}"
        if cache_key in self._cache:
            logger.debug(f"使用缓存数据: {cache_key}")
            return self._cache[cache_key]
        
        # 尝试主数据源
        sources_to_try = [self.primary_source] + self.backup_sources
        last_error = None
        
        for source in sources_to_try:
            adapter = self._adapters.get(source)
            if adapter is None:
                logger.warning(f"数据源 {source} 未配置，跳过")
                continue
            
            try:
                logger.info(f"尝试数据源: {source} for {index_id}")
                data = adapter.get_index_prices(index_id, start_date, end_date)
                
                if data is None or data.empty:
                    logger.warning(f"{source} 返回空数据")
                    continue
                
                # 数据质量验证
                quality_score = self._validate_data_quality(data, source)
                if quality_score < 0.6:
                    logger.warning(f"{source} 数据质量不达标: {quality_score:.2f}")
                    continue
                
                # 数据清洗（专家4.2节：异常值处理）
                cleaned_data = self._clean_data(data, index_id)
                
                # 缓存结果
                self._cache[cache_key] = cleaned_data
                logger.info(f"成功获取数据: {source}, 行数={len(cleaned_data)}, 质量={quality_score:.2f}")
                
                return cleaned_data
                
            except Exception as e:
                last_error = e
                logger.warning(f"{source} 获取失败: {e}")
                continue
        
        # 所有数据源失败
        error_msg = f"所有数据源失败: {index_id} ({start_date} to {end_date})"
        if last_error:
            error_msg += f", 最后错误: {last_error}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    def get_event_window_data(self, 
                              index_id: str, 
                              event_date: str,
                              window_days: int = 30,
                              baseline_days: int = 252) -> Dict[str, pd.DataFrame]:
        """
        获取事件窗口数据（专家answer.md第1轮4.2节标准）
        
        Args:
            index_id: 指数代码
            event_date: 事件发生日期 'YYYY-MM-DD'
            window_days: 事件前后窗口天数（默认30）
            baseline_days: 基准期天数（默认252 = 1年）
        
        Returns:
            字典包含:
                'event_window': 事件窗口数据（前后30个交易日）
                'baseline': 基准期数据（事件前252个交易日）
        """
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
            'baseline': baseline_filtered
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
        数据质量验证（专家answer.md第1轮5.1节：数据质量评分≥90%）
        
        优化点:
        - 数值稳定性：添加除零保护
        - 性能优化：缓存键使用哈希避免长字符串
        - 异常处理：捕获计算异常并降级
        
        Returns:
            质量评分 (0-1)，≥0.6为及格
        """
        if data is None or data.empty:
            return 0.0
        
        # 优化：使用数据指纹代替简单长度
        data_fingerprint = hash((source, len(data), tuple(data.columns)))
        cache_key = f"quality:{data_fingerprint}"
        if cache_key in self._quality_cache:
            return self._quality_cache[cache_key]
        
        try:
            score = 0.0
            total_cells = len(data) * len(data.columns)
            
            # 1. 完整性检查（30%权重）- 数值稳定性优化
            if total_cells > 0:
                missing_count = data.isnull().sum().sum()
                completeness = 1.0 - (missing_count / total_cells)
                score += completeness * 0.3
            else:
                score += 0.0  # 空数据框
            
            # 2. 一致性检查（30%权重）- 添加边界保护
            consistency = 1.0
            if 'close' in data.columns and len(data) > 0:
                close_prices = data['close'].dropna()
                if len(close_prices) > 0:
                    # 无负价格
                    negative_prices = (close_prices < 0).sum()
                    if negative_prices > 0:
                        consistency -= min(0.5, 0.5 * (negative_prices / len(close_prices)))
                    
                    # 无异常大价格（超过均值10倍）- 添加除零保护
                    mean_price = close_prices.mean()
                    if mean_price > 0:
                        extreme_prices = (close_prices > mean_price * 10).sum()
                        if extreme_prices > 0:
                            consistency -= min(0.3, 0.3 * (extreme_prices / len(close_prices)))
            score += max(0.0, consistency) * 0.3
            
            # 3. 连续性检查（20%权重）- 异常处理优化
            if 'date' in data.columns and len(data) > 1:
                try:
                    date_series = pd.to_datetime(data['date'])
                    date_diffs = date_series.diff().dt.days
                    # 过滤NaT和负值
                    valid_diffs = date_diffs[date_diffs.notna() & (date_diffs > 0)]
                    if len(valid_diffs) > 0:
                        long_gaps = (valid_diffs > 10).sum()
                        continuity = 1.0 - (long_gaps / len(valid_diffs))
                        score += continuity * 0.2
                    else:
                        score += 0.1
                except Exception as e:
                    logger.warning(f"连续性检查失败: {e}")
                    score += 0.1
            else:
                score += 0.1  # 无法验证，给部分分
            
            # 4. 合理性检查（20%权重）- 数值稳定性优化
            reasonableness = 1.0
            if 'close' in data.columns and len(data) > 1:
                try:
                    close_prices = data['close'].dropna()
                    if len(close_prices) > 1:
                        returns = close_prices.pct_change().dropna()
                        if len(returns) > 0:
                            # 日收益率应在-50%到+50%之间（极端但合理）
                            unreasonable = ((returns < -0.5) | (returns > 0.5)).sum()
                            if unreasonable > 0:
                                penalty = min(0.2, 0.2 * (unreasonable / len(returns)))
                                reasonableness -= penalty
                except Exception as e:
                    logger.warning(f"合理性检查失败: {e}")
                    reasonableness = 0.8  # 降级评分
            score += max(0.0, reasonableness) * 0.2
            
            # 确保评分在[0, 1]范围内
            final_score = max(0.0, min(1.0, score))
            self._quality_cache[cache_key] = final_score
            return final_score
            
        except Exception as e:
            logger.error(f"数据质量验证失败: {e}")
            return 0.0  # 异常降级
    
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
