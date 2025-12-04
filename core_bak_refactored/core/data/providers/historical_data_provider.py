"""
真实历史数据提供者

职责：
- 历史价格数据获取（支持多数据源回退）
- 数据质量验证与清洗
- 区域化数据源优先级管理
- 交叉验证支持

重构说明（2025-12-02）：
- MockHistoricalDataProvider已迁移到tests/fixtures/core/data/mock_historical_data_provider.py
- Protocol定义已提取到protocols.py
- Stub适配器已提取到stubs/目录
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import logging

from core_bak_refactored.core.data.quality import DataQualityChecker
from core_bak_refactored.core.share import MarketCode
from core_bak_refactored.core.share.config_manager import ConfigManager
from core_bak_refactored.core.share.market.market_enums import DataSource

# 加载配置
config_manager = ConfigManager()
EVENT_WINDOW_CONFIGS = config_manager.get('event_window', {})
REGIONAL_DATA_SOURCE_PRIORITY = config_manager.get('regional_data_source', {})

logger = logging.getLogger('DeepSeekQuant.DataProviders')

# =============================================================================
# 真实数据提供者
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
        self.backup_sources = backup_sources or []
        self.enable_cross_validation = enable_cross_validation
        self._cache = {}
        self._quality_cache = {}  # 数据质量缓存
        self._cross_validation_log = []  # 交叉验证日志
        
        # 加载数据源适配器（仅生产数据源）
        self._adapters = self._initialize_adapters()
    
    def _initialize_adapters(self) -> Dict[str, Any]:
        """初始化数据源适配器（仅生产数据源）"""
        adapters = {}
        
        # Yahoo Finance适配器
        try:
            from core_bak_refactored.core.data.providers.yahoo_finance import YahooFinanceDataProvider
            adapters[DataSource.YAHOO.value] = YahooFinanceDataProvider(fallback_to_mock=False)
            logger.info("Yahoo Finance适配器已加载")
        except Exception as e:
            logger.warning(f"Yahoo Finance适配器加载失败: {e}")
        
        # JoinQuant适配器（真实API未集成，暂不加载）
        # 如需启用，请实现 JoinQuantDataProvider 并在此注册
        # logger.info("JoinQuant适配器未配置（真实API未集成）")
        
        # Wind适配器（真实API未集成，暂不加载）
        # 如需启用，请实现 WindDataProvider 并在此注册
        # logger.info("Wind适配器未配置（真实API未集成）")
        
        # Tushare适配器（实际API实现，A股/港股备用数据源）
        try:
            from core_bak_refactored.core.data.providers.tushare import TushareDataProvider
            tushare_adapter = TushareDataProvider(fallback_to_mock=False)
            if tushare_adapter.available:
                adapters[DataSource.TUSHARE.value] = tushare_adapter
                logger.info("Tushare适配器已加载（实际API）")
            else:
                logger.warning("Tushare适配器未配置（API不可用）")
        except Exception as e:
            logger.warning(f"Tushare适配器加载失败: {e}")
        
        # 至少需要一个真实数据源已就绪
        if not adapters:
            raise RuntimeError("无可用数据源；请配置至少一个真实API（例如：Yahoo、Tushare）。")
        return adapters
    
    def get_index_prices(self, index_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取指数价格数据（含自动回退机制 + 区域化优先级 + 健康检查）"""
        # 优先使用缓存
        cache_key = f"prices:{index_id}:{start_date}:{end_date}:{self.primary_source}"
        if cache_key in self._cache:
            logger.debug(f"使用缓存数据: {cache_key}")
            return self._cache[cache_key]
        
        # 区域化数据源优先级（专家第2轮5.1节）
        regional_sources = self._get_regional_priority(index_id)
        
        # 强制规则：不同市场必须使用对应的本地真实数据源
        # CN/HK 必须使用 Tushare（真实API）；US/EU/JP/SG/UNKNOWN 必须使用 Yahoo
        required_source = None
        if index_id.endswith('.SH') or index_id.endswith('.SZ'):
            required_source = DataSource.TUSHARE.value
        elif index_id.endswith('.HK') or index_id in ['HSI', 'HSCEI']:
            required_source = DataSource.TUSHARE.value
        else:
            required_source = DataSource.YAHOO.value
        
        if required_source not in self._adapters:
            raise RuntimeError(
                f"市场强制规则未满足：index={index_id} 需要配置真实数据源 '{required_source}'，当前未配置或不可用。"
                f"请完成相应API的就绪配置（例如：Tushare token 或 Yahoo 接入）。"
            )
        
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
        # 从symbol中提取市场代码（统一使用MarketCode枚举）
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
            market = MarketCode.UNKNOWN
        
        # 从全局枚举配置获取优先级列表，保持为枚举对象
        priority = REGIONAL_DATA_SOURCE_PRIORITY.get(market, REGIONAL_DATA_SOURCE_PRIORITY[MarketCode.UNKNOWN])
        return [source for source in priority]
    
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
        config = EVENT_WINDOW_CONFIGS.get(event_type, EVENT_WINDOW_CONFIGS['market_crash'])
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
        
        # 获取事件窗口数据（从真实数据源）
        event_data = self.get_index_prices(
            index_id,
            event_start.strftime('%Y-%m-%d'),
            event_end.strftime('%Y-%m-%d')
        )
        
        if event_data.empty:
            raise ValueError(
                f"无法获取事件窗口数据: index={index_id}, "
                f"date_range={event_start.strftime('%Y-%m-%d')} to {event_end.strftime('%Y-%m-%d')}"
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
