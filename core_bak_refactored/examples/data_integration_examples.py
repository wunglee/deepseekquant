"""
专家碎片组合使用示例

展示如何组合使用专家完整版(DataFetcher)与专家碎片的增量功能:
1. DataQualityEnhancer - 质量驱动的多源智能切换
2. RealHistoricalDataProvider - 区域化优先级与事件窗口
3. YahooFinanceDataProvider - 指数代码映射与标准化输出

TODO：专家碎片整合 - 待评审
作者: Qoder AI
日期: 2025-11-28
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

# 从专家完整版导入
from core_bak_refactored.core.data.data_fetcher import (
    DataFetcher,
    DataSourceType,
    DataFrequency
)

# 从专家碎片导入
from core_bak_refactored.core.data.data_quality_enhancer import DataQualityEnhancer
from core_bak_refactored.core.data.historical_data_provider import (
    RealHistoricalDataProvider,
    MockHistoricalDataProvider
)
from core_bak_refactored.core.data.providers.yahoo_finance import YahooFinanceDataProvider


# ============================================================================
# 示例1: 基础使用 - 仅使用专家完整版DataFetcher
# ============================================================================

async def example1_basic_data_fetcher():
    """
    示例1: 基础数据获取
    
    使用场景:
    - 基础的多源数据获取（失败驱动切换）
    - 内置缓存和性能监控
    - 适合稳定的生产环境
    """
    print("\n" + "=" * 80)
    print("示例1: 基础使用 - DataFetcher（专家完整版）")
    print("=" * 80)
    
    # 配置DataFetcher
    config = {
        'cache_enabled': True,
        'cache_duration': 300,
        'primary': DataSourceType.YAHOO_FINANCE.value,
        'fallback_sources': [
            DataSourceType.ALPHA_VANTAGE.value,
            DataSourceType.IEX_CLOUD.value
        ],
        'max_retries': 3,
        'request_timeout': 30
    }
    
    fetcher = DataFetcher(config)
    
    # 获取历史数据
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    try:
        data = await fetcher.get_historical_data(
            symbols=symbols,
            period='1mo',
            interval='1d',
            data_type='ohlcv',
            adjustments=True
        )
        
        print(f"\n✅ 成功获取 {len(data)} 个股票的数据")
        for symbol, market_data_list in data.items():
            print(f"   - {symbol}: {len(market_data_list)} 个数据点")
            
    except Exception as e:
        print(f"❌ 数据获取失败: {e}")


# ============================================================================
# 示例2: 质量驱动切换 - DataQualityEnhancer + YahooFinanceDataProvider
# ============================================================================

def example2_quality_driven_switching():
    """
    示例2: 质量驱动的多源智能切换
    
    使用场景:
    - 需要根据数据质量自动选择最佳数据源
    - 对数据质量要求较高的场景
    - 多源数据对比和验证
    
    增量功能:
    - 质量评分驱动切换（质量<0.8自动切换）
    - 质量对比选择（选择质量最高的源）
    - 细化质量评分（完整性+一致性+准确性+异常检测）
    - IQR异常值检测
    
    TODO：专家碎片整合 - 待评审
    """
    print("\n" + "=" * 80)
    print("示例2: 质量驱动切换 - DataQualityEnhancer（专家碎片）")
    print("=" * 80)
    
    # 主数据源: Yahoo Finance（带指数映射）
    primary_provider = YahooFinanceDataProvider()
    
    # 备用数据源: Mock（用于演示质量对比）
    backup_provider = MockHistoricalDataProvider(
        use_real_events=True,
        noise_level=0.02
    )
    
    # 创建质量增强器
    enhancer = DataQualityEnhancer(
        primary_provider=primary_provider,
        backup_providers=[backup_provider],
        quality_threshold=0.8  # 质量阈值：低于0.8触发切换
    )
    
    # 获取沪深300指数数据（2015年股灾期间）
    index_id = '000300.SH'  # 会自动映射到Yahoo的'000300.SS'
    start_date = '2015-06-01'
    end_date = '2015-09-01'
    
    try:
        # 质量驱动的数据获取
        data, quality_report = enhancer.get_enhanced_prices(
            index_id=index_id,
            start_date=start_date,
            end_date=end_date
        )
        
        print(f"\n✅ 获取数据成功")
        print(f"   - 指数代码: {index_id}")
        print(f"   - 数据点数: {len(data)}")
        print(f"\n📊 质量报告:")
        print(f"   - 总体评分: {quality_report.overall_score:.3f}")
        print(f"   - 完整性评分: {quality_report.completeness_score:.3f}")
        print(f"   - 一致性评分: {quality_report.consistency_score:.3f}")
        print(f"   - 准确性评分: {quality_report.accuracy_score:.3f}")
        print(f"   - 异常值数量: {quality_report.outlier_count}")
        print(f"   - 数据源: {quality_report.data_source}")
        
        # 演示质量对比逻辑
        if quality_report.overall_score < 0.8:
            print(f"\n⚠️ 主源质量不足 ({quality_report.overall_score:.3f} < 0.8)")
            print(f"   已自动切换到备用源: {quality_report.data_source}")
        else:
            print(f"\n✅ 主源质量合格 ({quality_report.overall_score:.3f} >= 0.8)")
            
    except Exception as e:
        print(f"❌ 数据获取失败: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# 示例3: 区域化优先级 - RealHistoricalDataProvider
# ============================================================================

def example3_regional_priority():
    """
    示例3: 区域化数据源优先级
    
    使用场景:
    - 需要根据市场区域自动选择最佳数据源
    - A股优先JoinQuant，美股优先Yahoo，港股优先Wind
    - 跨市场数据获取和对比
    
    增量功能:
    - Protocol标准化接口
    - 区域化优先级（CN/US/HK自动切换）
    - Phase 5B-5增强（事件窗口、停牌处理）
    - 交叉验证双维度
    
    TODO：专家碎片整合 - 待评审
    """
    print("\n" + "=" * 80)
    print("示例3: 区域化优先级 - RealHistoricalDataProvider（专家碎片）")
    print("=" * 80)
    
    # 创建区域化历史数据提供者
    provider = RealHistoricalDataProvider(
        primary_source='yahoo',  # 默认主源
        enable_cross_validation=True,  # 启用交叉验证
        event_window_days=30  # 事件窗口30天
    )
    
    # 测试不同市场的数据获取
    test_cases = [
        {
            'index_id': '000300.SH',  # 沪深300（A股）
            'market': 'CN',
            'expected_source': 'JoinQuant'
        },
        {
            'index_id': '^GSPC',  # S&P 500（美股）
            'market': 'US',
            'expected_source': 'Yahoo'
        },
        {
            'index_id': '^HSI',  # 恒生指数（港股）
            'market': 'HK',
            'expected_source': 'Wind'
        }
    ]
    
    start_date = '2020-01-01'
    end_date = '2020-06-30'
    
    for case in test_cases:
        print(f"\n--- {case['market']}市场测试 ---")
        print(f"指数代码: {case['index_id']}")
        print(f"预期数据源: {case['expected_source']}")
        
        try:
            data = provider.get_index_prices(
                index_id=case['index_id'],
                start_date=start_date,
                end_date=end_date
            )
            
            print(f"✅ 数据获取成功")
            print(f"   - 数据点数: {len(data)}")
            print(f"   - 实际数据源: {getattr(provider, '_last_source', 'unknown')}")
            print(f"   - 区域化优先级: 已应用")
            
        except Exception as e:
            print(f"❌ 数据获取失败: {e}")
            print(f"   注意: {case['expected_source']}可能需要API凭据")


# ============================================================================
# 示例4: 高级组合 - DataFetcher + Enhancer + Provider
# ============================================================================

async def example4_advanced_combination():
    """
    示例4: 高级组合使用
    
    使用场景:
    - 需要同时使用多个增量功能
    - 生产环境的高质量数据获取
    - 复杂的多源数据策略
    
    组合优势:
    - DataFetcher提供缓存和性能监控
    - DataQualityEnhancer提供质量驱动切换
    - YahooFinanceDataProvider提供指数映射和格式化
    
    TODO：专家碎片整合 - 待评审
    """
    print("\n" + "=" * 80)
    print("示例4: 高级组合 - DataFetcher + Enhancer + Provider")
    print("=" * 80)
    
    # 步骤1: 创建Yahoo专用Provider（带指数映射）
    yahoo_provider = YahooFinanceDataProvider()
    
    # 步骤2: 创建质量增强器（质量驱动）
    enhancer = DataQualityEnhancer(
        primary_provider=yahoo_provider,
        backup_providers=[MockHistoricalDataProvider()],
        quality_threshold=0.85  # 更高的质量要求
    )
    
    # 步骤3: 创建DataFetcher（缓存和监控）
    fetcher = DataFetcher({
        'cache_enabled': True,
        'cache_duration': 600,
        'primary': DataSourceType.YAHOO_FINANCE.value,
        'max_retries': 3
    })
    
    print("\n📦 组件初始化完成:")
    print("   ✅ YahooFinanceDataProvider (指数映射+标准化)")
    print("   ✅ DataQualityEnhancer (质量驱动切换)")
    print("   ✅ DataFetcher (缓存+性能监控)")
    
    # 步骤4: 组合使用示例
    index_id = '000300.SH'
    start_date = '2024-01-01'
    end_date = '2024-11-01'
    
    try:
        # 方式1: 使用Enhancer获取高质量数据
        print(f"\n🔄 使用质量增强器获取数据...")
        data, quality_report = enhancer.get_enhanced_prices(
            index_id=index_id,
            start_date=start_date,
            end_date=end_date
        )
        
        print(f"✅ 质量驱动获取成功")
        print(f"   - 质量评分: {quality_report.overall_score:.3f}")
        print(f"   - 数据源: {quality_report.data_source}")
        print(f"   - 数据点数: {len(data)}")
        
        # 方式2: 使用DataFetcher的缓存机制
        # （实际应用中可以先用Enhancer验证质量，再用Fetcher缓存）
        symbols = ['^GSPC', 'AAPL']
        print(f"\n🔄 使用DataFetcher获取美股数据...")
        fetcher_data = await fetcher.get_historical_data(
            symbols=symbols,
            period='1mo',
            interval='1d'
        )
        
        print(f"✅ 基础获取成功（带缓存）")
        for symbol, data_list in fetcher_data.items():
            print(f"   - {symbol}: {len(data_list)} 个数据点")
            
        # 显示性能指标
        print(f"\n📊 DataFetcher性能指标:")
        metrics = fetcher.performance_metrics
        print(f"   - 总请求数: {metrics['requests_total']}")
        print(f"   - 缓存命中: {metrics['cache_hits']}")
        print(f"   - 缓存未命中: {metrics['cache_misses']}")
        print(f"   - 平均响应时间: {metrics['avg_response_time']:.3f}秒")
        
    except Exception as e:
        print(f"❌ 组合使用失败: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# 示例5: 实战场景 - 2015股灾期间的质量对比
# ============================================================================

def example5_crisis_quality_comparison():
    """
    示例5: 实战场景 - 股市危机期间的数据质量对比
    
    使用场景:
    - 极端市场条件下的数据质量验证
    - 多源数据质量对比
    - 历史事件分析
    
    测试场景:
    - 2015年6-9月中国股灾
    - 2020年3月COVID-19暴跌
    - 2008年金融危机
    
    TODO：专家碎片整合 - 待评审
    """
    print("\n" + "=" * 80)
    print("示例5: 实战场景 - 极端市场下的质量对比")
    print("=" * 80)
    
    # 历史事件测试用例
    crisis_events = [
        {
            'name': '2015中国股灾',
            'index': '000300.SH',
            'start': '2015-06-01',
            'end': '2015-09-01'
        },
        {
            'name': '2020 COVID-19暴跌',
            'index': '^GSPC',
            'start': '2020-02-01',
            'end': '2020-04-30'
        }
    ]
    
    # 创建质量增强器
    yahoo_provider = YahooFinanceDataProvider()
    mock_provider = MockHistoricalDataProvider(use_real_events=True)
    enhancer = DataQualityEnhancer(
        primary_provider=yahoo_provider,
        backup_providers=[mock_provider],
        quality_threshold=0.75  # 危机期间适当降低阈值
    )
    
    for event in crisis_events:
        print(f"\n--- {event['name']} ---")
        print(f"时间范围: {event['start']} 至 {event['end']}")
        print(f"指数代码: {event['index']}")
        
        try:
            data, quality_report = enhancer.get_enhanced_prices(
                index_id=event['index'],
                start_date=event['start'],
                end_date=event['end']
            )
            
            print(f"\n✅ 数据获取成功")
            print(f"📊 质量报告:")
            print(f"   - 总体评分: {quality_report.overall_score:.3f}")
            print(f"   - 完整性: {quality_report.completeness_score:.3f}")
            print(f"   - 一致性: {quality_report.consistency_score:.3f}")
            print(f"   - 准确性: {quality_report.accuracy_score:.3f}")
            print(f"   - 异常值: {quality_report.outlier_count}")
            print(f"   - 数据源: {quality_report.data_source}")
            print(f"   - 数据点数: {len(data)}")
            
            # 评估数据质量等级
            score = quality_report.overall_score
            if score >= 0.9:
                grade = "优秀 ⭐⭐⭐⭐⭐"
            elif score >= 0.8:
                grade = "良好 ⭐⭐⭐⭐"
            elif score >= 0.7:
                grade = "合格 ⭐⭐⭐"
            else:
                grade = "需改进 ⭐⭐"
                
            print(f"\n📈 数据质量等级: {grade}")
            
        except Exception as e:
            print(f"❌ 数据获取失败: {e}")


# ============================================================================
# 主函数 - 运行所有示例
# ============================================================================

def main():
    """
    运行所有示例
    
    注意事项:
    1. 某些数据源（JoinQuant/Wind/Tushare）需要API凭据
    2. 示例会自动回退到Mock数据（如果真实数据源不可用）
    3. 建议按顺序运行示例，从简单到复杂
    """
    print("=" * 80)
    print("专家碎片组合使用示例集")
    print("=" * 80)
    print("\n展示如何组合使用专家完整版与专家碎片的增量功能")
    print("\n包含5个示例:")
    print("  1. 基础使用 - DataFetcher（专家完整版）")
    print("  2. 质量驱动切换 - DataQualityEnhancer（专家碎片）")
    print("  3. 区域化优先级 - RealHistoricalDataProvider（专家碎片）")
    print("  4. 高级组合 - DataFetcher + Enhancer + Provider")
    print("  5. 实战场景 - 极端市场质量对比")
    
    # 运行同步示例
    print("\n" + "=" * 80)
    print("开始运行同步示例...")
    print("=" * 80)
    
    example2_quality_driven_switching()
    example3_regional_priority()
    example5_crisis_quality_comparison()
    
    # 运行异步示例
    print("\n" + "=" * 80)
    print("开始运行异步示例...")
    print("=" * 80)
    
    asyncio.run(example1_basic_data_fetcher())
    asyncio.run(example4_advanced_combination())
    
    print("\n" + "=" * 80)
    print("所有示例运行完成！")
    print("=" * 80)
    print("\n💡 提示:")
    print("  - 示例代码位于: core_bak_refactored/examples/data_integration_examples.py")
    print("  - 可以单独运行每个示例函数进行测试")
    print("  - 部分功能需要API凭据（JoinQuant/Wind等）")
    print("  - 无凭据时会自动回退到Mock数据")


if __name__ == '__main__':
    main()
