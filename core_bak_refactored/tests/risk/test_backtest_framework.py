"""
历史回测框架测试
测试范围：
1. 模拟数据提供者功能
2. 合成组合构造器
3. 事件窗口回测引擎
4. 报告生成器

注意：使用模拟数据，真实数据集成后需补充集成测试
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

import sys
from pathlib import Path
# 添加core_bak_refactored到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.risk.backtest_framework import (
    MockHistoricalDataProvider,
    SyntheticPortfolioBuilder,
    EventWindowBacktester,
    BacktestReporter,
    BacktestResult
)


class TestMockHistoricalDataProvider:
    """测试模拟数据提供者"""
    
    def test_get_index_prices_basic(self):
        """测试基本价格数据生成"""
        provider = MockHistoricalDataProvider()
        df = provider.get_index_prices('000300.SH', '2020-01-01', '2020-01-31')
        
        assert len(df) > 0, "应该生成价格数据"
        assert list(df.columns) == ['date', 'close', 'volume']
        assert df['close'].notna().all(), "价格不应有缺失"
        assert (df['volume'] > 0).all(), "成交量应为正"
    
    def test_get_index_prices_event_window(self):
        """测试事件窗口内的数据生成（应反映事件参数）"""
        provider = MockHistoricalDataProvider()
        
        # 测试2015股灾窗口（多次采样验证趋势）
        total_returns = []
        for _ in range(10):  # 采样10次
            df = provider.get_index_prices('000300.SH', '2015-06-15', '2015-08-26')
            assert len(df) > 0
            
            initial_price = df['close'].iloc[0]
            final_price = df['close'].iloc[-1]
            total_return = (final_price - initial_price) / initial_price
            total_returns.append(total_return)
        
        # 平均收益率应接近-43%（允许一定误差）
        avg_return = np.mean(total_returns)
        expected_decline = -0.43
        
        # 放宽容差：平均值应在[-50%, -30%]范围内
        assert -0.50 < avg_return < -0.30, \
            f"事件窗口平均收益率应接近{expected_decline:.0%}: {avg_return:.2%}"
    
    def test_get_index_returns(self):
        """测试收益率序列生成"""
        provider = MockHistoricalDataProvider()
        returns = provider.get_index_returns('000300.SH', '2020-01-01', '2020-01-31')
        
        assert isinstance(returns, pd.Series)
        assert len(returns) > 0
        assert returns.index.dtype == 'datetime64[ns]'


class TestSyntheticPortfolioBuilder:
    """测试合成组合构造器"""
    
    def test_build_csi300_equal_weight(self):
        """测试沪深300等权重组合"""
        portfolio = SyntheticPortfolioBuilder.build_csi300_equal_weight()
        
        assert portfolio.portfolio_id == 'CSI300_EQ'
        assert portfolio.name == '沪深300等权重组合'
        assert '000300.SH' in portfolio.composition
        assert portfolio.total_value == 1000000.0
        assert portfolio.metadata['type'] == 'index_replication'
    
    def test_build_sector_rotation(self):
        """测试行业轮动组合"""
        portfolio = SyntheticPortfolioBuilder.build_sector_rotation()
        
        assert portfolio.portfolio_id == 'SECTOR_ROT'
        assert len(portfolio.composition) == 4  # 4个行业
        
        # 检查权重和为1
        total_weight = sum(portfolio.composition.values())
        assert abs(total_weight - 1.0) < 1e-6, "权重总和应为1"
        
        # 检查权重分配
        assert portfolio.composition['finance_index'] == 0.30
        assert portfolio.composition['consumer_index'] == 0.25
    
    def test_build_ah_hybrid(self):
        """测试A+H混合组合"""
        portfolio = SyntheticPortfolioBuilder.build_ah_hybrid()
        
        assert portfolio.portfolio_id == 'AH_HYBRID'
        assert '000300.SH' in portfolio.composition
        assert 'HSI' in portfolio.composition
        
        # 检查A股70% + 港股30%
        assert portfolio.composition['000300.SH'] == 0.70
        assert portfolio.composition['HSI'] == 0.30
        assert portfolio.metadata['type'] == 'cross_border'


class TestEventWindowBacktester:
    """测试事件窗口回测引擎"""
    
    def test_load_events(self):
        """测试事件加载"""
        provider = MockHistoricalDataProvider()
        backtester = EventWindowBacktester(provider)
        
        assert len(backtester.events) == 3, "应加载3个事件"
        
        event_ids = [e.event_id for e in backtester.events]
        assert '2015_china_market_crash' in event_ids
        assert 'covid_19_pandemic' in event_ids
        assert '2008_financial_crisis' in event_ids
    
    def test_calculate_actual_loss(self):
        """测试实际损失计算"""
        provider = MockHistoricalDataProvider()
        backtester = EventWindowBacktester(provider)
        
        portfolio = SyntheticPortfolioBuilder.build_csi300_equal_weight()
        event = backtester.events[0]  # 2015股灾
        
        actual_loss = backtester._calculate_actual_loss(
            portfolio, event, '000300.SH'
        )
        
        assert actual_loss != 0.0, "应计算出实际损失"
        assert actual_loss < 0, "事件窗口应为负收益（损失）"
    
    def test_calculate_predicted_loss(self):
        """测试预测损失计算（简化版本）"""
        provider = MockHistoricalDataProvider()
        backtester = EventWindowBacktester(provider)
        
        portfolio = SyntheticPortfolioBuilder.build_csi300_equal_weight()
        event = backtester.events[0]  # 2015股灾
        
        predicted_loss = backtester._calculate_predicted_loss(
            portfolio, event, stress_tester=None  # 简化版本不需要实例
        )
        
        assert predicted_loss != 0.0, "应计算出预测损失"
        assert predicted_loss < 0, "预测应为损失（负值）"
    
    def test_run_backtest_basic(self):
        """测试基本回测流程"""
        provider = MockHistoricalDataProvider()
        backtester = EventWindowBacktester(provider)
        
        portfolio = SyntheticPortfolioBuilder.build_csi300_equal_weight()
        
        # 运行回测（无真实StressTester，使用简化逻辑）
        results = backtester.run_backtest(portfolio, stress_tester=None)
        
        assert len(results) == 3, "应返回3个事件的回测结果"
        
        for result in results:
            assert isinstance(result, BacktestResult)
            assert result.portfolio_id == 'CSI300_EQ'
            assert result.actual_loss != 0.0
            assert result.predicted_loss != 0.0
            assert result.prediction_error >= 0.0, "误差应为非负"


class TestBacktestReporter:
    """测试回测报告生成器"""
    
    def test_generate_summary(self):
        """测试摘要生成"""
        # 构造测试结果
        results = [
            BacktestResult(
                event_id='event1',
                portfolio_id='port1',
                predicted_loss=-0.30,
                actual_loss=-0.35,
                prediction_error=0.14,  # |(-0.30) - (-0.35)| / |-0.35| = 0.14
                benchmark_index='000300.SH'
            ),
            BacktestResult(
                event_id='event2',
                portfolio_id='port1',
                predicted_loss=-0.20,
                actual_loss=-0.22,
                prediction_error=0.09,
                benchmark_index='000300.SH'
            )
        ]
        
        summary = BacktestReporter.generate_summary(results)
        
        assert summary['total_tests'] == 2
        assert 'avg_error' in summary
        assert 'max_error' in summary
        assert 'accuracy_20pct' in summary
        assert len(summary['results']) == 2
    
    def test_generate_summary_empty(self):
        """测试空结果处理"""
        summary = BacktestReporter.generate_summary([])
        assert summary['status'] == 'no_results'
    
    def test_print_summary(self, capsys):
        """测试报告打印"""
        results = [
            BacktestResult(
                event_id='2015_china_market_crash',
                portfolio_id='CSI300_EQ',
                predicted_loss=-0.30,
                actual_loss=-0.35,
                prediction_error=0.14,
                benchmark_index='000300.SH',
                metadata={'event_name': '2015中国股灾'}
            )
        ]
        
        summary = BacktestReporter.generate_summary(results)
        BacktestReporter.print_summary(summary)
        
        captured = capsys.readouterr()
        assert "历史回测验证报告" in captured.out
        assert "平均误差" in captured.out
        assert "2015_china_market_crash" in captured.out


class TestIntegration:
    """集成测试：端到端回测流程"""
    
    def test_end_to_end_backtest(self):
        """端到端回测流程测试"""
        # 1. 创建数据提供者
        provider = MockHistoricalDataProvider()
        
        # 2. 创建回测引擎
        backtester = EventWindowBacktester(provider)
        
        # 3. 构造合成组合
        portfolio = SyntheticPortfolioBuilder.build_csi300_equal_weight()
        
        # 4. 运行回测
        results = backtester.run_backtest(portfolio, stress_tester=None)
        
        # 5. 生成报告
        summary = BacktestReporter.generate_summary(results)
        
        # 验证完整流程
        assert len(results) == 3, "应完成3个事件回测"
        assert summary['total_tests'] == 3
        assert 'avg_error' in summary
        
        # 检查所有结果都有有效数据
        for result in results:
            assert result.actual_loss != 0.0
            assert result.predicted_loss != 0.0
            assert result.prediction_error >= 0.0
    
    def test_multiple_portfolios(self):
        """测试多个组合的回测"""
        provider = MockHistoricalDataProvider()
        backtester = EventWindowBacktester(provider)
        
        portfolios = [
            SyntheticPortfolioBuilder.build_csi300_equal_weight(),
            SyntheticPortfolioBuilder.build_sector_rotation(),
            SyntheticPortfolioBuilder.build_ah_hybrid()
        ]
        
        all_results = []
        for portfolio in portfolios:
            results = backtester.run_backtest(portfolio, stress_tester=None)
            all_results.extend(results)
        
        assert len(all_results) == 9, "3个组合 × 3个事件 = 9个结果"
        
        # 检查portfolio_id正确
        portfolio_ids = set(r.portfolio_id for r in all_results)
        assert len(portfolio_ids) == 3, "应包含3个不同的组合"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
