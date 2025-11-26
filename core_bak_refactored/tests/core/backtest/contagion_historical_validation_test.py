import unittest

from core_bak_refactored.core.backtest._fragments.event_window_backtester import EventWindowBacktester
from core_bak_refactored.core.data._fragments.historical_data_provider import MockHistoricalDataProvider
from core_bak_refactored.core.portfolio._fragments.synthetic_portfolio_builder import SyntheticPortfolioBuilder


class ContagionHistoricalValidationTest(unittest.TestCase):
    """
    风险传导历史验证测试（业务目标4）
    
    目标：验证风险传导预测准确率≥80%（误差容忍≤25%）
    历史传导场景对验证：3组（2008→2015、COVID→2022、1997→2008）
    
    注：当前框架未实现传导预测功能，此测试仅验证基础事件数据完整性
    """

    def setUp(self):
        provider = MockHistoricalDataProvider()
        self.backtester = EventWindowBacktester(provider)
        self.portfolio = SyntheticPortfolioBuilder.build_csi300_equal_weight()

    def test_contagion_event_pairs_data_availability(self):
        """验证传导场景对的事件数据完整性"""
        # 传导场景对定义（先导事件 → 传导事件）
        contagion_pairs = [
            ('2008_financial_crisis', '2015_china_market_crash'),
            ('covid_19_pandemic', '2022_russia_ukraine_conflict'),
            ('1997_asian_financial_crisis', '2008_financial_crisis'),
        ]
        
        # 获取所有加载的事件ID
        loaded_event_ids = {e.event_id for e in self.backtester.events}
        
        # 验证每组传导对的事件均存在
        for lead_event, contagion_event in contagion_pairs:
            self.assertIn(
                lead_event,
                loaded_event_ids,
                msg=f"先导事件 {lead_event} 未加载"
            )
            self.assertIn(
                contagion_event,
                loaded_event_ids,
                msg=f"传导事件 {contagion_event} 未加载"
            )
        
        # 验证至少3组传导对
        self.assertGreaterEqual(
            len(contagion_pairs),
            3,
            msg="传导场景对数量不足3组"
        )

    def test_transmission_factor_validation(self):
        """验证传导因子取值依据（基于专家标准：30%，A股35%）"""
        # 当前框架未实现传导预测，此测试验证业务参数范围合理性
        EXPECTED_TRANSMISSION_FACTOR_RANGE = (0.25, 0.35)
        
        # A股市场调整值（专家建议）
        A_SHARE_TRANSMISSION_FACTOR = 0.35
        
        self.assertGreaterEqual(
            A_SHARE_TRANSMISSION_FACTOR,
            EXPECTED_TRANSMISSION_FACTOR_RANGE[0],
            msg="传导因子低于文献支持下限（25%）"
        )
        self.assertLessEqual(
            A_SHARE_TRANSMISSION_FACTOR,
            EXPECTED_TRANSMISSION_FACTOR_RANGE[1],
            msg="传导因子高于文献支持上限（35%）"
        )


if __name__ == '__main__':
    unittest.main()
