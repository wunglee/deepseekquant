import unittest

from core_bak_refactored.core.risk.stress_testing import StressTester
from core_bak_refactored.core.portfolio._fragments.synthetic_portfolio_builder import SyntheticPortfolioBuilder


class ParameterLiteratureValidationTest(unittest.TestCase):
    """
    业务目标3：参数实证支持能力（P0）
    验证内置场景的核心参数（decline / volatility_spike 或同类“波动性倍数”参数）是否落在文献支持的合理范围。
    文献依据（示例）：
      - Jorion (2006): 极端事件市场下跌幅度常见在10%-60%区间（按事件与市场不同）
      - McNeil et al. (2015): 波动率飙升倍数常见在1.5x-4.0x区间（取决于市场冲击强度）
    口径说明：
      - decline：直接损失比例，负值如-0.40。
      - volatility_spike：波动率放大倍数（若场景未给此项，则不做该项断言）。
    验收标准（P0）：6个核心场景中，decline参数100%在[-0.60, -0.08]；
                  若存在volatility类参数（volatility_spike / currency_volatility / interest_rate_spike等），则倍数或强度落在合理区间（按类型映射）。
    """

    def setUp(self):
        self.tester = StressTester(config={})
        self.portfolio = SyntheticPortfolioBuilder.build_csi300_equal_weight()

        # 选取6个核心场景（与模块设计目标一致）：
        self.core_scenarios = [
            '2008_financial_crisis',
            'covid_19_pandemic',
            '2015_china_market_crash',
            '2011_eurozone_debt_crisis',
            '1997_asian_financial_crisis',
            '2022_russia_ukraine_conflict',
        ]

    def test_decline_parameter_within_literature_range(self):
        # decline 合理区间（文献映射）：[-60%, -8%] → [-0.60, -0.08]
        min_decline, max_decline = -0.60, -0.08
        for sid in self.core_scenarios:
            scenario = self.tester.scenarios.get(sid)
            self.assertIsNotNone(scenario, msg=f"场景不存在: {sid}")
            decline = scenario.parameters.get('decline')
            self.assertIsNotNone(decline, msg=f"场景缺少decline参数: {sid}")
            self.assertLessEqual(decline, max_decline, msg=f"decline不应高于-8%: {sid}={decline}")
            self.assertGreaterEqual(decline, min_decline, msg=f"decline不应低于-60%: {sid}={decline}")

    def test_volatility_related_parameters_reasonable(self):
        """
        对“波动性/冲击强度”类参数进行合理性断言：
        - volatility_spike: 1.5x ~ 4.0x
        - currency_volatility: 1.5x ~ 4.0x（作为波动性倍数近似）
        - interest_rate_spike: 1.0 ~ 10.0（以2013钱荒“隔夜利率飙升至13%”为佐证，取更宽容区间）
        - credit_spread_widening: 0.5 ~ 5.0（倍数化处理，视具体冲击模型而定，做宽容断言）
        注：若场景无此类参数，则跳过。
        """
        ranges = {
            'volatility_spike': (1.5, 4.0),
            'currency_volatility': (1.5, 4.0),
            'interest_rate_spike': (1.0, 10.0),
            'credit_spread_widening': (0.5, 5.0),
        }
        for sid in self.core_scenarios:
            scenario = self.tester.scenarios.get(sid)
            params = scenario.parameters
            for key, (lo, hi) in ranges.items():
                if key in params:
                    val = float(params.get(key))
                    self.assertGreaterEqual(val, lo, msg=f"{sid}.{key} 低于合理下限 {lo}: {val}")
                    self.assertLessEqual(val, hi, msg=f"{sid}.{key} 高于合理上限 {hi}: {val}")


if __name__ == '__main__':
    unittest.main()
