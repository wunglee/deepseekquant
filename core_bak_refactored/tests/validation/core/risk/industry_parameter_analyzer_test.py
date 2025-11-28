import unittest
import numpy as np

from core_bak_refactored.core.risk.stress_testing import IndustryParameterAnalyzer


class IndustryParameterAnalyzerTest(unittest.TestCase):
    """
    阶段B：行业参数统计与显著性检验（纯技术实现）
    验证：行业间冲击系数差异≥10%且t检验p<0.05（大样本近似）
    """

    def setUp(self):
        np.random.seed(42)
        # 构造四个行业的冲击样本（≥1000个样本）
        # 设定均值层次：金融>周期>科技>防御，且差异≥10%
        n = 1200
        self.samples = {
            'financial': (np.random.normal(loc=-0.150, scale=0.03, size=n)).tolist(),  # 金融：-15%
            'cyclical': (np.random.normal(loc=-0.130, scale=0.03, size=n)).tolist(),   # 周期：-13%
            'technology': (np.random.normal(loc=-0.115, scale=0.03, size=n)).tolist(), # 科技：-11.5%
            'defensive': (np.random.normal(loc=-0.095, scale=0.03, size=n)).tolist(),  # 防御：-9.5%
        }
        self.analyzer = IndustryParameterAnalyzer()

    def test_analyze_and_validate_industry_differences(self):
        # 端到端：统计→t检验→UAT验证
        result = self.analyzer.analyze_and_validate(self.samples)
        self.assertTrue(result.passed, msg=f"行业差异未通过：avg_diff={result.actual_value:.2%}, details={result.details}")
        # 基本断言：至少一个p值<0.05
        p_values = list(result.details['t_test_results'].values()) if 't_test_results' in result.details else []
        self.assertTrue(any(p < 0.05 for p in p_values), msg=f"p值显著性不足：{p_values}")


if __name__ == '__main__':
    unittest.main()
