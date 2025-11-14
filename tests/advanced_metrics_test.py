"""
高级风险度量模块测试
"""
import sys
sys.path.insert(0, '.')

from core.risk.advanced_metrics import (
    AdvancedRiskMetrics,
    VaRResult,
    StressTestResult
)


def test_var_historical():
    """测试历史VaR"""
    returns = [-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05] * 3
    
    result = AdvancedRiskMetrics.calculate_var(returns, confidence_level=0.95, method="historical")
    
    assert result is not None
    assert isinstance(result, VaRResult)
    assert result.confidence_level == 0.95
    assert result.method == "historical"
    assert result.value_at_risk < 0  # VaR应该是负值（损失）
    print(f"✓ 历史VaR测试通过: VaR={result.value_at_risk:.4f}")


def test_var_parametric():
    """测试参数法VaR"""
    returns = [-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05] * 3
    
    result = AdvancedRiskMetrics.calculate_var(returns, confidence_level=0.95, method="parametric")
    
    assert result is not None
    assert result.method == "parametric"
    assert result.value_at_risk < 0
    print(f"✓ 参数法VaR测试通过: VaR={result.value_at_risk:.4f}")


def test_var_insufficient_data():
    """测试数据不足时的VaR"""
    returns = [-0.05, 0.05]  # 只有2个数据点
    
    result = AdvancedRiskMetrics.calculate_var(returns, method="historical")
    
    assert result is not None
    assert result.value_at_risk == -0.1  # 应该返回默认值
    print("✓ VaR数据不足处理测试通过")


def test_expected_shortfall():
    """测试预期短缺(ES)"""
    returns = [-0.10, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08] * 3
    
    es = AdvancedRiskMetrics.calculate_expected_shortfall(returns, confidence_level=0.95)
    
    assert es is not None
    assert es < 0  # ES应该是负值（损失）
    print(f"✓ 预期短缺测试通过: ES={es:.4f}")


def test_max_drawdown():
    """测试最大回撤"""
    # 模拟一个先涨后跌的收益序列
    returns = [0.02] * 10 + [-0.03] * 15 + [0.01] * 10
    
    mdd = AdvancedRiskMetrics.calculate_max_drawdown(returns)
    
    assert mdd is not None
    assert mdd < 0  # 最大回撤应该是负值
    print(f"✓ 最大回撤测试通过: MDD={mdd:.4f}")


def test_stress_test_basic():
    """测试基本压力测试"""
    portfolio_weights = {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}
    scenario = {'AAPL': -0.20, 'MSFT': -0.15, 'GOOGL': -0.18}  # 市场崩盘情景
    
    result = AdvancedRiskMetrics.stress_test(portfolio_weights, scenario, "market_crash")
    
    assert isinstance(result, StressTestResult)
    assert result.scenario_name == "market_crash"
    assert result.portfolio_impact < 0  # 应该是负面影响
    print(f"✓ 压力测试通过: 影响={result.portfolio_impact:.2%}, 违规={len(result.breached_limits)}项")


def test_stress_test_severe():
    """测试严重压力测试"""
    portfolio_weights = {'AAPL': 0.5, 'MSFT': 0.5}
    scenario = {'AAPL': -0.40, 'MSFT': -0.35}  # 极端崩盘
    
    result = AdvancedRiskMetrics.stress_test(portfolio_weights, scenario)
    
    assert len(result.breached_limits) > 0  # 应该有违规
    assert len(result.recommendations) > 0  # 应该有建议
    print(f"✓ 严重压力测试通过: 影响={result.portfolio_impact:.2%}")


def test_tail_risk_analysis():
    """测试尾部风险分析"""
    returns = [-0.15, -0.10, -0.08, -0.05, -0.03] + [0.01] * 20 + [0.05] * 10
    
    tail_risk = AdvancedRiskMetrics.tail_risk_analysis(returns, tail_percentile=0.05)
    
    assert 'tail_mean' in tail_risk
    assert 'tail_worst' in tail_risk
    assert 'tail_count' in tail_risk
    assert tail_risk['tail_worst'] < 0  # 最差情况应该是负值
    print(f"✓ 尾部风险分析测试通过: 均值={tail_risk['tail_mean']:.4f}, 最差={tail_risk['tail_worst']:.4f}")


def test_concentration_risk_diversified():
    """测试分散化组合的集中度风险"""
    weights = {'A': 0.2, 'B': 0.2, 'C': 0.2, 'D': 0.2, 'E': 0.2}  # 完全分散
    
    risk = AdvancedRiskMetrics.concentration_risk(weights, hhi_threshold=0.25)
    
    assert 'hhi' in risk
    assert 'max_weight' in risk
    assert risk['exceeded'] == False  # 不应该超过阈值
    assert risk['effective_n'] >= 4.5  # 有效资产数应该接近5
    print(f"✓ 分散组合集中度测试通过: HHI={risk['hhi']:.4f}, 有效资产数={risk['effective_n']:.2f}")


def test_concentration_risk_concentrated():
    """测试集中化组合的集中度风险"""
    weights = {'A': 0.7, 'B': 0.2, 'C': 0.1}  # 高度集中
    
    risk = AdvancedRiskMetrics.concentration_risk(weights, hhi_threshold=0.25)
    
    assert risk['exceeded'] == True  # 应该超过阈值
    assert risk['max_weight'] == 0.7
    assert risk['effective_n'] < 2.0  # 有效资产数应该很低
    print(f"✓ 集中组合集中度测试通过: HHI={risk['hhi']:.4f}, 有效资产数={risk['effective_n']:.2f}")


if __name__ == "__main__":
    print("=== 高级风险度量模块测试 ===\n")
    
    test_var_historical()
    test_var_parametric()
    test_var_insufficient_data()
    test_expected_shortfall()
    test_max_drawdown()
    test_stress_test_basic()
    test_stress_test_severe()
    test_tail_risk_analysis()
    test_concentration_risk_diversified()
    test_concentration_risk_concentrated()
    
    print("\n=== 全部10项测试通过！ ===")
