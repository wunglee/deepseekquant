"""
高级组合优化模块测试
"""
import sys
sys.path.insert(0, '.')

from core.portfolio.advanced_optimizers import (
    AdvancedOptimizers,
    OptimizationInput
)


def test_black_litterman_with_views():
    """测试Black-Litterman with观点"""
    optimization_input = OptimizationInput(
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        expected_returns={'AAPL': 0.10, 'MSFT': 0.08, 'GOOGL': 0.12},
        covariance_matrix=[
            [0.04, 0.01, 0.02],
            [0.01, 0.03, 0.01],
            [0.02, 0.01, 0.05]
        ],
        constraints={}
    )
    
    market_weights = {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}
    views = {'AAPL': 0.05, 'MSFT': -0.02, 'GOOGL': 0.03}  # 观点收益率
    
    weights = AdvancedOptimizers.black_litterman(
        optimization_input, 
        market_cap_weights=market_weights,
        views=views
    )
    
    assert len(weights) == 3
    assert all(sym in weights for sym in ['AAPL', 'MSFT', 'GOOGL'])
    # 验证权重和为1
    assert abs(sum(weights.values()) - 1.0) < 0.01
    # 验证权重非负
    assert all(w >= 0 for w in weights.values())
    print(f"✓ Black-Litterman测试通过: 权重={weights}")


def test_black_litterman_no_views():
    """测试Black-Litterman without观点（回退到市值加权）"""
    optimization_input = OptimizationInput(
        symbols=['AAPL', 'MSFT'],
        expected_returns={'AAPL': 0.10, 'MSFT': 0.08},
        covariance_matrix=[[0.04, 0.01], [0.01, 0.03]],
        constraints={}
    )
    
    market_weights = {'AAPL': 0.6, 'MSFT': 0.4}
    
    weights = AdvancedOptimizers.black_litterman(
        optimization_input,
        market_cap_weights=market_weights,
        views=None
    )
    
    # 没有观点，应该返回市值加权
    assert weights == market_weights
    print("✓ Black-Litterman无观点回退测试通过")


def test_hierarchical_risk_parity():
    """测试HRP"""
    optimization_input = OptimizationInput(
        symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
        expected_returns={},
        covariance_matrix=[
            [0.04, 0.01, 0.02, 0.01],
            [0.01, 0.03, 0.01, 0.02],
            [0.02, 0.01, 0.05, 0.02],
            [0.01, 0.02, 0.02, 0.06]
        ],
        constraints={}
    )
    
    weights = AdvancedOptimizers.hierarchical_risk_parity(optimization_input)
    
    assert len(weights) == 4
    assert abs(sum(weights.values()) - 1.0) < 0.01
    assert all(w >= 0 for w in weights.values())
    # 高方差资产应该有更低的权重
    assert weights['AMZN'] < weights['MSFT']  # AMZN方差0.06 > MSFT方差0.03
    print(f"✓ HRP测试通过: 权重={weights}")


def test_critical_line_algorithm():
    """测试关键线算法"""
    optimization_input = OptimizationInput(
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        expected_returns={'AAPL': 0.12, 'MSFT': 0.08, 'GOOGL': 0.10},
        covariance_matrix=[
            [0.04, 0.01, 0.02],
            [0.01, 0.03, 0.01],
            [0.02, 0.01, 0.05]
        ],
        constraints={}
    )
    
    weights = AdvancedOptimizers.critical_line_algorithm(optimization_input)
    
    assert len(weights) == 3
    assert abs(sum(weights.values()) - 1.0) < 0.01
    assert all(w >= 0 for w in weights.values())
    # 最大夏普模式：高收益资产应该有更高权重
    assert weights['AAPL'] > weights['MSFT']  # AAPL收益0.12 > MSFT收益0.08
    print(f"✓ CLA测试通过: 权重={weights}")


def test_risk_parity_equal_budget():
    """测试风险平价（等风险预算）"""
    optimization_input = OptimizationInput(
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        expected_returns={},
        covariance_matrix=[
            [0.04, 0.01, 0.02],
            [0.01, 0.03, 0.01],
            [0.02, 0.01, 0.05]
        ],
        constraints={}
    )
    
    weights = AdvancedOptimizers.risk_parity(optimization_input)
    
    assert len(weights) == 3
    assert abs(sum(weights.values()) - 1.0) < 0.01
    assert all(w >= 0 for w in weights.values())
    # 高波动资产应该有更低权重
    assert weights['GOOGL'] < weights['MSFT']  # GOOGL波动sqrt(0.05) > MSFT波动sqrt(0.03)
    print(f"✓ 风险平价测试通过: 权重={weights}")


def test_risk_parity_custom_budget():
    """测试风险平价（自定义风险预算）"""
    optimization_input = OptimizationInput(
        symbols=['AAPL', 'MSFT'],
        expected_returns={},
        covariance_matrix=[[0.04, 0.01], [0.01, 0.03]],
        constraints={}
    )
    
    # AAPL承担70%风险，MSFT承担30%
    risk_budget = {'AAPL': 0.7, 'MSFT': 0.3}
    
    weights = AdvancedOptimizers.risk_parity(optimization_input, risk_budget=risk_budget)
    
    assert abs(sum(weights.values()) - 1.0) < 0.01
    # AAPL风险预算更高，应该有更高权重
    assert weights['AAPL'] > weights['MSFT']
    print(f"✓ 风险平价自定义预算测试通过: 权重={weights}")


def test_min_variance():
    """测试最小方差"""
    optimization_input = OptimizationInput(
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        expected_returns={},
        covariance_matrix=[
            [0.04, 0.01, 0.02],
            [0.01, 0.03, 0.01],
            [0.02, 0.01, 0.05]
        ],
        constraints={}
    )
    
    weights = AdvancedOptimizers.min_variance(optimization_input)
    
    assert len(weights) == 3
    assert abs(sum(weights.values()) - 1.0) < 0.01
    assert all(w >= 0 for w in weights.values())
    # 最低方差资产应该有最高权重
    assert weights['MSFT'] > weights['GOOGL']  # MSFT方差0.03 < GOOGL方差0.05
    print(f"✓ 最小方差测试通过: 权重={weights}")


def test_max_sharpe():
    """测试最大夏普比率"""
    optimization_input = OptimizationInput(
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        expected_returns={'AAPL': 0.15, 'MSFT': 0.08, 'GOOGL': 0.12},
        covariance_matrix=[
            [0.04, 0.01, 0.02],
            [0.01, 0.03, 0.01],
            [0.02, 0.01, 0.05]
        ],
        constraints={}
    )
    
    weights = AdvancedOptimizers.max_sharpe(optimization_input, risk_free_rate=0.02)
    
    assert len(weights) == 3
    assert abs(sum(weights.values()) - 1.0) < 0.01
    assert all(w >= 0 for w in weights.values())
    # 高收益/低方差资产应该有更高权重
    # AAPL: (0.15-0.02)/0.04 = 3.25
    # MSFT: (0.08-0.02)/0.03 = 2.0
    # GOOGL: (0.12-0.02)/0.05 = 2.0
    assert weights['AAPL'] > weights['MSFT']
    print(f"✓ 最大夏普比率测试通过: 权重={weights}")


def test_max_diversification():
    """测试最大分散化"""
    optimization_input = OptimizationInput(
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        expected_returns={},
        covariance_matrix=[
            [0.04, 0.001, 0.002],  # AAPL与其他资产低相关
            [0.001, 0.03, 0.015],  # MSFT与GOOGL高相关
            [0.002, 0.015, 0.05]
        ],
        constraints={}
    )
    
    weights = AdvancedOptimizers.max_diversification(optimization_input)
    
    assert len(weights) == 3
    assert abs(sum(weights.values()) - 1.0) < 0.01
    assert all(w >= 0 for w in weights.values())
    # 低相关性资产应该有更高权重
    assert weights['AAPL'] > weights['GOOGL']  # AAPL相关性更低
    print(f"✓ 最大分散化测试通过: 权重={weights}")


def test_single_asset():
    """测试单一资产边界情况"""
    optimization_input = OptimizationInput(
        symbols=['AAPL'],
        expected_returns={'AAPL': 0.10},
        covariance_matrix=[[0.04]],
        constraints={}
    )
    
    weights = AdvancedOptimizers.min_variance(optimization_input)
    assert weights == {'AAPL': 1.0}
    
    weights = AdvancedOptimizers.max_sharpe(optimization_input)
    assert weights == {'AAPL': 1.0}
    
    print("✓ 单一资产边界测试通过")


if __name__ == "__main__":
    print("=== 高级组合优化模块测试 ===\n")
    
    test_black_litterman_with_views()
    test_black_litterman_no_views()
    test_hierarchical_risk_parity()
    test_critical_line_algorithm()
    test_risk_parity_equal_budget()
    test_risk_parity_custom_budget()
    test_min_variance()
    test_max_sharpe()
    test_max_diversification()
    test_single_asset()
    
    print("\n=== 全部10项测试通过！ ===")
