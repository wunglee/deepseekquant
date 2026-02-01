"""
功能碎片：合成组合构造器
从 core/risk/backtest_framework.py 提取
状态：待整合到 core/portfolio 模块

职责：
- 测试组合构造（等权重、行业轮动、跨市场混合）
- 组合元数据管理
- 权重验证和归一化

迁移计划：
当 core_bak_refactored/core/portfolio 模块重构完成后，整合此文件到该模块

相关文件：
- 源文件：core/risk/backtest_framework.py (SyntheticPortfolio, SyntheticPortfolioBuilder)
- 调用者：core/risk/stress_test_validator.py (StressTestValidator)
"""

from dataclasses import dataclass, field
from typing import Dict, Any


# =============================================================================
# 数据模型（组合模块标准）
# =============================================================================

@dataclass
class SyntheticPortfolio:
    """
    合成组合定义（功能碎片数据模型）
    
    用于回测验证的标准化测试组合
    
    迁移计划：
    - 当 core/portfolio 重构完成后，可能需要与该模块的 Portfolio 类合并
    - 或作为 TestPortfolio 子类存在
    """
    portfolio_id: str
    name: str
    composition: Dict[str, float]  # {symbol/sector: weight}
    total_value: float = 1000000.0  # 100万基准
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """验证权重总和为1"""
        total_weight = sum(self.composition.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"权重总和必须为1.0，当前为{total_weight}")


# =============================================================================
# 组合构造器（组合模块业务逻辑）
# =============================================================================

class SyntheticPortfolioBuilder:
    """
    合成组合构造器（功能碎片业务逻辑）
    
    基于专家answer.md第5轮1.3节指导，构造3种典型组合：
    1. 沪深300等权重组合
    2. 行业轮动组合（金融30%+消费25%+科技20%+其他25%）
    3. A+H混合组合（A股70%+港股30%）
    
    迁移计划：
    - 当 core/portfolio 重构时，整合到 PortfolioFactory 或 PortfolioBuilder 类
    - 可能需要扩展支持更多组合类型（如市值加权、风险平价等）
    """
    
    @staticmethod
    def build_csi300_equal_weight() -> SyntheticPortfolio:
        """构造沪深300等权重组合"""
        return SyntheticPortfolio(
            portfolio_id='CSI300_EQ',
            name='沪深300等权重组合',
            composition={'000300.SH': 1.0},  # 简化：直接使用指数
            metadata={'type': 'index_replication', 'market': 'CN'}
        )
    
    @staticmethod
    def build_sector_rotation() -> SyntheticPortfolio:
        """构造行业轮动组合"""
        return SyntheticPortfolio(
            portfolio_id='SECTOR_ROT',
            name='行业轮动组合',
            composition={
                'finance_index': 0.30,    # 金融30%
                'consumer_index': 0.25,   # 消费25%
                'tech_index': 0.20,       # 科技20%
                'other_index': 0.25       # 其他25%
            },
            metadata={'type': 'sector_rotation', 'market': 'CN'}
        )
    
    @staticmethod
    def build_ah_hybrid() -> SyntheticPortfolio:
        """构造A+H混合组合"""
        return SyntheticPortfolio(
            portfolio_id='AH_HYBRID',
            name='A+H混合组合',
            composition={
                '000300.SH': 0.70,  # A股70%（沪深300）
                'HSI': 0.30         # 港股30%（恒生指数）
            },
            metadata={'type': 'cross_border', 'markets': ['CN', 'HK']}
        )
    
    @classmethod
    def build_by_type(cls, portfolio_type: str) -> SyntheticPortfolio:
        """
        根据类型构造组合（工厂方法）
        
        Args:
            portfolio_type: 组合类型 ('csi300'/'sector_rotation'/'ah_hybrid')
        
        Returns:
            SyntheticPortfolio: 对应的组合对象
        
        Raises:
            ValueError: 如果portfolio_type不支持
        """
        builders = {
            'csi300': cls.build_csi300_equal_weight,
            'sector_rotation': cls.build_sector_rotation,
            'ah_hybrid': cls.build_ah_hybrid
        }
        
        if portfolio_type not in builders:
            raise ValueError(
                f"不支持的组合类型: {portfolio_type}. "
                f"可选: {list(builders.keys())}"
            )
        
        return builders[portfolio_type]()


# =============================================================================
# 迁移检查清单
# =============================================================================

"""
功能碎片迁移检查清单（core/portfolio模块重构时使用）

□ 1. 与现有Portfolio类整合
    □ 确认 SyntheticPortfolio 与 core/portfolio/portfolio.py 的 Portfolio 类关系
    □ 决定是合并、继承还是保持独立
    
□ 2. 扩展组合类型
    □ 市值加权组合
    □ 风险平价组合
    □ 动量组合
    □ 价值组合
    
□ 3. 权重生成策略
    □ 等权重策略
    □ 市值加权策略
    □ 优化权重策略（MVO/Black-Litterman）
    □ 风险平价策略
    
□ 4. 组合约束支持
    □ 最大/最小权重约束
    □ 行业/国家集中度约束
    □ 杠杆约束
    
□ 5. 再平衡策略
    □ 定期再平衡
    □ 阈值触发再平衡
    □ 税收优化再平衡
    
□ 6. 测试覆盖
    □ 权重验证测试
    □ 组合构造测试
    □ 约束满足测试
    
□ 7. 调用者更新
    □ 更新 core/risk/stress_test_validator.py 的导入路径
    □ 更新 core/backtest 模块的调用
    □ 更新文档和示例
"""
