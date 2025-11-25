"""
Synthetic portfolio fragments for backtest validation.
Belongs to portfolio module; migrated from risk/backtest_framework.py.
Mark: To be merged when portfolio module is fully implemented.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class SyntheticPortfolio:
    """
    Standardized synthetic portfolio for backtest validation.
    """
    portfolio_id: str
    name: str
    composition: Dict[str, float]  # {index_id: weight}
    total_value: float = 1000000.0  # baseline 1,000,000
    metadata: Dict[str, Any] = field(default_factory=dict)


class SyntheticPortfolioBuilder:
    """
    Build typical synthetic portfolios used by event-window backtests.
    1) CSI300 equal-weight
    2) Sector rotation (Finance 30% + Consumer 25% + Tech 20% + Other 25%)
    3) A+H hybrid (A-share 70% + H-share 30%)
    """

    @staticmethod
    def build_csi300_equal_weight() -> SyntheticPortfolio:
        """Construct CSI300 equal-weight portfolio."""
        return SyntheticPortfolio(
            portfolio_id='CSI300_EQ',
            name='沪深300等权重组合',
            composition={'000300.SH': 1.0},  # simplified index replication
            metadata={'type': 'index_replication', 'market': 'CN'}
        )

    @staticmethod
    def build_sector_rotation() -> SyntheticPortfolio:
        """Construct sector rotation portfolio."""
        return SyntheticPortfolio(
            portfolio_id='SECTOR_ROT',
            name='Sector Rotation',
            composition={
                'finance_index': 0.30,
                'consumer_index': 0.25,
                'tech_index': 0.20,
                'other_index': 0.25
            },
            metadata={'type': 'sector_rotation', 'market': 'CN'}
        )

    @staticmethod
    def build_ah_hybrid() -> SyntheticPortfolio:
        """Construct A+H hybrid portfolio."""
        return SyntheticPortfolio(
            portfolio_id='AH_HYBRID',
            name='A+H Hybrid',
            composition={
                '000300.SH': 0.70,
                'HSI': 0.30
            },
            metadata={'type': 'cross_border', 'markets': ['CN', 'HK']}
        )
