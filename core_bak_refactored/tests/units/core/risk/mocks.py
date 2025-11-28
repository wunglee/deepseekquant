from dataclasses import dataclass
from typing import Dict
import random

# Mocks moved from core/risk/stress_test_validator.py for tests-only usage

@dataclass
class HistoricalEvent:
    event_id: str
    name: str
    period: tuple
    expected_decline: float
    scenario_params: Dict

class MockHistoricalDataSource:
    def get_event_returns(self, event: HistoricalEvent, asset_id: str) -> float:
        noise = random.uniform(-0.1, 0.1) * event.expected_decline
        return event.expected_decline + noise

class MockPortfolioBuilder:
    def build_test_portfolio(self, portfolio_type: str) -> Dict[str, float]:
        if portfolio_type == 'csi300':
            return {'000300.SH': 1.0}
        elif portfolio_type == 'sector_rotation':
            return {'finance': 0.30, 'consumer': 0.25, 'tech': 0.20, 'other': 0.25}
        elif portfolio_type == 'ah_hybrid':
            return {'000300.SH': 0.70, 'HSI': 0.30}
        else:
            return {'000300.SH': 1.0}
