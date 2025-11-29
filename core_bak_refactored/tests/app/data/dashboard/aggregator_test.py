from core_bak_refactored.app.data.dashboard.aggregator import aggregate


class DummyMonitor:
    def get_quality_history(self, hours): return []
    def get_alert_history(self, hours): return []
    def get_performance_statistics(self): return {}


def test_aggregator():
    res = aggregate(DummyMonitor())
    assert 'quality' in res and 'alerts' in res and 'performance' in res
