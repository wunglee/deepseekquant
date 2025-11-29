from core_bak_refactored.app.data.api.controllers import DataQualityControllers


class DummyMonitor:
    def get_quality_history(self, hours): return [{'timestamp': 't', 'overall_score': 0.9}]
    def get_alert_history(self, hours): return []


def test_controllers_quality_current():
    c = DataQualityControllers(DummyMonitor())
    res = c.get_quality_current(12)
    assert 'data' in res and 'alerts' in res
