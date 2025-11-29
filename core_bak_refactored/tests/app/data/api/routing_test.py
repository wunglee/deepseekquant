from core_bak_refactored.app.data.api.routing import DataQualityAPIRouting


class DummyMonitor:
    def get_quality_history(self, hours): return []
    def get_alert_history(self, hours): return []
    def get_performance_statistics(self): return {}
    def export_monitoring_data(self, path, fmt): return True


def test_api_routing_app_exists():
    routing = DataQualityAPIRouting(DummyMonitor())
    app = routing.get_app()
    assert app is not None
