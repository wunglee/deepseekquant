from core_bak_refactored.app.data.api.exporter import export_quality


class DummyMonitor:
    def export_monitoring_data(self, path, fmt): return True


def test_export_quality():
    ok = export_quality(DummyMonitor(), 'x.json', 'json')
    assert ok is True
