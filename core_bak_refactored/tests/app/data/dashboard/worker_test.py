from core_bak_refactored.app.data.dashboard.worker import UpdateWorker


class DummyMonitor:
    def get_quality_history(self, hours): return []
    def get_alert_history(self, hours): return []
    def get_performance_statistics(self): return {}


def test_worker_run_once():
    w = UpdateWorker()
    res = w.run_once(DummyMonitor())
    assert isinstance(res, dict)
