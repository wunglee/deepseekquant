import os
from core_bak_refactored.app.data.dashboard.config_io import export_config, import_config


def test_config_io(tmp_path):
    path = os.path.join(tmp_path, 'cfg.json')
    ok = export_config({'a': 1}, path)
    assert ok is True
    cfg = import_config(path)
    assert cfg['a'] == 1
