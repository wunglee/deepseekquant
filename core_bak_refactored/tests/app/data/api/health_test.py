from core_bak_refactored.app.data.api.health import check_health


def test_check_health():
    res = check_health(None)
    assert res['status'] == 'healthy'
