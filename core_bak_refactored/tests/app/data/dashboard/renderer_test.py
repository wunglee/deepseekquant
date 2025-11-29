from core_bak_refactored.app.data.dashboard.renderer import render_dashboard


def test_render_dashboard():
    html = render_dashboard()
    assert '<html>' in html
