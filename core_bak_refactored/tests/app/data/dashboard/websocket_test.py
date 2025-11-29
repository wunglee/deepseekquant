from core_bak_refactored.app.data.dashboard.websocket import WebSocketManager


def test_websocket_manager():
    ws = WebSocketManager()
    ws.broadcast('msg')
    assert True
