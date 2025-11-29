class WebSocketManager:
    """WebSocket管理（职责单一：占位）"""

    def __init__(self) -> None:
        self._clients = []

    def broadcast(self, message: str) -> None:
        for c in list(self._clients):
            try:
                c.send(message)
            except Exception:
                pass
