"""WebSocket连接处理器

[应用层 - Dashboard组件] 从dashboard.py拆分而来
状态: ✅ 第四轮迁移 - WebSocket实时通信
来源: dashboard_bak.py 相关方法
迁移时间: 2025-11-28

包含功能:
- WebSocket连接管理
- 消息处理
- 订阅管理
- 数据推送
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Set

logger = logging.getLogger('DeepSeekQuant.App.Dashboard.WebSocket')


class WebSocketHandler:
    """WebSocket处理器 - 管理实时通信"""

    def __init__(self, quality_monitor: Any) -> None:
        """初始化WebSocket处理器
        
        Args:
            quality_monitor: 质量监控器实例
        """
        self._qm = quality_monitor
        self.connections: Set = set()

    def handle_connection(self, ws):
        """处理WebSocket连接
        
        Args:
            ws: WebSocket连接对象
        """
        try:
            self.connections.add(ws)
            logger.info(f"New WebSocket connection. Total: {len(self.connections)}")
            
            while True:
                message = ws.receive()
                if message is None:
                    break
                self.handle_message(ws, message)
                    
        except Exception as e:
            logger.error(f"WebSocket连接错误: {e}")
        finally:
            self.connections.discard(ws)
            logger.info(f"WebSocket connection closed. Total: {len(self.connections)}")

    def handle_message(self, ws, message: str):
        """处理WebSocket消息
        
        Args:
            ws: WebSocket连接对象
            message: 消息内容
        """
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'subscribe':
                self.handle_subscription(ws, data.get('channels', []))
            elif msg_type == 'unsubscribe':
                self.handle_unsubscription(ws, data.get('channels', []))
            elif msg_type == 'request_data':
                self.send_requested_data(ws, data.get('data_type'))
            else:
                logger.warning(f"未知消息类型: {msg_type}")
                    
        except json.JSONDecodeError as e:
            logger.error(f"消息解析失败: {e}")
            ws.send(json.dumps({'error': 'Invalid JSON'}))
        except Exception as e:
            logger.error(f"消息处理失败: {e}")
            ws.send(json.dumps({'error': str(e)}))

    def handle_subscription(self, ws, channels: List[str]):
        """处理订阅请求
        
        Args:
            ws: WebSocket连接对象
            channels: 订阅频道列表
        """
        logger.info(f"订阅频道: {channels}")
        ws.send(json.dumps({'type': 'subscribed', 'channels': channels}))

    def handle_unsubscription(self, ws, channels: List[str]):
        """处理取消订阅请求
        
        Args:
            ws: WebSocket连接对象
            channels: 取消订阅频道列表
        """
        logger.info(f"取消订阅频道: {channels}")
        ws.send(json.dumps({'type': 'unsubscribed', 'channels': channels}))

    def send_requested_data(self, ws, data_type: str):
        """发送请求的数据
        
        Args:
            ws: WebSocket连接对象
            data_type: 数据类型
        """
        try:
            if data_type == 'quality':
                data = self._qm.get_quality_history(24)
            elif data_type == 'alerts':
                data = self._qm.get_alert_history(24)
            elif data_type == 'performance':
                data = self._qm.get_performance_statistics()
            else:
                data = {}
                
            ws.send(json.dumps({
                'type': 'data',
                'data_type': data_type,
                'data': data
            }))
        except Exception as e:
            logger.error(f"发送数据失败: {e}")
            ws.send(json.dumps({'error': str(e)}))

    def broadcast(self, message: dict):
        """广播消息到所有连接
        
        Args:
            message: 要广播的消息
        """
        json_message = json.dumps(message)
        for ws in self.connections.copy():
            try:
                ws.send(json_message)
            except Exception as e:
                logger.error(f"广播失败: {e}")
                self.connections.discard(ws)
