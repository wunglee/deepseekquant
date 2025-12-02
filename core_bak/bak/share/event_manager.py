"""
事件管理器（共享模块）

职责：提供标准化的事件管理接口
用途：统一管理事件生命周期、状态跟踪和通知
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import asyncio

logger = logging.getLogger('DeepSeekQuant.Core.Share.EventManager')


class EventStatus(str, Enum):
    """事件状态枚举"""
    PENDING = 'pending'      # 待处理
    PROCESSING = 'processing'  # 处理中
    COMPLETED = 'completed'   # 已完成
    FAILED = 'failed'        # 失败
    CANCELLED = 'cancelled'   # 已取消


class EventType(str, Enum):
    """事件类型枚举"""
    DATA_QUALITY_CHECK = 'data_quality_check'
    ALERT_TRIGGER = 'alert_trigger'
    SYSTEM_HEALTH_CHECK = 'system_health_check'
    DATA_SOURCE_SWITCH = 'data_source_switch'
    PERFORMANCE_ANALYSIS = 'performance_analysis'


@dataclass
class Event:
    """事件数据类"""
    event_id: str
    event_type: EventType
    status: EventStatus = EventStatus.PENDING
    priority: int = 1  # 1-最高优先级
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    data: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'status': self.status.value,
            'priority': self.priority,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'data': self.data,
            'result': self.result,
            'error_message': self.error_message,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }


class EventManager:
    """
    事件管理器
    
    职责：提供标准化的事件管理接口
    """
    
    def __init__(self):
        self._events: Dict[str, Event] = {}
        self._event_handlers: Dict[EventType, List[Callable]] = {}
        self._event_queue: List[Event] = []  # 事件队列
        self._max_concurrent_events: int = 10  # 最大并发事件数
        self._running_events: int = 0  # 正在运行的事件数
    
    def register_handler(self, event_type: EventType, handler: Callable) -> None:
        """
        注册事件处理器
        
        Args:
            event_type: 事件类型
            handler: 事件处理器函数
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        logger.info(f"事件处理器已注册: {event_type.value}")
    
    def create_event(self, 
                    event_id: str,
                    event_type: EventType,
                    data: Optional[Dict[str, Any]] = None,
                    priority: int = 1,
                    max_retries: int = 3) -> Event:
        """
        创建事件
        
        Args:
            event_id: 事件ID
            event_type: 事件类型
            data: 事件数据
            priority: 优先级
            max_retries: 最大重试次数
            
        Returns:
            创建的事件
        """
        event = Event(
            event_id=event_id,
            event_type=event_type,
            priority=priority,
            data=data or {},
            max_retries=max_retries
        )
        
        self._events[event_id] = event
        self._event_queue.append(event)
        
        # 按优先级排序（数字越小优先级越高）
        self._event_queue.sort(key=lambda e: e.priority)
        
        logger.info(f"事件已创建: {event_id} ({event_type.value}), 优先级: {priority}")
        return event
    
    def get_event(self, event_id: str) -> Optional[Event]:
        """
        获取事件
        
        Args:
            event_id: 事件ID
            
        Returns:
            事件对象或None
        """
        return self._events.get(event_id)
    
    def update_event_status(self, 
                          event_id: str, 
                          status: EventStatus,
                          result: Optional[Dict[str, Any]] = None,
                          error_message: Optional[str] = None) -> None:
        """
        更新事件状态
        
        Args:
            event_id: 事件ID
            status: 新状态
            result: 事件结果
            error_message: 错误信息
        """
        if event_id in self._events:
            event = self._events[event_id]
            event.status = status
            
            if status == EventStatus.PROCESSING:
                event.started_at = datetime.now()
            elif status in [EventStatus.COMPLETED, EventStatus.FAILED, EventStatus.CANCELLED]:
                event.completed_at = datetime.now()
            
            if result is not None:
                event.result = result
            if error_message is not None:
                event.error_message = error_message
            
            logger.info(f"事件状态已更新: {event_id} -> {status.value}")
        else:
            logger.warning(f"未知事件: {event_id}")
    
    async def process_events(self) -> None:
        """
        处理事件队列中的事件
        """
        while self._event_queue and self._running_events < self._max_concurrent_events:
            # 获取下一个待处理事件
            event = self._event_queue.pop(0)
            
            if event.status != EventStatus.PENDING:
                continue
            
            # 更新事件状态为处理中
            self.update_event_status(event.event_id, EventStatus.PROCESSING)
            self._running_events += 1
            
            # 异步处理事件
            asyncio.create_task(self._process_event_async(event))
    
    async def _process_event_async(self, event: Event) -> None:
        """
        异步处理单个事件
        
        Args:
            event: 事件对象
        """
        try:
            # 获取事件处理器
            handlers = self._event_handlers.get(event.event_type, [])
            if not handlers:
                logger.warning(f"无处理器处理事件类型: {event.event_type.value}")
                self.update_event_status(
                    event.event_id, 
                    EventStatus.FAILED, 
                    error_message=f"无处理器处理事件类型: {event.event_type.value}"
                )
                return
            
            # 执行所有处理器
            results = []
            for handler in handlers:
                try:
                    result = await handler(event) if asyncio.iscoroutinefunction(handler) else handler(event)
                    results.append(result)
                except Exception as e:
                    logger.error(f"事件处理器执行失败: {e}", exc_info=True)
                    # 继续执行其他处理器
            
            # 更新事件状态为完成
            self.update_event_status(
                event.event_id, 
                EventStatus.COMPLETED, 
                result={'handler_results': results}
            )
            
        except Exception as e:
            logger.error(f"事件处理失败: {e}", exc_info=True)
            
            # 检查是否需要重试
            if event.retry_count < event.max_retries:
                event.retry_count += 1
                logger.info(f"事件将重试 ({event.retry_count}/{event.max_retries}): {event.event_id}")
                self._event_queue.append(event)
                # 重新排序
                self._event_queue.sort(key=lambda e: e.priority)
            else:
                # 更新事件状态为失败
                self.update_event_status(
                    event.event_id, 
                    EventStatus.FAILED, 
                    error_message=str(e)
                )
        
        finally:
            self._running_events -= 1
    
    def cancel_event(self, event_id: str) -> bool:
        """
        取消事件
        
        Args:
            event_id: 事件ID
            
        Returns:
            是否成功取消
        """
        if event_id in self._events:
            event = self._events[event_id]
            if event.status in [EventStatus.PENDING, EventStatus.PROCESSING]:
                self.update_event_status(event_id, EventStatus.CANCELLED)
                
                # 从队列中移除（如果还在队列中）
                self._event_queue = [e for e in self._event_queue if e.event_id != event_id]
                
                logger.info(f"事件已取消: {event_id}")
                return True
        return False
    
    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        """
        根据类型获取事件列表
        
        Args:
            event_type: 事件类型
            
        Returns:
            事件列表
        """
        return [event for event in self._events.values() if event.event_type == event_type]
    
    def get_events_by_status(self, status: EventStatus) -> List[Event]:
        """
        根据状态获取事件列表
        
        Args:
            status: 事件状态
            
        Returns:
            事件列表
        """
        return [event for event in self._events.values() if event.status == status]
    
    def get_recent_events(self, limit: int = 50) -> List[Event]:
        """
        获取最近的事件
        
        Args:
            limit: 返回事件数量限制
            
        Returns:
            最近的事件列表
        """
        events = list(self._events.values())
        events.sort(key=lambda e: e.created_at, reverse=True)
        return events[:limit]
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """
        获取事件统计信息
        
        Returns:
            事件统计字典
        """
        total_events = len(self._events)
        status_counts = {}
        type_counts = {}
        
        for event in self._events.values():
            # 状态统计
            status = event.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # 类型统计
            event_type = event.event_type.value
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
        
        return {
            'total_events': total_events,
            'status_distribution': status_counts,
            'type_distribution': type_counts,
            'pending_events': len(self._event_queue),
            'running_events': self._running_events,
            'max_concurrent_events': self._max_concurrent_events
        }
    
    def cleanup_completed_events(self, older_than_hours: int = 24) -> int:
        """
        清理已完成的旧事件
        
        Args:
            older_than_hours: 清理多少小时前的完成事件
            
        Returns:
            清理的事件数量
        """
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        cleaned_count = 0
        
        # 找到需要清理的事件
        events_to_remove = []
        for event_id, event in self._events.items():
            if (event.status in [EventStatus.COMPLETED, EventStatus.FAILED, EventStatus.CANCELLED] and
                event.completed_at and event.completed_at < cutoff_time):
                events_to_remove.append(event_id)
        
        # 清理事件
        for event_id in events_to_remove:
            del self._events[event_id]
            cleaned_count += 1
        
        logger.info(f"已清理 {cleaned_count} 个旧事件")
        return cleaned_count