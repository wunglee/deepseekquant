"""
熔断器组件 - 负责处理熔断逻辑
"""

import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import threading

@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    failure_threshold: int = 5
    recovery_timeout: int = 300
    half_open_max_requests: int = 3

@dataclass
class CircuitBreakerState:
    """熔断器状态"""
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    failure_count: int = 0
    last_failure_time: Optional[str] = None
    next_retry_time: Optional[str] = None
    consecutive_successes: int = 0

class CircuitBreaker:
    """熔断器实现"""

    def __init__(self, config: CircuitBreakerConfig, processor_name: str):
        self.config = config
        self.processor_name = processor_name
        self.state = CircuitBreakerState()
        self.lock = threading.RLock()

    # 在 circuit_breaker.py 中修复熔断器逻辑
    def allow_request(self) -> bool:
        """检查是否允许请求"""
        with self.lock:
            if self.state.state == "OPEN":
                # 检查是否应该尝试恢复
                if self.state.next_retry_time:
                    try:
                        next_retry_time = datetime.fromisoformat(self.state.next_retry_time)
                        if datetime.now() >= next_retry_time:
                            self.state.state = "HALF_OPEN"
                            self.state.consecutive_successes = 0
                            return True
                    except (ValueError, TypeError):
                        # 如果时间格式无效，也允许尝试
                        self.state.state = "HALF_OPEN"
                        self.state.consecutive_successes = 0
                        return True
                return False

            elif self.state.state == "HALF_OPEN":
                # 在半开状态下，检查是否达到最大请求数
                if self.state.consecutive_successes >= self.config.half_open_max_requests:
                    self.state.state = "CLOSED"
                    self.state.failure_count = 0
                return True

            return True  # CLOSED 状态始终允许请求

    def record_success(self):
        """记录成功"""
        with self.lock:
            if self.state.state == "HALF_OPEN":
                self.state.consecutive_successes += 1
                # 检查是否应该关闭熔断器
                if self.state.consecutive_successes >= self.config.half_open_max_requests:
                    self.state.state = "CLOSED"
                    self.state.failure_count = 0
            else:
                # 在CLOSED状态下，重置失败计数
                self.state.failure_count = 0

    def record_failure(self):
        """记录失败"""
        with self.lock:
            self.state.failure_count += 1
            self.state.last_failure_time = datetime.now().isoformat()

            if self.state.state == "CLOSED":
                if self.state.failure_count >= self.config.failure_threshold:
                    self.state.state = "OPEN"
                    next_retry = time.time() + self.config.recovery_timeout
                    self.state.next_retry_time = datetime.fromtimestamp(next_retry).isoformat()

            elif self.state.state == "HALF_OPEN":
                self.state.consecutive_successes = 0

    def get_status(self) -> dict:
        """获取状态"""
        with self.lock:
            return {
                'state': self.state.state,
                'failure_count': self.state.failure_count,
                'last_failure_time': self.state.last_failure_time,
                'next_retry_time': self.state.next_retry_time,
                'consecutive_successes': self.state.consecutive_successes
            }

    def update_config(self, new_config: CircuitBreakerConfig):
        """更新配置"""
        with self.lock:
            self.config = new_config