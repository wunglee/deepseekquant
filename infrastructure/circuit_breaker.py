"""
熔断器组件 - 负责处理熔断逻辑
"""

import threading
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd


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
    last_failure_time: Optional[pd.Timestamp] = None
    next_retry_time: Optional[pd.Timestamp] = None
    consecutive_successes: int = 0

class CircuitBreaker:
    """熔断器实现"""

    def __init__(self, config: CircuitBreakerConfig, processor_name: str):
        self.config = config
        self.processor_name = processor_name
        self.state = CircuitBreakerState()
        self.lock = threading.RLock()

    # 🔧 熔断器逻辑：统一使用 pd.Timestamp
    def allow_request(self) -> bool:
        """检查是否允许请求"""
        with self.lock:
            if self.state.state == "OPEN":
                # 检查是否应该尝试恢复
                if self.state.next_retry_time:
                    try:
                        # 直接比较 pd.Timestamp 对象
                        if pd.Timestamp.now() >= self.state.next_retry_time:
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
            # 🔧 直接存储 pd.Timestamp 对象，不转换为字符串
            self.state.last_failure_time = pd.Timestamp.now()

            if self.state.state == "CLOSED":
                if self.state.failure_count >= self.config.failure_threshold:
                    self.state.state = "OPEN"
                    next_retry = time.time() + self.config.recovery_timeout
                    # 🔧 使用 pd.Timestamp.fromtimestamp
                    self.state.next_retry_time = pd.Timestamp.fromtimestamp(next_retry)

            elif self.state.state == "HALF_OPEN":
                self.state.consecutive_successes = 0

    def get_status(self) -> dict:
        """获取状态"""
        with self.lock:
            return {
                'state': self.state.state,
                'failure_count': self.state.failure_count,
                # 🔧 转换为 ISO 格式字符串用于输出
                'last_failure_time': self.state.last_failure_time.isoformat() if self.state.last_failure_time else None,
                'next_retry_time': self.state.next_retry_time.isoformat() if self.state.next_retry_time else None,
                'consecutive_successes': self.state.consecutive_successes
            }

    def update_config(self, new_config: CircuitBreakerConfig):
        """更新配置"""
        with self.lock:
            self.config = new_config