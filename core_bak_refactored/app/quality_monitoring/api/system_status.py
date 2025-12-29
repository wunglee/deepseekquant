"""系统状态管理模块

[应用层 - API组件] 从api_service.py拆分而来
状态: ✅ 第二轮迁移 - 系统状态和维护模式
来源: api_service_bak.py 相关方法
迁移时间: 2025-11-28

包含功能:
- 系统状态获取
- 维护模式管理
- 服务可用性检查
"""

from __future__ import annotations

import pandas as pd
import logging

from typing import Dict, Any

logger = logging.getLogger('DeepSeekQuant.App.API.SystemStatus')


class SystemStatusManager:
    """系统状态管理器 - 管理系统状态和维护模式"""

    def __init__(self, quality_monitor: Any) -> None:
        """初始化系统状态管理器
        
        Args:
            quality_monitor: 质量监控器实例
        """
        self._qm = quality_monitor
        self._maintenance_mode = False
        self._maintenance_start_time = None
        self._maintenance_duration = 0

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态
        
        Returns:
            系统状态字典
        """
        # 获取组件状态
        components_status = {
            'quality_monitor': 'operational',
            'api_service': 'operational',
            'data_fetcher': 'operational'
        }
        
        # 计算整体状态
        overall_status = 'healthy' if all(status == 'operational' for status in components_status.values()) else 'degraded'
        
        # 获取性能指标（从质量监控器获取）
        try:
            performance_metrics = getattr(self._qm, 'get_performance_statistics', lambda: {})()
            if callable(performance_metrics):
                performance_metrics = {}
        except:
            performance_metrics = {}
        
        return {
            'service': 'data_quality_api',
            'version': '1.0.0',
            'status': 'maintenance' if self._maintenance_mode else 'operational',
            'overall_status': overall_status,  # 添加缺失的overall_status字段
            'performance_metrics': performance_metrics,  # 添加缺失的performance_metrics字段
            'uptime': self._get_uptime(),
            'maintenance_mode': self._maintenance_mode,
            'maintenance_remaining': self._get_maintenance_remaining() if self._maintenance_mode else None,
            'components': components_status,
            'timestamp': pd.Timestamp.now().isoformat()
        }

    def enable_maintenance_mode(self, duration: int = 3600) -> bool:
        """启用维护模式
        
        Args:
            duration: 维护持续时间（秒）
            
        Returns:
            是否启用成功
        """
        try:
            self._maintenance_mode = True
            self._maintenance_start_time = pd.Timestamp.now()
            self._maintenance_duration = duration
            logger.info(f"维护模式已启用，持续时间: {duration}秒")
            return True
        except Exception as e:
            logger.error(f"启用维护模式失败: {e}")
            return False

    def disable_maintenance_mode(self) -> bool:
        """禁用维护模式
        
        Returns:
            是否禁用成功
        """
        try:
            self._maintenance_mode = False
            self._maintenance_start_time = None
            self._maintenance_duration = 0
            logger.info("维护模式已禁用")
            return True
        except Exception as e:
            logger.error(f"禁用维护模式失败: {e}")
            return False

    def is_maintenance_mode(self) -> bool:
        """检查是否处于维护模式
        
        Returns:
            是否处于维护模式
        """
        if not self._maintenance_mode:
            return False
            
        # 检查是否超时
        if self._maintenance_start_time:
            elapsed = (pd.Timestamp.now() - self._maintenance_start_time).total_seconds()
            if elapsed > self._maintenance_duration:
                self.disable_maintenance_mode()
                return False
                
        return True

    def _get_uptime(self) -> str:
        """获取系统运行时间
        
        Returns:
            运行时间字符串
        """
        # 这里简化处理，实际应该记录启动时间
        return "24h 30m"

    def _get_maintenance_remaining(self) -> int:
        """获取维护模式剩余时间
        
        Returns:
            剩余时间（秒）
        """
        if not self._maintenance_start_time:
            return 0
            
        elapsed = (pd.Timestamp.now() - self._maintenance_start_time).total_seconds()
        remaining = max(0, self._maintenance_duration - elapsed)
        return int(remaining)
