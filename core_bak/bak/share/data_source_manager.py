"""
数据源管理器（共享模块）

职责：提供标准化的数据源管理接口
用途：统一管理数据源配置、评分、切换和应急处理
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger('DeepSeekQuant.Core.Share.DataSourceManager')


@dataclass
class DataSourceInfo:
    """数据源信息数据类"""
    name: str
    type: str  # 'primary', 'backup', 'monitoring'
    url: str
    enabled: bool = True
    priority: int = 1  # 1-最高优先级
    reliability_score: int = 100  # 0-100分
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'type': self.type,
            'url': self.url,
            'enabled': self.enabled,
            'priority': self.priority,
            'reliability_score': self.reliability_score,
            'last_updated': self.last_updated.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class SourceSwitchRecommendation:
    """数据源切换建议"""
    switch_recommended: bool
    primary_source: str
    backup_source: str
    primary_score: int
    backup_score: int
    auto_switch_allowed: bool
    manual_approval_required: bool
    reason: str = ""


class DataSourceManager:
    """
    数据源管理器
    
    职责：提供标准化的数据源管理接口
    """
    
    def __init__(self):
        self._sources: Dict[str, DataSourceInfo] = {}
        self._source_ratings: Dict[str, int] = {}  # 数据源评分
        self._emergency_cache: Dict[str, Any] = {}  # 应急缓存
        self._cache_timestamps: Dict[str, datetime] = {}
    
    def register_source(self, source_info: DataSourceInfo) -> None:
        """
        注册数据源
        
        Args:
            source_info: 数据源信息
        """
        self._sources[source_info.name] = source_info
        self._source_ratings[source_info.name] = source_info.reliability_score
        logger.info(f"数据源已注册: {source_info.name} (类型: {source_info.type})")
    
    def get_source(self, name: str) -> Optional[DataSourceInfo]:
        """
        获取数据源信息
        
        Args:
            name: 数据源名称
            
        Returns:
            数据源信息或None
        """
        return self._sources.get(name)
    
    def get_sources_by_type(self, source_type: str) -> List[DataSourceInfo]:
        """
        根据类型获取数据源列表
        
        Args:
            source_type: 数据源类型
            
        Returns:
            数据源列表
        """
        return [source for source in self._sources.values() if source.type == source_type]
    
    def update_source_rating(self, source_name: str, score: int, reason: str = "") -> None:
        """
        更新数据源评分
        
        Args:
            source_name: 数据源名称
            score: 新评分 (0-100)
            reason: 评分变更原因
        """
        if source_name in self._source_ratings:
            old_score = self._source_ratings[source_name]
            self._source_ratings[source_name] = max(0, min(100, score))
            
            # 同步更新数据源信息
            if source_name in self._sources:
                self._sources[source_name].reliability_score = self._source_ratings[source_name]
                self._sources[source_name].last_updated = datetime.now()
            
            logger.info(f"数据源评分已更新: {source_name} {old_score} -> {score} ({reason})")
        else:
            logger.warning(f"未知数据源: {source_name}")
    
    def apply_penalty(self, source_name: str, penalty: int, reason: str = "") -> None:
        """
        对数据源应用惩罚
        
        Args:
            source_name: 数据源名称
            penalty: 惩罚分数
            reason: 惩罚原因
        """
        current_score = self._source_ratings.get(source_name, 100)
        new_score = max(0, current_score - penalty)
        self.update_source_rating(source_name, new_score, f"惩罚: {reason} (-{penalty}分)")
    
    def should_pause_source(self, source_name: str) -> bool:
        """
        检查数据源是否应该暂停使用
        
        Args:
            source_name: 数据源名称
            
        Returns:
            是否应该暂停
        """
        return self._source_ratings.get(source_name, 100) <= 60
    
    def get_source_switch_recommendation(self, 
                                      primary_source: str, 
                                      backup_source: str) -> SourceSwitchRecommendation:
        """
        获取数据源切换建议
        
        Args:
            primary_source: 主数据源名称
            backup_source: 备用数据源名称
            
        Returns:
            切换建议信息
        """
        primary_rating = self._source_ratings.get(primary_source, 100)
        backup_rating = self._source_ratings.get(backup_source, 100)
        
        return SourceSwitchRecommendation(
            switch_recommended=primary_rating <= 60 and backup_rating > 70,
            primary_source=primary_source,
            backup_source=backup_source,
            primary_score=primary_rating,
            backup_score=backup_rating,
            auto_switch_allowed=primary_rating <= 50,  # 严重危险可自动切换
            manual_approval_required=50 < primary_rating <= 60,  # 一般危险需人工确认
            reason=f"主数据源评分{primary_rating}，备用数据源评分{backup_rating}"
        )
    
    def cache_emergency_data(self, source_name: str, data: Any) -> None:
        """
        缓存应急数据
        
        Args:
            source_name: 数据源名称
            data: 数据
        """
        self._emergency_cache[source_name] = data
        self._cache_timestamps[source_name] = datetime.now()
        logger.info(f"已缓存数据源{source_name}的应急数据")
    
    def get_emergency_fallback_data(self, source_name: str, max_age_hours: int = 24) -> Optional[Any]:
        """
        获取应急回退数据
        
        Args:
            source_name: 数据源名称
            max_age_hours: 最大缓存年龄（小时）
            
        Returns:
            缓存数据或None
        """
        if source_name not in self._emergency_cache:
            logger.warning(f"数据源{source_name}无应急缓存")
            return None
        
        cache_age = (datetime.now() - self._cache_timestamps[source_name]).total_seconds() / 3600
        if cache_age > max_age_hours:
            logger.warning(f"数据源{source_name}缓存已过期（{cache_age:.1f}小时 > {max_age_hours}小时）")
            return None
        
        logger.info(f"使用数据源{source_name}的应急缓存（年龄：{cache_age:.1f}小时）")
        return self._emergency_cache[source_name]
    
    def get_source_health_summary(self,
                                primary_source: Optional[str] = None,
                                backup_source: Optional[str] = None,
                                monitoring_sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        获取数据源健康度摘要
        
        Args:
            primary_source: 主数据源名称
            backup_source: 备用数据源名称
            monitoring_sources: 监控数据源列表
            
        Returns:
            健康度摘要字典
        """
        def _get_level(score: int) -> str:
            if score >= 90:
                return '优秀'
            elif score >= 70:
                return '良好'
            elif score >= 60:
                return '警告'
            else:
                return '危险'
        
        summary: Dict[str, Any] = {}
        
        # 1. 主数据源
        if primary_source and primary_source in self._source_ratings:
            score = self._source_ratings[primary_source]
            summary['primary_source'] = {
                'name': primary_source, 
                'score': score, 
                'level': _get_level(score)
            }
        
        # 2. 备用数据源
        if backup_source and backup_source in self._source_ratings:
            score = self._source_ratings[backup_source]
            summary['backup_source'] = {
                'name': backup_source, 
                'score': score, 
                'level': _get_level(score)
            }
        
        # 3. 监控数据源
        if monitoring_sources:
            summary['monitoring_sources'] = [
                {
                    'name': src, 
                    'score': self._source_ratings[src], 
                    'level': _get_level(self._source_ratings[src])
                }
                for src in monitoring_sources if src in self._source_ratings
            ]
        
        # 4. 危险档位数据源（评分≤60）
        dangerous = [
            {
                'name': source, 
                'score': score, 
                'level': _get_level(score), 
                'action': '暂停参与新回测 + 显式披露'
            }
            for source, score in self._source_ratings.items() if score <= 60
        ]
        if dangerous:
            summary['dangerous_sources'] = dangerous
        
        # 5. 所有数据源总览
        summary['all_sources'] = {
            source: {
                'score': score, 
                'level': _get_level(score)
            } 
            for source, score in self._source_ratings.items()
        }
        
        return summary
    
    def get_all_sources(self) -> List[DataSourceInfo]:
        """
        获取所有数据源
        
        Returns:
            所有数据源列表
        """
        return list(self._sources.values())
    
    def disable_source(self, source_name: str) -> bool:
        """
        禁用数据源
        
        Args:
            source_name: 数据源名称
            
        Returns:
            是否成功禁用
        """
        if source_name in self._sources:
            self._sources[source_name].enabled = False
            self._sources[source_name].last_updated = datetime.now()
            logger.info(f"数据源已禁用: {source_name}")
            return True
        return False
    
    def enable_source(self, source_name: str) -> bool:
        """
        启用数据源
        
        Args:
            source_name: 数据源名称
            
        Returns:
            是否成功启用
        """
        if source_name in self._sources:
            self._sources[source_name].enabled = True
            self._sources[source_name].last_updated = datetime.now()
            logger.info(f"数据源已启用: {source_name}")
            return True
        return False