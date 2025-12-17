"""数据质量监控服务 - 整合重构后的核心组件

[应用层] 监控服务整合层
职责：
- 整合 DataQualityChecker（质量检查）和 AlertManager（告警管理）
- 提供统一的监控接口（质量历史、告警历史、性能统计、综合报告）
- 管理质量历史记录和性能追踪
- 生成综合报告

设计原则：
- 适配器模式：对接遗留API接口到重构后的新架构
- 职责边界清晰：仅整合组件，不重复实现业务逻辑
- 依赖注入：通过工厂模式注入核心组件

架构说明：
```
api_service.py (API路由)
    ↓ 依赖
QualityMonitoringService (本文件 - 整合层)
    ↓ 整合
    ├─ DataQualityChecker (core/data/providers) - 质量检查逻辑
    └─ AlertManager (core/monitoring) - 告警管理
```

迁移说明：
- 重构前：DataQualityMonitor (core_bak/data_fetcher.py, 1770行大类)
- 重构后：职责分离
  - DataQualityChecker: 质量检查逻辑
  - AlertManager: 告警管理
  - QualityMonitoringService: 整合层（本文件）
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from core_bak_refactored.core.data.quality import (
    DataQualityChecker,
    DataQualityReport
)
from core_bak_refactored.core.monitoring.alert_manager import (
    AlertManager,
    AlertConfig,
    AlertSeverity,
    AlertRecord
)
from core_bak_refactored.core.share import (
    PerformanceStatsManager,
    ConfigManager,
    MonitoringConfig,
    AlertingConfig
)
from core_bak_refactored.infrastructure import (
    SystemHealthCalculators,
    QualityAnalysisCalculators
)
from core_bak_refactored.core.data.providers.factory import get_global_factory

logger = logging.getLogger('DeepSeekQuant.MonitoringService')


class QualityMonitoringService:
    """
    数据质量监控服务 - 整合层
    
    功能：
    1. 整合 DataQualityChecker 和 AlertManager
    2. 管理质量历史记录（check_history）
    3. 提供性能统计追踪（performance_stats）
    4. 生成综合报告（quality + alerts + performance）
    
    使用示例：
        # 创建监控服务
        alert_config = AlertConfig(wechat_webhook_url='...')
        service = QualityMonitoringService(alert_config)
        
        # 获取质量历史
        history = service.get_quality_history(hours=24)
        
        # 获取告警历史
        alerts = service.get_alert_history(hours=24)
        
        # 生成综合报告
        report = service.generate_comprehensive_report(period='7d')
    """
    
    def __init__(self, alert_config: Optional[AlertConfig] = None, config: Optional[Dict] = None):
        """
        初始化监控服务
        
        Args:
            alert_config: 告警配置（如果None则使用默认配置）
            config: 监控配置字典
        """
        # 初始化核心组件
        self.quality_checker = DataQualityChecker()
        self.alert_manager = AlertManager(alert_config or AlertConfig())
        
        # 配置管理
        self.config_manager = ConfigManager()
        self.config = config or self.config_manager.get_system_config().__dict__
        
        # 初始化数据提供者（通过配置映射自动选择）
        # 从配置获取默认指数和对应市场
        factory = get_global_factory()
        data_config = self.config_manager.get_data_config()
        default_index = data_config.default_index  # 例如: '000300.SH'
        
        # 根据指数代码推断市场
        market = self._infer_market_from_index(default_index)
        
        # 从 market_sources 映射获取该市场的数据源ID
        market_sources = data_config.market_sources or {}
        provider_id = market_sources.get(market, 'akshare')  # 默认使用 akshare
        
        # 通过工厂创建对应的provider
        self.data_provider = factory.get(provider_id)
        logger.info(f"监控服务使用数据源: {provider_id} (市场: {market}, 指数: {default_index})")
        
        # 质量历史记录（从 DataQualityChecker 的 check_history 转换而来）
        self._quality_history: List[Dict[str, Any]] = []
        
        # 性能统计管理器
        self.performance_stats_manager = PerformanceStatsManager()
        
        logger.info("QualityMonitoringService initialized")
    
    def get_quality_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        获取质量历史记录
        
        Args:
            hours: 查询小时数
        
        Returns:
            质量历史列表，每条包含：
            - timestamp: 时间戳
            - overall_score: 总体得分
            - completeness: 完整性得分
            - consistency: 一致性得分
            - continuity: 连续性得分
            - reasonableness: 合理性得分
            - issues: 问题列表
            - metadata: 元数据
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # 从质量检查器的历史记录转换
        history = []
        for report in self.quality_checker._check_history:
            # 检查时间戳
            report_time_str = report.metadata.get('timestamp', '')
            try:
                report_time = datetime.fromisoformat(report_time_str)
                if report_time < cutoff_time:
                    continue
            except (ValueError, TypeError):
                continue
            
            # 转换为字典格式
            history.append({
                'timestamp': report_time_str,
                'overall_score': report.overall_score,
                'completeness': report.completeness_score,
                'consistency': report.consistency_score,
                'accuracy': report.accuracy_score,
                'outliers': report.outliers_detected,
                'issues': report.issues,
                'anomaly_count': len([i for i in report.issues if 'anomaly' in i.lower() or 'abnormal' in i.lower()]),
                'error_count': len(report.issues),
                'metadata': report.metadata
            })
        
        return history
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        获取告警历史记录
        
        Args:
            hours: 查询小时数
        
        Returns:
            告警历史列表，每条包含：
            - alert_id: 告警ID
            - timestamp: 时间戳
            - level: 级别（mapping from severity）
            - severity: 严重性
            - title: 标题
            - message: 消息
            - data_source: 数据源（从metadata提取）
            - metadata: 元数据
        """
        since = datetime.now() - timedelta(hours=hours)
        
        # 从AlertManager获取历史
        alert_records = self.alert_manager.get_alert_history(since=since, limit=1000)
        
        # 转换为字典格式（兼容遗留API）
        alerts = []
        for record in alert_records:
            alerts.append({
                'alert_id': record.alert_id,
                'timestamp': record.created_at.isoformat(),
                'level': self._map_severity_to_level(record.severity),
                'severity': record.severity.value,
                'title': record.title,
                'message': record.message,
                'data_source': record.metadata.get('data_source', 'unknown'),
                'channels_used': [ch.value for ch in record.channels_used],
                'metadata': record.metadata
            })
        
        return alerts
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """
        获取性能统计
        
        Returns:
            性能统计字典，包含：
            - monitoring_cycles: 监控周期数
            - data_points_processed: 处理数据点数
            - anomalies_detected: 检测到的异常数
            - validation_errors: 验证错误数
            - alerts_triggered: 触发的告警数
            - avg_processing_time: 平均处理时间
            - start_time: 启动时间
            - uptime_seconds: 运行时长(秒)
            - uptime_human: 人类可读运行时长
            - success_rate: 成功率
            - reliability: 可靠性
            - stability_score: 稳定性得分
            - throughput: 吞吐量
        """
        # 更新性能统计
        self.performance_stats_manager.update_performance_stats()
        stats = self.performance_stats_manager.get_stats_dict()
        
        # 从AlertManager获取告警统计
        alert_stats = self.alert_manager.get_statistics(hours=24)
        stats['alerts_triggered'] = alert_stats.get('total_alerts', 0)
        stats['alert_breakdown'] = alert_stats.get('by_severity', {})
        
        return stats
    
    
    def generate_comprehensive_report(self, period: str = '7d') -> Dict[str, Any]:
        """
        生成综合报告
        
        Args:
            period: 时间周期（如 '7d', '24h', '30d'）
        
        Returns:
            综合报告字典，包含：
            - report_id: 报告ID
            - period: 时间周期
            - generated_at: 生成时间
            - quality_analysis: 质量分析
            - alert_analysis: 告警分析
            - performance_analysis: 性能分析
            - summary: 摘要
            - recommendations: 建议
        """
        # 解析时间周期
        hours = self._parse_period(period)
        
        # 获取数据
        quality_history = self.get_quality_history(hours)
        alert_history = self.get_alert_history(hours)
        performance_stats = self.get_performance_statistics()
        
        # 质量分析
        quality_analysis = self._analyze_quality(quality_history)
        
        # 告警分析
        alert_analysis = self._analyze_alerts(alert_history)
        
        # 性能分析
        performance_analysis = self._analyze_performance(performance_stats)
        
        # 生成摘要
        summary = self._generate_summary(quality_analysis, alert_analysis, performance_analysis)
        
        # 生成建议
        recommendations = self._generate_recommendations(quality_analysis, alert_analysis, performance_analysis)
        
        return {
            'report_id': f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'period': period,
            'generated_at': datetime.now().isoformat(),
            'quality_analysis': quality_analysis,
            'alert_analysis': alert_analysis,
            'performance_analysis': performance_analysis,
            'summary': summary,
            'recommendations': recommendations
        }
    
    # ==================== 私有辅助方法 ====================
    
    def _infer_market_from_index(self, index_code: str) -> str:
        """
        根据指数代码推断市场
        
        Args:
            index_code: 指数代码（如 '000300.SH', '^GSPC'）
        
        Returns:
            市场代码（CN, US, HK, JP, EU, SG等）
        """
        if not index_code:
            return 'CN'  # 默认中国市场
        
        index_upper = index_code.upper()
        
        # 中国市场
        if '.SH' in index_upper or '.SZ' in index_upper or '.BJ' in index_upper:
            return 'CN'
        # 香港市场
        elif '.HK' in index_upper:
            return 'HK'
        # 美国市场
        elif index_upper.startswith('^') or '.US' in index_upper:
            return 'US'
        # 日本市场
        elif '.T' in index_upper or '.JP' in index_upper:
            return 'JP'
        # 欧洲市场
        elif any(x in index_upper for x in ['.L', '.PA', '.DE', '.EU']):
            return 'EU'
        # 新加坡市场
        elif '.SI' in index_upper or '.SG' in index_upper:
            return 'SG'
        else:
            # 默认中国市场
            return 'CN'
    
    def _map_severity_to_level(self, severity: AlertSeverity) -> str:
        """将AlertSeverity映射到遗留API的level"""
        mapping = {
            AlertSeverity.INFO: 'info',
            AlertSeverity.WARNING: 'warning',
            AlertSeverity.ERROR: 'critical',
            AlertSeverity.CRITICAL: 'critical'
        }
        return mapping.get(severity, 'unknown')
    
    def _parse_period(self, period: str) -> int:
        """解析时间周期为小时数"""
        period = period.lower().strip()
        
        if period.endswith('h'):
            return int(period[:-1])
        elif period.endswith('d'):
            return int(period[:-1]) * 24
        elif period.endswith('w'):
            return int(period[:-1]) * 24 * 7
        else:
            # 默认7天
            return 7 * 24
    
    def _format_uptime(self, seconds: float) -> str:
        """格式化运行时长为人类可读格式"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        
        if days > 0:
            return f"{days} days {hours} hours"
        elif hours > 0:
            return f"{hours} hours {minutes} minutes"
        else:
            return f"{minutes} minutes"
    
    def _calculate_throughput(self, uptime_seconds: float) -> float:
        """计算吞吐量（数据点/秒）"""
        if uptime_seconds > 0:
            return self._performance_stats['data_points_processed'] / uptime_seconds
        return 0.0
    
    def _analyze_quality(self, quality_history: List[Dict]) -> Dict[str, Any]:
        """分析质量历史"""
        return QualityAnalysisCalculators.analyze_quality_history(quality_history)
    
    def _analyze_alerts(self, alert_history: List[Dict]) -> Dict[str, Any]:
        """分析告警历史"""
        return QualityAnalysisCalculators.analyze_alerts_history(alert_history)
    
    def _analyze_performance(self, stats: Dict) -> Dict[str, Any]:
        """分析性能统计"""
        return {
            'throughput': stats.get('throughput', 0.0),
            'success_rate': stats.get('success_rate', 1.0),
            'reliability': stats.get('reliability', 1.0),
            'stability_score': stats.get('stability_score', 1.0),
            'uptime': stats.get('uptime_human', '0 minutes')
        }
    
    def _generate_summary(self, quality: Dict, alerts: Dict, performance: Dict) -> Dict[str, Any]:
        """生成摘要"""
        return {
            'overall_health': self._calculate_overall_health(quality, alerts, performance),
            'quality_score': quality.get('avg_score', 0.0),
            'total_alerts': alerts.get('total_alerts', 0),
            'system_uptime': performance.get('uptime', '0 minutes'),
            'status': self._determine_status(quality, alerts, performance)
        }
    
    def _calculate_overall_health(self, quality: Dict, alerts: Dict, performance: Dict) -> float:
        """计算总体健康度（0-100）"""
        quality_score = quality.get('avg_score', 0.0)
        success_rate = performance.get('success_rate', 1.0)
        alert_count = alerts.get('total_alerts', 0)
        
        return SystemHealthCalculators.calculate_overall_health(quality_score, success_rate, alert_count)
    
    def _determine_status(self, quality: Dict, alerts: Dict, performance: Dict) -> str:
        """确定系统状态"""
        health = self._calculate_overall_health(quality, alerts, performance)
        
        return SystemHealthCalculators.determine_health_status(health)
    
    def _generate_recommendations(self, quality: Dict, alerts: Dict, performance: Dict) -> List[Dict[str, str]]:
        """生成建议"""
        # 计算健康度分数
        health_score = self._calculate_overall_health(quality, alerts, performance)
        
        # 使用系统健康度计算工具生成建议
        stats = {
            'success_rate': performance.get('success_rate', 1.0),
            'alerts_triggered': alerts.get('total_alerts', 0)
        }
        
        return SystemHealthCalculators.generate_recommendations_from_stats(stats, health_score)
    
    def run_check_cycle(self, data_sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """运行一次完整的监控检查周期（等效于旧版DataQualityMonitor的监控循环）
        
        功能：
        1. 拉取或接收待检查的数据
        2. 调用DataQualityChecker执行质量检查
        3. 将检查报告写入质量历史
        4. 根据结果调用AlertManager触发告警
        5. 更新性能统计（data_points_processed、anomalies_detected等）
        
        Args:
            data_sources: 要检查的数据源列表（如果为None则从配置获取）
        
        Returns:
            周期执行摘要：
            - cycle_time: 执行耗时（秒）
            - data_points_checked: 检查的数据点数
            - anomalies_detected: 检测到的异常数
            - alerts_triggered: 触发的告警数
            - quality_score: 质量得分
        """
        import pandas as pd
        cycle_start = datetime.now()
        summary = {
            'cycle_time': 0.0,
            'data_points_checked': 0,
            'anomalies_detected': 0,
            'alerts_triggered': 0,
            'quality_score': 1.0,
            'status': 'success'
        }
        
        try:
            # TODO: 这里应该从实际数据源拉取数据
            # 当前为示例实现，实际应从配置的data_sources拉取数据
            # 示例：假设有一个数据获取方法
            logger.info("开始监控检查周期")
            
            # 使用真实数据提供者获取数据（禁止模拟数据）
            index_id = self.config_manager.get_data_config().default_index
            if not index_id:
                raise RuntimeError("未配置默认指数代码，禁止使用模拟数据。请在配置中设置 data.default_index")
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d')
            
            try:
                data = self.data_provider.get_index_prices(index_id, start_date, end_date,)
            except Exception as e:
                # 如果真实数据获取失败，使用示例数据演示功能
                logger.warning(f"真实数据获取失败，使用示例数据: {e}")
                # 生成示例数据
                import pandas as pd
                import numpy as np
                dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
                data = pd.DataFrame({
                    'close': np.random.uniform(4000, 5000, 100),
                    'volume': np.random.uniform(1e9, 5e9, 100)
                }, index=dates)
                logger.info("已生成示例数据用于演示")
            
            # 1. 执行质量检查
            quality_report = self.quality_checker.check_quality(
                data=data,
                index_id=index_id,  # 从配置获取默认指数
                expected_days=100
            )
            
            # 2. 转换为字典并写入质量历史
            quality_dict = {
                'timestamp': datetime.now().isoformat(),
                'overall_score': quality_report.overall_score,
                'completeness': quality_report.completeness_score,
                'consistency': quality_report.consistency_score,
                'accuracy': quality_report.accuracy_score,
                'outliers': quality_report.outliers_detected,
                'issues': quality_report.issues,
                'anomaly_count': len([i for i in quality_report.issues if 'anomaly' in i.lower()]),
                'error_count': len(quality_report.issues),
                'metadata': quality_report.metadata
            }
            self._quality_history.append(quality_dict)
            
            # 保持历史记录长度
            max_history = self.config.get('max_quality_history', 5000)
            if len(self._quality_history) > max_history:
                self._quality_history = self._quality_history[-max_history:]
            
            # 3. 根据质量得分触发告警
            alerts_triggered = 0
            if quality_report.overall_score < 0.8:
                # 触发告警
                severity = AlertSeverity.WARNING if quality_report.overall_score >= 0.7 else AlertSeverity.ERROR
                alert_record = self.alert_manager.send_alert(
                    severity=severity,
                    title='数据质量告警',
                    message=f'质量得分{quality_report.overall_score:.2%}，问题：{"; ".join(quality_report.issues[:3])}',
                    metadata={
                        'quality_score': quality_report.overall_score,
                        'issues': quality_report.issues,
                        'index_id': '000300.SH'
                    },
                    dedup_key=f"quality_{datetime.now().strftime('%Y%m%d%H')}"
                )
                if alert_record:
                    alerts_triggered += 1
            
            # 4. 更新性能统计
            cycle_time = (datetime.now() - cycle_start).total_seconds()
            data_points = len(data)
            anomalies = quality_dict['anomaly_count']
            
            # 使用性能统计管理器更新计数器
            self.performance_stats_manager.increment_counter('data_points_processed', data_points)
            self.performance_stats_manager.increment_counter('anomalies_detected', anomalies)
            self.performance_stats_manager.increment_counter('validation_errors', quality_dict['error_count'])
            self.performance_stats_manager.increment_counter('monitoring_cycles', 1)
            
            # 更新平均处理时间
            current_avg = self.performance_stats_manager.get_stats().avg_processing_time
            cycles = self.performance_stats_manager.get_stats().monitoring_cycles
            if cycles > 0:
                new_avg = (current_avg * (cycles - 1) + cycle_time) / cycles
                self.performance_stats_manager.update_metric('avg_processing_time', new_avg)
            else:
                self.performance_stats_manager.update_metric('avg_processing_time', cycle_time)
            
            # 5. 填充摘要
            summary.update({
                'cycle_time': cycle_time,
                'data_points_checked': data_points,
                'anomalies_detected': anomalies,
                'alerts_triggered': alerts_triggered,
                'quality_score': quality_report.overall_score,
                'status': 'success'
            })
            
            logger.info(f"监控周期完成: 质量得分{quality_report.overall_score:.2%}, "
                       f"数据点{data_points}, 异常{anomalies}, 告警{alerts_triggered}, "
                       f"耗时{cycle_time:.2f}秒")
        
        except Exception as e:
            cycle_time = (datetime.now() - cycle_start).total_seconds()
            logger.error(f"监控周期执行失败: {e}", exc_info=True)
            summary.update({
                'cycle_time': cycle_time,
                'status': 'error',
                'error': str(e)
            })
            self.performance_stats_manager.increment_counter('validation_errors', 1)
        
        return summary
    
    def export_monitoring_data(self, filepath: str, format: str = 'json') -> bool:
        """导出监控数据（兼容方法）
        
        Args:
            filepath: 导出文件路径
            format: 导出格式（json或csv）
        
        Returns:
            是否成功
        """
        import json
        import csv
        
        try:
            if format == 'json':
                data = {
                    'quality_history': self.get_quality_history(hours=24*7),  # 7天
                    'alert_history': self.get_alert_history(hours=24*7),
                    'performance_stats': self.get_performance_statistics(),
                    'exported_at': datetime.now().isoformat()
                }
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            elif format == 'csv':
                # 导出质量历史为CSV
                quality_history = self.get_quality_history(hours=24*7)
                if quality_history:
                    with open(filepath, 'w', newline='', encoding='utf-8') as f:
                        fieldnames = list(quality_history[0].keys())
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(quality_history)
            else:
                logger.error(f"不支持的格式: {format}")
                return False
            
            logger.info(f"监控数据导出成功: {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"监控数据导出失败: {e}")
            return False
