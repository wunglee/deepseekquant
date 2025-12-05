"""数据质量仪表板 - 提供可视化监控界面

[应用层] 第四轮迁移 - 委派到dashboard组件
状态: ✅ 使用委派模式，功能拆分到组件
来源: dashboard_bak.py (734行)
迁移时间: 2025-11-28
版本: 组件化版本 (约200行，委派到3个组件)

委派到组件:
- DashboardDataAggregator: 数据聚合和转换
- WebSocketHandler: WebSocket实时通信
- DashboardRenderer: HTML模板渲染

原始代码备份: dashboard_bak.py
注意: 本类仅依赖领域层接口，严格遵守分层架构原则
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Dict, Any, List, Set, TYPE_CHECKING

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

# 导入Dashboard组件
from core_bak_refactored.app.quality_monitoring.dashboard_components.aggregator import DashboardDataAggregator
from core_bak_refactored.app.quality_monitoring.dashboard_components.websocket_handler import WebSocketHandler

# 导入应用层监控服务（替代废弃的DataQualityMonitor）
if TYPE_CHECKING:
    from core_bak_refactored.app.quality_monitoring.monitoring_service import MonitoringService

logger = logging.getLogger('DeepSeekQuant.App.Dashboard')


class DataQualityDashboard:
    """数据质量仪表板 - 提供可视化监控界面
    
    .. deprecated:: 2025-11-29
        建议使用 MonitoringService + DataQualityChecker 组合
        
        迁移说明：
        - 旧的DataQualityMonitor已废弃
        - 应用层监控应由MonitoringService负责
        - 核心层质量检查由DataQualityChecker负责
        
        迁移示例：
        ```python
        # 旧代码（已废弃）
        # from core_bak_refactored.core.data.data_fetcher import DataQualityMonitor
        # from core_bak_refactored.app.quality_monitoring.dashboard import DataQualityDashboard
        # monitor = DataQualityMonitor(config)
        # dashboard = DataQualityDashboard(monitor)
        
        # 新代码
        from core_bak_refactored.app.quality_monitoring.monitoring_service import MonitoringService
        from core_bak_refactored.core.data.quality import DataQualityChecker
        
        checker = DataQualityChecker()
        monitoring_service = MonitoringService(checker, config)
        # 仪表板功能应集成在MonitoringService中
        ```
    """

    def __init__(self, quality_monitor: 'MonitoringService', scheduler=None):
        self.quality_monitor = quality_monitor
        self.scheduler = scheduler  # MonitoringScheduler实例（可选）
        self.dashboard_data = {}
        self.update_interval = 300  # 5分钟更新一次
        self.dashboard_config = self._load_dashboard_config()
        self.last_update_time = datetime.now()
        
        # 初始化组件
        self.data_aggregator = DashboardDataAggregator(quality_monitor)
        self.websocket_handler = WebSocketHandler(quality_monitor)

    def _load_dashboard_config(self) -> Dict[str, Any]:
        """加载仪表板配置"""
        # 从配置管理器加载配置
        from core_bak_refactored.core.share.config_manager import ConfigManager
        config_manager = ConfigManager()
        
        # 获取仪表板配置，如果不存在则使用默认配置
        dashboard_config = config_manager.get('dashboard', {})
        
        # 合并默认配置和用户配置
        default_config = {
            'host': '0.0.0.0',
            'port': 8080,
            'refresh_interval': 300,
            'max_data_points': 1000,
            'enable_websocket': True,
            'websocket_port': 8090,
            'chart_config': {
                'quality_score_chart': {
                    'type': 'line',
                    'title': '数据质量评分趋势',
                    'x_axis': 'timestamp',
                    'y_axis': 'overall_score',
                    'color': '#2196F3',
                    'fill': True
                },
                'anomaly_chart': {
                    'type': 'bar',
                    'title': '异常检测统计',
                    'x_axis': 'timestamp',
                    'y_axis': 'anomaly_count',
                    'color': '#FF5252'
                },
                'error_distribution_chart': {
                    'type': 'pie',
                    'title': '错误类型分布',
                    'data_field': 'error_types',
                    'color_scheme': 'category10'
                },
                'performance_metrics_chart': {
                    'type': 'radar',
                    'title': '性能指标雷达图',
                    'metrics': ['throughput', 'reliability', 'accuracy', 'timeliness', 'completeness'],
                    'max_value': 1.0
                }
            },
            'widgets': [
                {
                    'id': 'overall_quality',
                    'type': 'gauge',
                    'title': '总体质量评分',
                    'value_field': 'overall_score',
                    'ranges': [0.0, 0.6, 0.8, 0.9, 1.0],
                    'range_colors': ['#FF5252', '#FFB300', '#FFEB3B', '#4CAF50']
                },
                {
                    'id': 'anomaly_count',
                    'type': 'counter',
                    'title': '异常数量',
                    'value_field': 'total_anomalies',
                    'trend_field': 'anomaly_trend'
                },
                {
                    'id': 'data_throughput',
                    'type': 'metric',
                    'title': '数据处理吞吐量',
                    'value_field': 'throughput',
                    'unit': 'points/sec'
                }
            ],
            'alert_settings': {
                'show_critical_alerts': True,
                'show_warnings': True,
                'alert_history_length': 50,
                'auto_refresh_alerts': True
            }
        }
        
        # 深度合并配置
        def deep_merge(default, override):
            result = default.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        return deep_merge(default_config, dashboard_config)

    def start_dashboard(self, host: str = None, port: int = None):
        """启动仪表板服务器"""
        try:
            # 使用配置中的主机和端口，如果未提供则使用默认值
            if host is None:
                host = self.dashboard_config.get('host', '0.0.0.0')
            if port is None:
                port = self.dashboard_config.get('port', 8080)
                
            logger.info(f"启动数据质量仪表板: http://{host}:{port}")

            # 创建Flask应用
            app = Flask(__name__)
            CORS(app)

            # 设置路由
            @app.route('/')
            def index():
                # TODO: 实现HTML渲染逻辑，当前返回简单响应
                return '<html><body><h1>DeepSeekQuant Dashboard</h1><p>Dashboard service is running.</p></body></html>'

            @app.route('/api/quality-data')
            def get_quality_data():
                return jsonify(self.data_aggregator.get_current_quality_data())

            @app.route('/api/performance-stats')
            def get_performance_stats():
                return jsonify(self.quality_monitor.get_performance_statistics())

            @app.route('/api/alerts')
            def get_alerts():
                hours = request.args.get('hours', 24, type=int)
                return jsonify(self.quality_monitor.get_alert_history(hours))

            @app.route('/api/scheduler-status')
            def get_scheduler_status():
                """获取调度器状态（新增）"""
                if self.scheduler:
                    status = self.scheduler.get_status()
                    return jsonify({
                        'scheduler_enabled': True,
                        'running': status['running'],
                        'strategy': status['strategy'],
                        'check_interval': status['check_interval'],
                        'next_run': status['next_run']
                    })
                else:
                    return jsonify({
                        'scheduler_enabled': False,
                        'message': '调度器未配置'
                    })

            @app.route('/api/reports/<report_id>')
            def get_report(report_id):
                return jsonify(self.data_aggregator.get_report_data(report_id))

            @app.route('/api/export-data')
            def export_data():
                format = request.args.get('format', 'json')
                filename = f"quality_data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
                filepath = os.path.join('exports', filename)

                if self.quality_monitor.export_monitoring_data(filepath, format):
                    return send_file(filepath, as_attachment=True)
                else:
                    return jsonify({'error': '导出失败'}), 500

            @app.route('/ws')
            def websocket_endpoint():
                if request.environ.get('wsgi.websocket'):
                    ws = request.environ['wsgi.websocket']
                    self.websocket_handler.handle_connection(ws)
                return 'WebSocket endpoint'

            # 启动后台更新线程
            update_thread = threading.Thread(target=self._dashboard_update_worker, daemon=True)
            update_thread.start()

            # 启动Flask服务器
            app.run(host=host, port=port, debug=False)

        except Exception as e:
            logger.error(f"仪表板启动失败: {e}")
            raise

    # ============================================================
    # 第四轮迁移：已迁移到dashboard组件
    # ============================================================
    # 
    # 已迁移到 data_aggregator (DashboardDataAggregator):
    # - _get_current_quality_data -> data_aggregator.get_current_quality_data
    # - _calculate_quality_trend -> data_aggregator.calculate_quality_trend
    # - _prepare_anomaly_data -> data_aggregator.prepare_anomaly_data
    # - _determine_anomaly_level -> data_aggregator.determine_anomaly_level
    # - _prepare_performance_data -> data_aggregator.prepare_performance_data
    # - _calculate_error_distribution -> data_aggregator.calculate_error_distribution
    # - _group_alerts_by_level -> data_aggregator.group_alerts_by_level
    # - _get_report_data -> data_aggregator.get_report_data
    #
    # 已迁移到 websocket_handler (WebSocketHandler):
    # - _handle_websocket_connection -> websocket_handler.handle_connection
    # - _handle_websocket_message -> websocket_handler.handle_message
    # - _handle_subscription -> websocket_handler.handle_subscription
    # - _handle_unsubscription -> websocket_handler.handle_unsubscription
    # - _send_requested_data -> websocket_handler.send_requested_data
    # - _broadcast_to_websockets -> websocket_handler.broadcast
    #
    # 已迁移到 renderer (DashboardRenderer):
    # - _render_dashboard -> renderer.render_dashboard
    # ============================================================

    def _dashboard_update_worker(self):
        """仪表板更新工作线程"""
        while True:
            try:
                # 更新仪表板数据
                current_data = self.data_aggregator.get_current_quality_data()
                self.dashboard_data = current_data

                # 广播给所有WebSocket连接
                self.websocket_handler.broadcast({
                    'type': 'quality_update',
                    'data': current_data,
                    'timestamp': datetime.now().isoformat()
                })

                # 检查是否有新警报
                recent_alerts = self.quality_monitor.get_alert_history(hours=1)
                if recent_alerts:
                    self.websocket_handler.broadcast({
                        'type': 'alert_update',
                        'data': recent_alerts,
                        'timestamp': datetime.now().isoformat()
                    })

                # 等待下一次更新
                time.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"仪表板更新失败: {e}")
                time.sleep(60)  # 出错后等待1分钟

    def stop_dashboard(self):
        """停止仪表板"""
        logger.info("停止数据质量仪表板")
        # 关闭所有WebSocket连接
        for ws in self.websocket_handler.connections:
            try:
                ws.close()
            except:
                pass
        self.websocket_handler.connections.clear()

    def export_dashboard_config(self, filepath: str) -> bool:
        """导出仪表板配置"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.dashboard_config, f, indent=2)
            logger.info(f"仪表板配置导出成功: {filepath}")
            return True
        except Exception as e:
            logger.error(f"仪表板配置导出失败: {e}")
            return False

    def import_dashboard_config(self, filepath: str) -> bool:
        """导入仪表板配置"""
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            self.dashboard_config = config
            logger.info(f"仪表板配置导入成功: {filepath}")
            return True
        except Exception as e:
            logger.error(f"仪表板配置导入失败: {e}")
            return False

    def cleanup(self):
        """清理资源"""
        self.stop_dashboard()
        logger.info("数据质量仪表板清理完成")


# 数据质量API服务类
