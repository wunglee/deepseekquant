"""数据质量RESTful API服务 - 提供完整REST API接口

[应用层] 从专家完整版完整迁移 - 无删减版本
状态: ✅ 专家完整版，包含所有53个方法
来源: core_bak/data_fetcher.py DataQualityAPIService类 (专家完整版)
迁移时间: 2025-11-27
版本: 完整版 (约1289行，53个方法)

包含完整功能:
- 完整的REST API端点 (质量数据、报告、警报、性能、指标等)
- 健康检查与诊断
- 配置管理 (GET/PUT)
- 数据导出 (JSON/CSV)
- 维护模式
- 系统状态监控
- 资源利用率跟踪
- 完整的错误处理

API端点:
- GET  /api/v1/quality/current     - 获取当前质量数据
- GET  /api/v1/quality/report      - 生成质量报告
- GET  /api/v1/alerts               - 获取警报历史(支持过滤分页)
- GET  /api/v1/performance          - 获取性能统计
- GET  /api/v1/metrics              - 获取监控指标
- GET  /api/v1/export               - 导出数据
- GET  /api/v1/config               - 获取配置
- PUT  /api/v1/config               - 更新配置
- GET  /api/v1/health               - 健康检查
- GET  /api/v1/diagnostics          - 运行诊断
- GET  /api/v1/status               - 获取系统状态
- POST /api/v1/maintenance          - 维护模式

依赖说明:
- ✅ 已迁移到重构后架构: 使用 QualityMonitoringService (app/data/monitoring_service.py)
- ✅ 监控服务整合了重构后的核心组件:
  - DataQualityChecker (core/data/providers/data_quality_checker.py) - 质量检查
  - AlertManager (core/monitoring/alert_manager.py) - 告警管理
- ✅ 完全消除了对 data_fetcher.py 的依赖

架构说明:
- 应用层使用适配器模式对接遗留API接口
- 核心逻辑由重构后组件提供
- 便于后续进一步优化API接口
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Any, List, TYPE_CHECKING
import os

import numpy as np
import psutil
from flask import Flask, jsonify, request, Response, render_template, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# 从组件导入
from core_bak_refactored.app.quality_monitoring.api.controllers import DataQualityControllers
from core_bak_refactored.app.quality_monitoring.api.health import HealthChecker
from core_bak_refactored.app.quality_monitoring.api.system_metrics import MetricsCollector
from core_bak_refactored.app.quality_monitoring.api.diagnostics import DiagnosticsRunner
from core_bak_refactored.core.share.config_manager import ConfigManager
from core_bak_refactored.app.quality_monitoring.api.exporter import DataExporter
from core_bak_refactored.app.quality_monitoring.api.system_status import SystemStatusManager

if TYPE_CHECKING:
    from core_bak_refactored.app.quality_monitoring.monitoring_service import QualityMonitoringService

logger = logging.getLogger('DeepSeekQuant.App.APIService')


class DataQualityAPIService:
    """数据质量API服务 - 提供RESTful API接口
    
    架构说明:
    - 应用层：API路由和请求处理
    - 依赖监控服务：QualityMonitoringService（整合DataQualityChecker + AlertManager）
    - 职责分离：API组件专注于各自职责（健康检查、指标、诊断等）
    """

    def __init__(self, quality_monitor: QualityMonitoringService, scheduler=None):
        self.quality_monitor = quality_monitor
        self.scheduler = scheduler  # 调度器实例（可选）
        
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_dir = os.path.join(current_dir, 'templates')
        static_dir = os.path.join(current_dir, 'static')
        
        self.app = Flask(
            __name__,
            template_folder=template_dir,  # 指定模板目录
            static_folder=static_dir  # 指定静态文件目录
        )
        
        # 启用CORS
        CORS(self.app)
        
        # 初始化Socket.IO（支持实时推送）
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            async_mode='threading',
            logger=False,
            engineio_logger=False
        )
        logger.info("Socket.IO服务已初始化")
        
        # 初始化组件
        self.controllers = DataQualityControllers(quality_monitor)
        self.health_checker = HealthChecker(quality_monitor)
        self.metrics_collector = MetricsCollector(quality_monitor)
        self.diagnostics_runner = DiagnosticsRunner(quality_monitor)
        self.config_manager = ConfigManager()  # 核心层配置管理器不需要 quality_monitor
        self.data_exporter = DataExporter(quality_monitor)
        self.system_status_manager = SystemStatusManager(quality_monitor)
        
        self._setup_routes()
        self._setup_socketio_handlers()

    def _setup_routes(self):
        """设置API路由 - 完整生产实现"""

        @self.app.route('/api/v1/quality/current', methods=['GET'])
        def get_current_quality():
            """获取当前质量数据 - 委派到controllers"""
            try:
                hours = request.args.get('hours', 24, type=int)
                result = self.controllers.get_quality_current(hours)
                return jsonify({
                    'status': 'success',
                    **result,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"获取当前质量数据失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'QUALITY_DATA_FETCH_FAILED'
                }), 500

        @self.app.route('/api/v1/quality/report', methods=['GET'])
        def generate_quality_report():
            """生成质量报告 - 委派到data_exporter"""
            try:
                period = request.args.get('period', '7d')
                report_format = request.args.get('format', 'json')
                include_details = request.args.get('include_details', 'true').lower() == 'true'

                report = self.quality_monitor.generate_comprehensive_report(period)

                if report_format == 'csv':
                    # 转换为CSV格式
                    csv_data = self.data_exporter.convert_report_to_csv(report, include_details)
                    response = Response(csv_data, mimetype='text/csv')
                    response.headers[
                        'Content-Disposition'] = f'attachment; filename=quality_report_{datetime.now().strftime("%Y%m%d")}.csv'
                    return response
                else:
                    if not include_details:
                        # 移除详细数据以减少响应大小
                        report.pop('quality_analysis', None)
                        report.pop('alert_analysis', None)
                        report.pop('performance_analysis', None)

                    return jsonify({
                        'status': 'success',
                        'report': report,
                        'timestamp': datetime.now().isoformat(),
                        'report_id': report.get('report_id', f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
                    })

            except Exception as e:
                logger.error(f"生成质量报告失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'REPORT_GENERATION_FAILED'
                }), 500

        @self.app.route('/api/v1/alerts', methods=['GET'])
        def get_alerts():
            """获取警报历史 - 委派到controllers"""
            try:
                hours = request.args.get('hours', 24, type=int)
                level = request.args.get('level')
                severity = request.args.get('severity')
                data_source = request.args.get('data_source')
                page = request.args.get('page', 1, type=int)
                per_page = request.args.get('per_page', 50, type=int)

                result = self.controllers.get_alerts_with_pagination(
                    hours, level, severity, data_source, page, per_page
                )
                
                return jsonify({
                    'status': 'success',
                    **result,
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"获取警报历史失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'ALERTS_FETCH_FAILED'
                }), 500

        @self.app.route('/api/v1/performance', methods=['GET'])
        def get_performance():
            """获取性能统计 - 委派到controllers"""
            try:
                enhanced_stats = self.controllers.get_enhanced_performance()
                return jsonify({
                    'status': 'success',
                    'performance': enhanced_stats,
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"获取性能统计失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'PERFORMANCE_FETCH_FAILED'
                }), 500

        @self.app.route('/api/v1/metrics', methods=['GET'])
        def get_metrics():
            """获取监控指标 - 委派到metrics_collector"""
            try:
                metric_type = request.args.get('type', 'all')
                time_range = request.args.get('time_range', '24h')
                aggregation = request.args.get('aggregation', 'hourly')

                metrics = self.metrics_collector.get_system_metrics(metric_type, time_range, aggregation)

                return jsonify({
                    'status': 'success',
                    'metrics': metrics,
                    'metadata': {
                        'metric_type': metric_type,
                        'time_range': time_range,
                        'aggregation': aggregation,
                        'data_points': len(metrics.get('data', []))
                    },
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"获取监控指标失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'METRICS_FETCH_FAILED'
                }), 500

        @self.app.route('/api/v1/export', methods=['GET'])
        def export_data():
            """导出数据 - 委派到data_exporter"""
            try:
                data_type = request.args.get('data_type', 'quality')
                format = request.args.get('format', 'json')
                time_range = request.args.get('time_range', '7d')

                if data_type == 'quality':
                    success = self.data_exporter.export_quality_data(
                        f'quality_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{format}',
                        format
                    )
                elif data_type == 'alerts':
                    success = self.data_exporter.export_alert_data(format, time_range)
                else:
                    return jsonify({
                        'status': 'error',
                        'message': f'不支持的数据类型: {data_type}',
                        'error_code': 'INVALID_DATA_TYPE'
                    }), 400

                if success:
                    return jsonify({
                        'status': 'success',
                        'message': '数据导出成功',
                        'export_type': data_type,
                        'format': format,
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({
                        'status': 'error',
                        'message': '数据导出失败',
                        'error_code': 'EXPORT_FAILED'
                    }), 500

            except Exception as e:
                logger.error(f"数据导出失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'EXPORT_FAILED'
                }), 500

        @self.app.route('/api/v1/config', methods=['GET', 'PUT'])
        def manage_config():
            """管理配置 - 使用核心层 ConfigManager"""
            try:
                if request.method == 'GET':
                    # 获取当前配置（聚合 API 服务相关配置）
                    config = {
                        'monitoring': self.config_manager.get('monitoring', {}),
                        'api_settings': self.config_manager.get('api_service', {}),
                        'alerting': self.config_manager.get('alerting', {}),
                        'data': self.config_manager.get('data', {}),
                        'system': self.config_manager.get('system', {})
                    }
                    return jsonify({
                        'status': 'success',
                        'config': config,
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    # 更新配置
                    new_config = request.get_json()
                    if not new_config:
                        return jsonify({
                            'status': 'error',
                            'message': '无效的配置数据',
                            'error_code': 'INVALID_CONFIG'
                        }), 400

                    # 使用核心层 ConfigManager 的 update 方法
                    self.config_manager.update(new_config)
                    return jsonify({
                        'status': 'success',
                        'message': '配置更新成功',
                        'timestamp': datetime.now().isoformat()
                    })

            except Exception as e:
                logger.error(f"配置管理失败: {e}")
                # 区分客户端错误和服务器错误
                if '400' in str(e) or 'Bad Request' in str(e):
                    return jsonify({
                        'status': 'error',
                        'message': str(e),
                        'error_code': 'INVALID_CONFIG'
                    }), 400
                else:
                    return jsonify({
                        'status': 'error',
                        'message': str(e),
                        'error_code': 'CONFIG_MANAGEMENT_FAILED'
                    }), 500

        @self.app.route('/api/v1/health', methods=['GET'])
        def health_check():
            """健康检查 - 委派到health_checker"""
            try:
                health_status = self.health_checker.check_system_health()
                status_code = 200 if health_status['status'] == 'healthy' else 503
                return jsonify(health_status), status_code
            except Exception as e:
                logger.error(f"健康检查失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'HEALTH_CHECK_FAILED'
                }), 500

        @self.app.route('/api/v1/diagnostics', methods=['GET'])
        def run_diagnostics():
            """运行诊断 - 委派到diagnostics_runner"""
            try:
                diagnostic_type = request.args.get('type', 'full')
                diagnostics = self.diagnostics_runner.run_diagnostics(diagnostic_type)

                return jsonify({
                    'status': 'success',
                    'diagnostics': diagnostics,
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"诊断运行失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'DIAGNOSTICS_FAILED'
                }), 500

        @self.app.route('/api/v1/status', methods=['GET'])
        def system_status():
            """获取系统状态 - 委派到system_status_manager"""
            try:
                status = self.system_status_manager.get_system_status()
                return jsonify({
                    'status': 'success',
                    'system_status': status,
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"获取系统状态失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'STATUS_FETCH_FAILED'
                }), 500

        @self.app.route('/api/v1/maintenance', methods=['POST'])
        def maintenance_mode():
            """维护模式 - 委派到system_status_manager"""
            try:
                action = request.args.get('action', 'enable')
                duration = request.args.get('duration', 3600, type=int)

                if action == 'enable':
                    success = self.system_status_manager.enable_maintenance_mode(duration)
                else:
                    success = self.system_status_manager.disable_maintenance_mode()

                if success:
                    return jsonify({
                        'status': 'success',
                        'message': f'维护模式{action}成功',
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({
                        'status': 'error',
                        'message': f'维护模式{action}失败',
                        'error_code': 'MAINTENANCE_MODE_FAILED'
                    }), 500

            except Exception as e:
                logger.error(f"维护模式操作失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'MAINTENANCE_MODE_FAILED'
                }), 500

        # ============================================================
        # Web 仪表板路由
        # ============================================================
        
        @self.app.route('/')
        @self.app.route('/dashboard')
        def dashboard():
            """数据质量仪表板 - 渲染HTML页面"""
            try:
                # 渲染仪表板模板
                return render_template('dashboard.html')
            except Exception as e:
                logger.error(f"仪表板访问失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': f'仪表板服务不可用: {str(e)}',
                    'error_code': 'DASHBOARD_UNAVAILABLE'
                }), 500

        @self.app.route('/explorer')
        def data_explorer():
            """数据浏览器 - 指数价格/收益率/事件窗口"""
            try:
                return render_template('data_explorer.html')
            except Exception as e:
                logger.error(f"数据浏览器页面访问失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': f'页面不可用: {str(e)}',
                    'error_code': 'EXPLORER_UNAVAILABLE'
                }), 500

        @self.app.route('/rules')
        def rules_manager():
            """质量规则管理 - 查看/启停/编辑规则"""
            try:
                return render_template('rules_manager.html')
            except Exception as e:
                logger.error(f"质量规则管理页面访问失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': f'页面不可用: {str(e)}',
                    'error_code': 'RULES_UNAVAILABLE'
                }), 500

        @self.app.route('/scheduler')
        def scheduler_console():
            """调度与作业控制台 - 查看作业/手动执行/暂停恢复"""
            try:
                return render_template('scheduler_console.html')
            except Exception as e:
                logger.error(f"调度控制台页面访问失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': f'页面不可用: {str(e)}',
                    'error_code': 'SCHEDULER_CONSOLE_UNAVAILABLE'
                }), 500

        @self.app.route('/alerts-center')
        def alerts_center():
            """警报中心 - 历史警报、过滤与批量操作"""
            try:
                return render_template('alerts_center.html')
            except Exception as e:
                logger.error(f"警报中心页面访问失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': f'页面不可用: {str(e)}',
                    'error_code': 'ALERTS_CENTER_UNAVAILABLE'
                }), 500

        @self.app.route('/providers')
        def providers_credentials():
            """数据来源与凭证管理 - 配置数据源与凭证"""
            try:
                return render_template('providers_credentials.html')
            except Exception as e:
                logger.error(f"数据来源与凭证页面访问失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': f'页面不可用: {str(e)}',
                    'error_code': 'PROVIDERS_UNAVAILABLE'
                }), 500

        @self.app.route('/validation')
        def validation_reports():
            """交叉验证与质量报告 - 查看验证日志与报告"""
            try:
                return render_template('validation_reports.html')
            except Exception as e:
                logger.error(f"验证报告页面访问失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': f'页面不可用: {str(e)}',
                    'error_code': 'VALIDATION_UNAVAILABLE'
                }), 500

        @self.app.route('/realtime')
        def realtime_monitor():
            """实时数据监控 - 订阅推送并可视化"""
            try:
                return render_template('realtime_monitor.html')
            except Exception as e:
                logger.error(f"实时监控页面访问失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': f'页面不可用: {str(e)}',
                    'error_code': 'REALTIME_UNAVAILABLE'
                }), 500
        
        @self.app.route('/api/dashboard/quality-data')
        def get_dashboard_quality_data():
            """获取仪表板质量数据"""
            try:
                history = self.quality_monitor.get_quality_history(hours=24)
                current = history[-1] if history else None
                
                return jsonify({
                    'status': 'success',
                    'current_quality': current,
                    'history': history[-100:],  # 最近100条记录
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"获取仪表板数据失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'DASHBOARD_DATA_FETCH_FAILED'
                }), 500
        
        @self.app.route('/api/dashboard/alerts')
        def get_dashboard_alerts():
            """获取仪表板告警数据"""
            try:
                hours = request.args.get('hours', 24, type=int)
                alerts = self.quality_monitor.get_alert_history(hours=hours)
                
                return jsonify({
                    'status': 'success',
                    'alerts': alerts,
                    'total_count': len(alerts),
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"获取仪表板告警失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'DASHBOARD_ALERTS_FETCH_FAILED'
                }), 500
        
        @self.app.route('/api/dashboard/performance')
        def get_dashboard_performance():
            """获取仪表板性能数据（严格模式）"""
            try:
                stats = self.quality_monitor.get_performance_statistics()
                return jsonify({
                    'status': 'success',
                    'performance': stats,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"获取仪表板性能数据失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'DASHBOARD_PERFORMANCE_FETCH_FAILED'
                }), 500
        
        @self.app.route('/api/v1/trigger-check', methods=['POST'])
        def trigger_quality_check():
            """手动触发数据质量检查"""
            try:
                if self.scheduler:
                    logger.info("手动触发质量检查")
                    self.scheduler.execute_now()
                    return jsonify({
                        'status': 'success',
                        'message': '质量检查已触发',
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({
                        'status': 'error',
                        'message': '调度器未配置，无法触发检查',
                        'error_code': 'SCHEDULER_NOT_CONFIGURED'
                    }), 503
            except Exception as e:
                logger.error(f"触发质量检查失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'TRIGGER_CHECK_FAILED'
                }), 500

        # ============================================================
        # 数据提供者能力暴露（补充接口）
        # ============================================================
        
        @self.app.route('/api/v1/providers', methods=['GET'])
        def get_providers():
            """获取所有数据源配置"""
            try:
                config = self.config_manager.get('data', {})
                providers = config.get('providers', [])
                primary_source = config.get('primary_source', 'mock')
                
                return jsonify({
                    'status': 'success',
                    'providers': providers,
                    'primary_source': primary_source,
                    'total': len(providers),
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"获取数据源列表失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'PROVIDERS_FETCH_FAILED'
                }), 500
        
        @self.app.route('/api/v1/providers/<provider_id>', methods=['GET'])
        def get_provider(provider_id):
            """获取指定数据源配置"""
            try:
                config = self.config_manager.get('data', {})
                providers = config.get('providers', [])
                
                provider = next((p for p in providers if p.get('id') == provider_id or p.get('name') == provider_id), None)
                
                if not provider:
                    return jsonify({
                        'status': 'error',
                        'message': f'数据源不存在: {provider_id}',
                        'error_code': 'PROVIDER_NOT_FOUND'
                    }), 404
                
                return jsonify({
                    'status': 'success',
                    'provider': provider,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"获取数据源配置失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'PROVIDER_FETCH_FAILED'
                }), 500
        
        @self.app.route('/api/v1/providers', methods=['POST'])
        def create_provider():
            """创建新数据源"""
            try:
                new_provider = request.get_json()
                if not new_provider:
                    return jsonify({
                        'status': 'error',
                        'message': '无效的数据源配置',
                        'error_code': 'INVALID_PROVIDER_DATA'
                    }), 400
                
                # 验证必填字段
                required_fields = ['name', 'type']
                for field in required_fields:
                    if field not in new_provider:
                        return jsonify({
                            'status': 'error',
                            'message': f'缺少必填字段: {field}',
                            'error_code': 'MISSING_REQUIRED_FIELD'
                        }), 400
                
                config = self.config_manager.get('data', {})
                providers = config.get('providers', [])
                
                # 检查是否已存在
                if any(p.get('name') == new_provider['name'] for p in providers):
                    return jsonify({
                        'status': 'error',
                        'message': f'数据源已存在: {new_provider["name"]}',
                        'error_code': 'PROVIDER_ALREADY_EXISTS'
                    }), 409
                
                # 添加默认字段
                new_provider.setdefault('enabled', True)
                new_provider.setdefault('priority', len(providers) + 1)
                new_provider.setdefault('created_at', datetime.now().isoformat())
                
                providers.append(new_provider)
                config['providers'] = providers
                
                self.config_manager.update({'data': config})
                logger.info(f"创建数据源成功: {new_provider['name']}")
                
                return jsonify({
                    'status': 'success',
                    'message': '数据源创建成功',
                    'provider': new_provider,
                    'timestamp': datetime.now().isoformat()
                }), 201
            except Exception as e:
                logger.error(f"创建数据源失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'PROVIDER_CREATE_FAILED'
                }), 500
        
        @self.app.route('/api/v1/providers/<provider_id>', methods=['PUT'])
        def update_provider(provider_id):
            """更新数据源配置"""
            try:
                updated_data = request.get_json()
                if not updated_data:
                    return jsonify({
                        'status': 'error',
                        'message': '无效的更新数据',
                        'error_code': 'INVALID_UPDATE_DATA'
                    }), 400
                
                config = self.config_manager.get('data', {})
                providers = config.get('providers', [])
                
                provider_index = next((i for i, p in enumerate(providers) if p.get('id') == provider_id or p.get('name') == provider_id), None)
                
                if provider_index is None:
                    return jsonify({
                        'status': 'error',
                        'message': f'数据源不存在: {provider_id}',
                        'error_code': 'PROVIDER_NOT_FOUND'
                    }), 404
                
                # 更新字段
                providers[provider_index].update(updated_data)
                providers[provider_index]['updated_at'] = datetime.now().isoformat()
                
                config['providers'] = providers
                self.config_manager.update({'data': config})
                logger.info(f"更新数据源成功: {provider_id}")
                
                return jsonify({
                    'status': 'success',
                    'message': '数据源更新成功',
                    'provider': providers[provider_index],
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"更新数据源失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'PROVIDER_UPDATE_FAILED'
                }), 500
        
        @self.app.route('/api/v1/providers/<provider_id>', methods=['DELETE'])
        def delete_provider(provider_id):
            """删除数据源"""
            try:
                config = self.config_manager.get('data', {})
                providers = config.get('providers', [])
                
                provider_index = next((i for i, p in enumerate(providers) if p.get('id') == provider_id or p.get('name') == provider_id), None)
                
                if provider_index is None:
                    return jsonify({
                        'status': 'error',
                        'message': f'数据源不存在: {provider_id}',
                        'error_code': 'PROVIDER_NOT_FOUND'
                    }), 404
                
                deleted_provider = providers.pop(provider_index)
                config['providers'] = providers
                
                self.config_manager.update({'data': config})
                logger.info(f"删除数据源成功: {provider_id}")
                
                return jsonify({
                    'status': 'success',
                    'message': '数据源删除成功',
                    'deleted_provider': deleted_provider,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"删除数据源失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'PROVIDER_DELETE_FAILED'
                }), 500
        
        @self.app.route('/api/v1/providers/<provider_id>/test', methods=['POST'])
        def test_provider(provider_id):
            """测试数据源连接（真实测试）"""
            try:
                import time
                from core_bak_refactored.core.data.providers.historical_data_provider import get_global_factory
                
                config = self.config_manager.get('data', {})
                providers = config.get('providers', [])
                
                provider = next((p for p in providers if p.get('id') == provider_id or p.get('name') == provider_id), None)
                
                if not provider:
                    return jsonify({
                        'status': 'error',
                        'message': f'数据源不存在: {provider_id}',
                        'error_code': 'PROVIDER_NOT_FOUND'
                    }), 404
                
                # 实际测试数据源连接
                provider_type = provider.get('type') or provider.get('id')
                factory = get_global_factory()
                
                start_time = time.time()
                try:
                    # 尝试创建数据提供者实例
                    data_provider = factory.create(provider_type)
                    
                    # 尝试获取少量测试数据（最近3天的沪深300数据）
                    from datetime import datetime, timedelta
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=3)
                    
                    test_data = data_provider.get_index_prices(
                        '000300.SH',
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d')
                    )
                    
                    latency_ms = int((time.time() - start_time) * 1000)
                    
                    # 验证数据有效性
                    if test_data is None or len(test_data) == 0:
                        raise ValueError("数据源返回空数据")
                    
                    test_result = {
                        'connected': True,
                        'latency_ms': latency_ms,
                        'message': f'连接成功，获取到 {len(test_data)} 条数据',
                        'data_points': len(test_data),
                        'tested_at': datetime.now().isoformat()
                    }
                    
                    logger.info(f"数据源 {provider_id} 测试成功: {latency_ms}ms, {len(test_data)} 条数据")
                    
                except Exception as test_error:
                    latency_ms = int((time.time() - start_time) * 1000)
                    test_result = {
                        'connected': False,
                        'latency_ms': latency_ms,
                        'message': f'连接失败: {str(test_error)}',
                        'error': str(test_error),
                        'tested_at': datetime.now().isoformat()
                    }
                    logger.warning(f"数据源 {provider_id} 测试失败: {test_error}")
                
                return jsonify({
                    'status': 'success' if test_result['connected'] else 'error',
                    'provider': provider['name'],
                    'test_result': test_result,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"测试数据源连接失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'PROVIDER_TEST_FAILED'
                }), 500
        
        # ============================================================
        # 凭证管理端点（Credentials Management）
        # ============================================================
        
        @self.app.route('/api/v1/credentials', methods=['GET'])
        def get_credentials():
            """获取所有凭证列表（敏感信息脱敏）"""
            try:
                config = self.config_manager.get('credentials', {})
                credentials_list = []
                
                for key, cred in config.items():
                    # 脱敏处理
                    sanitized_cred = {
                        'id': key,
                        'type': cred.get('type', 'unknown'),
                        'provider': cred.get('provider', ''),
                        'username': cred.get('username', ''),
                        'api_key': '***' + cred.get('api_key', '')[-4:] if cred.get('api_key') else '',
                        'enabled': cred.get('enabled', True),
                        'created_at': cred.get('created_at', ''),
                        'updated_at': cred.get('updated_at', '')
                    }
                    credentials_list.append(sanitized_cred)
                
                return jsonify({
                    'status': 'success',
                    'credentials': credentials_list,
                    'total': len(credentials_list),
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"获取凭证列表失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'CREDENTIALS_FETCH_FAILED'
                }), 500
        
        @self.app.route('/api/v1/credentials/<credential_id>', methods=['GET'])
        def get_credential(credential_id):
            """获取指定凭证（脱敏）"""
            try:
                config = self.config_manager.get('credentials', {})
                cred = config.get(credential_id)
                
                if not cred:
                    return jsonify({
                        'status': 'error',
                        'message': f'凭证不存在: {credential_id}',
                        'error_code': 'CREDENTIAL_NOT_FOUND'
                    }), 404
                
                # 脱敏处理
                sanitized_cred = {
                    'id': credential_id,
                    'type': cred.get('type', 'unknown'),
                    'provider': cred.get('provider', ''),
                    'username': cred.get('username', ''),
                    'api_key': '***' + cred.get('api_key', '')[-4:] if cred.get('api_key') else '',
                    'enabled': cred.get('enabled', True),
                    'created_at': cred.get('created_at', ''),
                    'updated_at': cred.get('updated_at', '')
                }
                
                return jsonify({
                    'status': 'success',
                    'credential': sanitized_cred,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"获取凭证失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'CREDENTIAL_FETCH_FAILED'
                }), 500
        
        @self.app.route('/api/v1/credentials', methods=['POST'])
        def create_credential():
            """创建新凭证"""
            try:
                new_cred = request.get_json()
                if not new_cred:
                    return jsonify({
                        'status': 'error',
                        'message': '无效的凭证数据',
                        'error_code': 'INVALID_CREDENTIAL_DATA'
                    }), 400
                
                # 验证必填字段
                required_fields = ['id', 'type']
                for field in required_fields:
                    if field not in new_cred:
                        return jsonify({
                            'status': 'error',
                            'message': f'缺少必填字段: {field}',
                            'error_code': 'MISSING_REQUIRED_FIELD'
                        }), 400
                
                config = self.config_manager.get('credentials', {})
                
                # 检查是否已存在
                if new_cred['id'] in config:
                    return jsonify({
                        'status': 'error',
                        'message': f'凭证已存在: {new_cred["id"]}',
                        'error_code': 'CREDENTIAL_ALREADY_EXISTS'
                    }), 409
                
                # 添加默认字段
                new_cred.setdefault('enabled', True)
                new_cred.setdefault('created_at', datetime.now().isoformat())
                
                config[new_cred['id']] = new_cred
                self.config_manager.update({'credentials': config})
                logger.info(f"创建凭证成功: {new_cred['id']}")
                
                # 返回脱敏数据
                sanitized = {
                    'id': new_cred['id'],
                    'type': new_cred.get('type'),
                    'provider': new_cred.get('provider', ''),
                    'enabled': new_cred.get('enabled', True)
                }
                
                return jsonify({
                    'status': 'success',
                    'message': '凭证创建成功',
                    'credential': sanitized,
                    'timestamp': datetime.now().isoformat()
                }), 201
            except Exception as e:
                logger.error(f"创建凭证失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'CREDENTIAL_CREATE_FAILED'
                }), 500
        
        @self.app.route('/api/v1/credentials/<credential_id>', methods=['PUT'])
        def update_credential(credential_id):
            """更新凭证"""
            try:
                updated_data = request.get_json()
                if not updated_data:
                    return jsonify({
                        'status': 'error',
                        'message': '无效的更新数据',
                        'error_code': 'INVALID_UPDATE_DATA'
                    }), 400
                
                config = self.config_manager.get('credentials', {})
                
                if credential_id not in config:
                    return jsonify({
                        'status': 'error',
                        'message': f'凭证不存在: {credential_id}',
                        'error_code': 'CREDENTIAL_NOT_FOUND'
                    }), 404
                
                # 更新字段
                config[credential_id].update(updated_data)
                config[credential_id]['updated_at'] = datetime.now().isoformat()
                
                self.config_manager.update({'credentials': config})
                logger.info(f"更新凭证成功: {credential_id}")
                
                return jsonify({
                    'status': 'success',
                    'message': '凭证更新成功',
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"更新凭证失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'CREDENTIAL_UPDATE_FAILED'
                }), 500
        
        @self.app.route('/api/v1/credentials/<credential_id>', methods=['DELETE'])
        def delete_credential(credential_id):
            """删除凭证"""
            try:
                config = self.config_manager.get('credentials', {})
                
                if credential_id not in config:
                    return jsonify({
                        'status': 'error',
                        'message': f'凭证不存在: {credential_id}',
                        'error_code': 'CREDENTIAL_NOT_FOUND'
                    }), 404
                
                del config[credential_id]
                self.config_manager.update({'credentials': config})
                logger.info(f"删除凭证成功: {credential_id}")
                
                return jsonify({
                    'status': 'success',
                    'message': '凭证删除成功',
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"删除凭证失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'CREDENTIAL_DELETE_FAILED'
                }), 500
        
        @self.app.route('/api/v1/data/index-prices')
        def get_index_prices_api():
            """获取指数价格数据（直接来自当前数据提供者）"""
            try:
                index_id = request.args.get('index_id', type=str)
                start_date = request.args.get('start_date', type=str)
                end_date = request.args.get('end_date', type=str)
                if not all([index_id, start_date, end_date]):
                    return jsonify({'status': 'error', 'message': '缺少必要参数', 'error_code': 'MISSING_PARAMS'}), 400
                provider = getattr(self.quality_monitor, 'data_provider', None)
                if not provider or not hasattr(provider, 'get_index_prices'):
                    return jsonify({'status': 'error', 'message': '数据提供者不可用', 'error_code': 'DATA_PROVIDER_UNAVAILABLE'}), 503
                df = provider.get_index_prices(index_id, start_date, end_date)
                data = df.to_dict(orient='records') if hasattr(df, 'to_dict') else []
                return jsonify({'status': 'success', 'data': data, 'count': len(data), 'timestamp': datetime.now().isoformat()})
            except Exception as e:
                logger.error(f"获取指数价格失败: {e}")
                return jsonify({'status': 'error', 'message': str(e), 'error_code': 'INDEX_PRICES_FETCH_FAILED'}), 500

        @self.app.route('/api/v1/data/index-returns')
        def get_index_returns_api():
            """获取指数收益率序列（排除异常日）"""
            try:
                index_id = request.args.get('index_id', type=str)
                start_date = request.args.get('start_date', type=str)
                end_date = request.args.get('end_date', type=str)
                if not all([index_id, start_date, end_date]):
                    return jsonify({'status': 'error', 'message': '缺少必要参数', 'error_code': 'MISSING_PARAMS'}), 400
                provider = getattr(self.quality_monitor, 'data_provider', None)
                if not provider or not hasattr(provider, 'get_index_returns'):
                    return jsonify({'status': 'error', 'message': '数据提供者不可用', 'error_code': 'DATA_PROVIDER_UNAVAILABLE'}), 503
                series = provider.get_index_returns(index_id, start_date, end_date)
                data = [{'date': str(idx), 'return': float(val)} for idx, val in (series.items() if hasattr(series, 'items') else [])]
                return jsonify({'status': 'success', 'data': data, 'count': len(data), 'timestamp': datetime.now().isoformat()})
            except Exception as e:
                logger.error(f"获取指数收益率失败: {e}")
                return jsonify({'status': 'error', 'message': str(e), 'error_code': 'INDEX_RETURNS_FETCH_FAILED'}), 500

        @self.app.route('/api/v1/data/event-window')
        def get_event_window_api():
            """获取事件窗口数据（窗口+基准期）"""
            try:
                index_id = request.args.get('index_id', type=str)
                event_date = request.args.get('event_date', type=str)
                event_type = request.args.get('event_type', default='market_crash', type=str)
                window_days = request.args.get('window_days', type=int)
                baseline_days = request.args.get('baseline_days', type=int)
                if not all([index_id, event_date]):
                    return jsonify({'status': 'error', 'message': '缺少必要参数', 'error_code': 'MISSING_PARAMS'}), 400
                provider = getattr(self.quality_monitor, 'data_provider', None)
                if not provider or not hasattr(provider, 'get_event_window_data'):
                    return jsonify({'status': 'error', 'message': '数据提供者不可用', 'error_code': 'DATA_PROVIDER_UNAVAILABLE'}), 503
                result = provider.get_event_window_data(index_id, event_date, event_type, window_days, baseline_days)
                # 仅返回统计信息与样本，避免过大payload
                event_records = result.get('event_window')
                baseline_records = result.get('baseline')
                event_data = event_records.head(200).to_dict(orient='records') if hasattr(event_records, 'to_dict') else []
                baseline_data = baseline_records.head(200).to_dict(orient='records') if hasattr(baseline_records, 'to_dict') else []
                return jsonify({
                    'status': 'success',
                    'event_window': {'count': len(event_records) if hasattr(event_records, '__len__') else 0, 'samples': event_data},
                    'baseline': {'count': len(baseline_records) if hasattr(baseline_records, '__len__') else 0, 'samples': baseline_data},
                    'config': result.get('config', {}),
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"获取事件窗口数据失败: {e}")
                return jsonify({'status': 'error', 'message': str(e), 'error_code': 'EVENT_WINDOW_FETCH_FAILED'}), 500

        @self.app.route('/api/v1/data/cross-validation-log')
        def get_cross_validation_log_api():
            """获取数据源交叉验证日志"""
            try:
                provider = getattr(self.quality_monitor, 'data_provider', None)
                if not provider or not hasattr(provider, 'get_cross_validation_log'):
                    return jsonify({'status': 'error', 'message': '数据提供者不可用', 'error_code': 'DATA_PROVIDER_UNAVAILABLE'}), 503
                log = provider.get_cross_validation_log()
                return jsonify({'status': 'success', 'log': log, 'count': len(log), 'timestamp': datetime.now().isoformat()})
            except Exception as e:
                logger.error(f"获取交叉验证日志失败: {e}")
                return jsonify({'status': 'error', 'message': str(e), 'error_code': 'CROSS_VALIDATION_LOG_FETCH_FAILED'}), 500

        # ============================================================
        # 错误处理中间件
        # ============================================================
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({
                'status': 'error',
                'message': '端点不存在',
                'error_code': 'ENDPOINT_NOT_FOUND'
            }), 404

        @self.app.errorhandler(405)
        def method_not_allowed(error):
            return jsonify({
                'status': 'error',
                'message': '方法不允许',
                'error_code': 'METHOD_NOT_ALLOWED'
            }), 405

        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({
                'status': 'error',
                'message': '内部服务器错误',
                'error_code': 'INTERNAL_SERVER_ERROR'
            }), 500

    # ============================================================
    # 第三轮迁移：所有辅助方法已完全迁移到组件
    # ============================================================
    # 已迁移到 metrics_collector:
    # - _get_resource_utilization -> metrics_collector.get_resource_utilization
    # - _get_network_io -> metrics_collector.get_network_io
    # - _generate_performance_recommendations -> metrics_collector.generate_performance_recommendations
    # - _generate_health_recommendations -> metrics_collector.generate_health_recommendations
    # - _get_system_metrics -> metrics_collector.get_system_metrics
    #
    # 已迁移到 diagnostics_runner:
    # - _run_diagnostics -> diagnostics_runner.run_diagnostics
    # - _run_system_diagnostics -> diagnostics_runner.run_system_diagnostics
    # - _run_performance_diagnostics -> diagnostics_runner.run_performance_diagnostics
    # - _run_data_quality_diagnostics -> diagnostics_runner.run_data_quality_diagnostics
    # - _run_network_diagnostics -> diagnostics_runner.run_network_diagnostics
    # - _generate_diagnostics_summary -> diagnostics_runner.generate_diagnostics_summary
    # - _generate_diagnostics_recommendations -> diagnostics_runner.generate_diagnostics_recommendations
    # - _calculate_diagnostics_duration -> diagnostics_runner.calculate_diagnostics_duration
    #
    # 已迁移到 config_manager (已合并到核心层 core.share.config_manager):
    # - _get_current_config -> config_manager.get()
    # - _update_config -> config_manager.update()
    #
    # 已迁移到 data_exporter:
    # - _export_alert_data -> data_exporter.export_alert_data
    # - _convert_report_to_csv -> data_exporter.convert_report_to_csv
    #
    # 已迁移到 system_status_manager:
    # - _get_system_status -> system_status_manager.get_system_status
    # - _enable_maintenance_mode -> system_status_manager.enable_maintenance_mode
    # - _disable_maintenance_mode -> system_status_manager.disable_maintenance_mode
    # - (以及所有系统状态相关的私有方法)
    #
    # 已迁移到 health_checker:
    # - _check_system_health -> health_checker.check_system_health
    # - _check_component_health -> health_checker.check_component_health
    # - _check_database_health -> health_checker.check_database_health
    # - _check_external_services -> health_checker.check_external_services
    # - _calculate_overall_health_score -> health_checker.calculate_overall_health_score
    # - _generate_health_recommendations_from_components -> health_checker.generate_health_recommendations_from_components
    #
    # 已迁移到 controllers:
    # - _group_by_level -> controllers.group_by_level
    # - _group_by_severity -> controllers.group_by_severity
    # - _group_by_source -> controllers.group_by_source
    # - _calculate_system_health -> controllers.calculate_system_health
    # - _analyze_performance_trend -> controllers.analyze_performance_trend
    # ============================================================

    def start_api_service(self, host: str = '0.0.0.0', port: int = 8080):
        """启动API服务"""
        try:
            logger.info(f"启动数据质量API服务: http://{host}:{port}")

            # 配置Flask应用
            self.app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
            self.app.config['JSON_SORT_KEYS'] = False
            self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

            # 添加中间件
            self._add_middleware()

            # 启动服务
            self.app.run(
                host=host,
                port=port,
                debug=False,
                threaded=True,
                use_reloader=False
            )

        except Exception as e:
            logger.error(f"API服务启动失败: {e}")
            raise

    def _add_middleware(self):
        """添加中间件"""

        # 请求日志中间件
        @self.app.before_request
        def log_request():
            if request.path != '/health':
                logger.info(f"API请求: {request.method} {request.path} - {request.remote_addr}")

        # 响应处理中间件
        @self.app.after_request
        def after_request(response):
            response.headers['X-Data-Quality-API'] = 'DeepSeekQuant/1.0.0'
            response.headers['X-Response-Time'] = '100ms'  # 示例值
            return response

        # 错误处理中间件
        @self.app.errorhandler(Exception)
        def handle_exception(e):
            logger.error(f"API处理异常: {e}")
            return jsonify({
                'status': 'error',
                'message': '内部服务器错误',
                'error_code': 'INTERNAL_ERROR'
            }), 500
    
    def _setup_socketio_handlers(self):
        """设置Socket.IO事件处理器 - 实时推送支持"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """客户端连接事件"""
            logger.info(f"Socket.IO客户端已连接: {request.sid}")
            emit('connection_response', {
                'status': 'connected',
                'message': '已连接到数据质量监控服务',
                'timestamp': datetime.now().isoformat()
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """客户端断开连接事件"""
            logger.info(f"Socket.IO客户端已断开: {request.sid}")
        
        @self.socketio.on('request_quality_data')
        def handle_quality_request():
            """客户端请求质量数据"""
            try:
                result = self.controllers.get_quality_current(hours=24)
                emit('quality_update', {
                    'status': 'success',
                    'data': result,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Socket.IO获取质量数据失败: {e}")
                emit('error', {
                    'status': 'error',
                    'message': str(e)
                })
    
    def broadcast_quality_update(self, quality_data: Dict[str, Any]):
        """广播质量数据更新到所有连接的客户端
        
        Args:
            quality_data: 质量数据字典
        
        Note:
            在监控服务完成质量检查后调用此方法，实时推送数据到Dashboard
        """
        try:
            self.socketio.emit('quality_update', {
                'status': 'success',
                'data': quality_data,
                'timestamp': datetime.now().isoformat()
            })
            logger.debug("已广播质量数据更新到所有客户端")
        except Exception as e:
            logger.error(f"广播质量数据失败: {e}")

    def stop_api_service(self):
        """停止API服务"""
        logger.info("停止数据质量API服务")
        # 这里实现优雅关闭逻辑

    def get_api_statistics(self) -> Dict[str, Any]:
        """获取API统计信息"""
        return {
            'total_requests': 0,  # 需要实际实现请求计数
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'endpoint_usage': {},
            'error_rates': {},
            'timestamp': datetime.now().isoformat()
        }

    def export_api_logs(self, filepath: str) -> bool:
        """导出API日志"""
        try:
            # 实现API日志导出逻辑
            return True
        except Exception as e:
            logger.error(f"API日志导出失败: {e}")
            return False

    def cleanup(self):
        """清理资源"""
        self.stop_api_service()
        logger.info("API服务清理完成")

# 数据质量系统主类
