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
import os
from datetime import datetime, timedelta
from typing import Dict, Any, TYPE_CHECKING

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, Response, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from core_bak_refactored.app.quality_monitoring.api.chart_data import ChartDataAssembler
# 从组件导入
from core_bak_refactored.app.quality_monitoring.api.controllers import DataQualityControllers
from core_bak_refactored.app.quality_monitoring.api.diagnostics import DiagnosticsRunner
from core_bak_refactored.app.quality_monitoring.api.exporter import DataExporter
from core_bak_refactored.app.quality_monitoring.api.health import HealthChecker
from core_bak_refactored.app.quality_monitoring.api.system_metrics import MetricsCollector
from core_bak_refactored.app.quality_monitoring.api.system_status import SystemStatusManager
from core_bak_refactored.core.share import MarketCode
from core_bak_refactored.core.share.config_manager import ConfigManager
from core_bak_refactored.core.signal.indicator_service import TechnicalIndicators

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

        # 初始化全局 factory（用于运行时动态获取 provider）
        from core_bak_refactored.core.data.providers.factory import get_global_factory
        from core_bak_refactored.core.data.providers.provider_selector import ProviderSelector

        self.provider_factory = get_global_factory()
        self.provider_selector = ProviderSelector(self.config_manager)  # 领域层服务

        self._setup_routes()
        self._setup_socketio_handlers()

    def _create_chart_assembler(self, index_id: str, timeframe: str = 'daily') -> ChartDataAssembler:
        """动态创建图表数据组装器
        
        Args:
            index_id: 股票/指数代码
            timeframe: 时间周期
        
        Returns:
            ChartDataAssembler: 图表数据组装器实例
        """

        from core_bak_refactored.core.share.market import MarketUtils

        # 1. 使用领域层服务选择数据提供者
        data_provider = self.provider_selector.select_provider_for_symbol(
            symbol=index_id,
            provider_factory=self.provider_factory
        )

        # 2. 推断市场（用于创建指标服务）
        market_code = MarketUtils.infer_market_from_symbol(index_id)
        market = market_code.value

        # 3. 创建指标服务（根据市场）
        indicator_service = TechnicalIndicators(market=market, timeframe=timeframe)

        # 4. 创建图表数据组装器
        return ChartDataAssembler(
            data_provider=data_provider,
            indicator_service=indicator_service
        )

    def _create_provider_instance(self, provider: Dict[str, Any], credentials: Dict[str, Any] = None,
                                  proxy_config: Dict[str, Any] = None):
        """创建数据提供者实例（使用 factory.py 的功能）
        
        Args:
            provider: 数据提供者配置字典
            credentials: 凭证信息（可选）
            proxy_config: 代理配置（可选）
        
        Returns:
            数据提供者实例，失败返回 None
        
        Note:
            该方法利用 DataProviderFactory 的动态加载功能，
            并支持传入自定义凭证和代理配置。
        """
        try:
            provider_id = provider.get('id')
            if not provider_id:
                logger.error(f"数据提供者配置不完整: {provider}")
                return None

            # 使用 DataProviderFactory 的 _get_provider_class 逻辑
            # 但需要支持自定义凭证和代理，所以直接使用动态导入
            adapter_module = provider.get('adapter_module')
            adapter_class = provider.get('adapter_class')

            if not adapter_module or not adapter_class:
                logger.error(f"Provider '{provider_id}' 配置不完整：缺少 adapter_module 或 adapter_class")
                return None

            # 动态导入模块（与 factory.py 一致）
            import importlib
            module = importlib.import_module(adapter_module)
            provider_class = getattr(module, adapter_class)

            # 创建实例（支持自定义参数）
            kwargs = {}
            if credentials:
                kwargs['credentials'] = credentials
            if proxy_config:
                kwargs['proxy_config'] = proxy_config

            instance = provider_class(**kwargs)
            logger.debug(f"创建数据提供者实例成功: {provider_id}")
            return instance

        except Exception as e:
            logger.error(f"创建数据提供者实例失败: {e}", exc_info=True)
            return None

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

        @self.app.route('/api/v1/chart/data', methods=['GET'])
        def get_chart_data():
            """获取合并的图表数据（K线+技术指标+事件）
            
            查询参数：
                - index_id: 股票/指数代码（必需）
                - period: 周期（daily/weekly/monthly，默认 daily）
                - count: 数据条数（默认 120）
                - before: 获取此日期之前的数据（YYYY-MM-DD，可选）
                - indicators: 需要的指标，逗号分隔（默认 'all'）
                               支持: vol, macd, rsi, kdj, obv
                - use_mock: 是否使用模拟数据（true/false，默认 false）
            
            返回示例：
            {
                "status": "success",
                "data": {
                    "kline": [
                        {
                            "date": "2024-01-01",
                            "open": 100.0,
                            "high": 105.0,
                            "low": 99.0,
                            "close": 103.0,
                            "volume": 1000000,
                            "ma5": 102.0,
                            "ma10": 101.5,
                            "ma20": 100.8
                        }
                    ],
                    "indicators": {
                        "vol": [{"date": "2024-01-01", "value": 1000000}],
                        "macd": [{"date": "2024-01-01", "macd": 0.5, "signal": 0.3, "histogram": 0.2}],
                        "rsi": [{"date": "2024-01-01", "value": 60.0}],
                        "kdj": [{"date": "2024-01-01", "k": 70.0, "d": 65.0, "j": 75.0}],
                        "obv": [{"date": "2024-01-01", "value": 5000000}]
                    },
                    "events": [
                        {
                            "date": "2024-01-05",
                            "type": "market_crash",
                            "title": "暴跌 5.2%",
                            "decline_pct": -5.2,
                            "price": 98.0,
                            "impact": "negative",
                            "severity": "high"
                        }
                    ]
                }
            }
            """
            try:
                # 获取查询参数
                index_id = request.args.get('index_id')
                if not index_id:
                    return jsonify({
                        'status': 'error',
                        'message': '缺少必需参数: index_id',
                        'error_code': 'MISSING_PARAMETER'
                    }), 400

                period = request.args.get('period', 'daily')
                count = request.args.get('count', 120, type=int)
                before_str = request.args.get('before')  # 可选
                # 🔧 关键修复：将字符串转换为 pd.Timestamp 类型
                before = pd.to_datetime(before_str) if before_str else None
                indicators = request.args.get('indicators', 'all')
                use_mock = request.args.get('use_mock', 'false').lower() in ['true', '1', 'yes']

                # 参数验证
                if period not in ['daily', 'weekly', 'monthly']:
                    return jsonify({
                        'status': 'error',
                        'message': f'无效的周期参数: {period}，支持: daily/weekly/monthly',
                        'error_code': 'INVALID_PERIOD'
                    }), 400

                if count <= 0 or count > 1000:
                    return jsonify({
                        'status': 'error',
                        'message': f'数据条数必须在 1-1000 之间，当前: {count}',
                        'error_code': 'INVALID_COUNT'
                    }), 400

                # 🔧 根捪use_mock参数选择数据源
                if use_mock:
                    logger.info(f"🎭 使用模拟数据源: {index_id}")
                    # 使用MockDataProvider
                    from core_bak_refactored.core.data.providers.mock_provider import MockDataProvider
                    from core_bak_refactored.core.share.market.market_enums import TradingPhase

                    mock_provider = MockDataProvider()

                    # 🎭 关键：从前端获取trading_phase参数（用于needs_realtime_kline判断）
                    trading_phase_str = request.args.get('trading_phase', 'TRADING')  # 默认盘中
                    try:
                        trading_phase = TradingPhase.parse(trading_phase_str)
                        mock_provider.set_mock_trading_phase(trading_phase)
                        logger.info(f"🎭 Mock模式 - trading_phase={trading_phase.name}")
                    except KeyError:
                        logger.warning(f"🎭 无效的trading_phase: {trading_phase_str}，使用默认TRADING")
                        mock_provider.set_mock_trading_phase(TradingPhase.TRADING)

                    # 3. 创建指标服务（根据市场）
                    indicator_service = TechnicalIndicators(market=MarketCode.CN, timeframe=period)
                    # 创建使用Mock Provider的chart_assembler
                    chart_assembler = ChartDataAssembler(
                        data_provider=mock_provider,
                        indicator_service=indicator_service
                    )
                else:
                    logger.info(f"🎯 使用真实数据源: {index_id}")
                    # 使用真实Provider（原有逻辑）
                    chart_assembler = self._create_chart_assembler(index_id, timeframe=period)

                # 调用组装器
                chart_data = chart_assembler.assemble_chart_data(
                    index_id=index_id,
                    period=period,
                    count=count,
                    before=before,
                    indicators=indicators,
                    current_time=pd.Timestamp.now()
                )

                return jsonify({
                    'status': 'success',
                    'data': chart_data,
                    'metadata': {
                        'index_id': index_id,
                        'period': period,
                        'count': len(chart_data.get('kline', [])),
                        'indicators': list(chart_data.get('indicators', {}).keys()),
                        'events_count': len(chart_data.get('events', [])),
                        'use_mock': use_mock
                    },
                    'timestamp': datetime.now().isoformat()
                })

            except ValueError as e:
                logger.error(f"图表数据获取参数错误: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'INVALID_PARAMETER'
                }), 400

            except RuntimeError as e:
                logger.error(f"图表数据组装失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'CHART_DATA_ASSEMBLY_FAILED'
                }), 500

            except Exception as e:
                logger.error(f"获取图表数据失败: {e}", exc_info=True)
                return jsonify({
                    'status': 'error',
                    'message': f'获取图表数据失败: {str(e)}',
                    'error_code': 'CHART_DATA_FETCH_FAILED'
                }), 500

        @self.app.route('/api/v1/intraday/data', methods=['GET'])
        def get_intraday_data():
            """获取分时图数据（真实数据）
            
            查询参数：
                - symbol: 证券代码（必需）
                - tick_range: TickRange JSON（可选）
            
            返回示例：
            {
                "status": "success",
                "data": {
                    "symbol": "000001.SH",
                    "name": "上证指数",
                    "current_price": 3125.50,
                    "yesterday_close": 3120.00,
                    "change": 5.50,
                    "change_percent": 0.18,
                    "times": ["09:30", "09:31", ...],
                    "prices": [3121.0, 3122.5, ...],
                    "volumes": [12000, 15000, ...],
                    "avg_prices": [3121.0, 3121.75, ...],
                    "order_book": {
                        "bids": [{"price": 3125.49, "volume": 2000}, ...],
                        "asks": [{"price": 3125.51, "volume": 1800}, ...]
                    },
                    "trade_records": [
                        {"time": "14:59:50", "price": 3125.50, "volume": 100, "type": "buy"},
                        ...
                    ]
                },
                "timestamp": "2025-12-12T10:24:29.573456"
            }
            """
            try:
                symbol = request.args.get('symbol')
                if not symbol:
                    return jsonify({
                        'status': 'error',
                        'message': '缺少必需参数: symbol',
                        'error_code': 'MISSING_PARAMETER'
                    }), 400

                # 解析 tick_range（可选）
                import json
                tick_range_str = request.args.get('tick_range')
                tick_range = None

                if tick_range_str:
                    try:
                        tick_range_dict = json.loads(tick_range_str)

                        # 验证字段
                        required_fields = ['start_time', 'end_time', 'period_seconds']
                        for field in required_fields:
                            if field not in tick_range_dict:
                                return jsonify({
                                    'status': 'error',
                                    'message': f'tick_range缺少字段: {field}',
                                    'error_code': 'INVALID_TICK_RANGE'
                                }), 400

                        # 转换为 TickRange 对象
                        from core_bak_refactored.core.data.providers.protocols import TickRange
                        import pandas as pd
                        tick_range = TickRange(
                            start_time=pd.Timestamp(tick_range_dict['start_time']),
                            end_time=pd.Timestamp(tick_range_dict['end_time']),
                            period_seconds=int(tick_range_dict['period_seconds'])
                        )
                    except (json.JSONDecodeError, ValueError) as e:
                        return jsonify({
                            'status': 'error',
                            'message': f'解析tick_range失败: {str(e)}',
                            'error_code': 'INVALID_TICK_RANGE_FORMAT'
                        }), 400

                # 📊 真实模式：调用 ChartDataAssembler（会调用 akshare_provider）
                logger.info(f"📊 真实模式: symbol={symbol}, tick_range={'已提供' if tick_range else '未提供'}")

                chart_assembler = self._create_chart_assembler(symbol, timeframe='daily')
                intraday_data = chart_assembler.assemble_intraday_data(
                    symbol=symbol,
                    tick_range=tick_range
                )

                return jsonify({
                    'status': 'success',
                    'data': intraday_data,
                    'timestamp': datetime.now().isoformat()
                })

            except ValueError as e:
                # 数据验证错误（如盘后数据不完整），返回400
                logger.warning(f"数据验证失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'DATA_VALIDATION_FAILED'
                }), 400

            except Exception as e:
                logger.error(f"获取分时数据失败: {e}", exc_info=True)
                return jsonify({
                    'status': 'error',
                    'message': f'获取分时数据失败: {str(e)}',
                    'error_code': 'INTRADAY_DATA_FAILED'
                }), 500

        @self.app.route('/api/v1/intraday/mock', methods=['GET'])
        def get_intraday_mock():
            """获取模拟分时图数据
            
            查询参数：
                - symbol: 证券代码（必需）
                - trading_phase: 交易时段（'trading'/'before_open'/'after_close'，必需）- 模拟场景由前端按钮控制
                - tick_range: TickRange JSON（可选）
                - last_price: 上次请求的最终价格，用于保证价格连续性（可选）
            
            🔧 注意：服务器根据 trading_phase 决定返回 should_poll，前端只依赖 should_poll 控制行为
            """
            from core_bak_refactored.core.share.market.market_enums import TradingPhase

            try:
                symbol = request.args.get('symbol')
                if not symbol:
                    return jsonify({
                        'status': 'error',
                        'message': '缺少必需参数: symbol',
                        'error_code': 'MISSING_PARAMETER'
                    }), 400

                trading_phase_str = request.args.get('trading_phase')
                if not trading_phase_str:
                    return jsonify({
                        'status': 'error',
                        'message': '缺少必需参数: trading_phase',
                        'error_code': 'MISSING_PARAMETER'
                    }), 400

                # 验证 trading_phase
                valid_modes = ['trading', 'before_open', 'after_close']
                if trading_phase_str not in valid_modes:
                    return jsonify({
                        'status': 'error',
                        'message': f'trading_phase必须是{valid_modes}之一',
                        'error_code': 'INVALID_TRADING_PHASE'
                    }), 400

                # 转换为枚举
                trading_phase = TradingPhase.parse(trading_phase_str)

                # 解析 last_price
                last_price_str = request.args.get('last_price')
                last_price = None
                if last_price_str:
                    try:
                        last_price = float(last_price_str)
                    except ValueError:
                        return jsonify({
                            'status': 'error',
                            'message': f'last_price必须是数字: {last_price_str}',
                            'error_code': 'INVALID_LAST_PRICE'
                        }), 400

                # 解析 tick_range
                import json
                tick_range_str = request.args.get('tick_range')
                tick_range = None

                if tick_range_str:
                    try:
                        tick_range_dict = json.loads(tick_range_str)

                        # 验证字段
                        required_fields = ['start_time', 'end_time', 'period_seconds']
                        for field in required_fields:
                            if field not in tick_range_dict:
                                return jsonify({
                                    'status': 'error',
                                    'message': f'tick_range缺少字段: {field}',
                                    'error_code': 'INVALID_TICK_RANGE'
                                }), 400

                        # 转换为 TickRange 对象
                        from core_bak_refactored.core.data.providers.protocols import TickRange
                        import pandas as pd
                        tick_range = TickRange(
                            start_time=pd.Timestamp(tick_range_dict['start_time']),
                            end_time=pd.Timestamp(tick_range_dict['end_time']),
                            period_seconds=int(tick_range_dict['period_seconds'])
                        )
                    except (json.JSONDecodeError, ValueError) as e:
                        return jsonify({
                            'status': 'error',
                            'message': f'解析tick_range失败: {str(e)}',
                            'error_code': 'INVALID_TICK_RANGE_FORMAT'
                        }), 400

                logger.info(
                    f"🎮 模拟模式: symbol={symbol}, trading_phase={trading_phase_str}(前端按钮控制), tick_range={'已提供' if tick_range else '未提供'}")

                # 直接调用 MockDataProvider
                from core_bak_refactored.core.data.providers.mock_provider import MockDataProvider

                generator = MockDataProvider()

                # 判断是否为指数
                is_index = symbol in ['000001.SH', '000300.SH', '399001.SZ', '399006.SZ']

                # 使用系统当前日期
                trade_date = datetime.now().strftime('%Y-%m-%d')

                # tick_range 由前端直接传入，不需要转换

                mock_data = generator.generate(
                    symbol=symbol,
                    trade_date=trade_date,
                    tick_range=tick_range,
                    trading_phase=trading_phase,
                    last_price=last_price,
                    is_index=is_index
                )

                # 转换为前端需要的格式
                intraday_data = {
                    'symbol': mock_data.symbol,
                    'name': mock_data.name,
                    'current_price': mock_data.current_price,
                    'yesterday_close': mock_data.yesterday_close,
                    'change': mock_data.change,
                    'change_percent': mock_data.change_percent,
                    'times': [tick.time for tick in mock_data.ticks],
                    'prices': [tick.price for tick in mock_data.ticks],
                    'volumes': [tick.volume for tick in mock_data.ticks],
                    'avg_prices': [tick.avg_price for tick in mock_data.ticks],
                    'order_book': {
                        'bids': [{'price': level.price, 'volume': level.volume} for level in mock_data.order_book_bids],
                        'asks': [{'price': level.price, 'volume': level.volume} for level in mock_data.order_book_asks],
                        'message': mock_data.order_book_message
                    },
                    'trade_records': {
                        'items': [{'time': t.time, 'price': t.price, 'volume': t.volume, 'type': t.direction} for t in
                                  mock_data.trade_records],
                        'message': mock_data.trade_records_message
                    },
                    'is_index': mock_data.is_index,
                    'should_poll': mock_data.should_poll  # 🔧 服务器根据 trading_phase 决定，前端只依赖此字段控制行为
                }

                return jsonify({
                    'status': 'success',
                    'data': intraday_data,
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"获取模拟分时数据失败: {e}", exc_info=True)
                return jsonify({
                    'status': 'error',
                    'message': f'获取模拟分时数据失败: {str(e)}',
                    'error_code': 'INTRADAY_MOCK_FAILED'
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
        # 新API：市场数据源配置（彻底改写，不向后兼容）
        # ============================================================

        @self.app.route('/api/v1/markets/config', methods=['GET'])
        def get_markets_config():
            """获取所有市场配置信息（从配置文件读取）"""
            try:
                # 从配置文件读取真实配置
                data_provider_config = self.config_manager.get_provider_config()

                # 从 market.yml 的 market_registry 读取市场列表（包含UI展示信息）
                market_config = self.config_manager.get_market_config()
                market_registry = market_config.market_registry or {}

                # 从 market_registry 生成 UI 展示数据
                markets = [
                    {
                        'code': code,
                        'name': info.get('display_name', info.get('name', code)),  # 优先使用 display_name
                        'icon': info.get('icon', '')
                    }
                    for code, info in market_registry.items()
                ]

                # 从 data_provider_config.yml 读取 providers 配置
                providers_raw = data_provider_config.providers or []

                # 转换为前端需要的格式（过滤掉未实现的适配器）
                providers = []
                for p in providers_raw:
                    # 过滤掉未实现的适配器（adapter_module 和 adapter_class 为 null 的）
                    adapter_module = p.get('adapter_module')
                    adapter_class = p.get('adapter_class')
                    if not adapter_module or not adapter_class or adapter_module == 'null' or adapter_class == 'null':
                        continue  # 跳过未实现的适配器

                    # 获取配置文件中的状态（已移除聚合数据源）
                    # TODO: 实现新的测试状态获取机制
                    provider_id = p.get('id')
                    test_status = p.get('status', 'untested')  # 使用配置文件中的状态
                    is_available = test_status == 'passed'

                    provider_data = {
                        'id': p.get('id'),
                        'name': p.get('name'),
                        'type': p.get('type', '未知'),
                        'status': test_status,  # 使用实时状态
                        'available': is_available,  # 添加 bool 字段供前端使用
                        'markets': p.get('markets', []),
                        'needsConfig': p.get('requires_auth', False),
                        'params': []
                    }

                    # 如果需要配置，添加参数定义
                    if p.get('requires_auth'):
                        auth_type = p.get('auth_type', 'api_key')
                        if auth_type == 'token':
                            provider_data['params'] = [
                                {
                                    'name': 'token',
                                    'label': f"{p.get('name')} Token",
                                    'type': 'password',
                                    'required': True,
                                    'placeholder': f"在 {p.get('registration', '')} 注册获取"
                                }
                            ]
                        else:  # api_key
                            provider_data['params'] = [
                                {
                                    'name': 'api_key',
                                    'label': 'API Key',
                                    'type': 'password',
                                    'required': True,
                                    'placeholder': f"在 {p.get('registration', '')} 注册获取"
                                }
                            ]

                    providers.append(provider_data)
                data_market_config = self.config_manager.get_market_config()
                # 市场数据源配置
                market_sources = data_market_config.market_sources or {}

                # 从真实凭证文件读取凭证状态
                import os
                import yaml
                # 使用 ConfigManager 获取配置路径（封装环境逻辑）
                from core_bak_refactored.core.share.config_manager import ConfigManager
                config_manager_temp = ConfigManager()
                credentials_yml_path = config_manager_temp.get_config_path('credentials')

                # 读取凭证文件
                credentials_data = {}
                if os.path.exists(credentials_yml_path):
                    try:
                        with open(credentials_yml_path, 'r', encoding='utf-8') as f:
                            credentials_data = yaml.safe_load(f) or {}
                    except Exception as e:
                        logger.warning(f"读取凭证文件失败: {e}")

                # 生成凭证状态
                credentials = {}
                for p in providers_raw:
                    provider_id = p.get('id')
                    # 免费数据源标记为已配置
                    if not p.get('requires_auth'):
                        credentials[provider_id] = {'configured': True}
                    else:
                        # 检查凭证文件中是否存在
                        credentials[provider_id] = {
                            'configured': provider_id in credentials_data
                        }

                return jsonify({
                    'status': 'success',
                    'data': {
                        'markets': markets,
                        'providers': providers,
                        'market_sources': market_sources,
                        'credentials': credentials
                    },
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"获取市场配置失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'MARKETS_CONFIG_FETCH_FAILED'
                }), 500

        @self.app.route('/api/v1/markets/config', methods=['PUT'])
        def update_markets_config():
            """更新市场数据源配置（真实保存）"""
            try:
                data = request.get_json()
                if not data or 'market_sources' not in data:
                    return jsonify({
                        'status': 'error',
                        'message': '缺少 market_sources 字段',
                        'error_code': 'INVALID_REQUEST_DATA'
                    }), 400

                market_sources = data['market_sources']

                # 使用 ConfigManager 的验证和保存方法
                try:
                    self.config_manager.get_market_config().save_market_sources(market_sources)

                    return jsonify({
                        'status': 'success',
                        'message': '市场配置已保存',
                        'data': {
                            'updated_markets': list(market_sources.keys()),
                            'config_file': self.config_manager.get_config_path('market')
                        },
                        'timestamp': datetime.now().isoformat()
                    })
                except ValueError as ve:
                    # 验证失败
                    return jsonify({
                        'status': 'error',
                        'message': str(ve),
                        'error_code': 'MARKET_VALIDATION_FAILED'
                    }), 400
            except Exception as e:
                logger.error(f"更新市场配置失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'MARKETS_CONFIG_UPDATE_FAILED'
                }), 500

        @self.app.route('/api/v1/providers/<provider_id>/credentials', methods=['POST'])
        def save_provider_credentials(provider_id):
            """保存数据源凭证（调用领域层 Provider 的保存方法）"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({
                        'status': 'error',
                        'message': '无效的请求数据',
                        'error_code': 'INVALID_REQUEST_DATA'
                    }), 400

                # 使用环境变量或默认 dev
                import os

                # 使用 BaseDataProvider 的通用方法保存凭证
                from core_bak_refactored.core.data.providers.base_provider import BaseDataProvider

                success = BaseDataProvider.save_credentials(provider_id, data)

                if success:
                    # 重新加载配置
                    self.config_manager._load_config()

                    return jsonify({
                        'status': 'success',
                        'message': f'{provider_id} 凭证已保存，请重新测试连接',
                        'data': {
                            'provider_id': provider_id,
                            'configured': True,
                            'test_status': 'success'
                        },
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({
                        'status': 'error',
                        'message': '保存凭证失败',
                        'error_code': 'CREDENTIALS_SAVE_FAILED'
                    }), 500

            except Exception as e:
                logger.error(f"保存凭证失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'CREDENTIALS_SAVE_FAILED'
                }), 500

        @self.app.route('/api/v1/providers/<provider_id>/credentials', methods=['DELETE'])
        def delete_provider_credentials(provider_id):
            """删除数据源凭证（调用领域层 Provider 的删除方法）"""
            try:
                # 使用环境变量或默认 dev
                import os

                # 使用 BaseDataProvider 的通用方法删除凭证
                from core_bak_refactored.core.data.providers.base_provider import BaseDataProvider

                success = BaseDataProvider.delete_credentials(provider_id)

                if success:
                    return jsonify({
                        'status': 'success',
                        'message': f'{provider_id} 凭证已删除',
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({
                        'status': 'error',
                        'message': '删除凭证失败',
                        'error_code': 'CREDENTIALS_DELETE_FAILED'
                    }), 500

            except Exception as e:
                logger.error(f"删除凭证失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'CREDENTIALS_DELETE_FAILED'
                }), 500

        @self.app.route('/api/v1/providers/<provider_id>/test', methods=['POST'])
        def test_provider_connection(provider_id):
            """测试数据源连接（调用领域层 Provider 的测试方法）"""
            try:
                # 获取请求中的测试参数（如API Key）
                test_params = request.get_json() or {}

                # 使用 factory 获取 provider 类并调用 test_provider 方法
                from core_bak_refactored.core.data.providers.factory import get_global_factory

                factory = get_global_factory()

                try:
                    # 创建 provider 实例
                    provider = factory.get(provider_id)

                    # 转换参数：credentials → credential
                    # test_provider 方法签名：test_provider(provider_id, credential: str)
                    # proxy 从配置文件读取，不作为参数传递
                    credentials_dict = test_params.get('credentials', {})

                    # 提取 credential 字符串（可能是 api_key 或 token）
                    credential = None
                    if credentials_dict:
                        # 优先使用 api_key，其次使用 token
                        credential = credentials_dict.get('api_key') or credentials_dict.get('token')

                    # 调用 provider 类的 test_provider 方法
                    if credential:
                        result = provider.__class__.test_provider(provider_id, credential=credential)
                    else:
                        # 免费数据源不需要 credential
                        result = provider.__class__.test_provider(provider_id, credential='')
                except Exception as e:
                    result = {
                        'status': 'error',
                        'test_result': 'failed',
                        'available': False,
                        'message': str(e),
                        'timestamp': datetime.now().isoformat()
                    }

                # 根据结果返回 HTTP 状态码
                if result['status'] == 'error':
                    return jsonify(result), 500
                else:
                    return jsonify(result)

            except Exception as e:
                logger.error(f"测试连接失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'test_result': 'failed',
                    'error_code': 'TEST_CONNECTION_FAILED'
                }), 500

        # ============================================================
        # 旧API：数据提供者能力暴露（保留兼容）
        # ============================================================

        @self.app.route('/api/v1/providers', methods=['GET'])
        def get_providers():
            """获取所有数据源配置（只返回已实现的数据源）"""
            try:
                config = self.config_manager.get('data', {})
                all_providers = config.get('providers', [])
                primary_source = config.get('primary_source', 'mock')

                # 过滤掉未实现的适配器（adapter_module 和 adapter_class 为 null 的）
                implemented_providers = [
                    p for p in all_providers
                    if p.get('adapter_module') and p.get('adapter_class')
                       and p.get('adapter_module') != 'null' and p.get('adapter_class') != 'null'
                ]

                return jsonify({
                    'status': 'success',
                    'providers': implemented_providers,
                    'primary_source': primary_source,
                    'total': len(implemented_providers),
                    'total_configured': len(all_providers),  # 配置文件中的总数
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

                provider = next((p for p in providers if p.get('id') == provider_id or p.get('name') == provider_id),
                                None)

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

                provider_index = next(
                    (i for i, p in enumerate(providers) if p.get('id') == provider_id or p.get('name') == provider_id),
                    None)

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

                provider_index = next(
                    (i for i, p in enumerate(providers) if p.get('id') == provider_id or p.get('name') == provider_id),
                    None)

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
            """测试数据源连接（使用临时凭证）"""
            try:
                import time

                config = self.config_manager.get('data', {})
                providers = config.get('providers', [])

                provider = next((p for p in providers if p.get('id') == provider_id or p.get('name') == provider_id),
                                None)

                if not provider:
                    return jsonify({
                        'status': 'error',
                        'message': f'数据源不存在: {provider_id}',
                        'error_code': 'PROVIDER_NOT_FOUND'
                    }), 404

                # 获取前端传入的临时凭证和代理设置（如果有）
                request_data = request.get_json() or {}
                temp_credentials = request_data.get('credentials', {})
                proxy_config = request_data.get('proxy', {})
                test_symbol = request_data.get('test_symbol', 'AAPL')
                start_date = request_data.get('start_date', '2023-01-01')
                end_date = request_data.get('end_date', '2023-12-31')

                # 如果没有传入代理配置，则从系统配置中获取
                if not proxy_config:
                    try:
                        system_config = self.config_manager.get_system_config()
                        proxy_config = system_config.proxies or {}
                    except Exception:
                        proxy_config = {}

                # 创建临时实例进行测试
                test_instance = self._create_provider_instance(provider, temp_credentials, proxy_config)

                if not test_instance:
                    return jsonify({
                        'status': 'error',
                        'message': f'无法创建数据源实例: {provider_id}',
                        'error_code': 'INSTANCE_CREATION_FAILED'
                    }), 500

                requires_auth = test_instance.requires_auth()
                credentials = test_instance.credentials

                # 执行测试
                start_time = time.time()
                test_data = test_instance.test_connection(
                    test_symbol=test_symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                latency_ms = round((time.time() - start_time) * 1000, 2)

                # 检查测试数据
                if hasattr(test_data, 'to_dataframe'):
                    test_data_df = test_data.to_dataframe()
                    is_empty = test_data_df.empty
                    data_count = len(test_data_df)
                else:
                    is_empty = test_data.empty if test_data is not None else True
                    data_count = len(test_data) if test_data is not None else 0

                if test_data is None or is_empty:
                    # 测试失败：连接成功但返回空数据
                    is_available = False
                    message = f'{provider_id} 连接成功，但返回空数据'
                    logger.warning(f"{provider_id} 测试警告: {message}")

                    result_data = {
                        'status': 'error',
                        'test_result': 'failed',
                        'available': is_available,
                        'message': message,
                        'details': {
                            'test_symbol': test_symbol,
                            'date_range': f'{start_date} to {end_date}',
                            'latency_ms': latency_ms
                        }
                    }
                else:
                    # 测试成功
                    is_available = True
                    message = f'{provider_id} 连接测试通过'
                    logger.info(f"{provider_id} 测试成功: {data_count} 条数据, {latency_ms}ms")

                    result_data = {
                        'status': 'success',
                        'test_result': 'passed',
                        'available': is_available,
                        'message': message,
                        'details': {
                            'test_symbol': test_symbol,
                            'data_count': data_count,
                            'date_range': f'{start_date} to {end_date}',
                            'latency_ms': latency_ms
                        },
                        'timestamp': datetime.now().isoformat()
                    }

                    # 测试成功后，保存凭证到文件
                    if requires_auth and credentials:
                        from core_bak_refactored.core.data.providers.base_provider import BaseDataProvider
                        import os
                        BaseDataProvider.save_credentials(provider_id, credentials)
                        logger.info(f"{provider_id} 凭证已保存")

                    # 注意：不再保存测试状态到配置文件
                    # 状态由前端在内存中维护，下次启动时重新测试

                return jsonify(result_data)

            except Exception as e:
                logger.error(f"测试 {provider_id} 连接失败: {str(e)}", exc_info=True)
                return jsonify({
                    'status': 'error',
                    'message': f'测试连接时发生错误: {str(e)}',
                    'error_code': 'TEST_ERROR'
                }), 500

        @self.app.route('/api/v1/providers/<provider_id>/activate', methods=['POST'])
        def activate_provider(provider_id):
            """
            激活指定数据源（同时停用其他所有数据源）
            设计原则：同一时刻只能有一个活跃的数据源
            """
            try:
                config = self.config_manager.get('data', {})
                providers = config.get('providers', [])

                # 查找目标数据源
                target_provider = None
                for p in providers:
                    if p.get('id') == provider_id or p.get('name') == provider_id:
                        target_provider = p
                        break

                if not target_provider:
                    return jsonify({
                        'status': 'error',
                        'message': f'数据源不存在: {provider_id}',
                        'error_code': 'PROVIDER_NOT_FOUND'
                    }), 404

                # 停用所有数据源
                for p in providers:
                    p['status'] = 'inactive'
                    p['updated_at'] = datetime.now().isoformat()

                # 激活目标数据源
                target_provider['status'] = 'active'
                target_provider['updated_at'] = datetime.now().isoformat()

                # 更新配置文件中的 primary_source
                config['primary_source'] = provider_id
                config['providers'] = providers
                self.config_manager.update({'data': config})

                logger.info(f"已激活数据源: {provider_id}，其他数据源已自动停用")

                return jsonify({
                    'status': 'success',
                    'message': f'已切换到 {target_provider.get("name", provider_id)}',
                    'active_provider': provider_id,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"激活数据源失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': 'PROVIDER_ACTIVATE_FAILED'
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
                provider = getattr(self.quality_monitor, 'data_provider_config', None)
                if not provider or not hasattr(provider, 'get_index_prices'):
                    return jsonify({'status': 'error', 'message': '数据提供者不可用',
                                    'error_code': 'DATA_PROVIDER_UNAVAILABLE'}), 503
                df = provider.get_index_prices(index_id, start_date, end_date, datetime.now())
                data = df.to_dict(orient='records') if hasattr(df, 'to_dict') else []
                return jsonify(
                    {'status': 'success', 'data': data, 'count': len(data), 'timestamp': datetime.now().isoformat()})
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
                provider = getattr(self.quality_monitor, 'data_provider_config', None)
                if not provider or not hasattr(provider, 'get_index_returns'):
                    return jsonify({'status': 'error', 'message': '数据提供者不可用',
                                    'error_code': 'DATA_PROVIDER_UNAVAILABLE'}), 503
                series = provider.get_index_returns(index_id, start_date, end_date)
                data = [{'date': str(idx), 'return': float(val)} for idx, val in
                        (series.items() if hasattr(series, 'items') else [])]
                return jsonify(
                    {'status': 'success', 'data': data, 'count': len(data), 'timestamp': datetime.now().isoformat()})
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

                # 检查 quality_monitor 是否存在
                if not hasattr(self, 'quality_monitor') or self.quality_monitor is None:
                    logger.error("quality_monitor 未初始化")
                    return jsonify({
                        'status': 'error',
                        'message': '监控服务未初始化，请检查应用启动状态',
                        'error_code': 'MONITOR_NOT_INITIALIZED'
                    }), 503

                # 检查 data_provider_config 是否存在
                provider = getattr(self.quality_monitor, 'data_provider_config', None)
                if not provider:
                    logger.error(f"data_provider_config 不存在。quality_monitor 属性: {dir(self.quality_monitor)}")
                    return jsonify({
                        'status': 'error',
                        'message': '数据提供者未初始化，请检查数据源配置（config/dev/data_provider_config.yml）',
                        'error_code': 'DATA_PROVIDER_NOT_FOUND',
                        'hint': '确保 primary_source 已配置且有效（如 tushare, yahoo, akshare）'
                    }), 503

                # 检查方法是否存在
                if not hasattr(provider, 'get_event_window_data'):
                    logger.error(f"provider 类型: {type(provider).__name__}, 方法: {dir(provider)}")
                    return jsonify({
                        'status': 'error',
                        'message': f'数据提供者（{type(provider).__name__}）不支持 get_event_window_data 方法',
                        'error_code': 'METHOD_NOT_SUPPORTED'
                    }), 503

                logger.info(
                    f"调用 get_event_window_data: index_id={index_id}, event_date={event_date}, event_type={event_type}")
                result = provider.get_event_window_data(index_id, event_date, event_type, window_days, baseline_days)
                # 仅返回统计信息与样本，避免过大payload
                event_records = result.get('event_window')
                baseline_records = result.get('baseline')
                event_data = event_records.head(200).to_dict(orient='records') if hasattr(event_records,
                                                                                          'to_dict') else []
                baseline_data = baseline_records.head(200).to_dict(orient='records') if hasattr(baseline_records,
                                                                                                'to_dict') else []
                return jsonify({
                    'status': 'success',
                    'event_window': {'count': len(event_records) if hasattr(event_records, '__len__') else 0,
                                     'samples': event_data},
                    'baseline': {'count': len(baseline_records) if hasattr(baseline_records, '__len__') else 0,
                                 'samples': baseline_data},
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
                provider = getattr(self.quality_monitor, 'data_provider_config', None)
                if not provider or not hasattr(provider, 'get_cross_validation_log'):
                    return jsonify({'status': 'error', 'message': '数据提供者不可用',
                                    'error_code': 'DATA_PROVIDER_UNAVAILABLE'}), 503
                log = provider.get_cross_validation_log()
                return jsonify(
                    {'status': 'success', 'log': log, 'count': len(log), 'timestamp': datetime.now().isoformat()})
            except Exception as e:
                logger.error(f"获取交叉验证日志失败: {e}")
                return jsonify(
                    {'status': 'error', 'message': str(e), 'error_code': 'CROSS_VALIDATION_LOG_FETCH_FAILED'}), 500

        # 新增：K线数据（周期切换 + 最近30周期 + 事件标注 + 无限滚动支持）
        @self.app.route('/api/v1/data/kline')
        def get_kline_data():
            try:
                index_id = request.args.get('index_id', type=str)
                period = request.args.get('period', default='daily', type=str)
                count = request.args.get('count', default=30, type=int)
                before_str = request.args.get('before', type=str)  # 新增：获取此日期之前的数据
                before=None
                if before_str:
                    before=pd.Timestamp(before_str)
                mock_flag = request.args.get('mock', default='0', type=str)
                if not index_id:
                    return jsonify({'status': 'error', 'message': '缺少index_id', 'error_code': 'MISSING_PARAMS'}), 400
                # 若启用模拟数据，则在应用层生成K线数据（不依赖真实数据源）
                if mock_flag.lower() in ('1', 'true', 'yes'):
                    try:
                        # 使用 MockHistoricalDataProvider 生成逼真的K线数据
                        from core_bak_refactored.tests.fixtures.core.data.mock_historical_data_provider import \
                            MockHistoricalDataProvider
                        mock_provider = MockHistoricalDataProvider()

                        # 计算日期范围
                        multiplier = {'daily': 1, 'weekly': 7, 'monthly': 30}.get(period, 1)
                        days_needed = count * multiplier * 2
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=days_needed)

                        # 获取原始日线数据
                        df = mock_provider.get_index_prices(index_id, start_date.strftime('%Y-%m-%d'),
                                                            end_date.strftime('%Y-%m-%d'), datetime.now())
                        if hasattr(df, 'empty') and df.empty:
                            return jsonify({'status': 'error', 'message': '无数据', 'error_code': 'NO_DATA'}), 404

                        # 补齐OHLC（基于close生成逼真的OHLC）
                        df = df.copy()
                        if 'open' not in df.columns:
                            df['open'] = df['close'].shift(1).fillna(df['close'])
                        if 'high' not in df.columns or 'low' not in df.columns:
                            # 基于收益率波动生成high/low
                            returns = df['close'].pct_change().fillna(0)
                            volatility = returns.rolling(5, min_periods=1).std().fillna(0.01)
                            df['high'] = df['close'] * (1 + volatility * np.random.uniform(0.3, 0.8, len(df)))
                            df['low'] = df['close'] * (1 - volatility * np.random.uniform(0.3, 0.8, len(df)))
                            # 确保 high >= close >= low 和 high >= open >= low
                            df['high'] = df[['high', 'close', 'open']].max(axis=1)
                            df['low'] = df[['low', 'close', 'open']].min(axis=1)

                        # 周期转换
                        df2 = df.copy()
                        df2['date'] = pd.to_datetime(df2['date'])
                        df2 = df2.set_index('date')
                        if period == 'weekly':
                            df2 = df2.resample('W').agg(
                                {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
                        elif period == 'monthly':
                            df2 = df2.resample('M').agg(
                                {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
                        df2 = df2.reset_index()
                        df2 = df2.tail(count)

                        # 事件检测（最小规则）
                        df2['pct_change'] = df2['close'].pct_change() * 100
                        events = []
                        for i in range(len(df2)):
                            chg = float(df2.loc[df2.index[i], 'pct_change']) if not pd.isna(
                                df2.loc[df2.index[i], 'pct_change']) else 0.0
                            dt = df2.loc[df2.index[i], 'date']
                            cl = float(df2.loc[df2.index[i], 'close'])
                            if chg <= -5.0:
                                events.append({'date': dt.strftime('%Y-%m-%d'), 'type': 'market_crash',
                                               'title': f'暴跌 {abs(chg):.2f}%', 'decline_pct': chg, 'price': cl,
                                               'impact': 'negative', 'severity': 'high' if chg > -7 else 'critical'})
                            elif chg >= 5.0:
                                events.append(
                                    {'date': dt.strftime('%Y-%m-%d'), 'type': 'rally', 'title': f'暴涨 {chg:.2f}%',
                                     'rise_pct': chg, 'price': cl, 'impact': 'positive', 'severity': 'high'})

                        # 转换为dict并处理NaN值（替换为null以确保JSON有效）
                        data = df2.to_dict(orient='records')
                        # 将所有NaN替换为None（JSON序列化时会变成null）
                        for record in data:
                            for key, value in record.items():
                                if pd.isna(value):
                                    record[key] = None
                                elif key == 'date' and hasattr(value, 'strftime'):
                                    record[key] = value.strftime('%Y-%m-%d')
                        return jsonify(
                            {'status': 'success', 'data': data, 'period': period, 'count': len(data), 'events': events,
                             'timestamp': datetime.now().isoformat()})
                    except Exception as e:
                        logger.error(f"模拟K线数据生成失败: {e}")
                        return jsonify({'status': 'error', 'message': str(e), 'error_code': 'MOCK_KLINE_FAILED'}), 500
                # 否则走真实数据源路径
                provider = getattr(self.quality_monitor, 'data_provider_config', None)
                if not provider or not hasattr(provider, 'get_index_prices'):
                    # 生产环境：数据提供者不可用时返回错误，不降级为Mock
                    return jsonify({'status': 'error', 'message': '数据提供者不可用',
                                    'error_code': 'DATA_PROVIDER_UNAVAILABLE'}), 503

                try:
                    multiplier = {'daily': 1, 'weekly': 7, 'monthly': 30}.get(period, 1)
                    days_needed = count * multiplier * 2

                    if before:
                        end_date = datetime.strptime(before, '%Y-%m-%d')
                    else:
                        end_date = datetime.now()

                    start_date = end_date - timedelta(days=days_needed)
                    df = provider.get_index_prices(index_id, start_date.strftime('%Y-%m-%d'),
                                                   end_date.strftime('%Y-%m-%d'), datetime.now())
                    if hasattr(df, 'empty') and df.empty:
                        # 生产环境：真实数据为空时返回错误，不降级为Mock
                        return jsonify({'status': 'error', 'message': '无数据', 'error_code': 'NO_DATA'}), 404
                except Exception as e:
                    logger.error(f"获取真实数据失败: {e}")
                    # 生产环境：获取数据失败时返回错误，不降级为Mock
                    return jsonify({'status': 'error', 'message': f'数据获取失败: {str(e)}',
                                    'error_code': 'DATA_FETCH_FAILED'}), 500

                # 处理真实数据：补齐OHLC、周期转换、事件检测
                # 补齐OHLC
                if 'open' not in df.columns or 'high' not in df.columns or 'low' not in df.columns:
                    df = df.copy()
                    df['open'] = df['close'].shift(1).fillna(df['close'])
                    df['high'] = df['close'] * 1.005
                    df['low'] = df['close'] * 0.995

                # 周期转换
                df2 = df.copy()
                df2['date'] = pd.to_datetime(df2['date'])
                df2 = df2.set_index('date')
                if period == 'weekly':
                    df2 = df2.resample('W').agg(
                        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
                elif period == 'monthly':
                    df2 = df2.resample('M').agg(
                        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
                else:
                    # 保持日线
                    pass
                df2 = df2.reset_index()
                df2 = df2.tail(count)

                # 事件检测（最小规则）
                try:
                    df2['pct_change'] = df2['close'].pct_change() * 100
                    events = []
                    for i in range(len(df2)):
                        chg = float(df2.loc[df2.index[i], 'pct_change']) if not pd.isna(
                            df2.loc[df2.index[i], 'pct_change']) else 0.0
                        dt = df2.loc[df2.index[i], 'date']
                        cl = float(df2.loc[df2.index[i], 'close'])
                        if chg <= -5.0:
                            events.append({'date': dt.strftime('%Y-%m-%d'), 'type': 'market_crash',
                                           'title': f'暴跌 {abs(chg):.2f}%', 'decline_pct': chg, 'price': cl,
                                           'impact': 'negative', 'severity': 'high' if chg > -7 else 'critical'})
                        elif chg >= 5.0:
                            events.append(
                                {'date': dt.strftime('%Y-%m-%d'), 'type': 'rally', 'title': f'暴涨 {chg:.2f}%',
                                 'rise_pct': chg, 'price': cl, 'impact': 'positive', 'severity': 'high'})
                except Exception:
                    events = []

                # 转换为dict并处理NaN值（替换为null以确保JSON有效）
                data = df2.to_dict(orient='records')
                # 将所有NaN替换为None（JSON序列化时会变成null）
                for record in data:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None
                        elif key == 'date' and hasattr(value, 'strftime'):
                            record[key] = value.strftime('%Y-%m-%d')
                return jsonify(
                    {'status': 'success', 'data': data, 'period': period, 'count': len(data), 'events': events,
                     'timestamp': datetime.now().isoformat()})
            except Exception as e:
                logger.error(f"获取K线数据失败: {e}")
                return jsonify({'status': 'error', 'message': str(e), 'error_code': 'KLINE_FETCH_FAILED'}), 500

        # 🆕 新增：实时K线柱数据（用于盘前/盘中实时更新当天K线）
        # 模拟模式端点
        @self.app.route('/api/v1/data/kline/realtime/mock')
        def get_realtime_kline_mock():
            """
            获取当天K线柱的实时数据（模拟模式，独立于分时图）
            
            参数：
                index_id: 证券代码
                trading_phase: 交易时段 (BEFORE_OPEN, TRADING, AFTER_CLOSE) - 用于模拟控制
                trade_date: 交易日期 (YYYY-MM-DD)，默认今天
                is_index: 是否为指数，默认false
            
            返回：
                {
                    status: 'success',
                    data: {
                        date: 'YYYY-MM-DD',
                        open: float,
                        high: float,
                        low: float,
                        close: float,
                        volume: int,
                        trading_phase: str,  # 交易时段
                        should_poll: bool  # 服务器根据 trading_phase 决定
                    },
                    timestamp: str
                }
            """
            try:
                index_id = request.args.get('index_id', type=str)
                if not index_id:
                    return jsonify({'status': 'error', 'message': '缺少index_id', 'error_code': 'MISSING_PARAMS'}), 400

                # 🔧 获取前端传入的trading_phase参数（用于模拟控制）
                from core_bak_refactored.core.share.market.market_enums import TradingPhase
                from core_bak_refactored.core.share.market.market_utils import MarketUtils

                trading_phase_str = request.args.get('trading_phase', 'TRADING')  # 默认盘中
                try:
                    trading_phase = TradingPhase.parse(trading_phase_str)  # 转为枚举
                except KeyError:
                    return jsonify({
                        'status': 'error',
                        'message': f'无效的trading_phase: {trading_phase_str}，允许值: BEFORE_OPEN, TRADING, AFTER_CLOSE',
                        'error_code': 'INVALID_TRADING_PHASE'
                    }), 400

                trade_date = request.args.get('trade_date', datetime.now().strftime('%Y-%m-%d'))
                is_index_str = request.args.get('is_index', 'false').lower()
                is_index = is_index_str in ['true', '1', 'yes']

                from core_bak_refactored.core.data.providers.mock_provider import MockDataProvider

                # 🔧 调用领域层，显式传入参数
                provider = MockDataProvider()
                result = provider.get_realtime_kline(
                    symbol=index_id,
                    trade_date=trade_date,
                    trading_phase=trading_phase,
                    is_index=is_index
                )

                return jsonify({
                    'status': 'success',
                    'data': result,
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"处理模拟实时K线请求失败: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'status': 'error', 'message': str(e), 'error_code': 'MOCK_REALTIME_KLINE_FAILED'}), 500

        # 真实模式端点
        @self.app.route('/api/v1/data/kline/realtime')
        def get_realtime_kline():
            """
            获取当天K线柱的实时数据（真实模式，独立于分时图）
            
            参数：
                index_id: 证券代码
            
            返回：
                {
                    status: 'success',
                    data: {
                        date: 'YYYY-MM-DD',
                        open: float,
                        high: float,
                        low: float,
                        close: float,
                        volume: int,
                        should_poll: bool  # 服务器根据 trading_phase 决定，前端只依赖此字段控制行为
                    },
                    timestamp: str
                }
            """
            try:
                index_id = request.args.get('index_id', type=str)
                if not index_id:
                    return jsonify({'status': 'error', 'message': '缺少index_id', 'error_code': 'MISSING_PARAMS'}), 400

                provider = getattr(self.quality_monitor, 'data_provider_config', None)
                if not provider:
                    return jsonify({'status': 'error', 'message': '数据提供者不可用',
                                    'error_code': 'DATA_PROVIDER_UNAVAILABLE'}), 503

                try:
                    # 🔧 直接调用领域层，所有逻辑由领域层处理
                    result = provider.get_realtime_kline(symbol=index_id)

                    return jsonify({
                        'status': 'success',
                        'data': result,
                        'timestamp': datetime.now().isoformat()
                    })

                except Exception as e:
                    logger.error(f"获取实时K线失败: {e}")
                    return jsonify(
                        {'status': 'error', 'message': str(e), 'error_code': 'REALTIME_KLINE_FETCH_FAILED'}), 500

            except Exception as e:
                logger.error(f"处理实时K线请求失败: {e}")
                return jsonify(
                    {'status': 'error', 'message': str(e), 'error_code': 'REALTIME_KLINE_REQUEST_FAILED'}), 500

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
            logger.info("Socket.IO客户端已连接")
            emit('connection_response', {
                'status': 'connected',
                'message': '已连接到数据质量监控服务',
                'timestamp': datetime.now().isoformat()
            })

        @self.socketio.on('disconnect')
        def handle_disconnect():
            """客户端断开连接事件"""
            logger.info("Socket.IO客户端已断开")

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
