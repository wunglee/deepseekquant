"""
DeepSeekQuant 主系统模块
系统入口点和协调器
"""

import asyncio
import signal
import sys
import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import traceback
import psutil
from dataclasses import dataclass, asdict
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import hashlib
import uuid

# 导入内部模块
from config.config_manager import ConfigManager
from core.data_fetcher import DataFetcher
from core.signal_engine import SignalEngine, Signal, SignalType
from core.portfolio_manager import PortfolioManager, AllocationMethod
from core.risk_manager import RiskManager, RiskLevel, RiskAssessment
from core.execution_engine import ExecutionEngine, ExecutionStrategy, TradeCost
from core.bayesian_optimizer import BayesianOptimizer
from analytics.backtesting import BacktestingEngine
from analytics.performance import PerformanceAnalyzer
from infrastructure.monitoring import MonitoringSystem
from infrastructure.api_gateway import APIGateway
from infrastructure.disaster_recovery import DisasterRecoveryManager
from infrastructure.logging_system import LoggingSystem
from utils.helpers import validate_config, format_timestamp, calculate_hash
from utils.validators import validate_market_data, validate_signals, validate_positions

logger = logging.getLogger('DeepSeekQuant.Main')


class SystemState(Enum):
    """系统状态枚举"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    SAFE_MODE = "safe_mode"
    MAINTENANCE = "maintenance"
    RECOVERY = "recovery"


class TradingMode(Enum):
    """交易模式枚举"""
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"
    BACKTESTING = "backtesting"
    SIMULATION = "simulation"
    OPTIMIZATION = "optimization"


@dataclass
class SystemStatus:
    """系统状态数据类"""
    state: SystemState
    trading_mode: TradingMode
    uptime: float
    memory_usage_mb: float
    cpu_usage_percent: float
    active_modules: int
    error_count: int
    last_update: str
    performance_metrics: Dict[str, float]
    positions_count: int
    trading_volume: float
    risk_level: RiskLevel


@dataclass
class TradingCycleMetrics:
    """交易周期指标数据类"""
    cycle_id: str
    start_time: str
    end_time: str
    duration: float
    signals_generated: int
    signals_executed: int
    risk_assessment: Dict[str, Any]
    trades_executed: int
    trading_volume: float
    slippage: float
    commission: float
    performance_metrics: Dict[str, float]
    error_count: int
    status: str


class DeepSeekQuantSystem:
    """DeepSeekQuant 主系统类"""

    def __init__(self, config_path: str = None):
        """初始化系统"""
        self.config_path = config_path
        self.state = SystemState.UNINITIALIZED
        self.trading_mode = TradingMode.PAPER_TRADING
        self.start_time = None
        self.modules = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=20)
        self.loop = None
        self._shutdown_event = threading.Event()
        self._pause_event = threading.Event()
        self._recovery_mode = False

        # 系统组件
        self.config_manager = None
        self.data_fetcher = None
        self.signal_engine = None
        self.portfolio_manager = None
        self.risk_manager = None
        self.execution_engine = None
        self.bayesian_optimizer = None
        self.backtesting_engine = None
        self.performance_analyzer = None
        self.monitoring_system = None
        self.api_gateway = None
        self.disaster_recovery = None
        self.logging_system = None

        # 系统数据
        self.current_positions = {}
        self.trading_history = []
        self.performance_data = []
        self.error_log = []
        self.audit_log = []
        self.cycle_metrics = []
        self.signal_history = []
        self.risk_history = []

        # 性能指标
        self.initial_capital = 1000000.0
        self.current_capital = self.initial_capital
        self.total_trades = 0
        self.total_volume = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0

        # 初始化系统
        self._initialize_system()

    def _initialize_system(self):
        """初始化系统组件"""
        try:
            logger.info("开始初始化 DeepSeekQuant 系统")
            self.state = SystemState.INITIALIZING

            # 1. 初始化配置管理器
            self.config_manager = ConfigManager(self.config_path)
            self.config = self.config_manager.get_config()

            # 设置初始资金
            self.initial_capital = self.config.get('trading', {}).get('initial_capital', 1000000.0)
            self.current_capital = self.initial_capital

            logger.info(f"配置管理器初始化完成，初始资金: {self.initial_capital:,.2f}")

            # 2. 初始化日志系统
            self.logging_system = LoggingSystem(self.config.get('logging', {}))
            self.logging_system.setup_logging()
            logger.info("日志系统初始化完成")

            # 3. 初始化灾难恢复
            self.disaster_recovery = DisasterRecoveryManager(self.config.get('disaster_recovery', {}))
            logger.info("灾难恢复系统初始化完成")

            # 4. 初始化核心模块
            self._initialize_core_modules()
            logger.info("核心模块初始化完成")

            # 5. 初始化分析模块
            self._initialize_analytics_modules()
            logger.info("分析模块初始化完成")

            # 6. 初始化基础设施
            self._initialize_infrastructure()
            logger.info("基础设施初始化完成")

            # 7. 设置信号处理
            self._setup_signal_handlers()
            logger.info("信号处理器设置完成")

            # 8. 设置交易模式
            trading_mode_config = self.config.get('system', {}).get('trading_mode', 'paper_trading')
            self.trading_mode = TradingMode(trading_mode_config)
            logger.info(f"交易模式设置为: {self.trading_mode.value}")

            self.state = SystemState.INITIALIZED
            logger.info("DeepSeekQuant 系统初始化完成")

        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            self.state = SystemState.ERROR
            self._handle_initialization_error(e)

    def _initialize_core_modules(self):
        """初始化核心模块"""
        # 数据获取器
        self.data_fetcher = DataFetcher(self.config.get('data_sources', {}))
        self.modules['data_fetcher'] = self.data_fetcher

        # 信号引擎
        self.signal_engine = SignalEngine(self.config.get('signal_engine', {}))
        self.modules['signal_engine'] = self.signal_engine

        # 组合管理器
        self.portfolio_manager = PortfolioManager(self.config.get('portfolio_management', {}))
        self.modules['portfolio_manager'] = self.portfolio_manager

        # 风险管理器
        self.risk_manager = RiskManager(self.config.get('risk_management', {}))
        self.modules['risk_manager'] = self.risk_manager

        # 执行引擎
        self.execution_engine = ExecutionEngine(self.config.get('execution', {}))
        self.modules['execution_engine'] = self.execution_engine

        # 贝叶斯优化器
        self.bayesian_optimizer = BayesianOptimizer(self.config.get('optimization', {}))
        self.modules['bayesian_optimizer'] = self.bayesian_optimizer

    def _initialize_analytics_modules(self):
        """初始化分析模块"""
        # 回测引擎
        self.backtesting_engine = BacktestingEngine(self.config.get('backtesting', {}))
        self.modules['backtesting_engine'] = self.backtesting_engine

        # 性能分析器
        self.performance_analyzer = PerformanceAnalyzer(self.config.get('performance_analytics', {}))
        self.modules['performance_analyzer'] = self.performance_analyzer

    def _initialize_infrastructure(self):
        """初始化基础设施"""
        # 监控系统
        self.monitoring_system = MonitoringSystem(self.config.get('monitoring', {}))
        self.modules['monitoring_system'] = self.monitoring_system

        # API网关
        self.api_gateway = APIGateway(self.config.get('api_gateway', {}))
        self.modules['api_gateway'] = self.api_gateway

        # 设置模块监控
        for name, module in self.modules.items():
            self.monitoring_system.register_module(name, module)

    def _setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            logger.info(f"接收到信号 {signum}, 开始优雅关闭")
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGHUP, signal_handler)  # 挂起信号，用于重新加载配置

    def start(self):
        """启动系统"""
        try:
            if self.state != SystemState.INITIALIZED:
                raise RuntimeError(f"系统未就绪，当前状态: {self.state}")

            logger.info("启动 DeepSeekQuant 系统")
            self.state = SystemState.STARTING
            self.start_time = datetime.now()
            self._shutdown_event.clear()

            # 启动基础设施
            self.monitoring_system.start()
            self.api_gateway.start()

            # 启动核心模块
            for name, module in self.modules.items():
                if hasattr(module, 'start'):
                    module.start()
                    logger.info(f"模块 {name} 启动完成")

            # 设置交易循环
            self._setup_trading_cycle()

            # 设置性能监控循环
            self._setup_performance_monitoring()

            self.state = SystemState.RUNNING
            logger.info("DeepSeekQuant 系统启动完成")

            return True

        except Exception as e:
            logger.error(f"系统启动失败: {e}")
            self.state = SystemState.ERROR
            return False

    def pause(self):
        """暂停系统"""
        if self.state == SystemState.RUNNING:
            self.state = SystemState.PAUSED
            self._pause_event.set()
            logger.info("系统已暂停")
            return True
        return False

    def resume(self):
        """恢复系统运行"""
        if self.state == SystemState.PAUSED:
            self.state = SystemState.RUNNING
            self._pause_event.clear()
            logger.info("系统已恢复运行")
            return True
        return False

    def stop(self):
        """停止系统"""
        try:
            logger.info("停止 DeepSeekQuant 系统")
            self.state = SystemState.STOPPING

            # 设置关闭事件
            self._shutdown_event.set()

            # 停止基础设施
            if self.api_gateway:
                self.api_gateway.stop()

            if self.monitoring_system:
                self.monitoring_system.stop()

            # 停止核心模块
            for name, module in reversed(list(self.modules.items())):
                if hasattr(module, 'stop'):
                    module.stop()
                    logger.info(f"模块 {name} 停止完成")

            # 关闭线程池
            self.thread_pool.shutdown(wait=True)

            self.state = SystemState.STOPPED
            logger.info("DeepSeekQuant 系统停止完成")

            return True

        except Exception as e:
            logger.error(f"系统停止失败: {e}")
            return False

    def _setup_trading_cycle(self):
        """设置交易循环"""
        def trading_worker():
            while not self._shutdown_event.is_set():
                try:
                    if self.state == SystemState.RUNNING and not self._pause_event.is_set():
                        self._execute_trading_cycle()

                    # 等待下一个周期
                    cycle_interval = self.config.get('trading', {}).get('cycle_interval', 60)
                    self._shutdown_event.wait(cycle_interval)

                except Exception as e:
                    logger.error(f"交易循环执行失败: {e}")
                    self._handle_trading_error(e)

        # 启动交易循环线程
        trading_thread = threading.Thread(target=trading_worker, daemon=True)
        trading_thread.start()
        logger.info("交易循环线程启动完成")

    def _setup_performance_monitoring(self):
        """设置性能监控循环"""
        def performance_monitor():
            while not self._shutdown_event.is_set():
                try:
                    if self.state == SystemState.RUNNING:
                        self._update_performance_metrics()

                    # 每分钟更新一次性能指标
                    self._shutdown_event.wait(60)

                except Exception as e:
                    logger.error(f"性能监控失败: {e}")

        # 启动性能监控线程
        monitor_thread = threading.Thread(target=performance_monitor, daemon=True)
        monitor_thread.start()
        logger.info("性能监控线程启动完成")

    def _execute_trading_cycle(self):
        """执行交易周期"""
        cycle_start = datetime.now()
        cycle_id = f"cycle_{cycle_start.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        try:
            logger.info(f"开始交易周期 {cycle_id}")

            # 1. 获取市场数据
            market_data = self._fetch_market_data()
            if not market_data:
                logger.warning("市场数据获取失败，跳过本次周期")
                return

            # 2. 生成交易信号
            signals = self._generate_signals(market_data)
            if not signals:
                logger.warning("信号生成失败，跳过本次周期")
                return

            # 3. 风险评估
            risk_assessment = self._assess_risk(signals, self.current_positions, market_data)
            if not risk_assessment.approved:
                logger.warning(f"风险评估未通过: {risk_assessment.reason}")
                self._record_risk_rejection(risk_assessment)
                return

            # 4. 计算目标仓位
            target_positions = self._calculate_target_positions(signals, self.current_positions, market_data)
            if not target_positions:
                logger.warning("目标仓位计算失败，跳过本次周期")
                return

            # 5. 执行交易
            execution_result = self._execute_trades(target_positions, self.current_positions, market_data)
            if execution_result:
                self._update_positions(execution_result)
                self._record_trade(execution_result)
                self._update_capital(execution_result)

            # 6. 更新性能数据
            self._update_performance_data()

            cycle_end = datetime.now()
            cycle_duration = (cycle_end - cycle_start).total_seconds()

            logger.info(f"交易周期 {cycle_id} 完成，耗时: {cycle_duration:.2f}秒")

            # 记录周期数据
            cycle_metrics = self._create_cycle_metrics(
                cycle_id, cycle_start, cycle_end, cycle_duration,
                signals, risk_assessment, execution_result
            )
            self._record_cycle_metrics(cycle_metrics)

        except Exception as e:
            logger.error(f"交易周期执行失败: {e}")
            self._handle_cycle_error(e, cycle_id)

    def _create_cycle_metrics(self, cycle_id, start_time, end_time, duration,
                             signals, risk_assessment, execution_result):
        """创建周期指标"""
        return TradingCycleMetrics(
            cycle_id=cycle_id,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration=duration,
            signals_generated=len(signals),
            signals_executed=len(execution_result.get('trades', []) if execution_result else 0),
            risk_assessment=risk_assessment.to_dict(),
            trades_executed=len(execution_result.get('trades', []) if execution_result else 0),
            trading_volume=sum(trade.get('value', 0) for trade in execution_result.get('trades', [])
                              if execution_result else 0),
            slippage=execution_result.get('slippage', 0) if execution_result else 0,
            commission=execution_result.get('commission', 0) if execution_result else 0,
            performance_metrics=self._get_current_performance(),
            error_count=0,
            status='completed'
        )

    def _fetch_market_data(self) -> Dict:
        """获取市场数据"""
        try:
            symbols = self.config.get('trading', {}).get('symbols', ['SPY', 'QQQ', 'TLT'])
            period = self.config.get('data_sources', {}).get('lookback_period', '1mo')
            interval = self.config.get('data_sources', {}).get('interval', '1d')

            market_data = self.data_fetcher.get_historical_data(symbols, period, interval)

            if not market_data:
                logger.warning("市场数据为空")
                return {}

            # 验证数据质量
            if not validate_market_data(market_data):
                logger.warning("市场数据验证失败")
                return {}

            return market_data

        except Exception as e:
            logger.error(f"市场数据获取失败: {e}")
            return {}

    def _generate_signals(self, market_data: Dict) -> Dict:
        """生成交易信号"""
        try:
            signals = self.signal_engine.generate_signals(market_data)

            if not signals:
                logger.warning("未生成任何信号")
                return {}

            # 验证信号质量
            if not validate_signals(signals):
                logger.warning("信号验证失败")
                return {}

            # 记录信号历史
            signal_record = {
                'timestamp': datetime.now().isoformat(),
                'signals': signals,
                'market_conditions': self._get_market_conditions(market_data)
            }
            self.signal_history.append(signal_record)

            # 保持历史记录长度
            max_signal_history = self.config.get('system', {}).get('max_signal_history', 1000)
            if len(self.signal_history) > max_signal_history:
                self.signal_history = self.signal_history[-max_signal_history:]

            return signals

        except Exception as e:
            logger.error(f"信号生成失败: {e}")
            return {}

    def _get_market_conditions(self, market_data: Dict) -> Dict:
        """获取市场状况"""
        try:
            # 计算市场波动率、趋势等指标
            volatility = self._calculate_market_volatility(market_data)
            trend = self._assess_market_trend(market_data)

            return {
                'volatility': volatility,
                'trend': trend,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"市场状况分析失败: {e}")
            return {'error': str(e)}

    def _calculate_market_volatility(self, market_data: Dict) -> float:
        """计算市场波动率"""
        # 实现波动率计算逻辑
        try:
            # 使用标准差计算波动率
            returns = []
            for symbol, data in market_data.items():
                if 'close' in data and len(data['close']) > 1:
                    prices = data['close']
                    symbol_returns = np.diff(prices) / prices[:-1]
                    returns.extend(symbol_returns)

            if returns:
                return np.std(returns) * np.sqrt(252)  # 年化波动率
            return 0.0
        except:
            return 0.0

    def _assess_market_trend(self, market_data: Dict) -> str:
        """评估市场趋势"""
        # 实现趋势评估逻辑
        try:
            trends = []
            for symbol, data in market_data.items():
                if 'close' in data and len(data['close']) > 10:
                    prices = data['close'][-10:]  # 最近10个价格
                    if prices[-1] > prices[0] * 1.05:
                        trends.append('bullish')
                    elif prices[-1] < prices[0] * 0.95:
                        trends.append('bearish')
                    else:
                        trends.append('neutral')

            if trends:
                return max(set(trends), key=trends.count)
            return 'neutral'
        except:
            return 'unknown'

    def _assess_risk(self, signals: Dict, positions: Dict, market_data: Dict) -> RiskAssessment:
        """风险评估"""
        try:
            risk_assessment = self.risk_manager.assess_risk(signals, positions, market_data)

            # 记录风险评估历史
            risk_record = {
                'timestamp': datetime.now().isoformat(),
                'assessment': risk_assessment.to_dict(),
                'signals': signals,
                'positions': positions
            }
            self.risk_history.append(risk_record)

            # 保持历史记录长度
            max_risk_history = self.config.get('system', {}).get('max_risk_history', 500)
            if len(self.risk_history) > max_risk_history:
                self.risk_history = self.risk_history[-max_risk_history:]

            return risk_assessment

        except Exception as e:
            logger.error(f"风险评估失败: {e}")
            return RiskAssessment(
                approved=False,
                reason=f"风险评估异常: {str(e)}",
                risk_level=RiskLevel.HIGH,
                warnings=[f"风险评估异常: {str(e)}"]
            )

    def _calculate_target_positions(self, signals: Dict, positions: Dict, market_data: Dict) -> Dict:
        """计算目标仓位"""
        try:
            target_positions = self.portfolio_manager.calculate_target_positions(signals, positions, market_data)

            if not target_positions:
                logger.warning("目标仓位计算为空")
                return {}

            # 验证仓位合理性
            if not validate_positions(target_positions):
                logger.warning("目标仓位验证失败")
                return {}

            return target_positions

        except Exception as e:
            logger.error(f"目标仓位计算失败: {e}")
            return {}

    def _execute_trades(self, target_positions: Dict, current_positions: Dict, market_data: Dict) -> Dict:
        """执行交易"""
        try:
            execution_plan = self.execution_engine.create_execution_plan(target_positions, current_positions, market_data)

            if execution_plan.get('action') == 'HOLD':
                logger.info("执行计划建议保持仓位")
                return {'action': 'HOLD', 'trades': []}

            execution_result = self.execution_engine.execute_plan(execution_plan)
            return execution_result

        except Exception as e:
            logger.error(f"交易执行失败: {e}")
            return {'action': 'ERROR', 'error': str(e), 'trades': []}

    def _update_positions(self, execution_result: Dict):
        """更新仓位"""
        try:
            if execution_result.get('action') == 'TRADE':
                for trade in execution_result.get('trades', []):
                    symbol = trade['symbol']
                    if trade['action'] == 'BUY':
                        self.current_positions[symbol] = self.current_positions.get(symbol, 0) + trade['quantity']
                    elif trade['action'] == 'SELL':
                        self.current_positions[symbol] = self.current_positions.get(symbol, 0) - trade['quantity']

                    # 清理零仓位
                    if abs(self.current_positions.get(symbol, 0)) < 0.0001:
                        del self.current_positions[symbol]

        except Exception as e:
            logger.error(f"仓位更新失败: {e}")

    def _update_capital(self, execution_result: Dict):
        """更新资金"""
        try:
            if execution_result.get('action') == 'TRADE':
                for trade in execution_result.get('trades', []):
                    # 更新总交易量
                    trade_value = trade.get('price', 0) * trade.get('quantity', 0)
                    self.total_volume += trade_value

                    # 更新总交易次数
                    self.total_trades += 1

                    # 更新当前资金（简化处理，实际中需要更精确的计算）
                    if trade['action'] == 'BUY':
                        self.current_capital -= trade_value
                    elif trade['action'] == 'SELL':
                        self.current_capital += trade_value

        except Exception as e:
            logger.error(f"资金更新失败: {e}")

    def _record_trade(self, execution_result: Dict):
        """记录交易"""
        try:
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'execution_result': execution_result,
                'current_positions': self.current_positions.copy(),
                'current_capital': self.current_capital,
                'performance_snapshot': self._get_current_performance()
            }
            self.trading_history.append(trade_record)

            # 保持历史记录长度
            max_history = self.config.get('system', {}).get('max_trading_history', 1000)
            if len(self.trading_history) > max_history:
                self.trading_history = self.trading_history[-max_history:]

        except Exception as e:
            logger.error(f"交易记录失败: {e}")

    def _update_performance_data(self):
        """更新性能数据"""
        try:
            performance_snapshot = self._get_current_performance()
            self.performance_data.append(performance_snapshot)

            # 保持性能数据长度
            max_performance_data = self.config.get('system', {}).get('max_performance_data', 5000)
            if len(self.performance_data) > max_performance_data:
                self.performance_data = self.performance_data[-max_performance_data:]

            # 如果性能分析器已初始化，更新其数据
            if hasattr(self, 'performance_analyzer') and self.performance_analyzer:
                self.performance_analyzer.update_performance_data(performance_snapshot)

            logger.debug(f"性能数据已更新，当前记录数: {len(self.performance_data)}")

        except Exception as e:
            logger.error(f"性能数据更新失败: {e}")
            self._handle_performance_error(e)

    def _get_current_performance(self) -> Dict:
        """获取当前性能指标 - 完整实现"""
        try:
            current_time = datetime.now()
            uptime = (current_time - self.start_time).total_seconds() if self.start_time else 0

            # 计算当前投资组合价值
            portfolio_value = self._calculate_portfolio_value()

            # 计算总回报率
            total_return = (
                                       portfolio_value - self.initial_capital) / self.initial_capital if self.initial_capital > 0 else 0

            # 计算年化回报率
            annualized_return = self._calculate_annualized_return(total_return, uptime)

            # 计算夏普比率
            sharpe_ratio = self._calculate_sharpe_ratio()

            # 计算最大回撤
            max_drawdown = self._calculate_max_drawdown()

            # 计算波动率
            volatility = self._calculate_volatility()

            # 计算胜率
            win_rate = self._calculate_win_rate()

            # 计算盈亏比
            profit_factor = self._calculate_profit_factor()

            # 获取风险指标
            risk_metrics = self._get_risk_metrics()

            return {
                'timestamp': current_time.isoformat(),
                'uptime_seconds': uptime,
                'portfolio_value': portfolio_value,
                'cash_balance': self.current_capital,
                'total_assets': portfolio_value + self.current_capital,
                'initial_capital': self.initial_capital,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'positions_count': len(self.current_positions),
                'active_positions': sum(1 for pos in self.current_positions.values() if abs(pos) > 0.0001),
                'total_trades': self.total_trades,
                'trading_volume': self.total_volume,
                'risk_metrics': risk_metrics,
                'system_state': self.state.value,
                'trading_mode': self.trading_mode.value,
                'performance_score': self._calculate_performance_score(total_return, sharpe_ratio, max_drawdown),
                'resource_usage': self._get_resource_usage()
            }

        except Exception as e:
            logger.error(f"性能指标计算失败: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'system_state': self.state.value
            }

    def _calculate_portfolio_value(self) -> float:
        """计算投资组合价值"""
        try:
            total_value = self.current_capital

            # 获取当前市场价格
            if self.current_positions and hasattr(self, 'data_fetcher'):
                symbols = list(self.current_positions.keys())
                current_prices = self.data_fetcher.get_current_prices(symbols)

                for symbol, quantity in self.current_positions.items():
                    if symbol in current_prices and current_prices[symbol] is not None:
                        total_value += quantity * current_prices[symbol]

            return total_value

        except Exception as e:
            logger.error(f"投资组合价值计算失败: {e}")
            return self.current_capital

    def _calculate_annualized_return(self, total_return: float, uptime: float) -> float:
        """计算年化回报率"""
        if uptime <= 0:
            return 0.0

        # 将秒转换为年
        years = uptime / (365 * 24 * 3600)
        if years <= 0:
            return 0.0

        annualized_return = (1 + total_return) ** (1 / years) - 1
        return annualized_return

    def _calculate_sharpe_ratio(self) -> float:
        """计算夏普比率"""
        try:
            if len(self.performance_data) < 20:  # 需要足够的数据点
                return 0.0

            # 提取日回报率
            returns = []
            for i in range(1, len(self.performance_data)):
                current_value = self.performance_data[i]['total_assets']
                prev_value = self.performance_data[i - 1]['total_assets']
                if prev_value > 0:
                    daily_return = (current_value - prev_value) / prev_value
                    returns.append(daily_return)

            if not returns:
                return 0.0

            # 计算年化夏普比率（假设无风险利率为0）
            avg_return = np.mean(returns) * 252  # 年化平均回报
            std_dev = np.std(returns) * np.sqrt(252)  # 年化标准差

            if std_dev > 0:
                return avg_return / std_dev
            return 0.0

        except Exception as e:
            logger.error(f"夏普比率计算失败: {e}")
            return 0.0

    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        try:
            if len(self.performance_data) < 10:
                return 0.0

            # 提取资产价值序列
            values = [data['total_assets'] for data in self.performance_data]

            # 计算回撤
            peak = values[0]
            max_drawdown = 0.0

            for value in values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

            return max_drawdown

        except Exception as e:
            logger.error(f"最大回撤计算失败: {e}")
            return 0.0

    def _calculate_volatility(self) -> float:
        """计算波动率"""
        try:
            if len(self.performance_data) < 20:
                return 0.0

            # 提取日回报率
            returns = []
            for i in range(1, len(self.performance_data)):
                current_value = self.performance_data[i]['total_assets']
                prev_value = self.performance_data[i - 1]['total_assets']
                if prev_value > 0:
                    daily_return = (current_value - prev_value) / prev_value
                    returns.append(daily_return)

            if not returns:
                return 0.0

            # 计算年化波动率
            volatility = np.std(returns) * np.sqrt(252)
            return volatility

        except Exception as e:
            logger.error(f"波动率计算失败: {e}")
            return 0.0

    def _calculate_win_rate(self) -> float:
        """计算胜率"""
        try:
            if not self.trading_history:
                return 0.0

            winning_trades = 0
            total_evaluated = 0

            for trade in self.trading_history[-100:]:  # 只看最近100笔交易
                if 'execution_result' in trade and 'trades' in trade['execution_result']:
                    for individual_trade in trade['execution_result']['trades']:
                        if 'profit' in individual_trade:
                            total_evaluated += 1
                            if individual_trade['profit'] > 0:
                                winning_trades += 1

            if total_evaluated > 0:
                return winning_trades / total_evaluated
            return 0.0

        except Exception as e:
            logger.error(f"胜率计算失败: {e}")
            return 0.0

    def _calculate_profit_factor(self) -> float:
        """计算盈亏比"""
        try:
            if not self.trading_history:
                return 0.0

            total_profit = 0.0
            total_loss = 0.0

            for trade in self.trading_history[-100:]:  # 只看最近100笔交易
                if 'execution_result' in trade and 'trades' in trade['execution_result']:
                    for individual_trade in trade['execution_result']['trades']:
                        if 'profit' in individual_trade:
                            profit = individual_trade['profit']
                            if profit > 0:
                                total_profit += profit
                            else:
                                total_loss += abs(profit)

            if total_loss > 0:
                return total_profit / total_loss
            elif total_profit > 0:
                return float('inf')  # 无亏损，只有盈利
            return 0.0

        except Exception as e:
            logger.error(f"盈亏比计算失败: {e}")
            return 0.0

    def _get_risk_metrics(self) -> Dict[str, float]:
        """获取风险指标"""
        try:
            if not hasattr(self, 'risk_manager') or not self.risk_manager:
                return {}

            return self.risk_manager.get_current_risk_metrics(
                self.current_positions,
                self.performance_data,
                self.trading_history
            )

        except Exception as e:
            logger.error(f"风险指标获取失败: {e}")
            return {}

    def _calculate_performance_score(self, total_return: float, sharpe_ratio: float, max_drawdown: float) -> float:
        """计算综合性能评分"""
        try:
            # 权重配置
            weights = {
                'return': 0.4,
                'sharpe': 0.3,
                'drawdown': 0.3
            }

            # 标准化各项指标
            return_score = min(max(total_return * 10, 0), 1)  # 将回报率映射到0-1范围
            sharpe_score = min(max(sharpe_ratio / 2, 0), 1)  # 将夏普比率映射到0-1范围
            drawdown_score = 1 - min(max_drawdown, 1)  # 回撤越小分数越高

            # 计算加权分数
            performance_score = (
                    weights['return'] * return_score +
                    weights['sharpe'] * sharpe_score +
                    weights['drawdown'] * drawdown_score
            )

            return round(performance_score, 3)

        except Exception as e:
            logger.error(f"性能评分计算失败: {e}")
            return 0.0

    def _get_resource_usage(self) -> Dict[str, float]:
        """获取资源使用情况"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                'memory_mb': memory_info.rss / 1024 / 1024,
                'memory_percent': process.memory_percent(),
                'cpu_percent': process.cpu_percent(interval=0.1),
                'thread_count': process.num_threads(),
                'open_files': len(process.open_files()),
                'io_counters': process.io_counters()._asdict() if process.io_counters() else {}
            }

        except Exception as e:
            logger.error(f"资源使用情况获取失败: {e}")
            return {}

    def _record_cycle_metrics(self, cycle_metrics: TradingCycleMetrics):
        """记录周期指标"""
        try:
            # 转换为字典格式
            metrics_dict = asdict(cycle_metrics)
            self.cycle_metrics.append(metrics_dict)

            # 保持历史记录长度
            max_cycle_metrics = self.config.get('system', {}).get('max_cycle_metrics', 1000)
            if len(self.cycle_metrics) > max_cycle_metrics:
                self.cycle_metrics = self.cycle_metrics[-max_cycle_metrics:]

            # 记录到审计日志
            self.audit_log.append({
                'type': 'trading_cycle',
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics_dict
            })

            logger.debug(f"周期指标记录完成: {cycle_metrics.cycle_id}")

        except Exception as e:
            logger.error(f"周期指标记录失败: {e}")

    def _handle_initialization_error(self, error: Exception):
        """处理初始化错误"""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'system_state': self.state.value,
            'component': 'system_initialization'
        }
        self.error_log.append(error_info)

        # 尝试灾难恢复
        if hasattr(self, 'disaster_recovery') and self.disaster_recovery:
            recovery_success = self.disaster_recovery.handle_initialization_failure(error_info)
            if recovery_success:
                logger.info("系统初始化错误已通过灾难恢复处理")
                self.state = SystemState.RECOVERY
            else:
                logger.error("灾难恢复处理失败")
                self.state = SystemState.ERROR
        else:
            self.state = SystemState.ERROR

    def _handle_trading_error(self, error: Exception):
        """处理交易错误"""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'system_state': self.state.value,
            'component': 'trading_cycle'
        }
        self.error_log.append(error_info)

        # 错误计数和自动恢复
        recent_errors = [e for e in self.error_log
                         if datetime.fromisoformat(e['timestamp']) > datetime.now() - timedelta(hours=1)]

        if len(recent_errors) > 10:  # 1小时内超过10个错误
            logger.error("错误频率过高，进入安全模式")
            self.state = SystemState.SAFE_MODE

            # 尝试自动恢复
            if hasattr(self, 'disaster_recovery') and self.disaster_recovery:
                recovery_success = self.disaster_recovery.attempt_auto_recovery()
                if recovery_success:
                    self.state = SystemState.RUNNING
                    logger.info("系统自动恢复成功")
                else:
                    logger.error("系统自动恢复失败")

    def _handle_cycle_error(self, error: Exception, cycle_id: str):
        """处理周期错误"""
        error_info = {
            'cycle_id': cycle_id,
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'system_state': self.state.value
        }
        self.error_log.append(error_info)

        # 记录失败的周期指标
        failed_cycle_metrics = TradingCycleMetrics(
            cycle_id=cycle_id,
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            duration=0,
            signals_generated=0,
            signals_executed=0,
            risk_assessment={},
            trades_executed=0,
            trading_volume=0,
            slippage=0,
            commission=0,
            performance_metrics={},
            error_count=1,
            status='failed'
        )
        self._record_cycle_metrics(failed_cycle_metrics)

        logger.error(f"交易周期 {cycle_id} 执行失败: {error}")

    def _handle_performance_error(self, error: Exception):
        """处理性能计算错误"""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'system_state': self.state.value,
            'component': 'performance_calculation'
        }
        self.error_log.append(error_info)
        logger.error(f"性能计算错误: {error}")

    def _record_risk_rejection(self, risk_assessment: RiskAssessment):
        """记录风险拒绝"""
        rejection_record = {
            'timestamp': datetime.now().isoformat(),
            'type': 'risk_rejection',
            'assessment': risk_assessment.to_dict(),
            'current_positions': self.current_positions.copy(),
            'current_capital': self.current_capital,
            'performance_snapshot': self._get_current_performance()
        }
        self.audit_log.append(rejection_record)

        # 保持审计日志长度
        max_audit_log = self.config.get('system', {}).get('max_audit_log', 5000)
        if len(self.audit_log) > max_audit_log:
            self.audit_log = self.audit_log[-max_audit_log:]

        logger.info(f"风险拒绝记录: {risk_assessment.reason}")

    def get_status(self) -> SystemStatus:
        """获取系统状态"""
        current_time = datetime.now()
        uptime = (current_time - self.start_time).total_seconds() if self.start_time else 0

        # 获取系统资源使用情况
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent(interval=0.1)

        # 计算活跃模块数量
        active_modules = sum(1 for module in self.modules.values()
                             if hasattr(module, 'is_active') and module.is_active())

        # 获取当前性能指标
        performance = self._get_current_performance()

        # 获取当前风险等级
        risk_level = self._get_current_risk_level()

        return SystemStatus(
            state=self.state,
            trading_mode=self.trading_mode,
            uptime=uptime,
            memory_usage_mb=memory_info.rss / 1024 / 1024,
            cpu_usage_percent=cpu_percent,
            active_modules=active_modules,
            error_count=len(self.error_log),
            last_update=current_time.isoformat(),
            performance_metrics=performance,
            positions_count=len(self.current_positions),
            trading_volume=self.total_volume,
            risk_level=risk_level
        )

    def _get_current_risk_level(self) -> RiskLevel:
        """获取当前风险等级"""
        try:
            if hasattr(self, 'risk_manager') and self.risk_manager:
                return self.risk_manager.get_current_risk_level(
                    self.current_positions,
                    self.performance_data
                )
            return RiskLevel.MODERATE
        except Exception as e:
            logger.error(f"风险等级获取失败: {e}")
            return RiskLevel.UNKNOWN

    def get_performance_report(self, period: str = '30d') -> Dict:
        """获取性能报告"""
        try:
            if hasattr(self, 'performance_analyzer') and self.performance_analyzer:
                return self.performance_analyzer.generate_report(
                    self.performance_data,
                    period,
                    self.trading_history
                )
            else:
                return self._generate_basic_performance_report(period)

        except Exception as e:
            logger.error(f"性能报告生成失败: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def _generate_basic_performance_report(self, period: str) -> Dict:
        """生成基础性能报告"""
        try:
            # 筛选指定期间的数据
            if period.endswith('d'):
                days = int(period[:-1])
                cutoff_date = datetime.now() - timedelta(days=days)
            else:
                cutoff_date = datetime.now() - timedelta(days=30)  # 默认30天

            recent_data = [d for d in self.performance_data
                           if datetime.fromisoformat(d['timestamp']) >= cutoff_date]

            if not recent_data:
                return {'error': 'No data available for the specified period'}

            # 提取关键指标
            returns = [d.get('total_return', 0) for d in recent_data]
            volatilities = [d.get('volatility', 0) for d in recent_data]
            sharpe_ratios = [d.get('sharpe_ratio', 0) for d in recent_data]
            drawdowns = [d.get('max_drawdown', 0) for d in recent_data]

            # 计算统计指标
            total_return = returns[-1] - returns[0] if returns else 0
            avg_daily_return = np.mean(np.diff(returns)) if len(returns) > 1 else 0
            annualized_return = avg_daily_return * 252 if avg_daily_return else 0
            avg_volatility = np.mean(volatilities) if volatilities else 0
            avg_sharpe = np.mean(sharpe_ratios) if sharpe_ratios else 0
            max_drawdown = max(drawdowns) if drawdowns else 0

            return {
                'period': period,
                'start_date': recent_data[0]['timestamp'],
                'end_date': recent_data[-1]['timestamp'],
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': avg_volatility,
                'sharpe_ratio': avg_sharpe,
                'max_drawdown': max_drawdown,
                'win_rate': self._calculate_win_rate(),
                'profit_factor': self._calculate_profit_factor(),
                'positions_count': len(self.current_positions),
                'trades_executed': self.total_trades,
                'trading_volume': self.total_volume,
                'risk_level': self._get_current_risk_level().value,
                'performance_score': self._calculate_performance_score(total_return, avg_sharpe, max_drawdown),
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"基础性能报告生成失败: {e}")
            return {'error': str(e)}

    def run_backtest(self, strategy_config: Dict) -> Dict:
        """运行回测"""
        try:
            if not hasattr(self, 'backtesting_engine') or not self.backtesting_engine:
                raise RuntimeError("回测引擎未初始化")

            # 确保系统处于可回测状态
            if self.state not in [SystemState.INITIALIZED, SystemState.RUNNING]:
                raise RuntimeError(f"系统状态 {self.state} 不支持回测")

            logger.info(f"开始回测策略: {strategy_config.get('name', 'unknown')}")

            # 运行回测
            backtest_result = self.backtesting_engine.run_backtest(
                strategy_config,
                self.data_fetcher,
                initial_capital=strategy_config.get('initial_capital', 1000000)
            )

            # 记录回测结果
            self._record_backtest_result(backtest_result)

            return backtest_result

        except Exception as e:
            logger.error(f"回测执行失败: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _record_backtest_result(self, result: Dict):
        """记录回测结果"""
        try:
            backtest_record = {
                'timestamp': datetime.now().isoformat(),
                'result': result,
                'strategy_name': result.get('strategy_name', 'unknown'),
                'performance_metrics': result.get('performance_metrics', {}),
                'parameters': result.get('parameters', {})
            }

            # 保存到审计日志
            self.audit_log.append({
                'type': 'backtest_result',
                'timestamp': datetime.now().isoformat(),
                'data': backtest_record
            })

            # 如果回测成功，更新优化历史
            if result.get('status') == 'completed' and hasattr(self, 'performance_analyzer'):
                self.performance_analyzer.record_backtest_result(result)

            logger.info(f"回测结果记录完成: {result.get('strategy_name', 'unknown')}")

        except Exception as e:
            logger.error(f"回测结果记录失败: {e}")

    def optimize_strategy(self, strategy_config: Dict,
                          parameter_space: Dict,
                          objective_metrics: List[str]) -> Dict:
        """优化策略参数"""
        try:
            if not hasattr(self, 'bayesian_optimizer') or not self.bayesian_optimizer:
                raise RuntimeError("贝叶斯优化器未初始化")

            logger.info(f"开始策略优化: {strategy_config.get('name', 'unknown')}")

            # 运行优化
            optimization_result = self.bayesian_optimizer.optimize_strategy(
                strategy_config,
                parameter_space,
                objective_metrics,
                self.data_fetcher
            )

            # 记录优化结果
            self._record_optimization_result(optimization_result)

            return optimization_result

        except Exception as e:
            logger.error(f"策略优化失败: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _record_optimization_result(self, result: Dict):
        """记录优化结果"""
        try:
            optimization_record = {
                'timestamp': datetime.now().isoformat(),
                'result': result,
                'strategy_name': result.get('strategy_name', 'unknown'),
                'best_parameters': result.get('best_parameters', {}),
                'best_score': result.get('best_score', 0),
                'optimization_time': result.get('optimization_time', 0),
                'iterations': result.get('iterations', 0),
                'objective_metrics': result.get('objective_metrics', []),
                'parameter_space': result.get('parameter_space', {})
            }

            # 保存到审计日志
            self.audit_log.append({
                'type': 'optimization_result',
                'timestamp': datetime.now().isoformat(),
                'data': optimization_record
            })

            # 如果优化成功，更新策略配置
            if result.get('status') == 'completed' and 'best_parameters' in result:
                self._update_strategy_config(result['strategy_name'], result['best_parameters'])

            logger.info(f"优化结果记录完成: {result.get('strategy_name', 'unknown')}")

        except Exception as e:
            logger.error(f"优化结果记录失败: {e}")

    def _update_strategy_config(self, strategy_name: str, best_parameters: Dict):
        """更新策略配置"""
        try:
            # 更新内存中的配置
            if 'strategies' in self.config and strategy_name in self.config['strategies']:
                self.config['strategies'][strategy_name]['parameters'] = best_parameters
                logger.info(f"策略 {strategy_name} 配置已更新")

            # 保存到配置文件
            if hasattr(self, 'config_manager') and self.config_manager:
                self.config_manager.update_config(self.config)
                logger.info(f"策略 {strategy_name} 配置已保存到文件")

        except Exception as e:
            logger.error(f"策略配置更新失败: {e}")

    def switch_trading_mode(self, new_mode: TradingMode):
        """切换交易模式"""
        try:
            if self.state != SystemState.RUNNING:
                raise RuntimeError("只有在运行状态下才能切换交易模式")

            logger.info(f"切换交易模式: {self.trading_mode.value} -> {new_mode.value}")

            old_mode = self.trading_mode
            self.trading_mode = new_mode

            # 更新配置
            self.config['system']['trading_mode'] = new_mode.value
            self.config_manager.update_config(self.config)

            # 记录模式切换
            self.audit_log.append({
                'type': 'mode_switch',
                'timestamp': datetime.now().isoformat(),
                'old_mode': old_mode.value,
                'new_mode': new_mode.value,
                'reason': 'manual_switch'
            })

            # 如果是切换到实盘模式，需要额外验证
            if new_mode == TradingMode.LIVE_TRADING:
                self._validate_live_trading_readiness()

            return True

        except Exception as e:
            logger.error(f"交易模式切换失败: {e}")
            return False

    def _validate_live_trading_readiness(self):
        """验证实盘交易准备状态"""
        checks = []

        # 检查经纪商连接
        if hasattr(self.execution_engine, 'check_broker_connection'):
            broker_ok = self.execution_engine.check_broker_connection()
            checks.append(('broker_connection', broker_ok))

        # 检查风险控制
        risk_ok = self.risk_manager.validate_live_trading_ready()
        checks.append(('risk_management', risk_ok))

        # 检查资金充足性
        capital_ok = self._check_capital_sufficiency()
        checks.append(('capital_sufficiency', capital_ok))

        # 检查系统稳定性
        stability_ok = self._check_system_stability()
        checks.append(('system_stability', stability_ok))

        # 检查数据质量
        data_quality_ok = self._check_data_quality()
        checks.append(('data_quality', data_quality_ok))

        # 检查所有检查是否通过
        all_checks_passed = all(check[1] for check in checks)

        if not all_checks_passed:
            failed_checks = [check[0] for check in checks if not check[1]]
            raise RuntimeError(f"实盘交易准备检查失败: {failed_checks}")

        logger.info("实盘交易准备检查全部通过")

    def _check_capital_sufficiency(self) -> bool:
        """检查资金充足性"""
        try:
            # 获取最小资金要求
            min_capital = self.config.get('trading', {}).get('min_capital', 10000)

            # 检查当前资金是否足够
            if self.current_capital < min_capital:
                logger.warning(f"资金不足: 当前 {self.current_capital:.2f}, 要求 {min_capital:.2f}")
                return False

            return True

        except Exception as e:
            logger.error(f"资金充足性检查失败: {e}")
            return False

    def _check_system_stability(self) -> bool:
        """检查系统稳定性"""
        try:
            # 检查错误率
            recent_errors = [e for e in self.error_log
                             if datetime.fromisoformat(e['timestamp']) > datetime.now() - timedelta(hours=24)]

            if len(recent_errors) > 10:  # 24小时内超过10个错误
                logger.warning(f"系统稳定性不足: 24小时内错误数 {len(recent_errors)}")
                return False

            # 检查运行时间
            if self.start_time:
                uptime = (datetime.now() - self.start_time).total_seconds()
                if uptime < 3600:  # 运行时间不足1小时
                    logger.warning(f"系统运行时间不足: {uptime:.0f}秒")
                    return False

            return True

        except Exception as e:
            logger.error(f"系统稳定性检查失败: {e}")
            return False

    def _check_data_quality(self) -> bool:
        """检查数据质量"""
        try:
            # 检查数据获取成功率
            if hasattr(self.data_fetcher, 'get_success_rate'):
                success_rate = self.data_fetcher.get_success_rate()
                if success_rate < 0.95:  # 成功率低于95%
                    logger.warning(f"数据质量不足: 成功率 {success_rate:.2%}")
                    return False

            # 检查数据延迟
            if hasattr(self.data_fetcher, 'get_data_latency'):
                latency = self.data_fetcher.get_data_latency()
                max_latency = self.config.get('data_sources', {}).get('max_latency', 60)  # 默认60秒
                if latency > max_latency:
                    logger.warning(f"数据延迟过高: {latency:.2f}秒 > {max_latency}秒")
                    return False

            return True

        except Exception as e:
            logger.error(f"数据质量检查失败: {e}")
            return False

    def get_module_status(self, module_name: str) -> Dict:
        """获取模块状态"""
        if module_name not in self.modules:
            return {'error': f'模块 {module_name} 不存在'}

        module = self.modules[module_name]
        status = {
            'name': module_name,
            'active': hasattr(module, 'is_active') and module.is_active(),
            'last_update': datetime.now().isoformat()
        }

        # 添加模块特定状态信息
        if hasattr(module, 'get_status'):
            module_status = module.get_status()
            status.update(module_status)

        return status

    def get_all_modules_status(self) -> Dict[str, Dict]:
        """获取所有模块状态"""
        status = {}
        for name, module in self.modules.items():
            status[name] = self.get_module_status(name)
        return status

    def restart_module(self, module_name: str) -> bool:
        """重启模块"""
        try:
            if module_name not in self.modules:
                raise ValueError(f"模块 {module_name} 不存在")

            module = self.modules[module_name]

            # 停止模块
            if hasattr(module, 'stop'):
                module.stop()

            # 重新初始化模块
            if hasattr(module, '__init__'):
                # 获取模块配置
                config_key = self._get_module_config_key(module_name)
                module_config = self.config.get(config_key, {})

                # 重新初始化
                module.__init__(module_config)

            # 启动模块
            if hasattr(module, 'start'):
                module.start()

            logger.info(f"模块 {module_name} 重启成功")
            return True

        except Exception as e:
            logger.error(f"模块 {module_name} 重启失败: {e}")
            return False

    def _get_module_config_key(self, module_name: str) -> str:
        """获取模块配置键名"""
        config_map = {
            'data_fetcher': 'data_sources',
            'signal_engine': 'signal_engine',
            'portfolio_manager': 'portfolio_management',
            'risk_manager': 'risk_management',
            'execution_engine': 'execution',
            'bayesian_optimizer': 'optimization',
            'backtesting_engine': 'backtesting',
            'performance_analyzer': 'performance_analytics',
            'monitoring_system': 'monitoring',
            'api_gateway': 'api_gateway'
        }
        return config_map.get(module_name, module_name)

    def export_data(self, data_type: str, filepath: str) -> bool:
        """导出数据"""
        try:
            if data_type == 'trading_history':
                data = self.trading_history
            elif data_type == 'performance_data':
                data = self.performance_data
            elif data_type == 'audit_log':
                data = self.audit_log
            elif data_type == 'error_log':
                data = self.error_log
            elif data_type == 'signal_history':
                data = self.signal_history
            elif data_type == 'risk_history':
                data = self.risk_history
            elif data_type == 'cycle_metrics':
                data = self.cycle_metrics
            else:
                raise ValueError(f"不支持的数据类型: {data_type}")

            # 导出到文件
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"数据导出成功: {data_type} -> {filepath}")
            return True

        except Exception as e:
            logger.error(f"数据导出失败: {e}")
            return False

    def import_data(self, data_type: str, filepath: str) -> bool:
        """导入数据"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"文件不存在: {filepath}")

            with open(filepath, 'r') as f:
                data = json.load(f)

            if data_type == 'trading_history':
                self.trading_history = data
            elif data_type == 'performance_data':
                self.performance_data = data
            elif data_type == 'audit_log':
                self.audit_log = data
            elif data_type == 'error_log':
                self.error_log = data
            elif data_type == 'signal_history':
                self.signal_history = data
            elif data_type == 'risk_history':
                self.risk_history = data
            elif data_type == 'cycle_metrics':
                self.cycle_metrics = data
            else:
                raise ValueError(f"不支持的数据类型: {data_type}")

            logger.info(f"数据导入成功: {filepath} -> {data_type}")
            return True

        except Exception as e:
            logger.error(f"数据导入失败: {e}")
            return False

    def run_maintenance(self):
        """运行系统维护"""
        try:
            logger.info("开始系统维护...")

            maintenance_tasks = [
                self._cleanup_old_data,
                self._optimize_performance,
                self._backup_system,
                self._check_security,
                self._update_system
            ]

            for task in maintenance_tasks:
                try:
                    task()
                    logger.info(f"维护任务完成: {task.__name__}")
                except Exception as e:
                    logger.error(f"维护任务失败 {task.__name__}: {e}")

            logger.info("系统维护完成")

        except Exception as e:
            logger.error(f"系统维护失败: {e}")

    def _cleanup_old_data(self):
        """清理旧数据"""
        try:
            logger.info("清理旧数据...")

            # 清理性能数据
            max_performance_data = self.config.get('system', {}).get('max_performance_data', 5000)
            if len(self.performance_data) > max_performance_data:
                self.performance_data = self.performance_data[-max_performance_data:]
                logger.info(f"性能数据清理完成，保留 {len(self.performance_data)} 条记录")

            # 清理交易历史
            max_trading_history = self.config.get('system', {}).get('max_trading_history', 1000)
            if len(self.trading_history) > max_trading_history:
                self.trading_history = self.trading_history[-max_trading_history:]
                logger.info(f"交易历史清理完成，保留 {len(self.trading_history)} 条记录")

            # 清理错误日志
            max_error_log = self.config.get('system', {}).get('max_error_log', 1000)
            if len(self.error_log) > max_error_log:
                self.error_log = self.error_log[-max_error_log:]
                logger.info(f"错误日志清理完成，保留 {len(self.error_log)} 条记录")

            # 清理审计日志
            max_audit_log = self.config.get('system', {}).get('max_audit_log', 5000)
            if len(self.audit_log) > max_audit_log:
                self.audit_log = self.audit_log[-max_audit_log:]
                logger.info(f"审计日志清理完成，保留 {len(self.audit_log)} 条记录")

            # 清理其他历史数据
            for history_name in ['signal_history', 'risk_history', 'cycle_metrics']:
                history = getattr(self, history_name)
                max_history = self.config.get('system', {}).get(f'max_{history_name}', 1000)
                if len(history) > max_history:
                    setattr(self, history_name, history[-max_history:])
                    logger.info(f"{history_name} 清理完成，保留 {len(history)} 条记录")

            # 清理临时文件
            self._cleanup_temp_files()

            logger.info("旧数据清理完成")

        except Exception as e:
            logger.error(f"旧数据清理失败: {e}")
            raise

    def _cleanup_temp_files(self):
        """清理临时文件"""
        try:
            temp_dirs = ['temp/', 'cache/', 'exports/']
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    # 删除超过7天的临时文件
                    for filename in os.listdir(temp_dir):
                        filepath = os.path.join(temp_dir, filename)
                        if os.path.isfile(filepath):
                            file_age = time.time() - os.path.getmtime(filepath)
                            if file_age > 604800:  # 7天
                                os.remove(filepath)
                                logger.debug(f"删除临时文件: {filepath}")

        except Exception as e:
            logger.warning(f"临时文件清理失败: {e}")

    def _optimize_performance(self):
        """优化性能"""
        try:
            logger.info("优化系统性能...")

            # 优化数据库性能
            if hasattr(self, 'data_fetcher') and hasattr(self.data_fetcher, 'optimize_performance'):
                self.data_fetcher.optimize_performance()

            # 优化内存使用
            self._optimize_memory()

            # 优化网络连接
            if hasattr(self, 'api_gateway') and hasattr(self.api_gateway, 'optimize_connections'):
                self.api_gateway.optimize_connections()

            # 优化缓存
            if hasattr(self, 'performance_analyzer') and hasattr(self.performance_analyzer, 'optimize_cache'):
                self.performance_analyzer.optimize_cache()

            logger.info("性能优化完成")

        except Exception as e:
            logger.error(f"性能优化失败: {e}")
            raise

    def _optimize_memory(self):
        """优化内存使用"""
        try:
            # 清理缓存
            if hasattr(self.data_fetcher, 'clear_cache'):
                self.data_fetcher.clear_cache()

            # 调用垃圾回收
            import gc
            gc.collect()

            logger.info("内存优化完成")

        except Exception as e:
            logger.warning(f"内存优化失败: {e}")

    def _backup_system(self):
        """备份系统"""
        try:
            logger.info("备份系统数据...")

            backup_dir = self.config.get('system', {}).get('backup_dir', 'backups')
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = os.path.join(backup_dir, f'system_backup_{timestamp}.zip')

            # 创建备份
            import zipfile
            with zipfile.ZipFile(backup_file, 'w') as zipf:
                # 备份配置
                zipf.writestr('config.json', json.dumps(self.config, indent=2))

                # 备份数据
                data_types = ['trading_history', 'performance_data', 'audit_log', 'error_log']
                for data_type in data_types:
                    data = getattr(self, data_type)
                    zipf.writestr(f'{data_type}.json', json.dumps(data, indent=2))

                # 备份模块状态
                modules_status = self.get_all_modules_status()
                zipf.writestr('modules_status.json', json.dumps(modules_status, indent=2))

            logger.info(f"系统备份完成: {backup_file}")

            # 清理旧备份
            self._cleanup_old_backups(backup_dir)

        except Exception as e:
            logger.error(f"系统备份失败: {e}")
            raise

    def _cleanup_old_backups(self, backup_dir: str):
        """清理旧备份"""
        try:
            max_backups = self.config.get('system', {}).get('max_backups', 30)
            backup_files = [f for f in os.listdir(backup_dir) if
                            f.startswith('system_backup_') and f.endswith('.zip')]
            backup_files.sort(key=lambda x: os.path.getmtime(os.path.join(backup_dir, x)))

            if len(backup_files) > max_backups:
                for old_backup in backup_files[:-max_backups]:
                    os.remove(os.path.join(backup_dir, old_backup))
                    logger.info(f"删除旧备份: {old_backup}")

        except Exception as e:
            logger.warning(f"旧备份清理失败: {e}")

    def _check_security(self):
        """安全检查"""
        try:
            logger.info("运行安全检查...")

            security_checks = [
                self._check_authentication,
                self._check_authorization,
                self._check_data_encryption,
                self._check_network_security
            ]

            security_issues = []
            for check in security_checks:
                try:
                    issues = check()
                    security_issues.extend(issues)
                except Exception as e:
                    logger.warning(f"安全检查失败 {check.__name__}: {e}")

            if security_issues:
                logger.warning(f"发现 {len(security_issues)} 个安全问题")
                for issue in security_issues:
                    logger.warning(f"安全问题: {issue}")
            else:
                logger.info("安全检查完成，未发现问题")

        except Exception as e:
            logger.error(f"安全检查失败: {e}")
            raise

    def _check_authentication(self) -> List[str]:
        """检查认证安全"""
        issues = []
        # 实现认证检查逻辑
        return issues

    def _check_authorization(self) -> List[str]:
        """检查授权安全"""
        issues = []
        # 实现授权检查逻辑
        return issues

    def _check_data_encryption(self) -> List[str]:
        """检查数据加密"""
        issues = []
        # 实现加密检查逻辑
        return issues

    def _check_network_security(self) -> List[str]:
        """检查网络安全"""
        issues = []
        # 实现网络安全检查逻辑
        return issues

    def _update_system(self):
        """更新系统"""
        try:
            logger.info("检查系统更新...")

            # 检查配置更新
            if hasattr(self, 'config_manager'):
                new_config = self.config_manager.check_for_updates()
                if new_config:
                    logger.info("检测到配置更新，重新加载配置")
                    self.config = new_config
                    self._apply_config_updates()

            # 检查模块更新
            self._check_module_updates()

            logger.info("系统更新检查完成")

        except Exception as e:
            logger.error(f"系统更新检查失败: {e}")
            raise

    def _apply_config_updates(self):
        """应用配置更新"""
        try:
            # 更新所有模块的配置
            for name, module in self.modules.items():
                config_key = self._get_module_config_key(name)
                module_config = self.config.get(config_key, {})

                if hasattr(module, 'update_config'):
                    module.update_config(module_config)
                    logger.info(f"模块 {name} 配置已更新")

            logger.info("配置更新应用完成")

        except Exception as e:
            logger.error(f"配置更新应用失败: {e}")

    def _check_module_updates(self):
        """检查模块更新"""
        # 实现模块更新检查逻辑
        pass

    def cleanup(self):
        """清理系统资源"""
        try:
            logger.info("开始清理系统资源")

            # 停止所有模块
            for name, module in self.modules.items():
                if hasattr(module, 'cleanup'):
                    module.cleanup()
                    logger.info(f"模块 {name} 清理完成")

            # 清理数据
            self.trading_history.clear()
            self.performance_data.clear()
            self.audit_log.clear()
            self.error_log.clear()
            self.signal_history.clear()
            self.risk_history.clear()
            self.cycle_metrics.clear()
            self.current_positions.clear()

            # 关闭线程池
            if self.thread_pool:
                self.thread_pool.shutdown(wait=False)

            self.state = SystemState.STOPPED
            logger.info("系统资源清理完成")

        except Exception as e:
            logger.error(f"系统资源清理失败: {e}")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.cleanup()

    def __del__(self):
        """析构函数"""
        if hasattr(self, 'state') and self.state != SystemState.STOPPED:
            try:
                self.cleanup()
            except:
                pass  # 避免析构函数中的异常

    def main():
        """主函数 - 命令行入口点"""
        import argparse

        parser = argparse.ArgumentParser(description='DeepSeekQuant 量化交易系统')
        parser.add_argument('--config', '-c', help='配置文件路径', default='config.json')
        parser.add_argument('--mode', '-m', choices=['run', 'backtest', 'optimize', 'maintenance'],
                            default='run', help='运行模式')
        parser.add_argument('--strategy', '-s', help='策略配置文件路径')
        parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
        parser.add_argument('--daemon', '-d', action='store_true', help='守护进程模式')
        parser.add_argument('--log-level', '-l', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                            default='INFO', help='日志级别')

        args = parser.parse_args()

        # 设置日志级别
        log_level = getattr(logging, args.log_level)
        logging.basicConfig(level=log_level,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        try:
            # 初始化系统
            system = DeepSeekQuantSystem(args.config)

            if args.mode == 'run':
                # 运行模式
                if system.start():
                    logger.info("系统启动成功，按 Ctrl+C 停止")

                    # 守护进程模式
                    if args.daemon:
                        while True:
                            time.sleep(1)
                    else:
                        # 等待用户中断
                        try:
                            while True:
                                time.sleep(1)
                        except KeyboardInterrupt:
                            logger.info("接收到中断信号，停止系统")
                            system.stop()

            elif args.mode == 'backtest':
                # 回测模式
                if not args.strategy:
                    raise ValueError("回测模式需要指定策略配置文件")

                with open(args.strategy, 'r') as f:
                    strategy_config = json.load(f)

                result = system.run_backtest(strategy_config)
                print(json.dumps(result, indent=2))

            elif args.mode == 'optimize':
                # 优化模式
                if not args.strategy:
                    raise ValueError("优化模式需要指定策略配置文件")

                with open(args.strategy, 'r') as f:
                    strategy_config = json.load(f)

                # 这里需要参数空间定义，简化处理
                parameter_space = strategy_config.get('parameter_space', {})
                result = system.optimize_strategy(strategy_config, parameter_space, ['sharpe_ratio'])
                print(json.dumps(result, indent=2))

            elif args.mode == 'maintenance':
                # 维护模式
                system.run_maintenance()

        except Exception as e:
            logger.error(f"系统执行失败: {e}")
            if args.verbose:
                traceback.print_exc()
            sys.exit(1)

    if __name__ == "__main__":
        main()