"""
系统核心 - 业务层
从 core_bak/main.py 拆分
职责: 系统主流程、模块协调
"""

from typing import Dict, Any
import logging

logger = logging.getLogger("DeepSeekQuant.SystemCore")


class DeepSeekQuantSystem:
    """DeepSeekQuant 量化交易系统核心"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("DeepSeekQuant系统初始化")

    def start(self):
        """启动系统"""
        logger.info("系统启动中...")

    def stop(self):
        """停止系统"""
        logger.info("系统停止")

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
