"""
系统管理器 - 业务层
从 core_bak/main.py 拆分
职责: 系统生命周期管理、模块协调
"""

from typing import Dict
import logging

logger = logging.getLogger("DeepSeekQuant.SystemManager")


class SystemManager:
    """系统管理器"""

    def __init__(self, config: Dict):
        self.config = config


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
