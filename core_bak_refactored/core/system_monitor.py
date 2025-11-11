"""
系统监控器 - 业务层
从 core_bak/main.py 拆分
职责: 系统状态监控、性能监控
"""

from typing import Dict
import logging

logger = logging.getLogger("DeepSeekQuant.SystemMonitor")


class SystemMonitor:
    """系统监控器"""

    def __init__(self, config: Dict):
        self.config = config


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