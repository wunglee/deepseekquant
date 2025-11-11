"""
组合构建器 - 业务层
从 core_bak/portfolio_manager.py 拆分
职责: 组合构建、资产配置
"""

from typing import Dict, List
import logging

logger = logging.getLogger("DeepSeekQuant.PortfolioBuilder")


class PortfolioBuilder:
    """组合构建器"""

    def __init__(self, config: Dict):
        self.config = config

        # 线程池
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.portfolio_config.get('max_optimization_workers', 4)
        )

        # 初始化优化器
        self._initialize_optimizers()

        logger.info("组合管理器初始化完成")

    def _initialize_optimizers(self):
        """初始化优化器"""
        self.optimizers = {
            AllocationMethod.EQUAL_WEIGHT: self._equal_weight_optimization,
            AllocationMethod.MARKET_CAP: self._market_cap_optimization,
            AllocationMethod.MIN_VARIANCE: self._min_variance_optimization,
            AllocationMethod.MAX_SHARPE: self._max_sharpe_optimization,
            AllocationMethod.RISK_PARITY: self._risk_parity_optimization,
            AllocationMethod.BLACK_LITTERMAN: self._black_litterman_optimization,
            AllocationMethod.HRP: self._hierarchical_risk_parity,
            AllocationMethod.CLA: self._critical_line_algorithm,
            AllocationMethod.CUSTOM: self._custom_optimization
        }

        # 初始化风险模型
        self.risk_models = {
            RiskModel.SAMPLE_COVARIANCE: risk_models.sample_cov,
            RiskModel.LEDOIT_WOLF: risk_models.ledoit_wolf,
            RiskModel.ORACLE_APPROXIMATING: risk_models.oracle_approximating,
            RiskModel.CONSTANT_CORRELATION: risk_models.constant_correlation,
            RiskModel.EXPONENTIALLY_WEIGHTED: risk_models.exp_cov,
            RiskModel.GARCH: self._garch_covariance,
            RiskModel.DCC_GARCH: self._dcc_garch_covariance
        }

        logger.info(f"已加载 {len(self.optimizers)} 种优化方法和 {len(self.risk_models)} 种风险模型")

    def process(self, signals: Dict[str, Any], market_data: Dict[str, Any],
                current_positions: Dict[str, float]) -> Dict[str, Any]:
        """
        处理信号和市场数据，生成优化后的组合

        Args:
            signals: 交易信号字典
            market_data: 市场数据
            current_positions: 当前持仓

        Returns:
            优化后的组合配置
        """
        start_time = time.time()

        try:
            # 验证输入数据
            if not self._validate_inputs(signals, market_data, current_positions):
                logger.warning("输入数据验证失败")
                return {}

            # 更新市场数据缓存
            self._update_market_data(market_data)

            # 计算预期收益和风险
            self._calculate_expected_returns(market_data)
            self._calculate_covariance_matrix(market_data)

            # 构建初始组合
            initial_portfolio = self._build_initial_portfolio(current_positions, market_data)

            # 优化组合
            optimized_portfolio = self._optimize_portfolio(initial_portfolio, signals, market_data)

            # 应用约束条件
            constrained_portfolio = self._apply_constraints(optimized_portfolio, market_data)

            # 风险检查
            risk_assessment = self._assess_portfolio_risk(constrained_portfolio, market_data)
            if not risk_assessment['approved']:
                logger.warning(f"组合风险检查未通过: {risk_assessment['reason']}")
                # 应用风险控制
                constrained_portfolio = self._apply_risk_control(constrained_portfolio, risk_assessment)

            # 生成调仓指令
            rebalance_instructions = self._generate_rebalance_instructions(
                self.current_portfolio, constrained_portfolio, market_data
            )

            # 更新当前组合状态
            self._update_portfolio_state(constrained_portfolio, rebalance_instructions)

            processing_time = time.time() - start_time
            logger.info(
                f"组合优化完成: 耗时 {processing_time:.3f}s, 换手率: {rebalance_instructions['turnover_rate']:.2%}")

            return {
                'optimized_portfolio': constrained_portfolio.to_dict(),
                'rebalance_instructions': rebalance_instructions,
                'risk_assessment': risk_assessment,
                'performance_metrics': self._calculate_performance_metrics(constrained_portfolio)
            }

        except Exception as e:
            logger.error(f"组合处理失败: {e}")
            self._handle_processing_error(e)
            return {}

    def _validate_inputs(self, signals: Dict, market_data: Dict,
                         current_positions: Dict) -> bool:
        """验证输入数据"""
        try:
            # 检查必要字段
            required_market_fields = ['timestamp', 'symbols', 'prices']
            if not all(field in market_data for field in required_market_fields):
                logger.warning("市场数据缺少必要字段")
                return False

            # 检查价格数据完整性
            for symbol in market_data['symbols']:
                if symbol not in market_data['prices']:
                    logger.warning(f"缺少价格数据: {symbol}")
                    return False

                price_data = market_data['prices'][symbol]
                if not all(key in price_data for key in ['open', 'high', 'low', 'close']):
                    logger.warning(f"价格数据不完整: {symbol}")
                    return False

            # 检查持仓数据
            if not isinstance(current_positions, dict):
                logger.warning("持仓数据格式错误")
                return False

            # 检查信号数据
            if signals and not isinstance(signals, dict):
                logger.warning("信号数据格式错误")
                return False

            return True

        except Exception as e:
            logger.error(f"输入数据验证失败: {e}")
            return False

    def _update_market_data(self, market_data: Dict):
        """更新市场数据缓存"""
        try:
            timestamp = market_data['timestamp']

            # 更新个股数据
            for symbol in market_data['symbols']:
                if symbol not in self.market_data_cache:
                    self.market_data_cache[symbol] = {}

                # 存储价格数据
                price_data = market_data['prices'][symbol]
                self.market_data_cache[symbol][timestamp] = {
                    'open': price_data['open'],
                    'high': price_data['high'],
                    'low': price_data['low'],
                    'close': price_data['close'],
                    'volume': market_data.get('volumes', {}).get(symbol, {}).get('volume', 0)
                }

            # 限制缓存大小
            max_cache_size = self.portfolio_config.get('max_market_data_cache', 1000)
            if len(self.market_data_cache) > max_cache_size:
                # 移除最旧的数据
                oldest_symbol = min(self.market_data_cache.keys(),
                                    key=lambda s: len(self.market_data_cache[s]))
                del self.market_data_cache[oldest_symbol]

        except Exception as e:
            logger.error(f"市场数据更新失败: {e}")

    def _calculate_expected_returns(self, market_data: Dict):
        """计算预期收益"""
        try:
            symbols = market_data['symbols']
            returns_data = {}

            for symbol in symbols:
                if symbol in market_data['prices']:
                    closes = market_data['prices'][symbol]['close']
                    if len(closes) > 1:
                        # 计算对数收益
                        log_returns = np.diff(np.log(closes))
                        returns_data[symbol] = np.mean(log_returns) * 252  # 年化收益

            self.expected_returns = pd.Series(returns_data)

        except Exception as e:
            logger.error(f"预期收益计算失败: {e}")
            self.expected_returns = None

    def _calculate_covariance_matrix(self, market_data: Dict):
        """计算协方差矩阵"""
        try:
            symbols = market_data['symbols']
            returns_matrix = []
            valid_symbols = []

            # 构建收益矩阵
            for symbol in symbols:
                if symbol in market_data['prices']:
                    closes = market_data['prices'][symbol]['close']
                    if len(closes) > 2:  # 至少需要3个点计算收益
                        log_returns = np.diff(np.log(closes))
                        returns_matrix.append(log_returns)
                        valid_symbols.append(symbol)

            if len(returns_matrix) < 2:
                logger.warning("不足够的数据计算协方差矩阵")
                return

            returns_matrix = np.array(returns_matrix)

            # 选择风险模型
            risk_model_name = self.optimization_config.get('risk_model', RiskModel.LEDOIT_WOLF)
            risk_model_func = self.risk_models.get(risk_model_name, risk_models.ledoit_wolf)

            # 计算协方差矩阵
            if risk_model_name in [RiskModel.SAMPLE_COVARIANCE, RiskModel.LEDOIT_WOLF,
                                   RiskModel.ORACLE_APPROXIMATING, RiskModel.CONSTANT_CORRELATION]:
                cov_matrix = risk_model_func(pd.DataFrame(returns_matrix.T, columns=valid_symbols))
            elif risk_model_name == RiskModel.EXPONENTIALLY_WEIGHTED:
                span = self.optimization_config.get('ewm_span', 60)
                cov_matrix = risk_model_func(pd.DataFrame(returns_matrix.T, columns=valid_symbols), span=span)
            else:
                # 使用默认的Ledoit-Wolf模型
                cov_matrix = risk_models.ledoit_wolf(pd.DataFrame(returns_matrix.T, columns=valid_symbols))

            self.covariance_matrix = cov_matrix
            self.correlation_matrix = self._calculate_correlation_matrix(cov_matrix)

        except Exception as e:
            logger.error(f"协方差矩阵计算失败: {e}")
            self.covariance_matrix = None
            self.correlation_matrix = None

    def _calculate_correlation_matrix(self, cov_matrix: pd.DataFrame) -> pd.DataFrame:
        """从协方差矩阵计算相关矩阵"""
        try:
            std_dev = np.sqrt(np.diag(cov_matrix))
            correlation = cov_matrix / np.outer(std_dev, std_dev)
            return pd.DataFrame(correlation, index=cov_matrix.index, columns=cov_matrix.columns)
        except Exception as e:
            logger.error(f"相关矩阵计算失败: {e}")
            return None

    def _build_initial_portfolio(self, current_positions: Dict,
                                 market_data: Dict) -> PortfolioState:
        """构建初始组合状态"""
        try:
            total_value = sum(
                current_positions[symbol] * market_data['prices'][symbol]['close'][-1]
                for symbol in current_positions if symbol in market_data['prices']
            )

            # 添加现金余额
            cash_balance = current_positions.get('CASH', 0)
            total_value += cash_balance

            # 构建资产配置
            allocations = {}
            for symbol, shares in current_positions.items():
                if symbol == 'CASH':
                    continue

                if symbol in market_data['prices']:
                    current_price = market_data['prices'][symbol]['close'][-1]
                    current_value = shares * current_price
                    weight = current_value / total_value if total_value > 0 else 0

                    allocation = AssetAllocation(
                        symbol=symbol,
                        weight=weight,
                        target_weight=weight,
                        current_value=current_value,
                        target_value=current_value,
                        notional=current_value,
                        sector=self._get_asset_sector(symbol),
                        asset_class=self._get_asset_class(symbol),
                        region=self._get_asset_region(symbol),
                        currency=self._get_asset_currency(symbol),
                        liquidity_tier=self._get_liquidity_tier(symbol, market_data),
                        risk_contribution=0.0,
                        marginal_risk=0.0,
                        expected_return=self._get_expected_return(symbol),
                        expected_risk=self._get_expected_risk(symbol),
                        transaction_cost=self._calculate_transaction_cost(symbol, shares),
                        tax_implication=self._calculate_tax_implication(symbol, shares, current_price),
                        constraints=self._get_asset_constraints(symbol)
                    )

                    allocations[symbol] = allocation

            # 创建组合状态
            portfolio = PortfolioState(
                portfolio_id=f"portfolio_{int(time.time())}",
                total_value=total_value,
                cash_balance=cash_balance,
                leveraged_value=total_value,
                allocations=allocations,
                metadata=PortfolioMetadata(),
                performance=self._calculate_initial_performance(allocations),
                risk_metrics=self._calculate_initial_risk_metrics(allocations),
                constraints=PortfolioConstraints(),
                benchmark=self.portfolio_config.get('benchmark', 'SPY')
            )

            return portfolio

        except Exception as e:
            logger.error(f"初始组合构建失败: {e}")
            raise

    def _optimize_portfolio(self, portfolio: PortfolioState, signals: Dict,
                            market_data: Dict) -> PortfolioState:
        """优化投资组合"""
        start_time = time.time()

        try:
            # 获取优化配置
            optimization_method = AllocationMethod(
                self.optimization_config.get('method', 'max_sharpe')
            )
            objective = PortfolioObjective(
                self.optimization_config.get('objective', 'maximize_sharpe')
            )

            # 选择优化方法
            if optimization_method in self.optimizers:
                optimizer_func = self.optimizers[optimization_method]
                optimized_weights = optimizer_func(portfolio, signals, market_data, objective)
            else:
                logger.warning(f"未知的优化方法: {optimization_method}, 使用默认的最大夏普比率优化")
                optimized_weights = self._max_sharpe_optimization(portfolio, signals, market_data, objective)

            # 应用优化权重
            optimized_portfolio = self._apply_optimized_weights(portfolio, optimized_weights, market_data)

            # 计算优化后的风险指标
            optimized_portfolio.risk_metrics = self._calculate_portfolio_risk_metrics(
                optimized_portfolio, market_data
            )

            # 更新元数据
            optimized_portfolio.metadata.optimization_method = optimization_method
            optimized_portfolio.metadata.objective = objective
            optimized_portfolio.metadata.last_rebalanced = datetime.now().isoformat()
            optimized_portfolio.metadata.expected_return = optimized_portfolio.risk_metrics.get('expected_return', 0.0)
            optimized_portfolio.metadata.expected_risk = optimized_portfolio.risk_metrics.get('volatility', 0.0)
            optimized_portfolio.metadata.sharpe_ratio = optimized_portfolio.risk_metrics.get('sharpe_ratio', 0.0)

            optimization_time = time.time() - start_time
            self.performance_stats['optimizations_performed'] += 1
            self.performance_stats['avg_optimization_time'] = (
                                                                      self.performance_stats[
                                                                          'avg_optimization_time'] * (
                                                                                  self.performance_stats[
                                                                                      'optimizations_performed'] - 1) +
                                                                      optimization_time
                                                              ) / self.performance_stats['optimizations_performed']

            logger.info(f"组合优化完成: 方法={optimization_method.value}, 耗时={optimization_time:.3f}s")

            return optimized_portfolio

        except Exception as e:
            logger.error(f"组合优化失败: {e}")
            # 返回原始组合作为备选
            return portfolio

    def _equal_weight_optimization(self, portfolio: PortfolioState, signals: Dict,
                                   market_data: Dict, objective: PortfolioObjective) -> Dict[str, float]:
        """等权重优化"""
        try:
            symbols = list(portfolio.allocations.keys())
            if not symbols:
                return {}

            # 等权重分配
            n_assets = len(symbols)
            equal_weight = 1.0 / n_assets

            optimized_weights = {symbol: equal_weight for symbol in symbols}

            # 考虑现金余额
            cash_weight = portfolio.cash_balance / portfolio.total_value if portfolio.total_value > 0 else 0
            if cash_weight > 0:
                # 调整权重，考虑现金
                adjustment_factor = 1 - cash_weight
                optimized_weights = {symbol: weight * adjustment_factor for symbol, weight in optimized_weights.items()}
                optimized_weights['CASH'] = cash_weight

            return optimized_weights

        except Exception as e:
            logger.error(f"等权重优化失败: {e}")
            return {}

