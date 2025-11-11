"""
组合分析器 - 业务层
从 core_bak/portfolio_manager.py 拆分
职责: 组合绩效分析、归因分析
"""

from typing import Dict
import logging

logger = logging.getLogger("DeepSeekQuant.PortfolioAnalytics")


class PortfolioAnalytics:
    """组合分析器"""

    def __init__(self, config: Dict):
        self.config = config


            if optimization_type == 'signal_based':
                return self._signal_based_optimization(portfolio, signals, market_data)
            elif optimization_type == 'risk_budget':
                return self._risk_budget_optimization(portfolio, signals, market_data)
            elif optimization_type == 'factor_based':
                return self._factor_based_optimization(portfolio, signals, market_data)
            else:
                logger.warning(f"未知的自定义优化类型: {optimization_type}, 使用风险平价优化")
                return self._risk_parity_optimization(portfolio, signals, market_data, objective)

        except Exception as e:
            logger.error(f"自定义优化失败: {e}")
            return self._critical_line_algorithm(portfolio, signals, market_data, objective)

    def _signal_based_optimization(self, portfolio: PortfolioState, signals: Dict,
                                   market_data: Dict) -> Dict[str, float]:
        """基于信号的优化"""
        try:
            symbols = list(portfolio.allocations.keys())
            if not signals or not symbols:
                return self._equal_weight_optimization(portfolio, signals, market_data,
                                                       PortfolioObjective.MAXIMIZE_SHARPE)

            # 基于信号强度计算权重
            signal_weights = {}
            total_signal_strength = 0

            for symbol in symbols:
                symbol_signals = signals.get(symbol, [])
                if symbol_signals:
                    # 计算信号综合强度
                    signal_strength = sum(
                        signal.weight * signal.metadata.confidence
                        for signal in symbol_signals
                    )
                    signal_weights[symbol] = signal_strength
                    total_signal_strength += signal_strength
                else:
                    signal_weights[symbol] = 0

            if total_signal_strength == 0:
                return self._equal_weight_optimization(portfolio, signals, market_data,
                                                       PortfolioObjective.MAXIMIZE_SHARPE)

            # 归一化权重
            optimized_weights = {}
            for symbol, strength in signal_weights.items():
                optimized_weights[symbol] = strength / total_signal_strength

            # 考虑现金余额
            cash_weight = portfolio.cash_balance / portfolio.total_value if portfolio.total_value > 0 else 0
            if cash_weight > 0:
                adjustment_factor = 1 - cash_weight
                optimized_weights = {symbol: weight * adjustment_factor for symbol, weight in optimized_weights.items()}
                optimized_weights['CASH'] = cash_weight

            return optimized_weights

        except Exception as e:
            logger.error(f"信号优化失败: {e}")
            return self._equal_weight_optimization(portfolio, signals, market_data,
                                                   PortfolioObjective.MAXIMIZE_SHARPE)

    def _apply_optimized_weights(self, portfolio: PortfolioState, optimized_weights: Dict[str, float],
                                 market_data: Dict) -> PortfolioState:
        """应用优化权重"""
        try:
            # 创建新的组合状态
            new_portfolio = copy.deepcopy(portfolio)
            new_portfolio.version += 1
            new_portfolio.timestamp = datetime.now().isoformat()

            # 更新资产配置
            total_value = portfolio.total_value
            new_allocations = {}

            for symbol, target_weight in optimized_weights.items():
                if symbol == 'CASH':
                    continue

                current_allocation = portfolio.allocations.get(symbol)
                if current_allocation:
                    target_value = total_value * target_weight

                    new_allocation = AssetAllocation(
                        symbol=symbol,
                        weight=target_weight,
                        target_weight=target_weight,
                        current_value=current_allocation.current_value,
                        target_value=target_value,
                        notional=target_value,
                        sector=current_allocation.sector,
                        asset_class=current_allocation.asset_class,
                        region=current_allocation.region,
                        currency=current_allocation.currency,
                        liquidity_tier=current_allocation.liquidity_tier,
                        risk_contribution=0.0,  # 将在后续计算
                        marginal_risk=0.0,
                        expected_return=current_allocation.expected_return,
                        expected_risk=current_allocation.expected_risk,
                        transaction_cost=self._calculate_transaction_cost(symbol, target_value),
                        tax_implication=current_allocation.tax_implication,
                        constraints=current_allocation.constraints,
                        metadata=current_allocation.metadata
                    )

                    new_allocations[symbol] = new_allocation

            # 处理现金
            cash_weight = optimized_weights.get('CASH', 0)
            new_portfolio.cash_balance = total_value * cash_weight
            new_portfolio.allocations = new_allocations

            return new_portfolio

        except Exception as e:
            logger.error(f"优化权重应用失败: {e}")
            return portfolio

    def _apply_constraints(self, portfolio: PortfolioState, market_data: Dict) -> PortfolioState:
        """应用约束条件"""
        try:
            constraints = self.portfolio_config.get('constraints', {})
            if not constraints:
                return portfolio

            # 创建约束后的组合副本
            constrained_portfolio = copy.deepcopy(portfolio)

            # 应用各种约束
            constrained_portfolio = self._apply_weight_constraints(constrained_portfolio, constraints)
            constrained_portfolio = self._apply_sector_constraints(constrained_portfolio, constraints, market_data)
            constrained_portfolio = self._apply_liquidity_constraints(constrained_portfolio, constraints, market_data)
            constrained_portfolio = self._apply_leverage_constraints(constrained_portfolio, constraints)
            constrained_portfolio = self._apply_regulatory_constraints(constrained_portfolio, constraints)

            # 重新归一化权重
            constrained_portfolio = self._normalize_weights(constrained_portfolio)

            return constrained_portfolio

        except Exception as e:
            logger.error(f"约束应用失败: {e}")
            return portfolio

    def _apply_weight_constraints(self, portfolio: PortfolioState, constraints: Dict) -> PortfolioState:
        """应用权重约束"""
        try:
            max_weight = constraints.get('max_asset_weight', 0.2)
            min_weight = constraints.get('min_asset_weight', 0.0)

            for symbol, allocation in portfolio.allocations.items():
                current_weight = allocation.weight

                # 应用上下限约束
                if current_weight > max_weight:
                    allocation.weight = max_weight
                    allocation.target_weight = max_weight
                elif current_weight < min_weight and current_weight > 0:  # 只调整正权重
                    allocation.weight = min_weight
                    allocation.target_weight = min_weight

            return portfolio

        except Exception as e:
            logger.error(f"权重约束应用失败: {e}")
            return portfolio

    def _apply_sector_constraints(self, portfolio: PortfolioState, constraints: Dict,
                                  market_data: Dict) -> PortfolioState:
        """应用行业约束"""
        try:
            max_sector_weight = constraints.get('max_sector_weight', 0.3)

            # 计算各行业当前权重
            sector_weights = {}
            for allocation in portfolio.allocations.values():
                sector = allocation.sector
                if sector not in sector_weights:
                    sector_weights[sector] = 0
                sector_weights[sector] += allocation.weight

            # 检查并调整超限行业
            for sector, weight in sector_weights.items():
                if weight > max_sector_weight:
                    # 计算需要减少的权重
                    excess_weight = weight - max_sector_weight
                    reduction_factor = max_sector_weight / weight

                    # 按比例减少该行业所有资产的权重
                    for allocation in portfolio.allocations.values():
                        if allocation.sector == sector:
                            allocation.weight *= reduction_factor
                            allocation.target_weight *= reduction_factor

            return portfolio

        except Exception as e:
            logger.error(f"行业约束应用失败: {e}")
            return portfolio

    def _assess_portfolio_risk(self, portfolio: PortfolioState, market_data: Dict) -> Dict[str, Any]:
        """评估组合风险"""
        try:
            risk_assessment = {
                'approved': True,
                'reason': '',
                'risk_score': 0.0,
                'violations': [],
                'stress_test_results': {},
                'scenario_analysis': {}
