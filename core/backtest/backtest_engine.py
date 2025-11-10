from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import math

from infrastructure.interfaces import InfrastructureProvider

@dataclass
class BacktestMetrics:
    """回测性能指标"""
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    # 组合特定指标
    turnover: float = 0.0
    total_costs: float = 0.0
    calmar_ratio: float = 0.0
    recovery_factor: float = 0.0

@dataclass
class BacktestConfig:
    """回测配置"""
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    commission: float = 0.001  # 手续费率
    slippage: float = 0.0005  # 滑点
    rebalance_frequency: str = "monthly"  # daily, weekly, monthly, quarterly
    transaction_cost: float = 0.001  # 交易成本率
    benchmark: Optional[str] = None  # 基准标的

class BacktestEngine:
    """
    回测引擎
    从 core_bak/portfolio_manager.py:backtest_portfolio_strategy 提取 (line 2830-2918)
    """
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = InfrastructureProvider.get('logging').get_logger('DeepSeekQuant.BacktestEngine')
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []
        self.current_capital = config.initial_capital
        self.weights_history: List[Dict[str, float]] = []
        self.rebalance_dates: List[str] = []

    def run(self, strategy_fn: Callable, market_data: List[Dict[str, Any]]) -> BacktestMetrics:
        """
        执行回测
        从 core_bak/portfolio_manager.py:backtest_portfolio_strategy 提取核心逻辑
        
        Args:
            strategy_fn: 策略函数，接收市场数据，返回交易信号
            market_data: 按时间排序的市场数据列表
        
        Returns:
            BacktestMetrics: 回测指标
        """
        self.logger.info(f"回测开始: {self.config.start_date} -> {self.config.end_date}")
        self.logger.info(f"初始资金: ${self.config.initial_capital:,.2f}")
        
        # 初始化权益曲线
        self.equity_curve = [self.config.initial_capital]
        
        for i, data_point in enumerate(market_data):
            # 生成交易信号
            signal = strategy_fn(data_point)
            
            if signal:
                self._execute_trade(signal, data_point)
            
            # 记录当前权益
            self.equity_curve.append(self.current_capital)
        
        # 计算回测指标
        metrics = self._calculate_metrics()
        
        self.logger.info(f"回测完成: 总收益率={metrics.total_return:.2%}, 总交易={metrics.total_trades}")
        self.logger.info(f"夏普比率={metrics.sharpe_ratio:.3f}, 最大回撤={metrics.max_drawdown:.2%}")
        
        return metrics

    def _execute_trade(self, signal: Dict[str, Any], data_point: Dict[str, Any]):
        """模拟执行交易"""
        price = data_point.get('price', 0.0)
        quantity = signal.get('quantity', 1)
        side = signal.get('side', 'buy')
        
        notional = price * quantity
        commission_cost = notional * self.config.commission
        slippage_cost = notional * self.config.slippage
        total_costs = commission_cost + slippage_cost
        if side == 'buy':
            self.current_capital -= (notional + total_costs)
        else:
            self.current_capital += (notional - total_costs)

        self.trades.append({
            'timestamp': data_point.get('timestamp', ''),
            'side': side,
            'price': price,
            'quantity': quantity,
            'commission': self.config.commission,
            'slippage': self.config.slippage,
            'costs': round(total_costs, 6),
            'capital': self.current_capital
        })

    def _calculate_metrics(self) -> BacktestMetrics:
        """计算回测指标（单策略模式）"""
        if not self.equity_curve or len(self.equity_curve) < 2:
            return BacktestMetrics()

        total_return = (self.current_capital - self.config.initial_capital) / self.config.initial_capital
        
        # 计算收益序列
        returns = []
        for i in range(1, len(self.equity_curve)):
            ret = (self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1] if self.equity_curve[i-1] > 0 else 0
            returns.append(ret)
        
        # 夏普比率
        sharpe = 0.0
        if returns and len(returns) > 1:
            avg_ret = sum(returns) / len(returns)
            variance = sum((r - avg_ret)**2 for r in returns) / len(returns)
            std_ret = math.sqrt(variance)
            sharpe = (avg_ret / std_ret * math.sqrt(252)) if std_ret > 0 else 0.0

        # 最大回撤
        max_dd = self._calculate_max_drawdown_from_values(self.equity_curve)

        # 胜率
        profitable_trades = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
        win_rate = profitable_trades / len(self.trades) if self.trades else 0.0

        return BacktestMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            total_trades=len(self.trades),
            profit_factor=1.0,  # 简化
            avg_trade_return=total_return / len(self.trades) if self.trades else 0.0
        )

    def run_portfolio_backtest(self, 
                               weights_fn: Callable,
                               historical_data: Dict[str, Dict[str, Any]],
                               initial_weights: Optional[Dict[str, float]] = None) -> BacktestMetrics:
        """
        执行组合回测（完整迁移自core_bak）
        从 core_bak/portfolio_manager.py:backtest_portfolio_strategy 提取 (line 2830-2918)
        
        Args:
            weights_fn: 权重生成函数，接收(当前权重, 市场数据)，返回目标权重
            historical_data: {日期: {prices: {...}, volumes: {...}}} 格式的历史数据
            initial_weights: 初始权重字典，None则等权重
        
        Returns:
            BacktestMetrics: 回测指标
        """
        self.logger.info(f"组合回测开始: {self.config.start_date} -> {self.config.end_date}")
        
        # 准备回测数据
        dates = sorted(historical_data.keys())
        portfolio_values = [self.config.initial_capital]
        self.weights_history = []
        self.trades = []
        
        # 初始化权重
        if initial_weights:
            current_weights = initial_weights
        else:
            # 等权重初始化
            if dates and 'prices' in historical_data[dates[0]]:
                symbols = list(historical_data[dates[0]]['prices'].keys())
                current_weights = {symbol: 1.0 / len(symbols) for symbol in symbols}
            else:
                raise ValueError("无法从历史数据中提取资产列表")
        
        # 执行回测（从core_bak line 2851-2892提取）
        for i in range(1, len(dates)):
            current_date = dates[i]
            prev_date = dates[i - 1]
            
            # 计算期间收益
            period_returns = self._calculate_period_returns(
                historical_data[prev_date].get('prices', {}),
                historical_data[current_date].get('prices', {})
            )
            
            # 更新组合价值
            portfolio_return = sum(
                current_weights.get(symbol, 0) * period_returns.get(symbol, 0)
                for symbol in current_weights.keys()
            )
            new_value = portfolio_values[-1] * (1 + portfolio_return)
            portfolio_values.append(new_value)
            
            # 检查再平衡条件
            if self._should_rebalance(current_date, i):
                # 生成新权重
                target_weights = weights_fn(current_weights, historical_data[current_date])
                
                # 计算换手成本和交易
                turnover, trades, costs = self._calculate_rebalance_costs(
                    current_weights, target_weights, new_value
                )
                
                # 应用成本和更新权重
                new_value -= costs
                portfolio_values[-1] = new_value
                current_weights = target_weights
                
                # 记录交易和再平衡
                self.trades.append({
                    'date': current_date,
                    'trades': trades,
                    'costs': costs,
                    'turnover': turnover
                })
                self.rebalance_dates.append(current_date)
            
            self.weights_history.append(current_weights.copy())
        
        # 保存权益曲线
        self.equity_curve = portfolio_values
        
        # 计算回测指标
        metrics = self._calculate_portfolio_metrics(portfolio_values, dates)
        
        self.logger.info(f"组合回测完成: 年化收益={metrics.annualized_return:.2%}, 夏普={metrics.sharpe_ratio:.3f}")
        self.logger.info(f"最大回撤={metrics.max_drawdown:.2%}, 换手率={metrics.turnover:.2%}")
        
        return metrics
    
    def _calculate_period_returns(self, 
                                  prev_prices: Dict[str, float],
                                  curr_prices: Dict[str, float]) -> Dict[str, float]:
        """
        计算期间收益率
        从 core_bak/portfolio_manager.py 提取 (line 2856-2859)
        """
        returns = {}
        for symbol in prev_prices.keys():
            if symbol in curr_prices and prev_prices[symbol] > 0:
                returns[symbol] = (curr_prices[symbol] - prev_prices[symbol]) / prev_prices[symbol]
            else:
                returns[symbol] = 0.0
        return returns
    
    def _should_rebalance(self, current_date: str, date_index: int) -> bool:
        """
        检查是否应该再平衡
        从 core_bak/portfolio_manager.py 提取 (line 2868)
        """
        frequency = self.config.rebalance_frequency.lower()
        
        if frequency == "daily":
            return True
        elif frequency == "weekly":
            return date_index % 5 == 0  # 简化：每5个交易日
        elif frequency == "monthly":
            return date_index % 21 == 0  # 简化：每21个交易日
        elif frequency == "quarterly":
            return date_index % 63 == 0  # 简化：每63个交易日
        else:
            return False
    
    def _calculate_rebalance_costs(self,
                                   current_weights: Dict[str, float],
                                   target_weights: Dict[str, float],
                                   portfolio_value: float) -> tuple:
        """
        计算再平衡成本和交易明细
        从 core_bak/portfolio_manager.py 提取 (line 2875-2877)
        
        Returns:
            (换手率, 交易列表, 总成本)
        """
        trades = []
        total_turnover = 0.0
        
        # 合并所有资产
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        
        for symbol in all_symbols:
            current_w = current_weights.get(symbol, 0.0)
            target_w = target_weights.get(symbol, 0.0)
            
            weight_change = abs(target_w - current_w)
            if weight_change > 1e-6:  # 忽略极小变化
                notional = weight_change * portfolio_value
                total_turnover += weight_change
                
                trades.append({
                    'symbol': symbol,
                    'from_weight': current_w,
                    'to_weight': target_w,
                    'notional': notional
                })
        
        # 计算交易成本
        total_costs = total_turnover * portfolio_value * self.config.transaction_cost
        
        return total_turnover, trades, total_costs
    
    def _calculate_portfolio_metrics(self, 
                                     portfolio_values: List[float],
                                     dates: List[str]) -> BacktestMetrics:
        """
        计算组合回测指标
        从 core_bak/portfolio_manager.py 提取 (line 2895-2912)
        """
        if len(portfolio_values) < 2:
            return BacktestMetrics()
        
        initial_capital = portfolio_values[0]
        final_value = portfolio_values[-1]
        
        # 计算收益序列
        returns = [
            (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            if portfolio_values[i-1] > 0 else 0.0
            for i in range(1, len(portfolio_values))
        ]
        
        # 总收益率
        total_return = (final_value - initial_capital) / initial_capital
        
        # 年化收益率
        trading_days = len(dates)
        years = trading_days / 252.0
        annualized_return = (final_value / initial_capital) ** (1 / years) - 1 if years > 0 else 0.0
        
        # 波动率（年化）
        if len(returns) > 1:
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            volatility = math.sqrt(variance) * math.sqrt(252)
        else:
            volatility = 0.0
        
        # 夏普比率
        if volatility > 0:
            sharpe_ratio = (sum(returns) / len(returns) * 252) / volatility
        else:
            sharpe_ratio = 0.0
        
        # Sortino比率
        sortino_ratio = self._calculate_sortino_ratio(returns)
        
        # 最大回撤
        max_drawdown = self._calculate_max_drawdown_from_values(portfolio_values)
        
        # Calmar比率
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        # 换手率和成本
        total_turnover = sum(trade['turnover'] for trade in self.trades) if self.trades else 0.0
        avg_turnover = total_turnover / len(self.trades) if self.trades else 0.0
        total_costs = sum(trade['costs'] for trade in self.trades) if self.trades else 0.0
        
        return BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            win_rate=0.0,  # 组合回测中不适用
            total_trades=len(self.trades),
            profit_factor=1.0 + total_return,
            avg_trade_return=0.0,
            turnover=avg_turnover,
            total_costs=total_costs,
            calmar_ratio=calmar_ratio,
            recovery_factor=total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        )
    
    def _calculate_sortino_ratio(self, returns: List[float], target_return: float = 0.0) -> float:
        """
        计算Sortino比率
        从 core_bak/portfolio_manager.py 提取 (line 2905)
        """
        if not returns:
            return 0.0
        
        # 计算下行偏差
        downside_returns = [r - target_return for r in returns if r < target_return]
        
        if not downside_returns:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        downside_variance = sum(r ** 2 for r in downside_returns) / len(downside_returns)
        downside_std = math.sqrt(downside_variance)
        
        if downside_std == 0:
            return 0.0
        
        # 年化Sortino比率
        sortino = (mean_return * 252) / (downside_std * math.sqrt(252))
        return sortino
    
    def _calculate_max_drawdown_from_values(self, values: List[float]) -> float:
        """
        计算最大回撤（从权益曲线）
        从 core_bak/portfolio_manager.py 提取 (line 2902)
        """
        if not values or len(values) < 2:
            return 0.0
        
        max_dd = 0.0
        peak = values[0]
        
        for value in values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, drawdown)
        
        return max_dd
