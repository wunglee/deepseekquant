from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from infrastructure.interfaces import InfrastructureProvider

@dataclass
class BacktestMetrics:
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0

@dataclass
class BacktestConfig:
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    commission: float = 0.001
    slippage: float = 0.0005

class BacktestEngine:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = InfrastructureProvider.get('logging').get_logger('DeepSeekQuant.BacktestEngine')
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []
        self.current_capital = config.initial_capital

    def run(self, strategy_fn, market_data: List[Dict[str, Any]]) -> BacktestMetrics:
        """
        执行回测
        """
        self.logger.info(f"回测开始: {self.config.start_date} -> {self.config.end_date}")
        
        for data_point in market_data:
            signal = strategy_fn(data_point)
            if signal:
                self._execute_trade(signal, data_point)
            self.equity_curve.append(self.current_capital)

        metrics = self._calculate_metrics()
        self.logger.info(f"回测完成: 总收益率={metrics.total_return:.2%}, 总交易={metrics.total_trades}")
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
        """计算回测指标"""
        if not self.equity_curve:
            return BacktestMetrics()

        total_return = (self.current_capital - self.config.initial_capital) / self.config.initial_capital
        
        # 简化计算
        returns = []
        for i in range(1, len(self.equity_curve)):
            ret = (self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1] if self.equity_curve[i-1] > 0 else 0
            returns.append(ret)
        
        sharpe = 0.0
        if returns:
            import math
            avg_ret = sum(returns) / len(returns)
            std_ret = math.sqrt(sum((r - avg_ret)**2 for r in returns) / len(returns)) if len(returns) > 1 else 0
            sharpe = (avg_ret / std_ret * math.sqrt(252)) if std_ret > 0 else 0.0

        max_dd = 0.0
        peak = self.equity_curve[0]
        for val in self.equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        win_trades = sum(1 for t in self.trades if t.get('capital', 0) > self.config.initial_capital)
        win_rate = win_trades / len(self.trades) if self.trades else 0.0

        return BacktestMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            total_trades=len(self.trades),
            profit_factor=1.0,  # 简化
            avg_trade_return=total_return / len(self.trades) if self.trades else 0.0
        )
