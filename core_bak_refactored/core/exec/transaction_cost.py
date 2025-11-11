"""
交易成本模型 - 业务层
从 core_bak/execution_engine.py 拆分
职责: 计算交易成本、滑点、佣金
"""

from typing import Dict
import logging

logger = logging.getLogger('DeepSeekQuant.TransactionCost')


class TransactionCostModel:
    """交易成本模型"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        """计算佣金"""
        try:
            broker_name = order.execution_params.algo_parameters.get('broker', self.default_broker)
            connection = self.broker_connections.get(broker_name)

            if not connection:
                return 0.0

            notional = price * quantity

            if connection.broker_type == BrokerType.IBKR:
                # IBKR佣金结构：每股0.0035美元，最低1美元
                commission_per_share = 0.0035
                commission = max(1.0, quantity * commission_per_share)
                return min(commission, notional * 0.01)  # 不超过名义价值的1%

            elif connection.broker_type == BrokerType.ALPACA:
                # Alpaca免佣金，只有极低费用
                return 0.0

            elif connection.broker_type == BrokerType.BINANCE:
                # Binance费用
                maker_fee = 0.001  # 0.1%
                taker_fee = 0.002  # 0.2%
                fee_rate = maker_fee if random.random() > 0.7 else taker_fee
                return notional * fee_rate

            else:
                # 默认佣金：0.1%
                return notional * 0.001

        except Exception as e:
            logger.error(f"佣金计算失败: {e}")
            return notional * 0.001  # 默认佣金


        """计算额外费用"""
        try:
            notional = price * quantity

            # 监管费用（SEC费用）
            sec_fee = notional * 0.0000229  # 0.00229%

            # 交易活动费（Trading Activity Fee）
            taf_fee = 0.000119 * quantity  # 每股0.000119美元
            taf_fee = min(taf_fee, 5.95)  # 最高5.95美元

            # 清算费用
            clearing_fee = notional * 0.00002  # 0.002%

            # 交易所费用（根据交易所不同）
            exchange_fee = self._calculate_exchange_fee(order.symbol, notional)

            # 路由费用（如果指定了特定路由）
            routing_fee = self._calculate_routing_fee(order)

            # 总费用
            total_fees = sec_fee + taf_fee + clearing_fee + exchange_fee + routing_fee

            # 确保费用合理（不超过名义价值的0.1%）
            max_fees = notional * 0.001
            return min(total_fees, max_fees)

        except Exception as e:
            logger.error(f"费用计算失败: {e}")
            return notional * 0.0001  # 默认费用


        """计算交易所费用"""
        try:
            # 根据交易所类型计算不同费用
            # 简化实现：假设所有股票在主要交易所交易
            if symbol.endswith('.NYSE'):
                return notional * 0.000015  # NYSE: 0.0015%
            elif symbol.endswith('.NASDAQ'):
                return notional * 0.000013  # NASDAQ: 0.0013%
            else:
                return notional * 0.000012  # 默认: 0.0012%
        except:
            return notional * 0.000012


        """计算路由费用"""
        try:
            # 如果指定了特定路由，可能有额外费用
            routing = order.parameters.routing_instructions.get('venue')
            if routing and routing != 'auto':
                return order.quantity * 0.0005  # 每股0.0005美元
            return 0.0
        except:
            return 0.0


        """计算滑点"""
        try:
            symbol = order.symbol
            side = order.side
            quantity = order.quantity

            # 获取基准价格（通常使用下单时的市场价格）
            benchmark_price = market_data['prices'][symbol]['close'][0]

            # 计算原始滑点
            raw_slippage = execution_price - benchmark_price
            if side == OrderSide.SELL:
                raw_slippage = -raw_slippage  # 对于卖出，价格下跌是正滑点

            # 根据市场条件调整滑点
            volatility = self._calculate_current_volatility(symbol, market_data)
            liquidity = self._assess_liquidity_conditions(symbol, market_data)

            # 滑点调整因子（高波动性和低流动性增加滑点）
            volatility_factor = 1 + (volatility / 0.2)  # 基准波动率20%
            liquidity_factor = 1 + (1 - liquidity) * 2  # 流动性越低，因子越大

            # 订单大小影响（大订单产生更大滑点）
            size_factor = min(1 + (quantity / 10000), 3)  # 每1万股增加1倍，最大3倍

            # 算法影响（某些算法可以减少滑点）
            algorithm_factor = self._get_algorithm_slippage_factor(order.execution_params.algorithm)

            adjusted_slippage = raw_slippage * volatility_factor * liquidity_factor * size_factor * algorithm_factor

            return float(adjusted_slippage)

        except Exception as e:
            logger.error(f"滑点计算失败: {e}")
            return 0.0


