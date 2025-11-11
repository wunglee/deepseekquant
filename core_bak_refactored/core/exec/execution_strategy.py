"""
执行策略 - 业务层
从 core_bak/execution_engine.py 拆分
职责: 订单执行策略、智能路由
"""

from typing import Dict, List
import logging

from .exec_models import Order, ExecutionAlgorithm, ExecutionParameters
from ...infrastructure.execution_algos import ExecutionAlgorithms

logger = logging.getLogger('DeepSeekQuant.ExecutionStrategy')


class ExecutionStrategy:
    """执行策略管理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.algos = ExecutionAlgorithms()
        
        """初始化执行算法"""
        try:
            algorithms = {
                'simple': self._execute_simple,
                'twap': self._execute_twap,
                'vwap': self._execute_vwap,
                'pov': self._execute_pov,
                'implementation_shortfall': self._execute_implementation_shortfall,
                'liquidity_seeking': self._execute_liquidity_seeking,
                'adaptive': self._execute_adaptive,
                'sniper': self._execute_sniper,
                'stealth': self._execute_stealth,
                'aggressive': self._execute_aggressive,
                'passive': self._execute_passive
            }

            # 加载配置参数
            for algo_name, algo_config in self.algo_config.items():
                if algo_name in algorithms:
                    # 可以在这里配置算法参数
                    logger.info(f"执行算法加载: {algo_name}")

            self.execution_algorithms = algorithms
            logger.info(f"已加载 {len(algorithms)} 个执行算法")

        except Exception as e:
            logger.error(f"执行算法初始化失败: {e}")


        """验证算法参数"""
        try:
            algorithm = order.execution_params.algorithm
            params = order.execution_params.algo_parameters

            # 通用参数验证
            if 'urgency' in params:
                urgency = params['urgency']
                if urgency not in ['low', 'medium', 'high', 'urgent']:
                    return False

            if 'max_participation_rate' in params:
                rate = params['max_participation_rate']
                if not 0 < rate <= 1.0:
                    return False

            if 'price_deviation_limit' in params:
                deviation = params['price_deviation_limit']
                if not 0 < deviation <= 0.1:  # 最大10%偏离
                    return False

            # 算法特定验证
            if algorithm == ExecutionAlgorithm.TWAP:
                if 'time_horizon' not in params or params['time_horizon'] <= 0:
                    return False

            elif algorithm == ExecutionAlgorithm.VWAP:
                if 'volume_participation' not in params or not 0 < params['volume_participation'] <= 1.0:
                    return False

            elif algorithm == ExecutionAlgorithm.POV:
                if 'participation_rate' not in params or not 0 < params['participation_rate'] <= 0.5:
                    return False

            return True

        except Exception as e:
            logger.error(f"算法参数验证失败 {order.order_id}: {e}")
            return False


        """简单执行算法"""
        try:
            symbol = order.symbol
            quantity = order.quantity
            side = order.side

            # 获取当前市场价格
            current_price = market_data['prices'][symbol]['close'][-1]

            # 计算执行价格（考虑买卖价差）
            spread = self._calculate_bid_ask_spread(symbol, market_data)
            if side == OrderSide.BUY:
                execution_price = current_price * (1 + spread / 2)  # 买入价（卖一价）
            else:
                execution_price = current_price * (1 - spread / 2)  # 卖出价（买一价）

            # 对于限价单，使用限价价格
            if order.parameters.order_type == OrderType.LIMIT:
                execution_price = order.parameters.limit_price
                # 检查限价是否可执行
                if (side == OrderSide.BUY and execution_price < current_price) or \
                        (side == OrderSide.SELL and execution_price > current_price):
                    execution_price = current_price  # 使用市价执行

            # 计算交易成本
            commission = self._calculate_commission(order, execution_price)
            fees = self._calculate_fees(order, execution_price)

            # 计算滑点
            slippage = self._calculate_slippage(order, market_data, execution_price)
            final_price = execution_price + slippage

            # 计算市场影响
            market_impact = self._calculate_market_impact(order, market_data, final_price)

            # 计算执行质量
            execution_quality = self._calculate_execution_quality(order, current_price, final_price, market_impact)

            # 模拟经纪商执行
            broker_result = self._execute_with_broker(order, final_price, quantity)

            result = {
                'executed_quantity': quantity,
                'average_price': final_price,
                'commission': commission,
                'fees': fees,
                'slippage': slippage,
                'market_impact': market_impact,
                'timing_risk': 0.0,  # 简单执行无时间风险
                'execution_quality': execution_quality,
                'broker_order_id': broker_result.get('order_id'),
                'exchange_order_id': broker_result.get('exchange_id'),
                'execution_reports': broker_result.get('reports', [])
            }

            return result

        except Exception as e:
            logger.error(f"简单执行算法失败 {order.order_id}: {e}")
            return {'error': str(e)}


        """TWAP执行算法"""
        try:
            symbol = order.symbol
            quantity = order.quantity
            time_horizon = order.execution_params.algo_parameters.get('time_horizon', 300)  # 默认5分钟
            slices = order.execution_params.algo_parameters.get('slices', 5)  # 默认分5片

            slice_quantity = quantity / slices
            slice_interval = time_horizon / slices

            total_executed = 0
            total_cost = 0
            execution_reports = []

            for slice_num in range(slices):
                # 等待到下一个执行时间
                if slice_num > 0:
                    time.sleep(slice_interval)

                # 获取当前市场数据
                current_price = market_data['prices'][symbol]['close'][-1]

                # 执行当前切片
                slice_order = copy.deepcopy(order)
                slice_order.quantity = slice_quantity
                slice_order.order_id = f"{order.order_id}_slice_{slice_num}"

                slice_result = self._execute_simple(slice_order, market_data)

                if slice_result.get('error'):
                    logger.warning(f"TWAP切片 {slice_num} 执行失败: {slice_result['error']}")
                    continue

                total_executed += slice_result['executed_quantity']
                total_cost += slice_result['average_price'] * slice_result['executed_quantity']
                execution_reports.append(slice_result)

                logger.debug(
                    f"TWAP切片 {slice_num} 执行完成: {slice_result['executed_quantity']} @ {slice_result['average_price']}")

            # 计算总体结果
            if total_executed > 0:
                average_price = total_cost / total_executed
            else:
                average_price = 0

            # 计算执行质量
            initial_price = market_data['prices'][symbol]['close'][0]
            price_improvement = average_price - initial_price if order.side == OrderSide.BUY else initial_price - average_price

            result = {
                'executed_quantity': total_executed,
                'average_price': average_price,
                'commission': sum(r['commission'] for r in execution_reports),
                'fees': sum(r['fees'] for r in execution_reports),
                'slippage': sum(r['slippage'] for r in execution_reports),
                'market_impact': sum(r['market_impact'] for r in execution_reports) / len(execution_reports),
                'timing_risk': self._calculate_twap_timing_risk(order, execution_reports),
                'execution_quality': price_improvement / initial_price if initial_price > 0 else 0,
                'execution_reports': execution_reports
            }

            return result

        except Exception as e:
            logger.error(f"TWAP执行算法失败 {order.order_id}: {e}")
            return {'error': str(e)}


        """VWAP执行算法"""
        try:
            symbol = order.symbol
            quantity = order.quantity

            # 获取历史成交量数据
            volume_data = market_data['volumes'][symbol].get('volume', [])
            if not volume_data:
                return self._execute_twap(order, market_data)  # 回退到TWAP

            # 计算成交量分布
            total_volume = sum(volume_data)
            if total_volume <= 0:
                return self._execute_twap(order, market_data)

            # 计算每个时间段的成交量权重
            volume_weights = [v / total_volume for v in volume_data]

            # 确定执行时间段
            time_horizon = order.execution_params.algo_parameters.get('time_horizon', len(volume_data))
            slices = min(time_horizon, len(volume_data))

            # 计算每个时间片的执行量
            slice_quantities = []
            for i in range(slices):
                slice_qty = quantity * volume_weights[i]
                slice_quantities.append(slice_qty)

            total_executed = 0
            total_cost = 0
            execution_reports = []

            for slice_num in range(slices):
                # 等待到下一个执行时间（模拟）
                if slice_num > 0:
                    time.sleep(1)  # 简化实现，实际应根据市场时间

                # 获取当前市场数据
                current_price = market_data['prices'][symbol]['close'][slice_num]

                # 执行当前切片
                slice_order = copy.deepcopy(order)
                slice_order.quantity = slice_quantities[slice_num]
                slice_order.order_id = f"{order.order_id}_vwap_slice_{slice_num}"

                slice_result = self._execute_simple(slice_order, market_data)

                if slice_result.get('error'):
                    logger.warning(f"VWAP切片 {slice_num} 执行失败: {slice_result['error']}")
                    continue

                total_executed += slice_result['executed_quantity']
                total_cost += slice_result['average_price'] * slice_result['executed_quantity']
                execution_reports.append(slice_result)

                logger.debug(
                    f"VWAP切片 {slice_num} 执行完成: {slice_result['executed_quantity']} @ {slice_result['average_price']}")

            # 计算总体结果
            if total_executed > 0:
                average_price = total_cost / total_executed
            else:
                average_price = 0

            # 计算VWAP基准
            vwap_benchmark = sum(p * w for p, w in zip(
                market_data['prices'][symbol]['close'][:slices],
                volume_weights[:slices]
            ))

            # 计算执行质量
            price_improvement = average_price - vwap_benchmark if order.side == OrderSide.BUY else vwap_benchmark - average_price

            result = {
                'executed_quantity': total_executed,
                'average_price': average_price,
                'commission': sum(r['commission'] for r in execution_reports),
                'fees': sum(r['fees'] for r in execution_reports),
                'slippage': sum(r['slippage'] for r in execution_reports),
                'market_impact': sum(r['market_impact'] for r in execution_reports) / len(execution_reports),
                'timing_risk': self._calculate_vwap_timing_risk(order, execution_reports, vwap_benchmark),
                'execution_quality': price_improvement / vwap_benchmark if vwap_benchmark > 0 else 0,
                'vwap_benchmark': vwap_benchmark,
                'execution_reports': execution_reports
            }

            return result

        except Exception as e:
            logger.error(f"VWAP执行算法失败 {order.order_id}: {e}")
            return {'error': str(e)}


        """参与率(POV)执行算法"""
        try:
            symbol = order.symbol
            quantity = order.quantity

            # 获取参与率参数
            participation_rate = order.execution_params.algo_parameters.get('participation_rate', 0.1)
            max_participation = order.execution_params.algo_parameters.get('max_participation_rate', 0.2)
            min_participation = order.execution_params.algo_parameters.get('min_participation_rate', 0.05)

            participation_rate = max(min(participation_rate, max_participation), min_participation)

            # 获取市场成交量数据
            volume_data = market_data['volumes'][symbol].get('volume', [])
            if not volume_data:
                return self._execute_twap(order, market_data)

            total_executed = 0
            total_cost = 0
            execution_reports = []
            remaining_quantity = quantity
            time_slices = min(len(volume_data), 10)  # 限制时间片数量

            for slice_num in range(time_slices):
                if remaining_quantity <= 0:
                    break

                # 计算当前市场成交量
                current_volume = volume_data[slice_num]
                if current_volume <= 0:
                    continue

                # 计算当前切片执行量
                slice_qty = min(remaining_quantity, current_volume * participation_rate)
                if slice_qty <= 0:
                    continue

                # 执行当前切片
                slice_order = copy.deepcopy(order)
                slice_order.quantity = slice_qty
                slice_order.order_id = f"{order.order_id}_pov_slice_{slice_num}"

                slice_result = self._execute_simple(slice_order, market_data)

                if slice_result.get('error'):
                    logger.warning(f"POV切片 {slice_num} 执行失败: {slice_result['error']}")
                    continue

                total_executed += slice_result['executed_quantity']
                total_cost += slice_result['average_price'] * slice_result['executed_quantity']
                remaining_quantity -= slice_result['executed_quantity']
                execution_reports.append(slice_result)

                logger.debug(
                    f"POV切片 {slice_num} 执行完成: {slice_result['executed_quantity']} @ {slice_result['average_price']}")

                # 动态调整参与率（基于市场条件和执行进度）
                if slice_num > 0 and len(execution_reports) >= 2:
                    # 计算执行效率
                    exec_efficiency = self._calculate_execution_efficiency(execution_reports[-2:])
                    if exec_efficiency < 0.8:  # 执行效率低
                        participation_rate = max(participation_rate * 0.9, min_participation)
                    elif exec_efficiency > 0.95:  # 执行效率高
                        participation_rate = min(participation_rate * 1.1, max_participation)

            # 计算总体结果
            if total_executed > 0:
                average_price = total_cost / total_executed
            else:
                average_price = 0

            # 计算执行质量
            arrival_price = market_data['prices'][symbol]['close'][0]
            price_improvement = average_price - arrival_price if order.side == OrderSide.BUY else arrival_price - average_price

            result = {
                'executed_quantity': total_executed,
                'average_price': average_price,
                'commission': sum(r['commission'] for r in execution_reports),
                'fees': sum(r['fees'] for r in execution_reports),
                'slippage': sum(r['slippage'] for r in execution_reports),
                'market_impact': sum(r['market_impact'] for r in execution_reports) / len(execution_reports),
                'timing_risk': self._calculate_pov_timing_risk(order, execution_reports, participation_rate),
                'execution_quality': price_improvement / arrival_price if arrival_price > 0 else 0,
                'final_participation_rate': participation_rate,
                'execution_reports': execution_reports
            }

            return result

        except Exception as e:
            logger.error(f"POV执行算法失败 {order.order_id}: {e}")
            return {'error': str(e)}


        """执行差额算法"""
        try:
            symbol = order.symbol
            quantity = order.quantity

            # 获取算法参数
            risk_aversion = order.execution_params.algo_parameters.get('risk_aversion', 0.5)
            urgency = order.execution_params.algo_parameters.get('urgency', 'medium')
            benchmark = order.execution_params.algo_parameters.get('benchmark', 'arrival_price')

            # 计算 urgency 权重
            urgency_weights = {
                'low': 0.3,
                'medium': 0.5,
                'high': 0.7,
                'urgent': 0.9
            }
            urgency_weight = urgency_weights.get(urgency, 0.5)

            # 获取基准价格
            if benchmark == 'arrival_price':
                benchmark_price = market_data['prices'][symbol]['close'][0]
            elif benchmark == 'vwap':
                # 计算VWAP基准
                volume_data = market_data['volumes'][symbol].get('volume', [])
                if volume_data:
                    total_volume = sum(volume_data)
                    vwap = sum(p * v for p, v in zip(
                        market_data['prices'][symbol]['close'][:len(volume_data)],
                        volume_data
                    )) / total_volume
                    benchmark_price = vwap
                else:
                    benchmark_price = market_data['prices'][symbol]['close'][0]
            else:
                benchmark_price = market_data['prices'][symbol]['close'][0]

            # 计算目标执行价格
            if order.side == OrderSide.BUY:
                target_price = benchmark_price * (1 - risk_aversion * urgency_weight)
            else:
                target_price = benchmark_price * (1 + risk_aversion * urgency_weight)

            # 执行订单
            result = self._execute_simple(order, market_data)

            if result.get('error'):
                return result

            # 计算执行差额
            implementation_shortfall = result['average_price'] - benchmark_price
            if order.side == OrderSide.SELL:
                implementation_shortfall = -implementation_shortfall

            # 计算执行质量
            execution_quality = 1 - (abs(implementation_shortfall) / benchmark_price) if benchmark_price > 0 else 0

            # 更新结果
            result.update({
                'implementation_shortfall': implementation_shortfall,
                'benchmark_price': benchmark_price,
                'target_price': target_price,
                'execution_quality': execution_quality,
                'risk_aversion': risk_aversion,
                'urgency': urgency
            })

            return result

        except Exception as e:
            logger.error(f"执行差额算法失败 {order.order_id}: {e}")
            return {'error': str(e)}


        """流动性寻找算法"""
        try:
            symbol = order.symbol
            quantity = order.quantity

            # 获取流动性数据
            liquidity_data = market_data.get('liquidity', {}).get(symbol, {})
            bid_ask_spread = liquidity_data.get('bid_ask_spread', 0.01)
            market_depth = liquidity_data.get('market_depth', 1000)
            volume_imbalance = liquidity_data.get('volume_imbalance', 0.5)

            # 计算最优执行策略
            if market_depth > quantity * 2 and bid_ask_spread < 0.02:
                # 流动性充足，使用简单执行
                return self._execute_simple(order, market_data)
