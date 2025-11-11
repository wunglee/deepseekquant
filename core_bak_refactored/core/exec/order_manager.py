"""
订单管理器 - 业务层
从 core_bak/execution_engine.py 拆分
职责: 订单生命周期管理、订单簿维护
"""

from typing import Dict, List, Optional
from datetime import datetime
import logging
from collections import defaultdict

from .exec_models import Order, OrderStatus, OrderType

logger = logging.getLogger('DeepSeekQuant.OrderManager')


class OrderManager:
    """订单管理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        
                       market_data: Dict[str, Any],
                       risk_assessment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行订单列表

        Args:
            orders: 订单列表
            market_data: 市场数据
            risk_assessment: 风险评估结果

        Returns:
            执行结果
        """
        start_time = time.time()

        try:
            if not self.is_connected:
                logger.warning("执行引擎未连接，尝试重新连接...")
                if not self.connect():
                    return {'error': '执行引擎未连接且重新连接失败'}

            # 验证订单
            validated_orders = self._validate_orders(orders, market_data, risk_assessment)
            if not validated_orders:
                return {'error': '没有有效的订单可执行'}

            # 分组订单（按符号、方向、算法等）
            order_groups = self._group_orders(validated_orders)

            # 并行执行订单组
            execution_results = {}
            with ThreadPoolExecutor(max_workers=min(len(order_groups), 4)) as executor:
                future_to_group = {
                    executor.submit(self._execute_order_group, group, market_data): group_id
                    for group_id, group in order_groups.items()
                }

                for future in as_completed(future_to_group):
                    group_id = future_to_group[future]
                    try:
                        result = future.result()
                        execution_results[group_id] = result
                    except Exception as e:
                        logger.error(f"订单组 {group_id} 执行失败: {e}")
                        execution_results[group_id] = {'error': str(e)}

            # 汇总执行结果
            overall_result = self._aggregate_execution_results(execution_results)
            processing_time = time.time() - start_time

            # 更新性能统计
            self._update_performance_stats(overall_result, processing_time)

            logger.info(f"订单执行完成: 处理订单={len(validated_orders)}, "
                        f"成功={overall_result.get('successful_orders', 0)}, "
                        f"失败={overall_result.get('failed_orders', 0)}, "
                        f"耗时={processing_time:.3f}s")

            return overall_result

        except Exception as e:
            logger.error(f"订单执行失败: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}


                         market_data: Dict[str, Any],
                         risk_assessment: Optional[Dict[str, Any]]) -> List[Order]:
        """验证订单有效性"""
        validated_orders = []

        try:
            for order in orders:
                try:
                    # 基本验证
                    if not self._validate_single_order(order):
                        logger.warning(f"订单验证失败: {order.order_id}")
                        order.status = OrderStatus.REJECTED
                        order.rejection_reason = "基本验证失败"
                        continue

                    # 市场数据验证
                    if not self._validate_market_data(order, market_data):
                        logger.warning(f"订单市场数据验证失败: {order.order_id}")
                        order.status = OrderStatus.REJECTED
                        order.rejection_reason = "市场数据不足"
                        continue

                    # 风险验证
                    if risk_assessment and not self._validate_risk_compliance(order, risk_assessment):
                        logger.warning(f"订单风险验证失败: {order.order_id}")
                        order.status = OrderStatus.REJECTED
                        order.rejection_reason = "风险控制限制"
                        continue

                    # 经纪商验证
                    if not self._validate_broker_compatibility(order):
                        logger.warning(f"订单经纪商验证失败: {order.order_id}")
                        order.status = OrderStatus.REJECTED
                        order.rejection_reason = "经纪商不支持"
                        continue

                    # 算法参数验证
                    if not self._validate_algorithm_parameters(order):
                        logger.warning(f"订单算法参数验证失败: {order.order_id}")
                        order.status = OrderStatus.REJECTED
                        order.rejection_reason = "算法参数无效"
                        continue

                    order.risk_check_passed = True
                    validated_orders.append(order)
                    logger.debug(f"订单验证通过: {order.order_id}")

                except Exception as e:
                    logger.error(f"订单验证错误 {order.order_id}: {e}")
                    order.status = OrderStatus.REJECTED
                    order.rejection_reason = f"验证错误: {str(e)}"

            return validated_orders

        except Exception as e:
            logger.error(f"订单验证过程失败: {e}")
            return []


        """验证单个订单"""
        try:
            # 检查必需字段
            if not all([order.order_id, order.portfolio_id, order.symbol, order.quantity]):
                return False

            # 检查数量有效性
            if order.quantity <= 0:
                return False

            # 检查价格有效性
            if order.parameters.order_type == OrderType.LIMIT and order.parameters.limit_price <= 0:
                return False

            if order.parameters.order_type == OrderType.STOP and order.parameters.stop_price <= 0:
                return False

            # 检查时间有效性
            if order.parameters.time_in_force == TimeInForce.GTD:
                if not order.parameters.good_till_date:
                    return False
                try:
                    expiry_date = datetime.fromisoformat(order.parameters.good_till_date)
                    if expiry_date <= datetime.now():
                        return False
                except ValueError:
                    return False

            # 检查冰山订单参数
            if order.parameters.iceberg:
                if not order.parameters.display_quantity or order.parameters.display_quantity <= 0:
                    return False
                if order.parameters.display_quantity >= order.quantity:
                    return False

            return True

        except Exception as e:
            logger.error(f"单个订单验证失败 {order.order_id}: {e}")
            return False


        """将订单分组"""
        groups = defaultdict(list)

        try:
            for order in orders:
                # 创建分组键
                group_key = self._create_order_group_key(order)
                groups[group_key].append(order)

            logger.info(f"订单分组完成: {len(groups)} 个组, {len(orders)} 个订单")
            return dict(groups)

        except Exception as e:
            logger.error(f"订单分组失败: {e}")
            # 失败时将所有订单放入一个组
            return {'default_group': orders}


        """创建订单分组键"""
        try:
            # 基于符号、方向、算法和经纪商分组
            key_parts = [
                order.symbol,
                order.side.value,
                order.execution_params.algorithm.value,
                order.execution_params.algo_parameters.get('broker', self.default_broker),
                str(order.parameters.order_type.value)
            ]

            return '_'.join(key_parts)

        except Exception as e:
            logger.error(f"订单分组键创建失败 {order.order_id}: {e}")
            return 'default_group'


        """执行订单组"""
        group_results = {
            'group_id': hashlib.md5(str(orders[0].order_id).encode()).hexdigest()[:8],
            'total_orders': len(orders),
            'successful_orders': 0,
            'failed_orders': 0,
            'partial_orders': 0,
            'total_quantity': 0,
            'executed_quantity': 0,
            'total_commission': 0,
            'total_slippage': 0,
            'avg_execution_quality': 0,
            'order_results': [],
            'start_time': datetime.now().isoformat(),
            'end_time': None
        }

        try:
            # 按优先级排序订单
            sorted_orders = self._prioritize_orders(orders)

            # 顺序执行订单（避免市场影响）
            for order in sorted_orders:
                try:
                    result = self._execute_single_order(order, market_data)
                    group_results['order_results'].append(result)

                    if result['status'] == 'filled':
                        group_results['successful_orders'] += 1
                    elif result['status'] == 'partially_filled':
                        group_results['partial_orders'] += 1
                    else:
                        group_results['failed_orders'] += 1

                    group_results['total_quantity'] += order.quantity
                    group_results['executed_quantity'] += result.get('executed_quantity', 0)
                    group_results['total_commission'] += result.get('commission', 0)
                    group_results['total_slippage'] += result.get('slippage', 0)

                except Exception as e:
                    logger.error(f"订单执行失败 {order.order_id}: {e}")
                    failed_result = {
                        'order_id': order.order_id,
                        'status': 'failed',
                        'error': str(e),
                        'executed_quantity': 0,
                        'commission': 0,
                        'slippage': 0
                    }
                    group_results['order_results'].append(failed_result)
                    group_results['failed_orders'] += 1

            # 计算平均执行质量
            successful_results = [r for r in group_results['order_results'] if
                                  r['status'] in ['filled', 'partially_filled']]
            if successful_results:
                avg_quality = sum(r.get('execution_quality', 0) for r in successful_results) / len(successful_results)
                group_results['avg_execution_quality'] = avg_quality

            group_results['end_time'] = datetime.now().isoformat()

            return group_results

        except Exception as e:
            logger.error(f"订单组执行失败: {e}")
            group_results['error'] = str(e)
            group_results['end_time'] = datetime.now().isoformat()
            return group_results


        """订单优先级排序"""
        try:
            def get_order_priority(order: Order) -> tuple:
                # 优先级因素：紧急程度、算法复杂度、订单大小
                urgency_score = {
                    'low': 1, 'medium': 2, 'high': 3, 'urgent': 4
                }.get(order.execution_params.urgency, 2)

                algorithm_complexity = {
                    ExecutionAlgorithm.SIMPLE: 1,
                    ExecutionAlgorithm.PASSIVE: 1,
                    ExecutionAlgorithm.AGGRESSIVE: 2,
                    ExecutionAlgorithm.VWAP: 3,
                    ExecutionAlgorithm.TWAP: 3,
                    ExecutionAlgorithm.POV: 4,
                    ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL: 4
                }.get(order.execution_params.algorithm, 2)

                # 大额订单优先执行（但需要考虑市场影响）
                size_priority = min(order.quantity / 1000, 5)  # 标准化到0-5

                return (-urgency_score, -algorithm_complexity, -size_priority)  # 负号因为优先级高排前面

            return sorted(orders, key=get_order_priority)

        except Exception as e:
            logger.error(f"订单优先级排序失败: {e}")
            return orders  # 返回原始顺序


        """执行单个订单"""
        start_time = time.time()

        try:
            # 更新订单状态
            order.status = OrderStatus.SUBMITTED
            order.updated_at = datetime.now().isoformat()

            # 选择执行算法
            algorithm_func = self.execution_algorithms.get(
                order.execution_params.algorithm.value,
                self._execute_simple
            )

            # 执行订单
            execution_result = algorithm_func(order, market_data)

            # 计算执行时间
            execution_time = time.time() - start_time

            # 更新订单信息
            order.updated_at = datetime.now().isoformat()
            order.filled_quantity = execution_result.get('executed_quantity', 0)
            order.average_fill_price = execution_result.get('average_price', 0)
            order.commission = execution_result.get('commission', 0)
            order.fees = execution_result.get('fees', 0)
            order.slippage = execution_result.get('slippage', 0)
            order.market_impact = execution_result.get('market_impact', 0)
            order.timing_risk = execution_result.get('timing_risk', 0)
            order.execution_quality = execution_result.get('execution_quality', 0)

            # 更新订单状态
            if order.filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
                status = 'filled'
            elif order.filled_quantity > 0:
                order.status = OrderStatus.PARTIALLY_FILLED
                status = 'partially_filled'
            else:
                order.status = OrderStatus.REJECTED
                status = 'failed'

            # 保存订单到历史
            self._save_order_to_history(order)

            # 生成执行报告
            execution_report = self._generate_execution_report(order, execution_result, execution_time)
            self._save_execution_report(execution_report)

            result = {
                'order_id': order.order_id,
                'status': status,
                'executed_quantity': order.filled_quantity,
                'average_price': order.average_fill_price,
                'commission': order.commission,
                'fees': order.fees,
                'slippage': order.slippage,
                'market_impact': order.market_impact,
                'timing_risk': order.timing_risk,
                'execution_quality': order.execution_quality,
                'execution_time': execution_time,
                'broker_order_id': execution_result.get('broker_order_id'),
                'exchange_order_id': execution_result.get('exchange_order_id'),
                'execution_reports': execution_result.get('execution_reports', [])
            }

            if status == 'failed':
                result['error'] = execution_result.get('error', '执行失败')

            logger.info(f"订单执行完成: {order.order_id}, 状态={status}, "
                        f"数量={order.filled_quantity}/{order.quantity}, "
                        f"质量={order.execution_quality:.3f}")

            return result

        except Exception as e:
            logger.error(f"单个订单执行失败 {order.order_id}: {e}")

            # 更新订单状态为失败
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.now().isoformat()
            order.rejection_reason = f"执行错误: {str(e)}"

            self._save_order_to_history(order)

            return {
                'order_id': order.order_id,
                'status': 'failed',
                'error': str(e),
                'executed_quantity': 0,
                'commission': 0,
                'slippage': 0,
                'execution_time': time.time() - start_time
            }


        """保存订单到历史记录"""
        try:
            # 添加到订单历史
            self.order_history.append(copy.deepcopy(order))

            # 更新订单状态映射
            if order.status == OrderStatus.FILLED:
                self.completed_orders[order.order_id] = order
                if order.order_id in self.active_orders:
                    del self.active_orders[order.order_id]
            elif order.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                self.active_orders[order.order_id] = order
                if order.order_id in self.pending_orders:
                    del self.pending_orders[order.order_id]
            elif order.status == OrderStatus.REJECTED:
                if order.order_id in self.pending_orders:
                    del self.pending_orders[order.order_id]

            # 限制历史记录大小
            max_history = self.execution_config.get('max_order_history', 10000)
            if len(self.order_history) > max_history:
                self.order_history = self.order_history[-max_history:]

            logger.debug(f"订单保存到历史: {order.order_id}, 状态={order.status.value}")

        except Exception as e:
            logger.error(f"订单保存失败 {order.order_id}: {e}")


        """获取订单状态"""
        try:
            # 检查所有订单状态
            for order_list in [self.pending_orders, self.active_orders, self.completed_orders]:
                if order_id in order_list:
                    order = order_list[order_id]
                    return {
                        'order_id': order.order_id,
                        'status': order.status.value,
                        'symbol': order.symbol,
                        'quantity': order.quantity,
                        'filled_quantity': order.filled_quantity,
                        'average_fill_price': order.average_fill_price,
                        'commission': order.commission,
                        'slippage': order.slippage,
                        'execution_quality': order.execution_quality,
                        'created_at': order.created_at,
                        'updated_at': order.updated_at,
                        'execution_reports': [r.to_dict() for r in self.execution_reports.get(order_id, [])],
                        'risk_check_passed': order.risk_check_passed,
                        'rejection_reason': order.rejection_reason,
                        'cancellation_reason': order.cancellation_reason
                    }

            return None

        except Exception as e:
            logger.error(f"订单状态查询失败 {order_id}: {e}")
            return None


        """取消订单"""
        try:
            # 查找订单
            order = None
            for order_list in [self.pending_orders, self.active_orders]:
                if order_id in order_list:
                    order = order_list[order_id]
                    break

            if not order:
                logger.warning(f"取消订单失败: 订单 {order_id} 不存在")
                return False

            # 更新订单状态
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now().isoformat()
            order.cancellation_reason = reason

            # 从活动订单移动到完成订单
            if order_id in self.active_orders:
                self.completed_orders[order_id] = order
                del self.active_orders[order_id]
            elif order_id in self.pending_orders:
                self.completed_orders[order_id] = order
                del self.pending_orders[order_id]

            # 生成取消报告
            cancel_report = ExecutionReport(
                report_id=f"CANCEL_{order_id}_{int(time.time())}",
                order_id=order_id,
                timestamp=datetime.now().isoformat(),
                fill_price=0,
                fill_quantity=0,
                remaining_quantity=order.quantity - order.filled_quantity,
                cumulative_quantity=order.filled_quantity,
                execution_type='cancelled',
                liquidity='none',
                venue='system',
                execution_id=f"CANCEL_{order_id}",
                transaction_time=datetime.now().isoformat(),
                commission=0,
                fees=0,
                slippage=0,
                market_impact=0,
                timing_risk=0,
                execution_quality=0,
                benchmark_comparison={},
                flags=['cancelled'],
                metadata={'cancellation_reason': reason}
            )

            self._save_execution_report(cancel_report)

            logger.info(f"订单取消成功: {order_id}, 原因={reason}")
            return True

        except Exception as e:
            logger.error(f"订单取消失败 {order_id}: {e}")
            return False


        """修改订单"""
        try:
            # 查找订单
            if order_id not in self.pending_orders and order_id not in self.active_orders:
                logger.warning(f"修改订单失败: 订单 {order_id} 不存在或不可修改")
                return False

            order = self.pending_orders.get(order_id) or self.active_orders.get(order_id)
            if not order:
                return False

            # 检查订单状态是否允许修改
            if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                logger.warning(f"订单 {order_id} 状态为 {order.status.value}，不允许修改")
                return False

            # 应用修改
            if 'quantity' in modifications:
                new_quantity = modifications['quantity']
                if new_quantity > 0 and new_quantity != order.quantity:
                    order.quantity = new_quantity
                    logger.debug(f"订单 {order_id} 数量修改: {order.quantity} -> {new_quantity}")

            if 'limit_price' in modifications:
                new_price = modifications['limit_price']
                if new_price > 0 and new_price != order.parameters.limit_price:
                    order.parameters.limit_price = new_price
                    logger.debug(f"订单 {order_id} 限价修改: {order.parameters.limit_price} -> {new_price}")

            if 'stop_price' in modifications:
                new_stop = modifications['stop_price']
                if new_stop > 0 and new_stop != order.parameters.stop_price:
                    order.parameters.stop_price = new_stop
                    logger.debug(f"订单 {order_id} 止损价修改: {order.parameters.stop_price} -> {new_stop}")

            order.updated_at = datetime.now().isoformat()

            # 生成修改报告
            modify_report = ExecutionReport(
                report_id=f"MODIFY_{order_id}_{int(time.time())}",
                order_id=order_id,
                timestamp=datetime.now().isoformat(),
                fill_price=0,
                fill_quantity=0,
                remaining_quantity=order.quantity,
                cumulative_quantity=order.filled_quantity,
                execution_type='modified',
                liquidity='none',
                venue='system',
                execution_id=f"MODIFY_{order_id}",
                transaction_time=datetime.now().isoformat(),
                commission=0,
                fees=0,
                slippage=0,
                market_impact=0,
                timing_risk=0,
                execution_quality=0,
                benchmark_comparison={},
                flags=['modified'],
                metadata={'modifications': modifications}
            )

            self._save_execution_report(modify_report)

            logger.info(f"订单修改成功: {order_id}")
            return True

        except Exception as e:
            logger.error(f"订单修改失败 {order_id}: {e}")
            return False


        """保存订单历史"""
        try:
            # 这里应该保存到数据库或文件
            # 简化实现：记录到日志
            total_orders = len(self.order_history)
            completed_orders = len([o for o in self.order_history if o.status == OrderStatus.FILLED])
            success_rate = completed_orders / total_orders if total_orders > 0 else 0

            logger.info(
                f"订单历史保存: 总订单={total_orders}, 完成订单={completed_orders}, 成功率={success_rate:.2%}")

        except Exception as e:
            logger.error(f"订单历史保存失败: {e}")


