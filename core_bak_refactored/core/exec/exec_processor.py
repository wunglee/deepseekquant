"""
执行处理器核心 - 业务层
从 core_bak/execution_engine.py 拆分
职责: 协调订单管理、券商连接、执行策略
"""

from typing import Dict, List, Any
from datetime import datetime
import logging

from .exec_models import Order, ExecutionReport
from .order_manager import OrderManager
from .broker_connector import BrokerConnector
from .execution_strategy import ExecutionStrategy
from .transaction_cost import TransactionCostModel

logger = logging.getLogger('DeepSeekQuant.ExecProcessor')


class ExecProcessor:
    """执行处理器 - 协调执行引擎各组件"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 初始化各个组件
        self.order_manager = OrderManager(config)
        self.broker_connector = BrokerConnector(config)
        self.execution_strategy = ExecutionStrategy(config)
        self.cost_model = TransactionCostModel(config)
        
        logger.info("执行处理器初始化完成")
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理执行请求
        
        Args:
            data: 输入数据
            
        Returns:
            处理结果
        """
        try:
            action = data.get('action')
            
            if action == 'submit_order':
                return self._handle_submit_order(data)
            elif action == 'cancel_order':
                return self._handle_cancel_order(data)
            elif action == 'query_status':
                return self._handle_query_status(data)
            else:
                return {'success': False, 'error': f'Unknown action: {action}'}
                
        except Exception as e:
            logger.error(f"执行处理失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _handle_submit_order(self, data: Dict) -> Dict:
        """处理订单提交"""
        # 1. 创建订单
        order = self.order_manager.create_order(data['order_params'])
        
        # 2. 选择执行策略
        execution_plan = self.execution_strategy.plan_execution(order)
        
        # 3. 提交到券商
        result = self.broker_connector.submit_order(order, execution_plan)
        
        return {
            'success': True,
            'order_id': order.order_id,
            'result': result
        }
    
    def _handle_cancel_order(self, data: Dict) -> Dict:
        """处理订单取消"""
        order_id = data['order_id']
        result = self.broker_connector.cancel_order(order_id)
        
        return {
            'success': True,
            'order_id': order_id,
            'cancelled': result
        }
    
    def _handle_query_status(self, data: Dict) -> Dict:
        """查询订单状态"""
        order_id = data.get('order_id')
        
        if order_id:
            order = self.order_manager.get_order(order_id)
            return {
                'success': True,
                'order': order.to_dict() if order else None
            }
        else:
            orders = self.order_manager.get_all_orders()
            return {
                'success': True,
                'orders': [o.to_dict() for o in orders]
            }
