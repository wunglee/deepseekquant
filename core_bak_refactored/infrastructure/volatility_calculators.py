"""
波动率计算工具 - 基础设施层

职责：提供与业务无关的纯数学/统计计算函数，用于波动率分析
- 日波动率计算算法
- 历史波动率计算算法
- 年化波动率计算算法

架构原则：
- 不包含任何业务领域概念
- 只接收纯数值数据
- 参数全部显式传入，不使用业务默认值
- 函数命名使用数学/统计术语，而非业务术语
"""

import numpy as np
from typing import List, Any
import logging

logger = logging.getLogger('DeepSeekQuant.Infrastructure.VolatilityCalculators')


class VolatilityCalculators:
    """波动率计算工具类（纯数学/统计），不包含业务术语"""
    
    @staticmethod
    def calculate_daily_volatility(data: List[Any]) -> float:
        """
        计算日波动率
        
        Args:
            data: 数据列表（MarketData对象或数据字典）
            
        Returns:
            日波动率值
        """
        if len(data) < 2:
            logger.warning("数据不足，无法计算波动率（至少需要2个数据点）")
            return 0.0

        try:
            # 提取收盘价
            closes = []
            for item in data:
                if hasattr(item, 'close'):
                    closes.append(item.close)
                elif isinstance(item, dict):
                    closes.append(item.get('close', 0))
                else:
                    logger.warning(f"无法从数据中提取收盘价: {type(item)}")
                    return 0.0

            if len(closes) < 2:
                return 0.0

            # 计算对数收益率
            returns = []
            for i in range(1, len(closes)):
                if closes[i - 1] > 0:
                    daily_return = (closes[i] - closes[i - 1]) / closes[i - 1]
                    returns.append(daily_return)

            if len(returns) < 1:
                return 0.0

            # 计算标准差
            std_dev = np.std(returns)

            # 年化（假设252个交易日）
            annualized_volatility = std_dev * np.sqrt(252)

            return annualized_volatility

        except Exception as e:
            logger.error(f"计算波动率失败: {e}")
            return 0.0
    
    @staticmethod
    def calculate_historical_volatility(prices: List[float], window: int = 30) -> float:
        """
        计算历史波动率
        
        Args:
            prices: 价格列表
            window: 计算窗口大小
            
        Returns:
            历史波动率值
        """
        if len(prices) < window or window < 2:
            return 0.0
            
        # 取最近window个价格
        recent_prices = prices[-window:]
        
        # 计算收益率
        returns = []
        for i in range(1, len(recent_prices)):
            if recent_prices[i-1] > 0:
                ret = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
                returns.append(ret)
        
        if not returns:
            return 0.0
            
        # 计算标准差并年化
        std_dev = np.std(returns)
        annualized_volatility = std_dev * np.sqrt(252)
        
        return annualized_volatility