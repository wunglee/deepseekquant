"""
历史回测框架 - 压力测试场景验证
从第5轮专家指导实施
职责: 历史事件回测、合成组合构造、预测误差验证

设计原则：
- 数据抽象层：预留真实数据集成点
- 模拟优先：使用模拟数据快速验证框架
- 接口稳定：真实数据集成时无需修改业务逻辑
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Protocol
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

logger = logging.getLogger('DeepSeekQuant.BacktestFramework')


# =============================================================================
# 数据接口抽象层（预留真实数据集成点）
# =============================================================================

class HistoricalDataProvider(Protocol):
    """
    历史数据提供者接口（抽象层）
    
    设计目的：
    - 解耦业务逻辑与数据来源
    - 支持模拟数据（当前）和真实数据（未来）无缝切换
    - 为core_bak_refactored/core/data模块集成预留标准接口
    """
    
    def get_index_prices(self, index_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取指数价格数据
        
        Args:
            index_id: 指数代码（如'000300.SH'沪深300）
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
        
        Returns:
            DataFrame with columns: ['date', 'close', 'volume']
        """
        ...
    
    def get_index_returns(self, index_id: str, start_date: str, end_date: str) -> pd.Series:
        """
        获取指数收益率序列
        
        Args:
            index_id: 指数代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            Series with date index and return values
        """
        ...


# =============================================================================
# 模拟数据提供者（Phase 3A实现）
# =============================================================================

class MockHistoricalDataProvider:
    """
    模拟历史数据提供者
    
    用途：
    - 在core/data模块未完成前，提供测试数据
    - 基于真实历史事件参数生成合理的模拟数据
    - 支持框架功能验证和测试
    
    注意：
    - 数据为模拟生成，仅用于框架验证
    - 真实回测需要替换为RealHistoricalDataProvider
    """
    
    def __init__(self):
        self.event_params = {
            '2015_china_market_crash': {
                'period': ('2015-06-15', '2015-08-26'),
                'expected_decline': -0.43,
                'volatility_multiplier': 2.5
            },
            'covid_19_pandemic': {
                'period': ('2020-02-20', '2020-03-23'),
                'expected_decline': -0.34,
                'volatility_multiplier': 3.0
            },
            '2008_financial_crisis': {
                'period': ('2008-09-15', '2008-11-20'),
                'expected_decline': -0.40,
                'volatility_multiplier': 3.5
            }
        }
    
    def get_index_prices(self, index_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """生成模拟的指数价格数据"""
        # 解析日期
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end, freq='B')  # 交易日
        
        # 检查是否在已知事件窗口内
        event_decline = 0.0
        event_vol = 1.0
        for event_id, params in self.event_params.items():
            event_start = pd.to_datetime(params['period'][0])
            event_end = pd.to_datetime(params['period'][1])
            if start >= event_start and end <= event_end:
                event_decline = params['expected_decline']
                event_vol = params['volatility_multiplier']
                logger.info(f"检测到事件窗口: {event_id}, decline={event_decline}, vol={event_vol}")
                break
        
        # 生成模拟价格序列
        n_days = len(dates)
        initial_price = 3000.0  # 沪深300典型水平
        
        # 基于事件参数生成收益率（确定性趋势 + 随机波动）
        base_volatility = 0.015  # 1.5%日波动率
        daily_volatility = base_volatility * event_vol
        
        # 生成确定性下跌趋势（事件期间）+ 随机波动
        if event_decline != 0.0 and n_days > 0:
            # 确保总收益率接近expected_decline
            daily_drift = event_decline / n_days
            # 随机部分使用较小的波动率，避免淹没趋势
            random_component = np.random.normal(0, daily_volatility * 0.5, n_days)
            daily_returns = daily_drift + random_component
        else:
            # 非事件期间：纯随机游走
            daily_returns = np.random.normal(0, daily_volatility, n_days)
        
        # 计算价格序列
        prices = initial_price * np.cumprod(1 + daily_returns)
        
        # 生成成交量（简化模拟）
        base_volume = 100000000  # 1亿手
        volumes = base_volume * (1 + np.random.uniform(-0.3, 0.5, n_days))
        
        df = pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': volumes
        })
        
        logger.debug(f"生成模拟数据: {index_id}, {len(df)}天, 总收益率={prices[-1]/prices[0]-1:.2%}")
        return df
    
    def get_index_returns(self, index_id: str, start_date: str, end_date: str) -> pd.Series:
        """获取指数收益率序列"""
        df = self.get_index_prices(index_id, start_date, end_date)
        returns = df['close'].pct_change().fillna(0)
        returns.index = df['date']
        return returns


# =============================================================================
# 合成组合构造器（基于专家answer.md 1.3节）
# =============================================================================

@dataclass
class SyntheticPortfolio:
    """
    合成组合定义
    
    用于回测验证的标准化组合
    """
    portfolio_id: str
    name: str
    composition: Dict[str, float]  # {index_id: weight}
    total_value: float = 1000000.0  # 100万基准
    metadata: Dict[str, Any] = field(default_factory=dict)


class SyntheticPortfolioBuilder:
    """
    合成组合构造器
    
    基于专家answer.md第5轮1.3节指导，构造3种典型组合：
    1. 沪深300等权重组合
    2. 行业轮动组合（金融30%+消费25%+科技20%+其他25%）
    3. A+H混合组合（A股70%+港股30%）
    """
    
    @staticmethod
    def build_csi300_equal_weight() -> SyntheticPortfolio:
        """构造沪深300等权重组合"""
        return SyntheticPortfolio(
            portfolio_id='CSI300_EQ',
            name='沪深300等权重组合',
            composition={'000300.SH': 1.0},  # 简化：直接使用指数
            metadata={'type': 'index_replication', 'market': 'CN'}
        )
    
    @staticmethod
    def build_sector_rotation() -> SyntheticPortfolio:
        """构造行业轮动组合"""
        return SyntheticPortfolio(
            portfolio_id='SECTOR_ROT',
            name='行业轮动组合',
            composition={
                'finance_index': 0.30,    # 金融30%
                'consumer_index': 0.25,   # 消费25%
                'tech_index': 0.20,       # 科技20%
                'other_index': 0.25       # 其他25%
            },
            metadata={'type': 'sector_rotation', 'market': 'CN'}
        )
    
    @staticmethod
    def build_ah_hybrid() -> SyntheticPortfolio:
        """构造A+H混合组合"""
        return SyntheticPortfolio(
            portfolio_id='AH_HYBRID',
            name='A+H混合组合',
            composition={
                '000300.SH': 0.70,  # A股70%（沪深300）
                'HSI': 0.30         # 港股30%（恒生指数）
            },
            metadata={'type': 'cross_border', 'markets': ['CN', 'HK']}
        )


# =============================================================================
# 事件窗口回测引擎（基于专家answer.md 1.3节）
# =============================================================================

@dataclass
class BacktestEvent:
    """回测事件定义"""
    event_id: str
    name: str
    period: tuple  # (start_date, end_date)
    expected_decline: float
    scenario_params: Dict[str, Any]


@dataclass
class BacktestResult:
    """回测结果"""
    event_id: str
    portfolio_id: str
    predicted_loss: float
    actual_loss: float
    prediction_error: float  # abs(predicted - actual) / abs(actual)
    benchmark_index: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventWindowBacktester:
    """
    事件窗口回测引擎
    
    基于专家answer.md第5轮1.3节指导：
    - 使用事件窗口法验证压力测试预测准确性
    - 对比预测损失vs实际损失
    - 计算预测误差（目标≤20%）
    """
    
    def __init__(self, data_provider: HistoricalDataProvider):
        """
        初始化回测引擎
        
        Args:
            data_provider: 历史数据提供者（模拟或真实）
        """
        self.data_provider = data_provider
        self.events = self._load_events()
    
    def _load_events(self) -> List[BacktestEvent]:
        """加载回测事件（基于专家指导的3个核心事件）"""
        return [
            BacktestEvent(
                event_id='2015_china_market_crash',
                name='2015中国股灾',
                period=('2015-06-15', '2015-08-26'),
                expected_decline=-0.43,
                scenario_params={'decline': -0.30, 'liquidity_dry_up': 0.8}
            ),
            BacktestEvent(
                event_id='covid_19_pandemic',
                name='COVID-19疫情',
                period=('2020-02-20', '2020-03-23'),
                expected_decline=-0.34,
                scenario_params={'decline': -0.20, 'volatility_spike': 2.0}
            ),
            BacktestEvent(
                event_id='2008_financial_crisis',
                name='2008金融危机',
                period=('2008-09-15', '2008-11-20'),
                expected_decline=-0.40,
                scenario_params={'decline': -0.40, 'volatility_spike': 3.5}
            )
        ]
    
    def run_backtest(self, portfolio: SyntheticPortfolio, 
                     stress_tester, benchmark_index: str = '000300.SH') -> List[BacktestResult]:
        """
        运行回测
        
        Args:
            portfolio: 合成组合
            stress_tester: 压力测试器实例（StressTester）
            benchmark_index: 基准指数
        
        Returns:
            回测结果列表
        """
        results = []
        
        for event in self.events:
            try:
                # 1. 获取实际历史数据
                actual_loss = self._calculate_actual_loss(
                    portfolio, event, benchmark_index
                )
                
                # 2. 使用压力测试器预测损失
                predicted_loss = self._calculate_predicted_loss(
                    portfolio, event, stress_tester
                )
                
                # 3. 计算预测误差
                if actual_loss != 0:
                    prediction_error = abs(predicted_loss - actual_loss) / abs(actual_loss)
                else:
                    prediction_error = 0.0
                
                result = BacktestResult(
                    event_id=event.event_id,
                    portfolio_id=portfolio.portfolio_id,
                    predicted_loss=predicted_loss,
                    actual_loss=actual_loss,
                    prediction_error=prediction_error,
                    benchmark_index=benchmark_index,
                    metadata={
                        'event_name': event.name,
                        'period': event.period,
                        'expected_decline': event.expected_decline
                    }
                )
                
                results.append(result)
                
                logger.info(f"回测完成: {event.name}, 预测={predicted_loss:.2%}, "
                          f"实际={actual_loss:.2%}, 误差={prediction_error:.2%}")
                
            except Exception as e:
                logger.error(f"回测失败: {event.name}, 错误: {e}", exc_info=True)
        
        return results
    
    def _calculate_actual_loss(self, portfolio: SyntheticPortfolio, 
                               event: BacktestEvent, benchmark_index: str) -> float:
        """计算实际历史损失"""
        start_date, end_date = event.period
        
        # 获取基准指数数据
        prices = self.data_provider.get_index_prices(benchmark_index, start_date, end_date)
        
        if len(prices) < 2:
            logger.warning(f"数据不足: {benchmark_index}, {start_date}-{end_date}")
            return 0.0
        
        # 计算总收益率
        initial_price = prices['close'].iloc[0]
        final_price = prices['close'].iloc[-1]
        actual_return = (final_price - initial_price) / initial_price
        
        logger.debug(f"实际损失计算: {benchmark_index}, 初始={initial_price:.2f}, "
                    f"最终={final_price:.2f}, 收益率={actual_return:.2%}")
        
        return float(actual_return)
    
    def _calculate_predicted_loss(self, portfolio: SyntheticPortfolio,
                                  event: BacktestEvent, stress_tester) -> float:
        """使用压力测试器计算预测损失
        
        注：当前简化版本直接使用场景参数的decline
        真实集成时，应调用StressTester.run_stress_test()方法
        """
        # 简化版本：直接返回场景参数中的预期下跌幅度
        # TODO: 集成真实StressTester后，构造完整的场景对象并调用压力测试
        predicted_loss = event.scenario_params.get('decline', -0.20)
        logger.debug(f"预测损失（简化版）: {event.name}, {predicted_loss:.2%}")
        return float(predicted_loss)


# =============================================================================
# 回测报告生成器
# =============================================================================

class BacktestReporter:
    """回测报告生成器"""
    
    @staticmethod
    def generate_summary(results: List[BacktestResult]) -> Dict[str, Any]:
        """
        生成回测摘要
        
        Args:
            results: 回测结果列表
        
        Returns:
            摘要字典，包含统计指标
        """
        if not results:
            return {'status': 'no_results'}
        
        errors = [r.prediction_error for r in results]
        
        summary = {
            'total_tests': len(results),
            'avg_error': np.mean(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'error_std': np.std(errors),
            'accuracy_20pct': sum(1 for e in errors if e <= 0.20) / len(errors),  # ≤20%误差比例
            'results': [
                {
                    'event': r.event_id,
                    'portfolio': r.portfolio_id,
                    'predicted': f"{r.predicted_loss:.2%}",
                    'actual': f"{r.actual_loss:.2%}",
                    'error': f"{r.prediction_error:.2%}"
                }
                for r in results
            ]
        }
        
        return summary
    
    @staticmethod
    def print_summary(summary: Dict[str, Any]):
        """打印回测摘要"""
        if summary.get('status') == 'no_results':
            print("无回测结果")
            return
        
        print("\n" + "="*60)
        print("历史回测验证报告")
        print("="*60)
        print(f"总测试数: {summary['total_tests']}")
        print(f"平均误差: {summary['avg_error']:.2%}")
        print(f"最大误差: {summary['max_error']:.2%}")
        print(f"最小误差: {summary['min_error']:.2%}")
        print(f"误差标准差: {summary['error_std']:.2%}")
        print(f"≤20%误差占比: {summary['accuracy_20pct']:.1%} (目标≥80%)")
        print("\n详细结果:")
        print("-"*60)
        for r in summary['results']:
            print(f"{r['event']:30s} | 预测: {r['predicted']:>8s} | "
                  f"实际: {r['actual']:>8s} | 误差: {r['error']:>8s}")
        print("="*60 + "\n")
