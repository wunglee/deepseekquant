"""
功能碎片：事件窗口回测器
从 core/risk/backtest_framework.py 提取
状态：待整合到 core/backtest 模块

职责：
- 事件窗口回测（Event Study方法）
- 实际损失计算
- 预测损失对比
- 回测报告生成

迁移计划：
当 core_bak_refactored/core/backtest 模块开发完成后，整合此文件到该模块

相关文件：
- 源文件：core/risk/backtest_framework.py (EventWindowBacktester, BacktestReporter)
- 原有实现：core/backtest/backtest_engine.py (BacktestEngine)
- 调用者：core/risk/stress_test_validator.py (StressTestValidator)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Protocol
import logging

logger = logging.getLogger('DeepSeekQuant.BacktestFragments')


# =============================================================================
# 数据模型（回测模块标准）
# =============================================================================

@dataclass
class BacktestEvent:
    """
    回测事件定义（事件研究法专用）
    
    用于Event Study方法，定义事件窗口和预期效应
    """
    event_id: str
    name: str
    period: tuple  # (start_date, end_date)
    expected_decline: float
    scenario_params: Dict[str, Any]


@dataclass
class BacktestResult:
    """
    回测结果（功能碎片数据模型）
    
    记录预测vs实际的对比结果
    """
    event_id: str
    portfolio_id: str
    predicted_loss: float
    actual_loss: float
    prediction_error: float
    benchmark_index: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 协议接口（外部依赖）
# =============================================================================

class HistoricalDataProvider(Protocol):
    """历史数据提供者协议（依赖core/data模块）"""
    
    def get_index_prices(self, index_id: str, start_date: str, end_date: str):
        """获取指数价格数据"""
        ...
    
    def get_index_returns(self, index_id: str, start_date: str, end_date: str):
        """获取指数收益率序列"""
        ...


# =============================================================================
# 事件窗口回测引擎（回测模块业务逻辑）
# =============================================================================

class EventWindowBacktester:
    """
    事件窗口回测引擎（功能碎片业务逻辑）
    
    基于专家answer.md第5轮1.3节指导：
    - 使用事件窗口法验证压力测试预测准确性
    - 对比预测损失vs实际损失
    - 计算预测误差（目标≤20%）
    
    设计模式：
    - Event Study方法：聚焦事件窗口，避免噪音干扰
    - 依赖注入：通过构造函数注入数据源
    
    迁移计划：
    - 整合到 core/backtest/backtest_engine.py
    - 与现有 BacktestEngine 类协同（通用回测 vs 事件回测）
    - 可能作为 BacktestEngine 的子类或策略模式实现
    """
    
    def __init__(self, data_provider: HistoricalDataProvider):
        """
        初始化回测引擎
        
        Args:
            data_provider: 历史数据提供者（依赖注入）
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
    
    def run_backtest(self, portfolio, stress_tester, benchmark_index: str = '000300.SH') -> List[BacktestResult]:
        """
        运行回测
        
        Args:
            portfolio: 合成组合（来自portfolio模块）
            stress_tester: 压力测试器实例（来自risk模块）
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
    
    def _calculate_actual_loss(self, portfolio, event: BacktestEvent, benchmark_index: str) -> float:
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
    
    def _calculate_predicted_loss(self, portfolio, event: BacktestEvent, stress_tester) -> float:
        """
        使用压力测试器计算预测损失
        
        注：当前简化版本直接使用场景参数的decline
        真实集成时，应调用StressTester.run_stress_test()方法
        """
        # 简化版本：直接返回场景参数中的预期下跌幅度
        # TODO: 集成真实StressTester后，构造完整的场景对象并调用压力测试
        predicted_loss = event.scenario_params.get('decline', -0.20)
        logger.debug(f"预测损失（简化版）: {event.name}, {predicted_loss:.2%}")
        return float(predicted_loss)


# =============================================================================
# 回测报告生成器（回测模块报告功能）
# =============================================================================

class BacktestReporter:
    """
    回测报告生成器（功能碎片业务逻辑）
    
    迁移计划：
    - 整合到 core/backtest 模块的报告生成功能
    - 与现有 BacktestMetrics 类协同
    """
    
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


# =============================================================================
# 迁移检查清单
# =============================================================================

"""
功能碎片迁移检查清单（core/backtest模块开发时使用）

□ 1. 与BacktestEngine整合
    □ 确认 EventWindowBacktester 与 BacktestEngine 的关系
    □ 决定是合并、继承还是策略模式
    
□ 2. 扩展回测方法
    □ 时间序列回测（传统方法）
    □ 事件窗口回测（Event Study）
    □ 蒙特卡洛回测
    □ 滚动窗口回测（Walk-Forward）
    
□ 3. 回测指标增强
    □ 与 BacktestMetrics 类整合
    □ 添加风险调整收益指标
    □ 添加交易成本分析
    
□ 4. 报告生成增强
    □ HTML报告
    □ PDF报告
    □ 可视化图表（权益曲线、回撤曲线）
    
□ 5. 性能优化
    □ 向量化计算
    □ 并行回测（多策略/多参数）
    □ 增量回测（仅回测新数据）
    
□ 6. 测试覆盖
    □ 单元测试（各回测方法）
    □ 集成测试（端到端回测流程）
    □ 性能测试（大规模数据）
    
□ 7. 调用者更新
    □ 更新 core/risk/stress_test_validator.py 的导入路径
    □ 更新示例脚本
    □ 更新文档
"""
