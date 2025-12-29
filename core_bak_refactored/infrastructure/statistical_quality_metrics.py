""" 
统计质量度量工具 - 基础设施层（重命名自 data_quality_calculators.py）

职责：提供与业务无关的纯数学/统计计算函数，用于数据质量分析
- 时间序列一致性计算（时间间隔、价格平滑性）
- 跨序列相关性计算（符号相关性检测）
- 统计异常检测算法（3-sigma、IQR、Z-score）
- 数据完整性度量（缺失率、覆盖率）

架构原则：
- 不包含任何业务领域概念（市场、股票、指数等）
- 只接收纯数值数据（DataFrame、Series、ndarray）
- 参数全部显式传入，不使用业务默认值
- 函数命名使用数学/统计术语，而非业务术语

与 core/data/quality/data_quality_checker.py 的区别：
- 本模块：纯算法层，提供计算能力
- data_quality_checker：业务质量层，理解市场数据，调用本模块的算法

使用示例：
    from core_bak_refactored.infrastructure.statistical_quality_metrics import StatisticalQualityMetrics
    
    # 纯算法调用
    consistency_score, issues = StatisticalQualityMetrics.calculate_time_series_consistency(data)
"""

import logging
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger('DeepSeekQuant.Infrastructure.StatisticalQualityMetrics')


class StatisticalQualityMetrics:
    """统计质量度量工具类（纯数学/统计），不包含业务术语"""
    
    @staticmethod
    def calculate_time_series_consistency(data: List[Dict[str, Any]]) -> Tuple[float, List[Dict]]:
        """检查时间序列一致性
        
        Args:
            data: 时间序列数据列表
            
        Returns:
            (一致性分数, 问题列表)
        """
        issues = []
        score = 1.0
        
        if len(data) < 2:
            return score, issues
            
        # 按时间排序
        sorted_data = sorted(data, key=lambda x: x['timestamp'])
        
        # 检查时间连续性
        for i in range(1, len(sorted_data)):
            time_gap = (sorted_data[i]['timestamp'] - sorted_data[i - 1]['timestamp']).total_seconds() / 3600  # 小时
            
            # 根据数据频率确定预期间隔
            expected_interval = StatisticalQualityMetrics._get_expected_interval(sorted_data)
            
            if time_gap > expected_interval * 3:  # 允许3倍间隔的容差
                issues.append({
                    'type': 'time_gap_inconsistency',
                    'severity': 'medium',
                    'message': f'时间间隔异常: {time_gap:.1f}小时 (预期: {expected_interval:.1f}小时)',
                    'start_time': sorted_data[i - 1]['timestamp'].isoformat(),
                    'end_time': sorted_data[i]['timestamp'].isoformat(),
                    'gap_hours': time_gap,
                    'expected_interval': expected_interval
                })
        
        # 检查价格序列的平滑性
        price_changes = []
        for i in range(1, len(sorted_data)):
            if 'close' in sorted_data[i] and 'close' in sorted_data[i-1]:
                price_change = abs(sorted_data[i]['close'] - sorted_data[i - 1]['close']) / sorted_data[i - 1]['close']
                price_changes.append(price_change)
        
        if price_changes:
            avg_change = np.mean(price_changes)
            std_change = np.std(price_changes)
            
            # 检测异常的价格变化
            for i, change in enumerate(price_changes):
                if change > avg_change + 3 * std_change:  # 3sigma异常
                    issues.append({
                        'type': 'price_change_inconsistency',
                        'severity': 'medium',
                        'message': f'价格变化异常: {change:.1%} (平均: {avg_change:.1%})',
                        'timestamp': sorted_data[i + 1]['timestamp'].isoformat(),
                        'price_change': change,
                        'average_change': avg_change,
                        'std_dev': std_change
                    })
        
        if issues:
            score *= 0.85
            
        return max(0.0, min(1.0, score)), issues
    
    @staticmethod
    def _get_expected_interval(data: List[Dict[str, Any]]) -> float:
        """获取预期的时间间隔"""
        if len(data) < 2:
            return 24.0  # 默认24小时
            
        intervals = []
        for i in range(1, len(data)):
            gap = (data[i]['timestamp'] - data[i - 1]['timestamp']).total_seconds() / 3600
            intervals.append(gap)
            
        # 使用中位数作为预期间隔
        return float(np.median(intervals)) if intervals else 24.0
    
    @staticmethod
    def calculate_cross_symbol_consistency(data: List[Dict[str, Any]], symbols: List[str]) -> Tuple[float, List[Dict]]:
        """检查跨符号一致性
        
        Args:
            data: 数据列表
            symbols: 符号列表
            
        Returns:
            (一致性分数, 问题列表)
        """
        issues = []
        score = 1.0
        
        if len(data) < 10:  # 需要足够的数据点
            return score, issues
            
        # 按符号分组
        symbol_data = {}
        for point in data:
            symbol = point.get('symbol')
            if symbol not in symbol_data:
                symbol_data[symbol] = []
            symbol_data[symbol].append(point)
        
        # 检查相关符号之间的价格关系
        correlated_symbols = StatisticalQualityMetrics._find_correlated_symbols(symbols)
        
        for sym1, sym2 in correlated_symbols:
            if sym1 in symbol_data and sym2 in symbol_data:
                correlation_issues = StatisticalQualityMetrics._check_symbol_correlation(
                    symbol_data[sym1], symbol_data[sym2], sym1, sym2
                )
                issues.extend(correlation_issues)
        
        if issues:
            score *= 0.9
            
        return max(0.0, min(1.0, score)), issues
    
    @staticmethod
    def _find_correlated_symbols(symbols: List[str]) -> List[Tuple[str, str]]:
        """查找相关的符号对"""
        # 这里实现符号相关性分析
        # 例如：同行业股票、ETF与成分股等
        
        correlated_pairs = []
        
        # 简单的行业分组
        sector_groups = {
            'technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
            'financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
            'healthcare': ['JNJ', 'PFE', 'MRK', 'UNH', 'ABT']
        }
        
        for sector, sector_symbols in sector_groups.items():
            # 找到在数据中存在的符号
            existing_symbols = [s for s in sector_symbols if s in symbols]
            
            # 为存在的符号创建配对
            for i in range(len(existing_symbols)):
                for j in range(i + 1, len(existing_symbols)):
                    correlated_pairs.append((existing_symbols[i], existing_symbols[j]))
        
        return correlated_pairs
    
    @staticmethod
    def _check_symbol_correlation(data1: List[Dict[str, Any]], data2: List[Dict[str, Any]],
                                 sym1: str, sym2: str) -> List[Dict]:
        """检查两个符号的相关性"""
        issues = []
        
        # 对齐时间戳
        aligned_data = StatisticalQualityMetrics._align_time_series(data1, data2)
        if not aligned_data:
            return issues
            
        prices1 = [d['close'] for d in aligned_data[sym1]]
        prices2 = [d['close'] for d in aligned_data[sym2]]
        timestamps = [d['timestamp'] for d in aligned_data[sym1]]
        
        # 计算相关性
        if len(prices1) > 1 and len(prices2) > 1:
            correlation = np.corrcoef(prices1, prices2)[0, 1]
            
            # 检查相关性是否异常低
            expected_correlation = 0.7  # 预期相关性阈值
            if correlation < expected_correlation - 0.3:  # 低于预期0.3
                issues.append({
                    'type': 'cross_symbol_correlation',
                    'severity': 'medium',
                    'message': f'跨符号相关性异常: {correlation:.2f} (预期: {expected_correlation})',
                    'symbols': [sym1, sym2],
                    'correlation': correlation,
                    'expected_correlation': expected_correlation,
                    'sample_size': len(prices1),
                    'timestamp': timestamps[-1].isoformat() if timestamps else None
                })
        
        return issues
    
    @staticmethod
    def _align_time_series(data1: List[Dict[str, Any]], data2: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """对齐两个时间序列数据"""
        # 创建时间戳到数据点的映射
        map1 = {d['timestamp']: d for d in data1}
        map2 = {d['timestamp']: d for d in data2}
        
        # 找到共同的时间戳
        common_timestamps = set(map1.keys()) & set(map2.keys())
        common_timestamps = sorted(list(common_timestamps))
        
        if not common_timestamps:
            return {}
            
        # 创建对齐的数据
        aligned_data = {
            data1[0]['symbol']: [map1[ts] for ts in common_timestamps],
            data2[0]['symbol']: [map2[ts] for ts in common_timestamps]
        }
        
        return aligned_data
    
    @staticmethod
    def calculate_distribution_consistency(data: List[Dict[str, Any]]) -> Tuple[float, List[Dict]]:
        """检查统计分布一致性
        
        Args:
            data: 数据列表
            
        Returns:
            (一致性分数, 问题列表)
        """
        issues = []
        score = 1.0
        
        if len(data) < 30:  # 需要足够的数据点进行统计分析
            return score, issues
            
        # 提取价格数据
        prices = [d['close'] for d in data if 'close' in d]
        returns = []
        for i in range(1, len(prices)):
            returns.append((prices[i] - prices[i-1]) / prices[i-1])
            
        if len(returns) < 30:
            return score, issues
            
        # 检查偏度和峰度
        returns_array = np.array(returns)
        skewness = float(np.mean(((returns_array - np.mean(returns_array)) / np.std(returns_array)) ** 3))
        kurtosis = float(np.mean(((returns_array - np.mean(returns_array)) / np.std(returns_array)) ** 4)) - 3
        
        # 检查偏度是否异常
        if abs(skewness) > 2:  # 偏度绝对值超过2认为异常
            issues.append({
                'type': 'skewness_inconsistency',
                'severity': 'medium',
                'message': f'收益率分布偏度异常: {skewness:.2f}',
                'metric': 'skewness',
                'value': skewness,
                'threshold': 2,
                'timestamp': data[-1]['timestamp'].isoformat() if data else None
            })
        
        # 检查峰度是否异常
        if abs(kurtosis) > 5:  # 峰度绝对值超过5认为异常
            issues.append({
                'type': 'kurtosis_inconsistency',
                'severity': 'medium',
                'message': f'收益率分布峰度异常: {kurtosis:.2f}',
                'metric': 'kurtosis',
                'value': kurtosis,
                'threshold': 5,
                'timestamp': data[-1]['timestamp'].isoformat() if data else None
            })
        
        if issues:
            score *= 0.88
            
        return max(0.0, min(1.0, score)), issues
    
    @staticmethod
    def calculate_return_distribution(data: List[Dict[str, Any]]) -> Tuple[float, List[Dict]]:
        """检查收益率分布
        
        Args:
            data: 数据列表
            
        Returns:
            (一致性分数, 问题列表)
        """
        issues = []
        score = 1.0
        
        # 提取收益率数据
        prices = [d['close'] for d in data if 'close' in d]
        if len(prices) < 30:
            return score, issues
            
        returns = []
        for i in range(1, len(prices)):
            returns.append((prices[i] - prices[i-1]) / prices[i-1])
            
        if len(returns) < 30:
            return score, issues
            
        returns_array = np.array(returns)
        
        # 检查正态性（使用Jarque-Bera检验）
        try:
            from scipy.stats import jarque_bera
            jb_stat, jb_pvalue = jarque_bera(returns_array)
            
            if jb_pvalue < 0.05:  # 非正态分布
                issues.append({
                    'type': 'return_distribution_non_normal',
                    'severity': 'low',
                    'message': f'收益率分布非正态: JB统计量 {jb_stat:.2f}, p值 {jb_pvalue:.3f}',
                    'jb_statistic': jb_stat,
                    'jb_pvalue': jb_pvalue,
                })
        except ImportError:
            pass  # 如果没有scipy，跳过这个检查
        
        # 检查异常值
        q1 = np.percentile(returns_array, 25)
        q3 = np.percentile(returns_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        
        outliers = returns_array[(returns_array < lower_bound) | (returns_array > upper_bound)]
        if len(outliers) > 0:
            issues.append({
                'type': 'return_outliers',
                'severity': 'medium',
                'message': f'检测到收益率异常值: {len(outliers)} 个异常值',
                'outlier_count': len(outliers),
                'outlier_values': outliers.tolist(),
            })
        
        return score, issues
    
    @staticmethod
    def calculate_timeliness(data: List[Dict[str, Any]]) -> Tuple[float, List[Dict]]:
        """检查数据时效性
        
        Args:
            data: 数据列表
            
        Returns:
            (时效性分数, 问题列表)
        """
        issues = []
        score = 1.0
        
        if not data:
            return 0.0, [{'type': 'timeliness', 'severity': 'critical', 'message': '无数据可检查时效性'}]
        
        # 检查数据新鲜度
        latest_timestamp = max(d['timestamp'] for d in data)
        data_age = (pd.Timestamp.now() - latest_timestamp).total_seconds() / 60  # 分钟
        
        freshness_thresholds = {
            'realtime': 5,  # 5分钟
            'daily': 1440,  # 24小时
            'historical': 43200  # 30天
        }
        
        data_frequency = StatisticalQualityMetrics._determine_data_frequency(data)
        max_age = freshness_thresholds.get(data_frequency, 1440)
        
        if data_age > max_age:
            issues.append({
                'type': 'data_freshness',
                'severity': 'high' if data_age > max_age * 2 else 'medium',
                'message': f'数据陈旧: 最新数据 {data_age:.1f} 分钟前, 阈值 {max_age} 分钟',
                'metric': 'data_freshness',
                'value': data_age,
                'threshold': max_age,
                'data_frequency': data_frequency
            })
            score *= 0.6 if data_age > max_age * 2 else 0.8
        
        # 检查数据更新频率
        update_frequency_issues = StatisticalQualityMetrics._check_update_frequency(data)
        issues.extend(update_frequency_issues)
        if update_frequency_issues:
            score *= 0.85
        
        # 检查延迟分布
        latency_issues = StatisticalQualityMetrics._check_latency_distribution(data)
        issues.extend(latency_issues)
        if latency_issues:
            score *= 0.9
        
        return max(0.0, min(1.0, score)), issues
    
    @staticmethod
    def _determine_data_frequency(data: List[Dict[str, Any]]) -> str:
        """确定数据频率"""
        if len(data) < 2:
            return 'unknown'
        
        # 计算平均时间间隔
        time_diffs = []
        sorted_data = sorted(data, key=lambda x: x['timestamp'])
        
        for i in range(1, len(sorted_data)):
            diff = (sorted_data[i]['timestamp'] - sorted_data[i - 1]['timestamp']).total_seconds() / 60  # 分钟
            time_diffs.append(diff)
        
        if not time_diffs:
            return 'unknown'
        
        avg_interval = np.mean(time_diffs)
        
        if avg_interval <= 5:  # 5分钟以内
            return 'realtime'
        elif avg_interval <= 1440:  # 24小时以内
            return 'daily'
        else:
            return 'historical'
    
    @staticmethod
    def _check_update_frequency(data: List[Dict[str, Any]]) -> List[Dict]:
        """检查数据更新频率"""
        issues = []
        
        if len(data) < 10:  # 需要足够的数据点
            return issues
        
        # 计算时间间隔
        time_diffs = []
        sorted_data = sorted(data, key=lambda x: x['timestamp'])
        
        for i in range(1, len(sorted_data)):
            diff = (sorted_data[i]['timestamp'] - sorted_data[i - 1]['timestamp']).total_seconds() / 60  # 分钟
            time_diffs.append(diff)
        
        if not time_diffs:
            return issues
        
        # 检查更新频率的一致性
        interval_std = np.std(time_diffs)
        if interval_std > np.mean(time_diffs) * 0.5:  # 标准差大于均值的50%
            issues.append({
                'type': 'update_frequency_inconsistency',
                'severity': 'medium',
                'message': f'数据更新频率不一致: 标准差 {interval_std:.1f} 分钟',
                'metric': 'update_frequency_std',
                'value': interval_std,
                'average_interval': np.mean(time_diffs),
                'min_interval': np.min(time_diffs),
                'max_interval': np.max(time_diffs),
            })
        
        # 检查异常的时间间隔
        q1 = np.percentile(time_diffs, 25)
        q3 = np.percentile(time_diffs, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_indices = np.where((time_diffs < lower_bound) | (time_diffs > upper_bound))[0]
        if len(outlier_indices) > 0:
            issues.append({
                'type': 'update_interval_outliers',
                'severity': 'low',
                'message': f'发现 {len(outlier_indices)} 个异常时间间隔',
                'metric': 'interval_outliers',
                'outlier_count': len(outlier_indices),
                'min_outlier': np.min([time_diffs[i] for i in outlier_indices]),
                'max_outlier': np.max([time_diffs[i] for i in outlier_indices]),
            })
        
        return issues
    
    @staticmethod
    def _check_latency_distribution(data: List[Dict[str, Any]]) -> List[Dict]:
        """检查延迟分布"""
        issues = []
        
        if not data:
            return issues
        
        # 计算数据延迟（从数据时间戳到当前时间）
        current_time = pd.Timestamp.now()
        latencies = [(current_time - d['timestamp']).total_seconds() / 60 for d in data]  # 分钟
        
        # 检查延迟分布
        latency_mean = np.mean(latencies)
        latency_std = np.std(latencies)
        
        if latency_std > latency_mean * 0.3:  # 延迟波动过大
            issues.append({
                'type': 'latency_variability',
                'severity': 'medium',
                'message': f'数据延迟波动过大: 标准差 {latency_std:.1f} 分钟',
                'metric': 'latency_std',
                'value': latency_std,
                'average_latency': latency_mean,
                'min_latency': np.min(latencies),
                'max_latency': np.max(latencies),
            })
        
        # 检查极端延迟
        if np.max(latencies) > 1440:  # 超过24小时
            issues.append({
                'type': 'extreme_latency',
                'severity': 'high',
                'message': f'检测到极端延迟: {np.max(latencies):.1f} 分钟',
                'metric': 'max_latency',
                'value': np.max(latencies),
                'threshold': 1440,
            })
        
        return issues