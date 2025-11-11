"""
数据质量指标 - 业务层
从 core_bak/data_fetcher.py 拆分
职责: 计算各类数据质量指标
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("DeepSeekQuant.DataQualityMetrics")


class DataQualityMetrics:
    """数据质量指标计算器"""

    @staticmethod
    def calculate_completeness(df: pd.DataFrame) -> float:
        """计算完整性"""
        return 1.0 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])

                    'type': 'time_gap_inconsistency',
                    'severity': 'medium',
                    'message': f'时间间隔异常: {time_gap:.1f}小时 (预期: {expected_interval:.1f}小时)',
                    'symbol': sorted_data[i].symbol,
                    'start_time': sorted_data[i - 1].timestamp.isoformat(),
                    'end_time': sorted_data[i].timestamp.isoformat(),
                    'gap_hours': time_gap,
                    'expected_interval': expected_interval
                })

        # 检查价格序列的平滑性
        price_changes = []
        for i in range(1, len(sorted_data)):
            price_change = abs(sorted_data[i].close - sorted_data[i - 1].close) / sorted_data[i - 1].close
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
                        'symbol': sorted_data[i + 1].symbol,
                        'timestamp': sorted_data[i + 1].timestamp.isoformat(),
                        'price_change': change,
                        'average_change': avg_change,
                        'std_dev': std_change
                    })

        return issues

    def _get_expected_interval(self, data: List[MarketData]) -> float:
        """获取预期的时间间隔"""
        if len(data) < 2:
            return 24.0  # 默认24小时

        intervals = []
        for i in range(1, len(data)):
            gap = (data[i].timestamp - data[i - 1].timestamp).total_seconds() / 3600
            intervals.append(gap)

        # 使用中位数作为预期间隔
        return float(np.median(intervals)) if intervals else 24.0

    def _check_cross_symbol_consistency(self, data: List[MarketData]) -> List[Dict]:
        """检查跨符号一致性"""
        issues = []

        if len(data) < 10:  # 需要足够的数据点
            return issues

        # 按符号分组
        symbols = {}
        for point in data:
            if point.symbol not in symbols:
                symbols[point.symbol] = []
            symbols[point.symbol].append(point)

        # 检查相关符号之间的价格关系
        correlated_symbols = self._find_correlated_symbols(symbols.keys())

        for sym1, sym2 in correlated_symbols:
            if sym1 in symbols and sym2 in symbols:
                correlation_issues = self._check_symbol_correlation(
                    symbols[sym1], symbols[sym2], sym1, sym2
                )
                issues.extend(correlation_issues)

        return issues

    def _find_correlated_symbols(self, symbols: List[str]) -> List[Tuple[str, str]]:
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

    def _check_symbol_correlation(self, data1: List[MarketData], data2: List[MarketData],
                                  sym1: str, sym2: str) -> List[Dict]:
        """检查两个符号的相关性"""
        issues = []

        # 对齐时间戳
        aligned_data = self._align_time_series(data1, data2)
        if not aligned_data:
            return issues

        prices1 = [d.close for d in aligned_data[sym1]]
        prices2 = [d.close for d in aligned_data[sym2]]
        timestamps = [d.timestamp for d in aligned_data[sym1]]

        # 计算相关性
        correlation = np.corrcoef(prices1, prices2)[0, 1]

        # 检查相关性是否异常低
        expected_correlation = 0.7  # 预期相关性阈值
        if correlation < expected_correlation - 0.3:  # 低于预期0.3
            issues.append({
                'type': 'cross_symbol_correlation',
                'severity': 'medium',
                'message': f'符号相关性异常低: {sym1}-{sym2} 相关性 {correlation:.2f}',
                'symbol1': sym1,
                'symbol2': sym2,
                'correlation': correlation,
                'expected_correlation': expected_correlation,
                'timestamp': timestamps[-1].isoformat() if timestamps else 'unknown'
            })

        # 检查价格比率异常
        price_ratios = [p1 / p2 for p1, p2 in zip(prices1, prices2)]
        ratio_std = np.std(price_ratios)

        if ratio_std > 0.2:  # 比率波动过大
            issues.append({
                'type': 'price_ratio_volatility',
                'severity': 'low',
                'message': f'价格比率波动过大: {sym1}/{sym2} 标准差 {ratio_std:.3f}',
                'symbol1': sym1,
                'symbol2': sym2,
                'ratio_std': ratio_std,
                'timestamp': timestamps[-1].isoformat() if timestamps else 'unknown'
            })

        return issues

    def _align_time_series(self, data1: List[MarketData], data2: List[MarketData]) -> Dict[str, List[MarketData]]:
        """对齐两个时间序列"""
        # 创建时间戳到数据的映射
        time_map1 = {d.timestamp: d for d in data1}
        time_map2 = {d.timestamp: d for d in data2}

        # 找到共同的时间戳
        common_times = sorted(set(time_map1.keys()) & set(time_map2.keys()))

        if not common_times:
            return {}

        return {
            'symbol1': [time_map1[t] for t in common_times],
            'symbol2': [time_map2[t] for t in common_times]
        }

    def _check_distribution_consistency(self, data: List[MarketData]) -> List[Dict]:
        """检查统计分布一致性"""
        issues = []

        if len(data) < 20:  # 需要足够的数据点
            return issues

        # 按符号分组
        symbols = {}
        for point in data:
            if point.symbol not in symbols:
                symbols[point.symbol] = []
            symbols[point.symbol].append(point)

        # 检查每个符号的收益率分布
        for symbol, symbol_data in symbols.items():
            if len(symbol_data) >= 20:
                returns = self._calculate_returns(symbol_data)
                distribution_issues = self._check_return_distribution(returns, symbol)
                issues.extend(distribution_issues)

        return issues

    def _calculate_returns(self, data: List[MarketData]) -> List[float]:
        """计算收益率序列"""
        returns = []
        for i in range(1, len(data)):
            ret = (data[i].close - data[i - 1].close) / data[i - 1].close
            returns.append(ret)
        return returns

    def _check_return_distribution(self, returns: List[float], symbol: str) -> List[Dict]:
        """检查收益率分布"""
        issues = []

        if len(returns) < 10:
            return issues

        # 检查正态性（使用Jarque-Bera检验）
        try:
            from scipy.stats import jarque_bera
            jb_stat, jb_pvalue = jarque_bera(returns)

            if jb_pvalue < 0.05:  # 非正态分布
                issues.append({
                    'type': 'return_distribution_non_normal',
                    'severity': 'low',
                    'message': f'收益率分布非正态: JB统计量 {jb_stat:.2f}, p值 {jb_pvalue:.3f}',
                    'symbol': symbol,
                    'jb_statistic': jb_stat,
                    'jb_pvalue': jb_pvalue,
                    'timestamp': datetime.now().isoformat()
                })
        except ImportError:
            pass  # 如果没有scipy，跳过这个检查

        # 检查异常值
        returns_array = np.array(returns)
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
                'symbol': symbol,
                'outlier_count': len(outliers),
                'outlier_values': outliers.tolist(),
                'timestamp': datetime.now().isoformat()
            })

        return issues

    def _check_timeliness(self, data: List[MarketData]) -> Tuple[float, List[Dict]]:
        """检查数据时效性"""
        issues = []
        score = 1.0

        if not data:
            return 0.0, [{'type': 'timeliness', 'severity': 'critical', 'message': '无数据可检查时效性'}]

        # 检查数据新鲜度
        latest_timestamp = max(d.timestamp for d in data)
        data_age = (datetime.now() - latest_timestamp).total_seconds() / 60  # 分钟

        freshness_thresholds = {
            'realtime': 5,  # 5分钟
            'daily': 1440,  # 24小时
            'historical': 43200  # 30天
        }

        data_frequency = self._determine_data_frequency(data)
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
        update_frequency_issues = self._check_update_frequency(data)
        issues.extend(update_frequency_issues)
        if update_frequency_issues:
            score *= 0.85

        # 检查延迟分布
        latency_issues = self._check_latency_distribution(data)
        issues.extend(latency_issues)
        if latency_issues:
            score *= 0.9

        return max(0.0, min(1.0, score)), issues

    def _determine_data_frequency(self, data: List[MarketData]) -> str:
        """确定数据频率"""
        if len(data) < 2:
            return 'unknown'

        # 计算平均时间间隔
        time_diffs = []
        sorted_data = sorted(data, key=lambda x: x.timestamp)

        for i in range(1, len(sorted_data)):
            diff = (sorted_data[i].timestamp - sorted_data[i - 1].timestamp).total_seconds() / 60  # 分钟
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

    def _check_update_frequency(self, data: List[MarketData]) -> List[Dict]:
        """检查数据更新频率"""
        issues = []

        if len(data) < 10:  # 需要足够的数据点
            return issues

        # 计算时间间隔
        time_diffs = []
        sorted_data = sorted(data, key=lambda x: x.timestamp)

        for i in range(1, len(sorted_data)):
            diff = (sorted_data[i].timestamp - sorted_data[i - 1].timestamp).total_seconds() / 60  # 分钟
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
                'timestamp': sorted_data[-1].timestamp.isoformat()
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
                'timestamp': sorted_data[-1].timestamp.isoformat()
            })

        return issues

    def _check_latency_distribution(self, data: List[MarketData]) -> List[Dict]:
        """检查延迟分布"""
        issues = []

        if not data:
            return issues

        # 计算数据延迟（从数据时间戳到当前时间）
        current_time = datetime.now()
        latencies = [(current_time - d.timestamp).total_seconds() / 60 for d in data]  # 分钟

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
                'timestamp': current_time.isoformat()
            })

        # 检查极端延迟
        if np.max(latencies) > 1440:  # 超过24小时
            issues.append({
                'type': 'extreme_latency',
                'severity': 'high',
                'message': f'发现极端数据延迟: 最大延迟 {np.max(latencies):.1f} 分钟',
                'metric': 'max_latency',
                'value': np.max(latencies),
                'threshold': 1440,
                'timestamp': current_time.isoformat()
            })

        return issues

    def _detect_anomalies(self, data: List[MarketData]) -> List[Dict]:
        """检测数据异常"""
        anomalies = []

        if not data:
            return anomalies

        # 统计异常检测
        statistical_anomalies = self._detect_statistical_anomalies(data)
        anomalies.extend(statistical_anomalies)

        # 时间序列异常检测
        timeseries_anomalies = self._detect_timeseries_anomalies(data)
        anomalies.extend(timeseries_anomalies)

        # 模式异常检测
        pattern_anomalies = self._detect_pattern_anomalies(data)
        anomalies.extend(pattern_anomalies)

        # 基于机器学习的异常检测
        if self.anomaly_detector['statistical_methods']['isolation_forest']['enabled']:
            ml_anomalies = self._detect_ml_anomalies(data)
            anomalies.extend(ml_anomalies)

        return anomalies

    def _detect_statistical_anomalies(self, data: List[MarketData]) -> List[Dict]:
        """使用统计方法检测异常"""
        anomalies = []

        if len(data) < 10:  # 需要足够的数据点
            return anomalies

        # 提取价格和成交量序列
        prices = np.array([d.close for d in data])
        volumes = np.array([d.volume for d in data])

        # Z-score 异常检测
        if self.anomaly_detector['statistical_methods']['z_score']['enabled']:
            z_threshold = self.anomaly_detector['statistical_methods']['z_score']['threshold']

            price_z_scores = np.abs((prices - np.mean(prices)) / np.std(prices))
            volume_z_scores = np.abs((volumes - np.mean(volumes)) / np.std(volumes))

            for i, (price_z, volume_z) in enumerate(zip(price_z_scores, volume_z_scores)):
                if price_z > z_threshold:
                    anomalies.append({
                        'type': 'statistical_anomaly',
                        'method': 'z_score',
                        'severity': 'high' if price_z > z_threshold * 2 else 'medium',
                        'message': f'价格Z-score异常: {price_z:.2f} (阈值: {z_threshold})',
                        'timestamp': data[i].timestamp.isoformat(),
                        'symbol': data[i].symbol,
                        'metric': 'price_z_score',
                        'value': float(price_z),
                        'threshold': z_threshold
                    })

                if volume_z > z_threshold:
                    anomalies.append({
                        'type': 'statistical_anomaly',
                        'method': 'z_score',
                        'severity': 'high' if volume_z > z_threshold * 2 else 'medium',
                        'message': f'成交量Z-score异常: {volume_z:.2f} (阈值: {z_threshold})',
                        'timestamp': data[i].timestamp.isoformat(),
                        'symbol': data[i].symbol,
                        'metric': 'volume_z_score',
                        'value': float(volume_z),
                        'threshold': z_threshold
                    })

        # IQR 异常检测
        if self.anomaly_detector['statistical_methods']['iqr']['enabled']:
            iqr_multiplier = self.anomaly_detector['statistical_methods']['iqr']['multiplier']

            price_q1 = np.percentile(prices, 25)
            price_q3 = np.percentile(prices, 75)
            price_iqr = price_q3 - price_q1
            price_lower_bound = price_q1 - iqr_multiplier * price_iqr
            price_upper_bound = price_q3 + iqr_multiplier * price_iqr

            volume_q1 = np.percentile(volumes, 25)
            volume_q3 = np.percentile(volumes, 75)
            volume_iqr = volume_q3 - volume_q1
            volume_lower_bound = volume_q1 - iqr_multiplier * volume_iqr
            volume_upper_bound = volume_q3 + iqr_multiplier * volume_iqr

            for i, data_point in enumerate(data):
                if data_point.close < price_lower_bound or data_point.close > price_upper_bound:
                    anomalies.append({
                        'type': 'statistical_anomaly',
                        'method': 'iqr',
                        'severity': 'medium',
                        'message': f'价格IQR异常: {data_point.close:.2f} (范围: [{price_lower_bound:.2f}, {price_upper_bound:.2f}])',
                        'timestamp': data_point.timestamp.isoformat(),
                        'symbol': data_point.symbol,
                        'metric': 'price_iqr',
                        'value': data_point.close,
                        'lower_bound': price_lower_bound,
                        'upper_bound': price_upper_bound
                    })

                if data_point.volume < volume_lower_bound or data_point.volume > volume_upper_bound:
                    anomalies.append({
                        'type': 'statistical_anomaly',
                        'method': 'iqr',
                        'severity': 'medium',
                        'message': f'成交量IQR异常: {data_point.volume:.0f} (范围: [{volume_lower_bound:.0f}, {volume_upper_bound:.0f}])',
                        'timestamp': data_point.timestamp.isoformat(),
                        'symbol': data_point.symbol,
                        'metric': 'volume_iqr',
                        'value': data_point.volume,
                        'lower_bound': volume_lower_bound,
                        'upper_bound': volume_upper_bound
                    })

        # 滚动标准差异常检测
        if self.anomaly_detector['statistical_methods']['rolling_std']['enabled']:
            window = self.anomaly_detector['statistical_methods']['rolling_std']['window']
            multiplier = self.anomaly_detector['statistical_methods']['rolling_std']['multiplier']

            if len(prices) >= window:
                rolling_mean = pd.Series(prices).rolling(window=window).mean()
                rolling_std = pd.Series(prices).rolling(window=window).std()

                for i in range(window, len(prices)):
                    if rolling_std.iloc[i] > 0:  # 避免除零
                        z_score = abs(prices[i] - rolling_mean.iloc[i]) / rolling_std.iloc[i]
                        if z_score > multiplier:
                            anomalies.append({
                                'type': 'statistical_anomaly',
                                'method': 'rolling_std',
                                'severity': 'medium',
                                'message': f'滚动标准差异常: z-score {z_score:.2f}',
                                'timestamp': data[i].timestamp.isoformat(),
                                'symbol': data[i].symbol,
                                'metric': 'rolling_z_score',
                                'value': z_score,
                                'threshold': multiplier,
                                'window_size': window
                            })

        return anomalies

    def _detect_timeseries_anomalies(self, data: List[MarketData]) -> List[Dict]:
        """检测时间序列异常"""
        anomalies = []

        if len(data) < 20:  # 需要足够的时间序列数据
            return anomalies

        # 检测缺失数据点
        if self.anomaly_detector['temporal_patterns']['missing_data']['enabled']:
            missing_data_anomalies = self._detect_missing_data(data)
            anomalies.extend(missing_data_anomalies)

        # 检测季节性异常
        if self.anomaly_detector['temporal_patterns']['seasonality']['detect_seasonality']:
            seasonality_anomalies = self._detect_seasonality_anomalies(data)
            anomalies.extend(seasonality_anomalies)

        # 检测趋势断裂
        if self.anomaly_detector['temporal_patterns']['trend_breaks']['detect_breaks']:
            trend_break_anomalies = self._detect_trend_breaks(data)
            anomalies.extend(trend_break_anomalies)

        return anomalies

    def _detect_missing_data(self, data: List[MarketData]) -> List[Dict]:
        """检测缺失数据"""
        anomalies = []
        max_consecutive_missing = self.anomaly_detector['temporal_patterns']['missing_data']['max_consecutive_missing']

        # 按时间排序
        sorted_data = sorted(data, key=lambda x: x.timestamp)

        # 检查时间间隔
        for i in range(1, len(sorted_data)):
            time_gap = (sorted_data[i].timestamp - sorted_data[i - 1].timestamp).total_seconds() / 3600  # 小时

            # 根据数据频率确定预期间隔
            expected_interval = self._get_expected_interval(sorted_data)

            if time_gap > expected_interval * max_consecutive_missing:
                anomalies.append({
                    'type': 'timeseries_anomaly',
                    'method': 'missing_data',
                    'severity': 'medium',
                    'message': f'检测到数据缺失: 时间间隔 {time_gap:.1f} 小时, 预期 {expected_interval:.1f} 小时',
                    'start_time': sorted_data[i - 1].timestamp.isoformat(),
                    'end_time': sorted_data[i].timestamp.isoformat(),
                    'gap_hours': time_gap,
                    'expected_interval': expected_interval,
                    'symbol': sorted_data[i].symbol
                })

        return anomalies

    def _detect_seasonality_anomalies(self, data: List[MarketData]) -> List[Dict]:
        """检测季节性异常"""
        anomalies = []

        try:
            # 使用STL分解检测季节性异常
            from statsmodels.tsa.seasonal import STL

            prices = [d.close for d in data]
            timestamps = [d.timestamp for d in data]

            # 创建时间序列索引
            if len(data) >= 100:  # 需要足够的数据进行季节性分析
                stl = STL(prices, period=20)  # 假设20个点的周期
                result = stl.fit()

                # 检测季节性残差异常
                seasonal_residual = result.resid
                seasonal_mean = np.mean(seasonal_residual)
                seasonal_std = np.std(seasonal_residual)

                for i, residual in enumerate(seasonal_residual):
                    if abs(residual - seasonal_mean) > 3 * seasonal_std:
                        anomalies.append({
                            'type': 'timeseries_anomaly',
                            'method': 'seasonality',
                            'severity': 'low',
                            'message': f'季节性异常: 残差 {residual:.4f}',
                            'timestamp': timestamps[i].isoformat(),
                            'symbol': data[i].symbol,
                            'value': residual,
                            'mean': seasonal_mean,
                            'std': seasonal_std
                        })

        except Exception as e:
            logger.warning(f"季节性异常检测失败: {e}")

        return anomalies

    def _detect_trend_breaks(self, data: List[MarketData]) -> List[Dict]:
        """检测趋势断裂"""
        anomalies = []

        try:
            prices = [d.close for d in data]
            timestamps = [d.timestamp for d in data]

            if len(prices) >= 50:  # 需要足够的数据点
                # 使用CUSUM算法检测均值变化
                from statsmodels.tsa.stattools import cusum_squares

                # 计算CUSUM统计量
                cusum_stat = cusum_squares(prices)

                # 检测显著的变化点
                threshold = 1.0  # 经验阈值
                change_points = np.where(np.abs(cusum_stat) > threshold)[0]

                for cp in change_points:
                    if 0 < cp < len(prices) - 1:
                        anomalies.append({
                            'type': 'timeseries_anomaly',
                            'method': 'trend_break',
                            'severity': 'medium',
                            'message': f'检测到趋势断裂点',
                            'timestamp': timestamps[cp].isoformat(),
                            'symbol': data[cp].symbol,
                            'cusum_statistic': float(cusum_stat[cp]),
                            'threshold': threshold,
                            'position': cp
                        })

        except Exception as e:
            logger.warning(f"趋势断裂检测失败: {e}")

        return anomalies

    def _detect_pattern_anomalies(self, data: List[MarketData]) -> List[Dict]:
        """检测模式异常"""
        anomalies = []

        # 检测重复数据模式
        duplicate_patterns = self._detect_duplicate_patterns(data)
        anomalies.extend(duplicate_patterns)

        # 检测异常波动模式
        volatility_patterns = self._detect_volatility_patterns(data)
        anomalies.extend(volatility_patterns)

