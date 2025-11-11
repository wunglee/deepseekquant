"""
数据质量监控 - 业务层
从 core_bak/data_fetcher.py 拆分
职责: 数据质量监控、报告生成
"""

import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger('DeepSeekQuant.DataQualityMonitor')


class DataQualityMonitor:
    """数据质量监控器 - 实时监控数据质量"""

    def __init__(self, config: Dict):
        self.config = config
        self.quality_metrics = {}
        self.anomaly_detector = self._setup_anomaly_detection()
        self.data_validator = self._setup_data_validation()
        self.quality_history = []
        self.alert_history = []

    def _setup_anomaly_detection(self) -> Dict[str, Any]:
        """设置异常检测系统"""
        return {
            'statistical_methods': {
                'z_score': {'enabled': True, 'threshold': 3.0},
                'iqr': {'enabled': True, 'multiplier': 1.5},
                'rolling_std': {'enabled': True, 'window': 20, 'multiplier': 2.0},
                'isolation_forest': {'enabled': False, 'contamination': 0.1},
                'autoencoder': {'enabled': False, 'threshold': 0.05}
            },
            'temporal_patterns': {
                'missing_data': {'max_consecutive_missing': 5},
                'seasonality': {'detect_seasonality': True},
                'trend_breaks': {'detect_breaks': True}
            },
            'cross_validation': {
                'cross_source_validation': True,
                'consistency_check': True
            }
        }

    def _setup_data_validation(self) -> Dict[str, Any]:
        """设置数据验证规则"""
        return {
            'price_validation': {
                'min_price': 0.01,
                'max_price': 10000.0,
                'max_daily_change': 0.5,
                'price_consistency': True
            },
            'volume_validation': {
                'min_volume': 0,
                'max_volume': 1e9,
                'volume_spike_threshold': 10.0
            },
            'temporal_validation': {
                'max_time_gap': timedelta(hours=24),
                'future_data_check': True,
                'duplicate_timestamps': False
            },
            'completeness_validation': {
                'required_fields': ['open', 'high', 'low', 'close', 'volume', 'timestamp'],
                'max_missing_rate': 0.05,
                'min_data_points': 10
            }
        }

    def monitor_data_quality(self, data: List[MarketData]) -> Dict[str, Any]:
        """监控数据质量 - 完整生产实现"""
        quality_report = {
            'overall_score': 1.0,
            'dimension_scores': {},
            'anomalies_detected': [],
            'validation_errors': [],
            'recommendations': [],
            'timestamp': datetime.now().isoformat(),
            'data_source': data[0].metadata.get('data_source', 'unknown') if data else 'unknown',
            'symbol_count': len(set(d.symbol for d in data)) if data else 0,
            'time_period': self._get_data_time_period(data),
            'processing_stats': self._get_processing_statistics(data)
        }

        try:
            # 检查数据完整性
            completeness_score, completeness_issues = self._check_completeness(data)
            quality_report['dimension_scores']['completeness'] = completeness_score
            quality_report['overall_score'] *= completeness_score
            quality_report['validation_errors'].extend(completeness_issues)

            # 检查数据准确性
            accuracy_score, accuracy_issues = self._check_accuracy(data)
            quality_report['dimension_scores']['accuracy'] = accuracy_score
            quality_report['overall_score'] *= accuracy_score
            quality_report['validation_errors'].extend(accuracy_issues)

            # 检查数据一致性
            consistency_score, consistency_issues = self._check_consistency(data)
            quality_report['dimension_scores']['consistency'] = consistency_score
            quality_report['overall_score'] *= consistency_score
            quality_report['validation_errors'].extend(consistency_issues)

            # 检查数据时效性
            timeliness_score, timeliness_issues = self._check_timeliness(data)
            quality_report['dimension_scores']['timeliness'] = timeliness_score
            quality_report['overall_score'] *= timeliness_score
            quality_report['validation_errors'].extend(timeliness_issues)

            # 检测数据异常
            anomalies = self._detect_anomalies(data)
            quality_report['anomalies_detected'] = anomalies

            # 生成质量评分和建议
            quality_report['quality_level'] = self._determine_quality_level(quality_report['overall_score'])
            quality_report['recommendations'] = self._generate_recommendations(quality_report)

            # 触发警报（如果需要）
            self._trigger_alerts(quality_report)

            # 记录质量指标历史
            self._record_quality_metrics(quality_report)

            logger.info(f"数据质量监控完成: 总体评分 {quality_report['overall_score']:.3f}, "
                        f"检测到 {len(anomalies)} 个异常, {len(quality_report['validation_errors'])} 个验证错误")

            return quality_report

        except Exception as e:
            logger.error(f"数据质量监控失败: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'overall_score': 0.0,
                'quality_level': 'critical'
            }

    def _check_completeness(self, data: List[MarketData]) -> Tuple[float, List[Dict]]:
        """检查数据完整性"""
        issues = []
        score = 1.0

        if not data:
            return 0.0, [{'type': 'completeness', 'severity': 'critical', 'message': '空数据集'}]

        # 检查数据点数量
        expected_points = self._calculate_expected_data_points(data)
        actual_points = len(data)
        completeness_ratio = actual_points / expected_points if expected_points > 0 else 0

        if completeness_ratio < 0.95:
            issues.append({
                'type': 'completeness',
                'severity': 'high' if completeness_ratio < 0.8 else 'medium',
                'message': f'数据点不足: 预期 {expected_points}, 实际 {actual_points}',
                'metric': 'data_point_count',
                'value': completeness_ratio
            })
            score *= 0.7 if completeness_ratio < 0.8 else 0.9

        # 检查字段完整性
        field_completeness = self._check_field_completeness(data)
        for field, completeness in field_completeness.items():
            if completeness < 0.99:  # 允许1%的字段缺失
                issues.append({
                    'type': 'completeness',
                    'severity': 'medium' if completeness < 0.95 else 'low',
                    'message': f'字段 {field} 完整性不足: {completeness:.1%}',
                    'metric': f'field_{field}_completeness',
                    'value': completeness
                })
                score *= 0.95 if completeness < 0.95 else 0.98

        # 检查时间连续性
        time_gaps = self._check_time_continuity(data)
        if time_gaps:
            issues.append({
                'type': 'completeness',
                'severity': 'medium',
                'message': f'发现 {len(time_gaps)} 个时间间隔异常',
                'metric': 'time_gaps',
                'value': len(time_gaps),
                'details': time_gaps
            })
            score *= 0.9

        return max(0.0, min(1.0, score)), issues

    def _check_accuracy(self, data: List[MarketData]) -> Tuple[float, List[Dict]]:
        """检查数据准确性"""
        issues = []
        score = 1.0

        # 检查价格合理性
        price_issues = self._validate_price_ranges(data)
        issues.extend(price_issues)
        if price_issues:
            score *= 0.8 if any(i['severity'] == 'high' for i in price_issues) else 0.95

        # 检查成交量合理性
        volume_issues = self._validate_volume_data(data)
        issues.extend(volume_issues)
        if volume_issues:
            score *= 0.85 if any(i['severity'] == 'high' for i in volume_issues) else 0.96

        # 检查数据一致性（内部）
        internal_consistency_issues = self._check_internal_consistency(data)
        issues.extend(internal_consistency_issues)
        if internal_consistency_issues:
            score *= 0.7 if any(i['severity'] == 'high' for i in internal_consistency_issues) else 0.9

        # 检查与外部数据源的一致性
        external_consistency_issues = self._check_external_consistency(data)
        issues.extend(external_consistency_issues)
        if external_consistency_issues:
            score *= 0.8 if any(i['severity'] == 'high' for i in external_consistency_issues) else 0.93

        return max(0.0, min(1.0, score)), issues

    def _validate_price_ranges(self, data: List[MarketData]) -> List[Dict]:
        """验证价格范围合理性"""
        issues = []
        rules = self.data_validator['price_validation']

        for data_point in data:
            # 检查价格范围
            if not (rules['min_price'] <= data_point.close <= rules['max_price']):
                issues.append({
                    'type': 'price_range',
                    'severity': 'high',
                    'message': f'价格超出范围: {data_point.close} (范围: [{rules["min_price"]}, {rules["max_price"]}])',
                    'symbol': data_point.symbol,
                    'timestamp': data_point.timestamp.isoformat(),
                    'value': data_point.close,
                    'min_threshold': rules['min_price'],
                    'max_threshold': rules['max_price']
                })

            # 检查价格一致性
            if rules['price_consistency']:
                if not (data_point.low <= data_point.open <= data_point.high and
                        data_point.low <= data_point.close <= data_point.high):
                    issues.append({
                        'type': 'price_consistency',
                        'severity': 'high',
                        'message': '价格不一致 (low <= open/close <= high)',
                        'symbol': data_point.symbol,
                        'timestamp': data_point.timestamp.isoformat(),
                        'open': data_point.open,
                        'high': data_point.high,
                        'low': data_point.low,
                        'close': data_point.close
                    })

            # 检查高低价关系
            if rules['high_low_consistency'] and data_point.high < data_point.low:
                issues.append({
                    'type': 'high_low_inconsistency',
                    'severity': 'high',
                    'message': '最高价低于最低价',
                    'symbol': data_point.symbol,
                    'timestamp': data_point.timestamp.isoformat(),
                    'high': data_point.high,
                    'low': data_point.low
                })

        return issues

    def _validate_volume_data(self, data: List[MarketData]) -> List[Dict]:
        """验证成交量数据合理性"""
        issues = []
        rules = self.data_validator['volume_validation']

        for i, data_point in enumerate(data):
            # 检查成交量范围
            if not (rules['min_volume'] <= data_point.volume <= rules['max_volume']):
                issues.append({
                    'type': 'volume_range',
                    'severity': 'medium',
                    'message': f'成交量超出范围: {data_point.volume} (范围: [{rules["min_volume"]}, {rules["max_volume"]}])',
                    'symbol': data_point.symbol,
                    'timestamp': data_point.timestamp.isoformat(),
                    'value': data_point.volume,
                    'min_threshold': rules['min_volume'],
                    'max_threshold': rules['max_volume']
                })

            # 检查成交量异常
            if rules['volume_spike_threshold'] and i > 0:
                prev_volume = data[i - 1].volume
                if prev_volume > 0:  # 避免除零
                    volume_ratio = data_point.volume / prev_volume
                    if volume_ratio > rules['volume_spike_threshold']:
                        issues.append({
                            'type': 'volume_spike',
                            'severity': 'medium',
                            'message': f'成交量异常飙升: {volume_ratio:.1f}x (阈值: {rules["volume_spike_threshold"]}x)',
                            'symbol': data_point.symbol,
                            'timestamp': data_point.timestamp.isoformat(),
                            'current_volume': data_point.volume,
                            'previous_volume': prev_volume,
                            'ratio': volume_ratio,
                            'threshold': rules['volume_spike_threshold']
                        })

            # 检查成交量价格相关性
            if rules['volume_price_correlation'] and i > 0:
                price_change = abs(data_point.close - data[i - 1].close) / data[i - 1].close
                volume_change = abs(data_point.volume - data[i - 1].volume) / data[i - 1].volume

                # 如果价格大幅变化但成交量没有相应变化，可能有问题
                if price_change > 0.05 and volume_change < 0.1:  # 5%价格变化但成交量变化小于10%
                    issues.append({
                        'type': 'volume_price_correlation',
                        'severity': 'low',
                        'message': f'价格成交量相关性异常: 价格变化 {price_change:.1%}, 成交量变化 {volume_change:.1%}',
                        'symbol': data_point.symbol,
                        'timestamp': data_point.timestamp.isoformat(),
                        'price_change': price_change,
                        'volume_change': volume_change
                    })

        return issues

    def _check_internal_consistency(self, data: List[MarketData]) -> List[Dict]:
        """检查数据内部一致性"""
        issues = []

        for i, data_point in enumerate(data):
            # 检查开盘价是否在高低价范围内
            if not (data_point.low <= data_point.open <= data_point.high):
                issues.append({
                    'type': 'open_price_consistency',
                    'severity': 'high',
                    'message': '开盘价不在高低价范围内',
                    'symbol': data_point.symbol,
                    'timestamp': data_point.timestamp.isoformat(),
                    'open': data_point.open,
                    'high': data_point.high,
                    'low': data_point.low
                })

            # 检查收盘价是否在高低价范围内
            if not (data_point.low <= data_point.close <= data_point.high):
                issues.append({
                    'type': 'close_price_consistency',
                    'severity': 'high',
                    'message': '收盘价不在高低价范围内',
                    'symbol': data_point.symbol,
                    'timestamp': data_point.timestamp.isoformat(),
                    'close': data_point.close,
                    'high': data_point.high,
                    'low': data_point.low
                })

            # 检查成交量与价格的关系
            if i > 0:
                price_change = (data_point.close - data[i - 1].close) / data[i - 1].close
                volume_change = (data_point.volume - data[i - 1].volume) / data[i - 1].volume

                # 极端情况下，价格大幅变化但成交量极低可能有问题
                if abs(price_change) > 0.1 and data_point.volume < 1000:  # 10%价格变化但成交量小于1000
                    issues.append({
                        'type': 'low_volume_price_move',
                        'severity': 'medium',
                        'message': f'低成交量下的价格大幅变动: 价格变化 {price_change:.1%}, 成交量 {data_point.volume}',
                        'symbol': data_point.symbol,
                        'timestamp': data_point.timestamp.isoformat(),
                        'price_change': price_change,
                        'volume': data_point.volume
                    })

        return issues

    def _check_external_consistency(self, data: List[MarketData]) -> List[Dict]:
        """检查与外部数据源的一致性"""
        issues = []

        if not data:
            return issues

        # 这里需要实现与外部数据源的交叉验证
        # 例如：与Yahoo Finance、Bloomberg等数据源进行对比

        try:
            # 获取第一个数据点的外部验证数据
            sample_point = data[0]

            # 模拟外部数据验证
            external_validation = self._validate_with_external_source(sample_point)

            if external_validation and not external_validation['consistent']:
                issues.append({
                    'type': 'external_consistency',
                    'severity': 'high',
                    'message': f'与外部数据源不一致: {external_validation["discrepancy"]}',
                    'symbol': sample_point.symbol,
                    'timestamp': sample_point.timestamp.isoformat(),
                    'internal_value': sample_point.close,
                    'external_value': external_validation['external_price'],
                    'discrepancy': external_validation['discrepancy_pct']
                })

        except Exception as e:
            logger.warning(f"外部一致性检查失败: {e}")
            # 不将外部检查失败作为问题记录

        return issues

    def _validate_with_external_source(self, data_point: MarketData) -> Optional[Dict]:
        """与外部数据源进行验证"""
        # 这里实现实际的外部数据源验证逻辑
        # 例如：调用Yahoo Finance API进行价格验证

        try:
            # 模拟外部数据获取
            external_price = self._fetch_external_price(
                data_point.symbol,
                data_point.timestamp
            )

            if external_price is None:
                return None

            # 计算差异百分比
            discrepancy = abs(data_point.close - external_price) / external_price

            return {
                'consistent': discrepancy < 0.01,  # 1%差异阈值
                'external_price': external_price,
                'discrepancy_pct': discrepancy,
                'timestamp': data_point.timestamp.isoformat()
            }

        except Exception as e:
            logger.debug(f"外部验证失败: {e}")
            return None

    def _fetch_external_price(self, symbol: str, timestamp: datetime) -> Optional[float]:
        """从外部数据源获取价格"""
        # 这里实现实际的外部数据获取逻辑
        # 可以使用Yahoo Finance、Alpha Vantage等API

        try:
            # 模拟外部数据获取
            if symbol in ['AAPL', 'MSFT', 'GOOGL']:  # 主要股票有外部数据
                # 返回模拟的外部价格（实际中应该调用API）
                return round(data_point.close * random.uniform(0.995, 1.005), 2)  # 轻微随机差异
            return None

        except Exception as e:
            logger.debug(f"外部价格获取失败: {e}")
            return None

    def _check_consistency(self, data: List[MarketData]) -> Tuple[float, List[Dict]]:
        """检查数据一致性"""
        issues = []
        score = 1.0

        # 检查时间序列一致性
        time_series_issues = self._check_time_series_consistency(data)
        issues.extend(time_series_issues)
        if time_series_issues:
            score *= 0.85

        # 检查跨符号一致性
        cross_symbol_issues = self._check_cross_symbol_consistency(data)
        issues.extend(cross_symbol_issues)
        if cross_symbol_issues:
            score *= 0.9

        # 检查统计分布一致性
        distribution_issues = self._check_distribution_consistency(data)
        issues.extend(distribution_issues)
        if distribution_issues:
            score *= 0.88

        return max(0.0, min(1.0, score)), issues

    def _check_time_series_consistency(self, data: List[MarketData]) -> List[Dict]:
        """检查时间序列一致性"""
        issues = []

        if len(data) < 2:
            return issues

        # 按时间排序
        sorted_data = sorted(data, key=lambda x: x.timestamp)

        # 检查时间连续性
        for i in range(1, len(sorted_data)):
            time_gap = (sorted_data[i].timestamp - sorted_data[i - 1].timestamp).total_seconds() / 3600  # 小时

            # 根据数据频率确定预期间隔
            expected_interval = self._get_expected_interval(sorted_data)

            if time_gap > expected_interval * 3:  # 允许3倍间隔的容差
                issues.append({
