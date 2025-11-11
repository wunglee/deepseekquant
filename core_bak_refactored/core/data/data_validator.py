"""
数据验证器 - 业务层
从 core_bak/data_fetcher.py 拆分
职责: 数据质量检查、异常检测
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger('DeepSeekQuant.DataValidator')


class DataValidator:
    """数据验证器 - 验证数据的完整性和正确性"""

    def __init__(self, config: Dict):
        self.config = config
        self.validation_rules = self._setup_validation_rules()
        self.validation_history = []

    def _setup_validation_rules(self) -> Dict[str, Any]:
        """设置验证规则"""
        return {
            'price_validation': {
                'min_price': 0.001,
                'max_price': 100000,
                'max_daily_change': 2.0,  # 200%
                'price_consistency': True,
                'high_low_consistency': True
            },
            'volume_validation': {
                'min_volume': 0,
                'max_volume': 1e12,
                'volume_spike_threshold': 50.0,
                'volume_price_correlation': True
            },
            'temporal_validation': {
                'future_data_check': True,
                'duplicate_timestamps': False,
                'time_gap_threshold': timedelta(hours=24),
                'chronological_order': True
            },
            'completeness_validation': {
                'required_fields': ['open', 'high', 'low', 'close', 'volume', 'timestamp'],
                'null_value_check': True,
                'data_range_check': True
            },
            'cross_validation': {
                'cross_source_validation': True,
                'statistical_consistency': True,
                'market_consistency': True
            }
        }

    def validate_market_data(self, data: List[MarketData]) -> Dict[str, Any]:
        """验证市场数据"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'passed_tests': 0,
            'total_tests': 0,
            'validation_time': datetime.now().isoformat()
        }

        try:
            if not data:
                validation_results['valid'] = False
                validation_results['errors'].append('空数据')
                return validation_results

            # 执行各种验证
            self._validate_prices(data, validation_results)
            self._validate_volumes(data, validation_results)
            self._validate_temporal(data, validation_results)
            self._validate_completeness(data, validation_results)
            self._validate_cross_source(data, validation_results)

            # 计算通过率
            validation_results['passed_tests'] = len([test for test in validation_results.keys()
                                                      if test.startswith('test_') and validation_results[test]])
            validation_results['total_tests'] = len([key for key in validation_results.keys()
                                                     if key.startswith('test_')])

            # 记录验证历史
            self._record_validation(validation_results)

            return validation_results

        except Exception as e:
            logger.error(f"数据验证失败: {e}")
            validation_results['valid'] = False
            validation_results['errors'].append(f'验证异常: {str(e)}')
            return validation_results

    def _validate_prices(self, data: List[MarketData], results: Dict):
        """验证价格数据"""
        rules = self.validation_rules['price_validation']

        for i, data_point in enumerate(data):
            # 检查价格范围
            if not (rules['min_price'] <= data_point.close <= rules['max_price']):
                results['errors'].append({
                    'type': 'price_range',
                    'severity': 'high',
                    'message': f'价格超出范围: {data_point.close}',
                    'symbol': data_point.symbol,
                    'timestamp': data_point.timestamp.isoformat()
                })
                results['valid'] = False

            # 检查价格一致性
            if rules['price_consistency']:
                if not (data_point.low <= data_point.open <= data_point.high and
                        data_point.low <= data_point.close <= data_point.high):
                    results['errors'].append({
                        'type': 'price_consistency',
                        'severity': 'high',
                        'message': '价格不一致 (low <= open/close <= high)',
                        'symbol': data_point.symbol,
                        'timestamp': data_point.timestamp.isoformat()
                    })
                    results['valid'] = False

            # 检查日价格变化
            if rules['max_daily_change'] and i > 0:
                prev_close = data[i - 1].close
                if prev_close > 0:  # 避免除零
                    daily_change = abs(data_point.close - prev_close) / prev_close
                    if daily_change > rules['max_daily_change']:
                        results['warnings'].append({
                            'type': 'price_change',
                            'severity': 'medium',
                            'message': f'日价格变化过大: {daily_change:.1%}',
                            'symbol': data_point.symbol,
                            'timestamp': data_point.timestamp.isoformat()
                        })

        results['test_price_validation'] = len(results['errors']) == 0

    def _validate_volumes(self, data: List[MarketData], results: Dict):
        """验证成交量数据"""
        rules = self.validation_rules['volume_validation']

        for i, data_point in enumerate(data):
            # 检查成交量范围
            if not (rules['min_volume'] <= data_point.volume <= rules['max_volume']):
                results['errors'].append({
                    'type': 'volume_range',
                    'severity': 'medium',
                    'message': f'成交量超出范围: {data_point.volume}',
                    'symbol': data_point.symbol,
                    'timestamp': data_point.timestamp.isoformat()
                })
                results['valid'] = False

            # 检查成交量异常
            if rules['volume_spike_threshold'] and i > 0:
                prev_volume = data[i - 1].volume
                if prev_volume > 0:  # 避免除零
                    volume_ratio = data_point.volume / prev_volume
                    if volume_ratio > rules['volume_spike_threshold']:
                        results['warnings'].append({
                            'type': 'volume_spike',
                            'severity': 'medium',
                            'message': f'成交量异常飙升: {volume_ratio:.1f}x',
                            'symbol': data_point.symbol,
                            'timestamp': data_point.timestamp.isoformat()
                        })

        results['test_volume_validation'] = len(results['errors']) == 0

    def _validate_temporal(self, data: List[MarketData], results: Dict):
        """验证时间数据"""
        rules = self.validation_rules['temporal_validation']

        # 检查时间顺序
        if rules['chronological_order']:
            sorted_data = sorted(data, key=lambda x: x.timestamp)
            for i in range(1, len(sorted_data)):
                if sorted_data[i].timestamp < sorted_data[i - 1].timestamp:
                    results['errors'].append({
                        'type': 'temporal_order',
                        'severity': 'high',
                        'message': '时间顺序错误',
                        'symbol': sorted_data[i].symbol,
                        'timestamp': sorted_data[i].timestamp.isoformat()
                    })
                    results['valid'] = False

        # 检查重复时间戳
        if rules['duplicate_timestamps']:
            timestamps = [d.timestamp for d in data]
            if len(timestamps) != len(set(timestamps)):
                results['errors'].append({
                    'type': 'duplicate_timestamps',
                    'severity': 'medium',
                    'message': '发现重复时间戳',
                    'symbol': data[0].symbol if data else 'unknown'
                })
                results['valid'] = False

        results['test_temporal_validation'] = len(results['errors']) == 0

    def _validate_completeness(self, data: List[MarketData], results: Dict):
        """验证数据完整性 - 完整生产实现"""
        rules = self.validation_rules['completeness_validation']

        if not data:
            results['errors'].append({
                'type': 'completeness',
                'severity': 'critical',
                'message': '空数据集',
                'timestamp': datetime.now().isoformat()
            })
            results['valid'] = False
            return

        # 检查必需字段
        required_fields = rules['required_fields']
        missing_fields_by_symbol = {}

        for data_point in data:
            missing_fields = []
            for field in required_fields:
                # 检查字段是否存在且不为空
                field_value = getattr(data_point, field, None)
                if field_value is None or (isinstance(field_value, (int, float)) and np.isnan(field_value)):
                    missing_fields.append(field)

            if missing_fields:
                if data_point.symbol not in missing_fields_by_symbol:
                    missing_fields_by_symbol[data_point.symbol] = []
                missing_fields_by_symbol[data_point.symbol].extend(missing_fields)

        # 记录缺失字段错误
        for symbol, fields in missing_fields_by_symbol.items():
            results['errors'].append({
                'type': 'completeness',
                'severity': 'high',
                'message': f'符号 {symbol} 缺失必需字段: {", ".join(fields)}',
                'symbol': symbol,
                'missing_fields': fields,
                'timestamp': datetime.now().isoformat()
            })
            results['valid'] = False

        # 检查空值
        if rules['null_value_check']:
            null_values = self._check_null_values(data)
            if null_values:
                results['errors'].extend(null_values)
                results['valid'] = False

        # 检查数据范围
        if rules['data_range_check']:
            range_issues = self._check_data_ranges(data)
            if range_issues:
                results['errors'].extend(range_issues)
                results['valid'] = False

        # 检查数据点数量
        if len(data) < rules.get('min_data_points', 1):
            results['errors'].append({
                'type': 'completeness',
                'severity': 'medium',
                'message': f'数据点数量不足: {len(data)} (最少需要 {rules["min_data_points"]})',
                'data_point_count': len(data),
                'min_required': rules['min_data_points'],
                'timestamp': datetime.now().isoformat()
            })
            results['valid'] = False

        results['test_completeness_validation'] = len([
            e for e in results['errors']
            if e['type'] == 'completeness'
        ]) == 0

    def _check_null_values(self, data: List[MarketData]) -> List[Dict]:
        """检查空值和异常值"""
        null_issues = []

        for data_point in data:
            # 检查数值字段是否为NaN或无限大
            numeric_fields = ['open', 'high', 'low', 'close', 'volume']
            for field in numeric_fields:
                value = getattr(data_point, field)
                if np.isnan(value) or np.isinf(value):
                    null_issues.append({
                        'type': 'completeness',
                        'severity': 'high',
                        'message': f'字段 {field} 包含无效数值: {value}',
                        'symbol': data_point.symbol,
                        'timestamp': data_point.timestamp.isoformat(),
                        'field': field,
                        'value': value
                    })

            # 检查时间戳有效性
            if data_point.timestamp is None or data_point.timestamp > datetime.now() + timedelta(days=1):
                null_issues.append({
                    'type': 'completeness',
                    'severity': 'high',
                    'message': f'无效时间戳: {data_point.timestamp}',
                    'symbol': data_point.symbol,
                    'timestamp': data_point.timestamp.isoformat() if data_point.timestamp else 'None',
                    'field': 'timestamp'
                })

        return null_issues

    def _check_data_ranges(self, data: List[MarketData]) -> List[Dict]:
        """检查数据范围合理性"""
        range_issues = []

        for data_point in data:
            # 价格范围检查
            if not (0.001 <= data_point.close <= 1000000):  # 合理的价格范围
                range_issues.append({
                    'type': 'completeness',
                    'severity': 'high',
                    'message': f'价格超出合理范围: {data_point.close}',
                    'symbol': data_point.symbol,
                    'timestamp': data_point.timestamp.isoformat(),
                    'field': 'close',
                    'value': data_point.close,
                    'valid_range': '[0.001, 1000000]'
                })

            # 成交量范围检查
            if not (0 <= data_point.volume <= 1e12):  # 合理的成交量范围
                range_issues.append({
                    'type': 'completeness',
                    'severity': 'medium',
                    'message': f'成交量超出合理范围: {data_point.volume}',
                    'symbol': data_point.symbol,
                    'timestamp': data_point.timestamp.isoformat(),
                    'field': 'volume',
                    'value': data_point.volume,
                    'valid_range': '[0, 1e12]'
                })

            # 高低价关系检查
            if data_point.high < data_point.low:
                range_issues.append({
                    'type': 'completeness',
                    'severity': 'high',
                    'message': f'最高价低于最低价: high={data_point.high}, low={data_point.low}',
                    'symbol': data_point.symbol,
                    'timestamp': data_point.timestamp.isoformat(),
                    'field': 'high_low',
                    'high_value': data_point.high,
                    'low_value': data_point.low
                })

        return range_issues

    def _validate_cross_source(self, data: List[MarketData], results: Dict):
        """交叉验证数据源一致性"""
        rules = self.validation_rules['cross_validation']

        if not rules['cross_source_validation'] or len(data) < 2:
            results['test_cross_validation'] = True
            return

        try:
            # 按符号分组数据
            symbols_data = {}
            for point in data:
                if point.symbol not in symbols_data:
                    symbols_data[point.symbol] = []
                symbols_data[point.symbol].append(point)

            # 对每个符号进行交叉验证
            cross_validation_issues = []
            for symbol, symbol_data in symbols_data.items():
                if len(symbol_data) >= 2:  # 需要至少2个数据点进行验证
                    issues = self._validate_symbol_cross_source(symbol_data, symbol)
                    cross_validation_issues.extend(issues)

            if cross_validation_issues:
                results['errors'].extend(cross_validation_issues)
                results['valid'] = False

            results['test_cross_validation'] = len(cross_validation_issues) == 0

        except Exception as e:
            logger.error(f"交叉验证失败: {e}")
            results['errors'].append({
                'type': 'cross_validation',
                'severity': 'medium',
                'message': f'交叉验证异常: {str(e)}',
                'timestamp': datetime.now().isoformat()
            })
            results['test_cross_validation'] = False

    def _validate_symbol_cross_source(self, data: List[MarketData], symbol: str) -> List[Dict]:
        """验证单个符号的交叉源一致性"""
        issues = []

        # 按数据源分组（假设metadata中有data_source字段）
        sources_data = {}
        for point in data:
            source = point.metadata.get('data_source', 'unknown')
            if source not in sources_data:
                sources_data[source] = []
            sources_data[source].append(point)

        # 如果有多个数据源，进行交叉验证
        if len(sources_data) >= 2:
            sources = list(sources_data.keys())
            base_source = sources[0]
            comparison_sources = sources[1:]

            for comp_source in comparison_sources:
                # 对齐时间戳进行比较
                comparison_issues = self._compare_data_sources(
                    sources_data[base_source],
                    sources_data[comp_source],
                    base_source,
                    comp_source,
                    symbol
                )
                issues.extend(comparison_issues)

        return issues

    def _compare_data_sources(self, base_data: List[MarketData], comp_data: List[MarketData],
                              base_source: str, comp_source: str, symbol: str) -> List[Dict]:
        """比较两个数据源的数据一致性"""
        issues = []

        # 创建时间戳映射
        base_timestamps = {d.timestamp: d for d in base_data}
        comp_timestamps = {d.timestamp: d for d in comp_data}

        # 找到共同的时间戳
        common_timestamps = sorted(set(base_timestamps.keys()) & set(comp_timestamps.keys()))

        if not common_timestamps:
            issues.append({
                'type': 'cross_validation',
                'severity': 'medium',
                'message': f'数据源 {base_source} 和 {comp_source} 无共同时间戳',
                'symbol': symbol,
                'base_source': base_source,
                'comparison_source': comp_source,
                'timestamp': datetime.now().isoformat()
            })
            return issues

        # 比较共同时间戳的数据
        for timestamp in common_timestamps:
            base_point = base_timestamps[timestamp]
            comp_point = comp_timestamps[timestamp]

            # 比较收盘价
            price_diff = abs(base_point.close - comp_point.close)
            price_diff_pct = price_diff / base_point.close if base_point.close > 0 else 0

            if price_diff_pct > 0.01:  # 1%差异阈值
                issues.append({
                    'type': 'cross_validation',
                    'severity': 'medium',
                    'message': f'数据源价格差异: {price_diff_pct:.2%}',
                    'symbol': symbol,
                    'timestamp': timestamp.isoformat(),
                    'base_source': base_source,
                    'base_price': base_point.close,
                    'comparison_source': comp_source,
                    'comparison_price': comp_point.close,
                    'price_difference': price_diff,
                    'price_difference_pct': price_diff_pct
                })

            # 比较成交量
            volume_diff = abs(base_point.volume - comp_point.volume)
            volume_diff_pct = volume_diff / base_point.volume if base_point.volume > 0 else 0

            if volume_diff_pct > 0.05:  # 5%差异阈值
                issues.append({
                    'type': 'cross_validation',
                    'severity': 'low',
                    'message': f'数据源成交量差异: {volume_diff_pct:.2%}',
                    'symbol': symbol,
                    'timestamp': timestamp.isoformat(),
                    'base_source': base_source,
                    'base_volume': base_point.volume,
                    'comparison_source': comp_source,
                    'comparison_volume': comp_point.volume,
                    'volume_difference': volume_diff,
                    'volume_difference_pct': volume_diff_pct
                })

        return issues

    def _record_validation(self, validation_results: Dict):
        """记录验证结果到历史"""
        validation_record = {
            'timestamp': datetime.now().isoformat(),
            'valid': validation_results['valid'],
            'error_count': len(validation_results['errors']),
            'warning_count': len(validation_results['warnings']),
            'passed_tests': validation_results['passed_tests'],
            'total_tests': validation_results['total_tests'],
            'symbol_count': len(set([e.get('symbol', 'unknown') for e in validation_results['errors']])),
            'details': {
                'errors': validation_results['errors'],
                'warnings': validation_results['warnings']
            }
        }

        # 添加到历史记录
        self.validation_history.append(validation_record)

        # 保持历史记录长度
        max_history = self.config.get('max_validation_history', 1000)
        if len(self.validation_history) > max_history:
            self.validation_history = self.validation_history[-max_history:]

        # 更新统计信息
        self._update_validation_statistics(validation_record)

    def _update_validation_statistics(self, validation_record: Dict):
        """更新验证统计信息"""
        # 实现统计信息更新逻辑
        pass

    def get_validation_history(self, hours: int = 24) -> List[Dict]:
        """获取验证历史记录"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [record for record in self.validation_history
                if datetime.fromisoformat(record['timestamp']) >= cutoff_time]

    def get_validation_statistics(self) -> Dict[str, Any]:
        """获取验证统计信息"""
        if not self.validation_history:
            return {'error': 'No validation data available'}

        valid_records = [r for r in self.validation_history if r['valid']]
        error_counts = [r['error_count'] for r in self.validation_history]
        warning_counts = [r['warning_count'] for r in self.validation_history]

        return {
            'total_validations': len(self.validation_history),
            'success_rate': len(valid_records) / len(self.validation_history) if self.validation_history else 0,
            'average_errors': np.mean(error_counts) if error_counts else 0,
            'average_warnings': np.mean(warning_counts) if warning_counts else 0,
            'max_errors': max(error_counts) if error_counts else 0,
            'common_error_types': self._get_common_error_types(),
            'validation_trend': self._calculate_validation_trend(),
            'period_covered': {
                'start': self.validation_history[0]['timestamp'],
                'end': self.validation_history[-1]['timestamp'],
                'days': (datetime.fromisoformat(self.validation_history[-1]['timestamp']) -
                         datetime.fromisoformat(self.validation_history[0]['timestamp'])).days
            }
        }

    def _get_common_error_types(self) -> Dict[str, int]:
        """获取常见错误类型"""
        error_types = {}
        for record in self.validation_history:
            for error in record.get('details', {}).get('errors', []):
                error_type = error.get('type', 'unknown')
                if error_type not in error_types:
                    error_types[error_type] = 0
                error_types[error_type] += 1

        return dict(sorted(error_types.items(), key=lambda x: x[1], reverse=True))

    def _calculate_validation_trend(self) -> str:
        """计算验证趋势"""
        if len(self.validation_history) < 10:
            return 'insufficient_data'

        recent_success = [1 if r['valid'] else 0 for r in self.validation_history[-10:]]
        success_rate = np.mean(recent_success)

        if success_rate >= 0.9:
            return 'excellent'
        elif success_rate >= 0.8:
            return 'good'
        elif success_rate >= 0.7:
            return 'fair'
        else:
            return 'poor'

    def generate_validation_report(self, period: str = '7d') -> Dict[str, Any]:
        """生成验证报告"""
        try:
            # 确定时间范围
            if period.endswith('d'):
                days = int(period[:-1])
                cutoff_time = datetime.now() - timedelta(days=days)
            else:
                cutoff_time = datetime.now() - timedelta(days=7)

            # 筛选时间段内的数据
            period_data = [r for r in self.validation_history
                           if datetime.fromisoformat(r['timestamp']) >= cutoff_time]

            if not period_data:
                return {'error': f'No validation data available for period {period}'}

            return {
                'period': period,
                'summary': self._generate_validation_summary(period_data),
                'error_analysis': self._analyze_validation_errors(period_data),
                'performance_metrics': self._calculate_performance_metrics(period_data),
                'recommendations': self._generate_validation_recommendations(period_data)
            }

        except Exception as e:
            logger.error(f"验证报告生成失败: {e}")
            return {'error': str(e)}

    def _generate_validation_summary(self, period_data: List[Dict]) -> Dict[str, Any]:
        """生成验证摘要"""
        valid_count = sum(1 for r in period_data if r['valid'])
        total_count = len(period_data)
        error_counts = [r['error_count'] for r in period_data]
        warning_counts = [r['warning_count'] for r in period_data]

        return {
            'total_validations': total_count,
            'success_rate': valid_count / total_count if total_count > 0 else 0,
            'average_errors': np.mean(error_counts) if error_counts else 0,
            'average_warnings': np.mean(warning_counts) if warning_counts else 0,
            'total_errors': sum(error_counts),
            'total_warnings': sum(warning_counts),
            'success_trend': self._calculate_period_trend(period_data)
        }

    def _calculate_period_trend(self, period_data: List[Dict]) -> str:
        """计算周期趋势"""
        if len(period_data) < 5:
            return 'insufficient_data'

        # 按时间排序
        sorted_data = sorted(period_data, key=lambda x: x['timestamp'])
        success_rates = []

        # 计算滑动窗口成功率
        window_size = min(5, len(sorted_data))
        for i in range(len(sorted_data) - window_size + 1):
            window = sorted_data[i:i + window_size]
            success_rate = sum(1 for r in window if r['valid']) / window_size
            success_rates.append(success_rate)

        if len(success_rates) < 2:
            return 'stable'

        # 计算趋势
        slope = np.polyfit(range(len(success_rates)), success_rates, 1)[0]

        if slope > 0.05:
            return 'improving'
        elif slope < -0.05:
            return 'declining'
        else:
            return 'stable'

    def _analyze_validation_errors(self, period_data: List[Dict]) -> Dict[str, Any]:
        """分析验证错误"""
        all_errors = []
        for record in period_data:
            all_errors.extend(record.get('details', {}).get('errors', []))

        if not all_errors:
            return {'total_errors': 0, 'error_types': {}}

        # 按类型分类错误
        error_types = {}
        for error in all_errors:
            error_type = error.get('type', 'unknown')
            severity = error.get('severity', 'medium')

            if error_type not in error_types:
                error_types[error_type] = {
                    'count': 0,
                    'severities': {},
                    'symbols': set(),
                    'examples': []
                }

            error_types[error_type]['count'] += 1
            error_types[error_type]['severities'][severity] = error_types[error_type]['severities'].get(severity,
                                                                                                        0) + 1
            error_types[error_type]['symbols'].add(error.get('symbol', 'unknown'))

            # 保留一些示例
            if len(error_types[error_type]['examples']) < 5:
                error_types[error_type]['examples'].append({
                    'message': error.get('message', ''),
                    'symbol': error.get('symbol', ''),
                    'timestamp': error.get('timestamp', ''),
                    'severity': severity
                })

        # 转换sets为lists
        for error_type in error_types:
            error_types[error_type]['symbols'] = list(error_types[error_type]['symbols'])

        return {
            'total_errors': len(all_errors),
            'error_types': error_types,
            'most_common_error': max(error_types.items(), key=lambda x: x[1]['count'])[
                0] if error_types else 'none',
            'severity_distribution': self._calculate_error_severity_distribution(all_errors)
        }

    def _calculate_error_severity_distribution(self, errors: List[Dict]) -> Dict[str, int]:
        """计算错误严重性分布"""
        distribution = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for error in errors:
            severity = error.get('severity', 'medium')
            if severity in distribution:
                distribution[severity] += 1
        return distribution

    def _calculate_performance_metrics(self, period_data: List[Dict]) -> Dict[str, Any]:
        """计算性能指标"""
        validation_times = []
        error_resolution_times = []

        # 这里可以添加实际的性能指标计算逻辑
        # 例如：平均验证时间、错误解决时间等

        return {
            'average_validation_time': 0.0,  # 需要实际实现
            'error_resolution_time': 0.0,  # 需要实际实现
            'throughput': len(period_data) / 7 if period_data else 0,  # 每天的平均验证次数
            'availability': 1.0  # 假设100%可用性
        }

    def _generate_validation_recommendations(self, period_data: List[Dict]) -> List[Dict]:
        """生成验证建议"""
        recommendations = []

        # 分析错误模式
        error_analysis = self._analyze_validation_errors(period_data)
        if error_analysis['total_errors'] > 0:
            most_common_error = error_analysis['most_common_error']
            recommendations.append({
                'priority': 'high',
                'category': 'error_reduction',
                'action': f'重点解决 {most_common_error} 类型的错误',
                'reason': f'"{most_common_error}" 是最常见的错误类型，共 {error_analysis["error_types"][most_common_error]["count"]} 次'
            })

        # 分析成功率趋势
        summary = self._generate_validation_summary(period_data)
        if summary['success_rate'] < 0.8:
            recommendations.append({
                'priority': 'medium',
                'category': 'quality_improvement',
                'action': '提高数据验证成功率',
                'reason': f'当前成功率 {summary["success_rate"]:.1%} 低于目标值 80%'
            })

        # 分析严重错误
        severity_dist = error_analysis.get('severity_distribution', {})
        if severity_dist.get('critical', 0) > 0 or severity_dist.get('high', 0) > 10:
            recommendations.append({
                'priority': 'critical',
                'category': 'risk_mitigation',
                'action': '立即处理严重和高严重性错误',
                'reason': f'发现 {severity_dist.get("critical", 0)} 个严重错误和 {severity_dist.get("high", 0)} 个高严重性错误'
            })

        return recommendations

    def export_validation_data(self, filepath: str, format: str = 'json') -> bool:
        """导出验证数据"""
        try:
            if format == 'json':
                with open(filepath, 'w') as f:
                    json.dump({
                        'validation_history': self.validation_history,
                        'statistics': self.get_validation_statistics(),
                        'export_timestamp': datetime.now().isoformat()
                    }, f, indent=2)

            elif format == 'csv':
                # 将验证历史转换为CSV
                if self.validation_history:
                    fieldnames = set()
                    for record in self.validation_history:
                        fieldnames.update(record.keys())

                    with open(filepath, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
                        writer.writeheader()
                        writer.writerows(self.validation_history)

            else:
                raise ValueError(f"不支持的格式: {format}")

            logger.info(f"验证数据导出成功: {filepath}")
            return True

        except Exception as e:
            logger.error(f"验证数据导出失败: {e}")
            return False

    def import_validation_data(self, filepath: str, format: str = 'json') -> bool:
        """导入验证数据"""
        try:
            if format == 'json':
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    self.validation_history = data.get('validation_history', [])

            elif format == 'csv':
                with open(filepath, 'r') as f:
                    reader = csv.DictReader(f)
                    self.validation_history = list(reader)

            else:
                raise ValueError(f"不支持的格式: {format}")

            logger.info(f"验证数据导入成功: {filepath}")
            return True

        except Exception as e:
            logger.error(f"验证数据导入失败: {e}")
            return False

    def reset_validation_data(self):
        """重置验证数据"""
        self.validation_history.clear()
        logger.info("验证数据已重置")

    def cleanup(self):
        """清理资源"""
        try:
            # 保存最后的验证数据
            if self.config.get('auto_save', True):
                self.export_validation_data('validation_data_backup.json')

            # 清空内存中的数据
            self.validation_history.clear()

            logger.info("数据验证器清理完成")

        except Exception as e:
            logger.error(f"数据验证器清理失败: {e}")

# 数据质量监控器继续实现
class DataQualityMonitor:
