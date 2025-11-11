"""
数据质量报告 - 业务层
从 core_bak/data_fetcher.py 拆分
职责: 生成数据质量报告
"""

from typing import Dict
import logging

logger = logging.getLogger("DeepSeekQuant.DataQualityReporter")


class DataQualityReporter:
    """数据质量报告生成器"""

    def __init__(self, config: Dict):
        self.config = config

        # 检测异常交易量模式
        volume_patterns = self._detect_volume_patterns(data)
        anomalies.extend(volume_patterns)

        return anomalies

    def _detect_duplicate_patterns(self, data: List[MarketData]) -> List[Dict]:
        """检测重复数据模式"""
        anomalies = []

        # 检查连续相同价格
        for i in range(1, len(data)):
            if data[i].close == data[i - 1].close and data[i].volume == data[i - 1].volume:
                # 检查是否形成连续模式
                consecutive_count = 1
                for j in range(i + 1, min(i + 5, len(data))):  # 检查后续5个点
                    if data[j].close == data[i].close and data[j].volume == data[i].volume:
                        consecutive_count += 1
                    else:
                        break

                if consecutive_count >= 3:  # 连续3个相同点
                    anomalies.append({
                        'type': 'pattern_anomaly',
                        'method': 'duplicate_data',
                        'severity': 'medium',
                        'message': f'检测到连续 {consecutive_count} 个相同数据点',
                        'timestamp': data[i].timestamp.isoformat(),
                        'symbol': data[i].symbol,
                        'consecutive_count': consecutive_count,
                        'price': data[i].close,
                        'volume': data[i].volume
                    })

        return anomalies

    def _detect_volatility_patterns(self, data: List[MarketData]) -> List[Dict]:
        """检测波动率模式异常"""
        anomalies = []

        if len(data) < 20:
            return anomalies

        # 计算滚动波动率
        prices = [d.close for d in data]
        returns = np.diff(prices) / prices[:-1]

        window = 10
        rolling_volatility = pd.Series(returns).rolling(window=window).std().dropna().values

        # 检测波动率异常变化
        volatility_mean = np.mean(rolling_volatility)
        volatility_std = np.std(rolling_volatility)

        for i, vol in enumerate(rolling_volatility):
            if abs(vol - volatility_mean) > 3 * volatility_std:
                anomalies.append({
                    'type': 'pattern_anomaly',
                    'method': 'volatility_spike',
                    'severity': 'medium',
                    'message': f'波动率异常: {vol:.4f} (平均: {volatility_mean:.4f})',
                    'timestamp': data[i + window].timestamp.isoformat(),  # 调整索引
                    'symbol': data[i + window].symbol,
                    'volatility': vol,
                    'average_volatility': volatility_mean,
                    'std_dev': volatility_std
                })

        return anomalies

    def _detect_volume_patterns(self, data: List[MarketData]) -> List[Dict]:
        """检测成交量模式异常"""
        anomalies = []

        if len(data) < 20:
            return anomalies

        volumes = [d.volume for d in data]

        # 检测异常成交量模式（例如：异常高低成交量交替）
        volume_changes = np.diff(volumes) / volumes[:-1]

        # 检测异常大的成交量变化
        for i, change in enumerate(volume_changes):
            if abs(change) > 5.0:  # 500%的变化
                anomalies.append({
                    'type': 'pattern_anomaly',
                    'method': 'volume_spike',
                    'severity': 'medium',
                    'message': f'成交量异常变化: {change:.1%}',
                    'timestamp': data[i + 1].timestamp.isoformat(),
                    'symbol': data[i + 1].symbol,
                    'volume_change': change,
                    'previous_volume': volumes[i],
                    'current_volume': volumes[i + 1]
                })

        return anomalies

    def _detect_ml_anomalies(self, data: List[MarketData]) -> List[Dict]:
        """使用机器学习方法检测异常"""
        anomalies = []

        try:
            # 这里实现基于机器学习的异常检测
            # 例如：Isolation Forest, Autoencoder, LOF等

            if self.anomaly_detector['statistical_methods']['isolation_forest']['enabled']:
                iforest_anomalies = self._detect_isolation_forest_anomalies(data)
                anomalies.extend(iforest_anomalies)

            if self.anomaly_detector['statistical_methods']['autoencoder']['enabled']:
                autoencoder_anomalies = self._detect_autoencoder_anomalies(data)
                anomalies.extend(autoencoder_anomalies)

        except Exception as e:
            logger.warning(f"机器学习异常检测失败: {e}")

        return anomalies

    def _detect_isolation_forest_anomalies(self, data: List[MarketData]) -> List[Dict]:
        """使用Isolation Forest检测异常"""
        anomalies = []

        try:
            from sklearn.ensemble import IsolationForest

            # 准备特征数据
            features = []
            for i, point in enumerate(data):
                feature_vector = [
                    point.close,
                    point.volume,
                    point.high - point.low,  # 价格范围
                    (point.close - point.open) / point.open if point.open > 0 else 0  # 日内收益率
                ]

                # 添加技术指标（如果可用）
                if hasattr(point, 'metadata') and point.metadata:
                    for indicator in ['rsi', 'macd', 'atr']:
                        if indicator in point.metadata:
                            feature_vector.append(point.metadata[indicator])

                features.append(feature_vector)

            # 训练Isolation Forest模型
            contamination = self.anomaly_detector['statistical_methods']['isolation_forest']['contamination']
            clf = IsolationForest(contamination=contamination, random_state=42)
            predictions = clf.fit_predict(features)

            # 检测异常点
            for i, prediction in enumerate(predictions):
                if prediction == -1:  # 异常点
                    anomalies.append({
                        'type': 'ml_anomaly',
                        'method': 'isolation_forest',
                        'severity': 'medium',
                        'message': 'Isolation Forest检测到异常点',
                        'timestamp': data[i].timestamp.isoformat(),
                        'symbol': data[i].symbol,
                        'anomaly_score': float(clf.score_samples([features[i]])[0]),
                        'contamination': contamination
                    })

        except Exception as e:
            logger.warning(f"Isolation Forest异常检测失败: {e}")

        return anomalies

    def _determine_quality_level(self, score: float) -> str:
        """确定质量等级"""
        if score >= 0.95:
            return 'excellent'
        elif score >= 0.85:
            return 'good'
        elif score >= 0.70:
            return 'fair'
        elif score >= 0.50:
            return 'poor'
        else:
            return 'critical'

    def _generate_recommendations(self, quality_report: Dict) -> List[Dict]:
        """生成改进建议"""
        recommendations = []

        # 基于完整性问题的建议
        completeness_issues = [issue for issue in quality_report['validation_errors']
                               if issue['type'] == 'completeness']
        if completeness_issues:
            recommendations.append({
                'priority': 'high' if any(i['severity'] == 'high' for i in completeness_issues) else 'medium',
                'category': 'completeness',
                'action': '检查数据源连接性和数据提取流程',
                'details': f'发现 {len(completeness_issues)} 个完整性问题'
            })

        # 基于准确性问题的建议
        accuracy_issues = [issue for issue in quality_report['validation_errors']
                           if issue['type'] == 'accuracy']
        if accuracy_issues:
            recommendations.append({
                'priority': 'high' if any(i['severity'] == 'high' for i in accuracy_issues) else 'medium',
                'category': 'accuracy',
                'action': '验证数据源准确性和数据清洗流程',
                'details': f'发现 {len(accuracy_issues)} 个准确性问题'
            })

        # 基于异常检测的建议
        if quality_report['anomalies_detected']:
            recommendations.append({
                'priority': 'medium',
                'category': 'anomaly_detection',
                'action': '调查检测到的数据异常',
                'details': f'发现 {len(quality_report['anomalies_detected'])} 个数据异常'
            })

        # 总体建议
        if quality_report['overall_score'] < 0.8:
            recommendations.append({
                'priority': 'high',
                'category': 'overall',
                'action': '全面检查数据质量流程',
                'details': f'总体数据质量评分较低: {quality_report['overall_score']:.3f}'
            })

        return recommendations

    def _trigger_alerts(self, quality_report: Dict):
        """触发质量警报"""
        alert_level = self._determine_alert_level(quality_report)

        if alert_level != 'none':
            alert_message = self._create_alert_message(quality_report, alert_level)
            self._send_alert(alert_message, alert_level)

            logger.warning(f"触发数据质量{alert_level}警报: {alert_message}")

    def _determine_alert_level(self, quality_report: Dict) -> str:
        """确定警报级别"""
        overall_score = quality_report.get('overall_score', 1.0)
        validation_errors = quality_report.get('validation_errors', [])
        anomalies = quality_report.get('anomalies_detected', [])

        # 检查是否有严重错误
        critical_errors = [e for e in validation_errors if e.get('severity') == 'critical']
        high_errors = [e for e in validation_errors if e.get('severity') == 'high']
        high_anomalies = [a for a in anomalies if a.get('severity') == 'high']

        if critical_errors or overall_score < 0.3:
            return 'critical'
        elif high_errors or high_anomalies or overall_score < 0.6:
            return 'high'
        elif overall_score < 0.8:
            return 'medium'
        elif overall_score < 0.9:
            return 'low'
        else:
            return 'none'

    def _create_alert_message(self, quality_report: Dict, level: str) -> str:
        """创建警报消息"""
        overall_score = quality_report.get('overall_score', 0.0)
        error_count = len(quality_report.get('validation_errors', []))
        anomaly_count = len(quality_report.get('anomalies_detected', []))
        quality_level = quality_report.get('quality_level', 'unknown')

        return (f"数据质量{level}警报: 总体评分 {overall_score:.3f} ({quality_level})\n"
                f"问题统计: {error_count} 个验证错误, {anomaly_count} 个异常\n"
                f"数据源: {quality_report.get('data_source', 'unknown')}, "
                f"符号数量: {quality_report.get('symbol_count', 0)}\n"
                f"时间: {quality_report.get('timestamp', 'unknown')}")

    def _send_alert(self, message: str, level: str):
        """发送警报"""
        notification_channels = self.config.get('alerting', {}).get('channels', {})

        for channel, config in notification_channels.items():
            if not config.get('enabled', False):
                continue

            try:
                if channel == 'email' and level in config.get('levels', []):
                    self._send_email_alert(message, level, config)
                elif channel == 'slack' and level in config.get('levels', []):
                    self._send_slack_alert(message, level, config)
                elif channel == 'sms' and level in config.get('levels', []):
                    self._send_sms_alert(message, level, config)
                elif channel == 'webhook' and level in config.get('levels', []):
                    self._send_webhook_alert(message, level, config)

                logger.info(f"{level}警报通过 {channel} 发送成功")

            except Exception as e:
                logger.error(f"警报发送失败 ({channel}): {e}")

    def _send_email_alert(self, message: str, level: str, config: Dict):
        """发送邮件警报"""
        # 实现邮件发送逻辑
        pass

    def _send_slack_alert(self, message: str, level: str, config: Dict):
        """发送Slack警报"""
        # 实现Slack消息发送
        pass

    def _send_sms_alert(self, message: str, level: str, config: Dict):
        """发送短信警报"""
        # 实现短信发送
        pass

    def _send_webhook_alert(self, message: str, level: str, config: Dict):
        """发送Webhook警报"""
        # 实现Webhook调用
        pass

    def _record_quality_metrics(self, quality_report: Dict):
        """记录质量指标历史"""
        quality_metrics = {
            'timestamp': datetime.now().isoformat(),
            'overall_score': quality_report['overall_score'],
            'dimension_scores': quality_report.get('dimension_scores', {}),
            'issue_count': len(quality_report.get('validation_errors', [])) +
                           len(quality_report.get('anomalies_detected', [])),
            'data_source': quality_report.get('data_source', 'unknown'),
            'symbol_count': quality_report.get('symbol_count', 0),
            'quality_level': quality_report.get('quality_level', 'unknown')
        }

        # 添加到历史记录
        self.quality_history.append(quality_metrics)

        # 保持历史记录长度
        max_history = self.config.get('max_quality_history', 1000)
        if len(self.quality_history) > max_history:
            self.quality_history = self.quality_history[-max_history:]

        # 更新性能统计
        self._update_quality_statistics(quality_metrics)

    def _update_quality_statistics(self, quality_metrics: Dict):
        """更新质量统计"""
        # 更新各种统计指标
        pass

    def get_quality_history(self, hours: int = 24) -> List[Dict]:
        """获取质量历史记录"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.quality_history
                if datetime.fromisoformat(m['timestamp']) >= cutoff_time]

    def get_quality_statistics(self) -> Dict[str, Any]:
        """获取质量统计信息"""
        if not self.quality_history:
            return {'error': 'No quality data available'}

        scores = [m['overall_score'] for m in self.quality_history]
        issues = [m['issue_count'] for m in self.quality_history]

        return {
            'average_score': float(np.mean(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'std_score': float(np.std(scores)),
            'average_issues': float(np.mean(issues)),
            'total_alerts': len(self.alert_history),
            'data_sources_covered': len(set(m['data_source'] for m in self.quality_history)),
            'trend': self._calculate_quality_trend(),
            'period_covered': {
                'start': self.quality_history[0]['timestamp'],
                'end': self.quality_history[-1]['timestamp'],
                'days': (datetime.fromisoformat(self.quality_history[-1]['timestamp']) -
                         datetime.fromisoformat(self.quality_history[0]['timestamp'])).days
            }
        }

    def _calculate_quality_trend(self) -> str:
        """计算质量趋势"""
        if len(self.quality_history) < 10:
            return 'insufficient_data'

        recent_scores = [m['overall_score'] for m in self.quality_history[-10:]]
        x = range(len(recent_scores))
        slope = np.polyfit(x, recent_scores, 1)[0]

        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'

    def generate_quality_report(self, period: str = '7d') -> Dict[str, Any]:
        """生成质量报告"""
        try:
            # 确定时间范围
            if period.endswith('d'):
                days = int(period[:-1])
                cutoff_time = datetime.now() - timedelta(days=days)
            elif period.endswith('h'):
                hours = int(period[:-1])
                cutoff_time = datetime.now() - timedelta(hours=hours)
            else:
                cutoff_time = datetime.now() - timedelta(days=7)

            # 筛选时间段内的数据
            period_data = [m for m in self.quality_history
                           if datetime.fromisoformat(m['timestamp']) >= cutoff_time]

            if not period_data:
                return {'error': f'No quality data available for period {period}'}

            return {
                'period': period,
                'summary': self._generate_summary(period_data),
                'trend_analysis': self._analyze_trends(period_data),
                'issue_analysis': self._analyze_issues(period_data),
                'recommendations': self._generate_recommendations_report(period_data),
                'alerts_summary': self._summarize_alerts(period)
            }

        except Exception as e:
            logger.error(f"质量报告生成失败: {e}")
            return {'error': str(e)}

    def _generate_summary(self, period_data: List[Dict]) -> Dict[str, Any]:
        """生成摘要信息"""
        scores = [m['overall_score'] for m in period_data]
        issues = [m['issue_count'] for m in period_data]

        return {
            'average_score': float(np.mean(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'total_issues': sum(issues),
            'data_points': len(period_data),
            'data_sources': len(set(m['data_source'] for m in period_data)),
            'quality_level': self._determine_period_quality_level(scores)
        }

    def _determine_period_quality_level(self, scores: List[float]) -> str:
        """确定周期质量等级"""
        avg_score = np.mean(scores)
        if avg_score >= 0.95:
            return 'excellent'
        elif avg_score >= 0.85:
            return 'good'
        elif avg_score >= 0.70:
            return 'fair'
        elif avg_score >= 0.50:
            return 'poor'
        else:
            return 'critical'

    def _analyze_trends(self, period_data: List[Dict]) -> Dict[str, Any]:
        """分析趋势"""
        if len(period_data) < 2:
            return {'status': 'insufficient_data'}

        scores = [m['overall_score'] for m in period_data]
        dates = [datetime.fromisoformat(m['timestamp']) for m in period_data]

        # 计算线性趋势
        days_since_start = [(d - dates[0]).days for d in dates]
        slope, intercept = np.polyfit(days_since_start, scores, 1)

        return {
            'linear_trend': {
                'slope': float(slope),
                'direction': 'improving' if slope > 0.001 else ('declining' if slope < -0.001 else 'stable'),
                'r_squared': float(np.corrcoef(days_since_start, scores)[0, 1] ** 2)
            },
            'volatility': float(np.std(scores)),
            'stability': 1.0 - min(1.0, np.std(scores) * 2)  # 标准差越小越稳定
        }

    def _analyze_issues(self, period_data: List[Dict]) -> Dict[str, Any]:
        """分析问题"""
        all_issues = []
        for record in period_data:
            all_issues.extend(record.get('validation_errors', []))
            all_issues.extend(record.get('anomalies_detected', []))

        if not all_issues:
            return {'total_issues': 0, 'issue_types': {}}

        # 按类型分类
        issue_types = {}
        for issue in all_issues:
            issue_type = issue.get('type', 'unknown')
            if issue_type not in issue_types:
                issue_types[issue_type] = {'count': 0, 'severities': {}}

            issue_types[issue_type]['count'] += 1
            severity = issue.get('severity', 'medium')
            if severity not in issue_types[issue_type]['severities']:
                issue_types[issue_type]['severities'][severity] = 0
            issue_types[issue_type]['severities'][severity] += 1

        return {
            'total_issues': len(all_issues),
            'issue_types': issue_types,
            'most_common_issue': max(issue_types.items(), key=lambda x: x[1]['count'])[
                0] if issue_types else 'none',
            'severity_distribution': self._calculate_severity_distribution(all_issues)
        }

    def _calculate_severity_distribution(self, issues: List[Dict]) -> Dict[str, int]:
        """计算严重性分布"""
        distribution = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for issue in issues:
            severity = issue.get('severity', 'medium')
            if severity in distribution:
                distribution[severity] += 1
        return distribution

    def _generate_recommendations_report(self, period_data: List[Dict]) -> List[Dict]:
        """生成报告建议"""
        recommendations = []

        # 分析趋势
        trend_analysis = self._analyze_trends(period_data)
        if trend_analysis.get('linear_trend', {}).get('direction') == 'declining':
            recommendations.append({
                'priority': 'high',
                'action': '调查质量下降原因并采取纠正措施',
                'reason': '检测到质量下降趋势'
            })

        # 分析问题模式
        issue_analysis = self._analyze_issues(period_data)
        if issue_analysis['total_issues'] > 0:
            most_common = issue_analysis['most_common_issue']
            recommendations.append({
                'priority': 'medium',
                'action': f'重点解决{most_common}类型的问题',
                'reason': f'"{most_common}"是最常见的问题类型'
            })

        return recommendations

    def _summarize_alerts(self, period: str) -> Dict[str, Any]:
        """汇总警报信息"""
        # 筛选时间段内的警报
        if period.endswith('d'):
            days = int(period[:-1])
            cutoff_time = datetime.now() - timedelta(days=days)
        else:
            cutoff_time = datetime.now() - timedelta(days=7)

        period_alerts = [alert for alert in self.alert_history
                         if datetime.fromisoformat(alert['timestamp']) >= cutoff_time]

        return {
            'total_alerts': len(period_alerts),
            'by_level': self._group_alerts_by_level(period_alerts),
            'by_source': self._group_alerts_by_source(period_alerts),
            'trend': self._calculate_alert_trend(period_alerts)
        }

    def _group_alerts_by_level(self, alerts: List[Dict]) -> Dict[str, int]:
        """按级别分组警报"""
        levels = {}
        for alert in alerts:
            level = alert.get('level', 'unknown')
            if level not in levels:
                levels[level] = 0
            levels[level] += 1
        return levels

    def _group_alerts_by_source(self, alerts: List[Dict]) -> Dict[str, int]:
        """按数据源分组警报"""
        sources = {}
        for alert in alerts:
            source = alert.get('data_source', 'unknown')
            if source not in sources:
                sources[source] = 0
            sources[source] += 1
        return sources

    def _calculate_alert_trend(self, alerts: List[Dict]) -> str:
        """计算警报趋势"""
        if len(alerts) < 5:
            return 'insufficient_data'

        # 按天分组
        daily_counts = {}
        for alert in alerts:
            date = datetime.fromisoformat(alert['timestamp']).strftime('%Y-%m-%d')
            if date not in daily_counts:
                daily_counts[date] = 0
            daily_counts[date] += 1

        if len(daily_counts) < 3:
            return 'insufficient_data'

        dates = sorted(daily_counts.keys())
        counts = [daily_counts[date] for date in dates]

        # 计算趋势
        x = range(len(counts))
        slope = np.polyfit(x, counts, 1)[0]

        if slope > 0.5:
            return 'increasing'
        elif slope < -0.5:
            return 'decreasing'
        else:
            return 'stable'

    def export_quality_data(self, filepath: str, format: str = 'json') -> bool:
        """导出质量数据"""
        try:
            if format == 'json':
                with open(filepath, 'w') as f:
                    json.dump({
                        'quality_history': self.quality_history,
                        'alert_history': self.alert_history,
                        'statistics': self.get_quality_statistics(),
                        'export_timestamp': datetime.now().isoformat()
                    }, f, indent=2)

            elif format == 'csv':
                # 将质量历史转换为CSV
                if self.quality_history:
                    fieldnames = set()
                    for record in self.quality_history:
                        fieldnames.update(record.keys())

                    with open(filepath, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
                        writer.writeheader()
                        writer.writerows(self.quality_history)

            else:
                raise ValueError(f"不支持的格式: {format}")

            logger.info(f"质量数据导出成功: {filepath}")
            return True

        except Exception as e:
            logger.error(f"质量数据导出失败: {e}")
            return False

    def import_quality_data(self, filepath: str, format: str = 'json') -> bool:
        """导入质量数据"""
        try:
            if format == 'json':
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    self.quality_history = data.get('quality_history', [])
                    self.alert_history = data.get('alert_history', [])

            elif format == 'csv':
                with open(filepath, 'r') as f:
                    reader = csv.DictReader(f)
                    self.quality_history = list(reader)

            else:
                raise ValueError(f"不支持的格式: {format}")

            logger.info(f"质量数据导入成功: {filepath}")
            return True

        except Exception as e:
            logger.error(f"质量数据导入失败: {e}")
            return False

    def reset_quality_data(self):
        """重置质量数据"""
        self.quality_history.clear()
        self.alert_history.clear()
        logger.info("质量数据已重置")

    def cleanup(self):
        """清理资源"""
        try:
            # 保存最后的质量数据
            if self.config.get('auto_save', True):
                self.export_quality_data('quality_data_backup.json')

            # 清空内存中的数据
            self.quality_history.clear()
            self.alert_history.clear()

            logger.info("数据质量监控器清理完成")

        except Exception as e:
            logger.error(f"数据质量监控器清理失败: {e}")

# 数据验证器类
class DataValidator:
