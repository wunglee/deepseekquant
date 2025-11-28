"""
第6轮专家反馈业务规则实现示例

演示内容：
1. 15年阈值动态MAPE计算
2. 跨市场一致性三级呈现结构
3. 关键行业配置文件加载
4. 数据源健康度三级披露
5. 7大决策节点审计追溯
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core_bak_refactored.core.backtest._fragments.uat_validator import UATValidator
from core_bak_refactored.core.data._fragments.data_quality_checker import DataQualityChecker


def demo_dynamic_threshold():
    """演示：15年阈值的动态MAPE计算（专家第6轮问题1）"""
    print("\n" + "="*80)
    print("演示1：15年阈值的动态MAPE计算")
    print("="*80)
    
    validator = UATValidator()
    
    # 测试不同年份事件
    test_events = [
        (1997, "1997亚洲金融危机"),
        (2008, "2008全球金融危机"),
        (2015, "2015中国股灾"),
        (2020, "2020新冠疫情"),
    ]
    
    for year, event_name in test_events:
        threshold = validator._calculate_dynamic_threshold(year)
        years_passed = datetime.now().year - year
        
        if years_passed > 15:
            status = "✓ 启用动态阈值"
        else:
            status = "○ 使用固定阈值15%"
        
        print(f"\n{event_name}：")
        print(f"  事件年份: {year}")
        print(f"  距今年限: {years_passed}年")
        print(f"  MAPE阈值: {threshold:.1%}")
        print(f"  状态: {status}")


def demo_cross_market_presentation():
    """演示：跨市场一致性三级呈现结构（专家第6轮问题2）"""
    print("\n" + "="*80)
    print("演示2：跨市场一致性三级呈现结构")
    print("="*80)
    
    validator = UATValidator()
    
    # 模拟跨市场验证数据
    events_data = [
        {
            'event_id': '2015_china_market_crash',
            'market': 'CN',
            'pearson': 0.87,
            'spearman': 0.85,
            'compared_markets': ['US', 'EU', 'HK'],
            'pearson_by_market': {'US': 0.87, 'EU': 0.82, 'HK': 0.89},
            'spearman_by_market': {'US': 0.85, 'EU': 0.79, 'HK': 0.88}
        },
        {
            'event_id': '2020_covid_crash',
            'market': 'US',
            'pearson': 0.92,
            'spearman': 0.88
        },
        {
            'event_id': '2008_financial_crisis',
            'market': 'US',
            'pearson': 0.86,
            'spearman': 0.83
        }
    ]
    
    result = validator.validate_cross_market_consistency_enhanced(events_data)
    
    # 提取三级呈现结构
    presentation = result.details.get('presentation_structure', {})
    
    print("\n摘要层：")
    summary = presentation.get('summary', {})
    print(f"  整体通过: {summary.get('overall_passed')}")
    print(f"  事件总数: {summary.get('total_events')}")
    print(f"  正常市场通过: {summary.get('normal_pass_count')}")
    print(f"  极端市场通过: {summary.get('extreme_pass_count')}")
    print(f"  总通过数: {summary.get('total_pass_count')}")
    
    print("\n明细层：")
    for detail in presentation.get('details', [])[:2]:  # 只显示前2个
        print(f"  事件: {detail['event_id']}")
        print(f"    Pearson: {detail['pearson']:.2f} ({'✓' if detail['pearson_passed'] else '✗'})")
        print(f"    Spearman: {detail['spearman']:.2f} ({'✓' if detail['spearman_passed'] else '✗'})")
    
    print("\n中国专项层：")
    china_section = presentation.get('china_specific_section', {})
    print(f"  监管要求: {china_section.get('requirement')}")
    print(f"  覆盖事件: {china_section.get('events_covered')}")
    print(f"  验证方法: {china_section.get('validation_methodology')}")


def demo_critical_industries_config():
    """演示：关键行业配置文件加载（专家第6轮问题3）"""
    print("\n" + "="*80)
    print("演示3：关键行业配置文件")
    print("="*80)
    
    config_path = Path(__file__).parent.parent / 'config' / 'critical_industries_config.json'
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"\n配置版本: {config.get('version')}")
        print(f"维护方: {config.get('maintainer')}")
        print(f"更新日期: {config.get('last_updated')}")
        
        print("\n关键行业列表：")
        for industry in config.get('critical_industries', []):
            print(f"\n  {industry['name']} ({industry['code']}):")
            print(f"    柔性阈值: {industry['flexible_threshold']:.0%}")
            print(f"    最小样本: {industry['min_sample_days']}天")
            print(f"    Bootstrap要求: {'是' if industry['bootstrap_required'] else '否'}")
            print(f"    审批文件: {industry['approval_document']}")
        
        print("\n审批流程：")
        for step in config.get('approval_process', {}).get('new_industry_requirements', []):
            print(f"  • {step}")
    else:
        print(f"配置文件不存在: {config_path}")


def demo_source_health_disclosure():
    """演示：数据源健康度三级披露（专家第6轮问题4）"""
    print("\n" + "="*80)
    print("演示4：数据源健康度三级披露")
    print("="*80)
    
    checker = DataQualityChecker()
    
    # 初始化模拟数据源评分
    checker._source_ratings = {
        'yahoo': 85,
        'joinquant': 92,
        'bloomberg': 78,
        'wind': 55,
        'tushare': 88
    }
    
    # 获取三级披露摘要
    summary = checker.get_source_health_summary(
        primary_source='yahoo',
        backup_source='joinquant',
        monitoring_sources=['bloomberg', 'tushare', 'wind']
    )
    
    print("\n主数据源：")
    if 'primary_source' in summary:
        ps = summary['primary_source']
        print(f"  名称: {ps['name']}")
        print(f"  评分: {ps['score']}")
        print(f"  档位: {ps['level']}")
    
    print("\n备用数据源：")
    if 'backup_source' in summary:
        bs = summary['backup_source']
        print(f"  名称: {bs['name']}")
        print(f"  评分: {bs['score']}")
        print(f"  档位: {bs['level']}")
    
    print("\n监控数据源：")
    for source in summary.get('monitoring_sources', []):
        print(f"  • {source['name']}: {source['score']}分 ({source['level']})")
    
    print("\n危险档位数据源：")
    for source in summary.get('dangerous_sources', []):
        print(f"  ⚠ {source['name']}: {source['score']}分")
        print(f"    处置措施: {source['action']}")


def demo_decision_path_audit():
    """演示：7大决策节点审计追溯（专家第6轮问题5）"""
    print("\n" + "="*80)
    print("演示5：7大决策节点审计追溯")
    print("="*80)
    
    validator = UATValidator()
    
    # 执行三级指标验证（会自动记录决策路径）
    predictions = [-0.10, -0.08, -0.12, -0.09, -0.11]
    actuals = [-0.09, -0.09, -0.11, -0.10, -0.10]
    
    results = validator.validate_triple_indicator_system(predictions, actuals, strict_mode=True, production_uat=True)
    
    print("\n已记录的决策节点：")
    
    # MAPE决策路径
    mape_result = results.get('mape')
    if mape_result and mape_result.decision_path:
        for step in mape_result.decision_path:
            print(f"\n节点: {step['step_name']}")
            print(f"  条件: {step['condition']}")
            print(f"  结果: {'✓ 通过' if step['result'] else '✗ 未通过'}")
            print(f"  参数: {step['parameters']}")
    
    # 方向准确率决策路径
    dir_result = results.get('direction_accuracy')
    if dir_result and dir_result.decision_path:
        for step in dir_result.decision_path:
            print(f"\n节点: {step['step_name']}")
            print(f"  条件: {step['condition']}")
            print(f"  结果: {'✓ 通过' if step['result'] else '✗ 未通过'}")
    
    # 尾部误差控制决策路径
    tail_result = results.get('tail_error_control')
    if tail_result and tail_result.decision_path:
        for step in tail_result.decision_path:
            print(f"\n节点: {step['step_name']}")
            print(f"  条件: {step['condition']}")
            print(f"  结果: {'✓ 通过' if step['result'] else '✗ 未通过'}")
    
    print("\n说明：7大决策节点包括：")
    mandatory_nodes = [
        'MAPE判定',
        '方向准确率判定',
        '尾部误差控制判定',
        '数据质量评分判定',
        '系统响应时间判定',
        '行业参数差异判定',
        '数据源健康度判定'
    ]
    for i, node in enumerate(mandatory_nodes, 1):
        print(f"  {i}. {node}")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("第6轮专家反馈业务规则实现示例")
    print("="*80)
    
    demo_dynamic_threshold()
    demo_cross_market_presentation()
    demo_critical_industries_config()
    demo_source_health_disclosure()
    demo_decision_path_audit()
    
    print("\n" + "="*80)
    print("演示完成")
    print("="*80 + "\n")
