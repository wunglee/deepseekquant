"""
基本面筛选器 - 基于策略配置筛选股票

职责：
- 使用ProviderSelector获取市场数据提供者
- 获取指定市场的所有股票列表
- 使用ConfigLoader加载策略配置（如芒格策略）
- 根据策略配置对股票进行基本面筛选
- 生成筛选结果并保存到文件

使用示例：
    from core_bak_refactored.core.share.market.market_enums import MarketCode
    
    # 筛选A股市场的芒格策略股票
    results = screen_fundamental_stocks(
        market=MarketCode.CN,
        strategy_name="munger",
        output_dir="core_bak_refactored/data/screening_results"
    )
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

from core_bak_refactored.core.data.providers.factory import get_global_factory
from core_bak_refactored.core.data.providers.provider_selector import ProviderSelector
from core_bak_refactored.core.share.market.market_enums import MarketCode
from core_bak_refactored.core.strategy.fundamentals.ConfigLoader import StrategyConfigLoader

logger = logging.getLogger(__name__)


def screen_fundamental_stocks(
    market: MarketCode,
    strategy_name: str = "munger",
    output_dir: str = None,
    max_stocks: int = None,
    save_format: str = "csv"
) -> Dict[str, Any]:
    """
    基本面筛选主函数
    
    Args:
        market: 市场枚举（如 MarketCode.CN）
        strategy_name: 策略名称（对应配置文件名，如 "munger"）
        output_dir: 结果输出目录（默认：core_bak_refactored/data/screening_results）
        max_stocks: 最多处理的股票数量（用于测试，None表示处理全部）
        save_format: 保存格式（"csv", "json", "both"）
    
    Returns:
        Dict[str, Any]: 筛选结果，包含：
            - total_stocks: 总股票数
            - screened_stocks: 筛选后的股票数
            - results: 筛选结果DataFrame
            - output_files: 输出的文件路径列表
            - strategy_config: 使用的策略配置
    """
    start_time = datetime.now()
    logger.info(f"=" * 80)
    logger.info(f"开始基本面筛选 - 市场: {market.value}, 策略: {strategy_name}")
    logger.info(f"=" * 80)
    
    # 1. 初始化组件
    logger.info("Step 1: 初始化组件...")
    selector = ProviderSelector()
    factory = get_global_factory()
    config_loader = StrategyConfigLoader()
    
    # 2. 选择数据提供者
    logger.info("Step 2: 选择数据提供者...")
    provider = selector.select_provider_for_market(market, factory)
    logger.info(f"✓ 使用提供者: {provider.__class__.__name__}")
    
    # 3. 加载策略配置
    logger.info("Step 3: 加载策略配置...")
    strategy_config = config_loader.load_strategy(strategy_name)
    logger.info(f"✓ 加载策略配置: {strategy_config.get('config_name', strategy_name)}")
    logger.info(f"  版本: {strategy_config.get('version', 'N/A')}")
    logger.info(f"  描述: {strategy_config.get('description', 'N/A')}")
    
    # 4. 获取股票列表
    logger.info("Step 4: 获取股票列表...")
    logger.info("正在获取所有股票，这可能需要一些时间...")
    stocks_df = provider.get_all_symbols(market)
    
    if stocks_df.empty:
        logger.error("❌ 无法获取股票列表")
        return {
            "status": "error",
            "message": "无法获取股票列表",
            "total_stocks": 0,
            "screened_stocks": 0,
            "results": pd.DataFrame(),
            "output_files": [],
            "strategy_config": strategy_config
        }
    
    total_stocks = len(stocks_df)
    logger.info(f"✓ 获取到 {total_stocks} 只股票")
    
    # 限制处理数量（用于测试）
    if max_stocks is not None and max_stocks < total_stocks:
        stocks_df = stocks_df.head(max_stocks)
        logger.info(f"  限制处理数量: {max_stocks} 只")
    
    # 5. 基本面筛选
    logger.info("Step 5: 执行基本面筛选...")
    logger.info("正在筛选股票，请稍候...")
    
    screened_results = []
    processed = 0
    errors = 0
    
    # 获取策略阈值配置
    thresholds = {
        'valuation': strategy_config.get('valuation_thresholds', {}),
        'profitability': strategy_config.get('profitability_thresholds', {}),
        'growth': strategy_config.get('growth_thresholds', {}),
        'asset_quality': strategy_config.get('asset_quality_thresholds', {}),
        'liquidity': strategy_config.get('liquidity_filters', {}),
        'death_combinations': strategy_config.get('death_combinations', [])
    }
    
    # 处理每只股票
    for _, stock in stocks_df.iterrows():
        symbol = stock['symbol']
        name = stock.get('name', 'N/A')
        processed += 1
        
        if processed % 100 == 0:
            logger.info(f"  进度: {processed}/{len(stocks_df)} ({processed/len(stocks_df)*100:.1f}%)")
        
        try:
            # 获取真实的基本面数据
            fundamental_data = provider.get_complete_fundamental_data(symbol)
            
            # 如果获取失败或数据为空，跳过
            if not fundamental_data:
                logger.debug(f"  跳过 {symbol}: 无法获取基本面数据")
                continue
            
            # 执行筛选
            score, passed, reasons = _apply_screening(fundamental_data, thresholds)
            
            if passed:
                screened_results.append({
                    'symbol': symbol,
                    'name': name,
                    'score': score,
                    'market': market.value,
                    'pb': fundamental_data.get('pb'),
                    'pe': fundamental_data.get('pe'),
                    'ps': fundamental_data.get('ps'),
                    'pcf': fundamental_data.get('pcf'),
                    'roe': fundamental_data.get('roe'),
                    '筛选理由': '; '.join(reasons) if reasons else '符合所有条件'
                })
                
        except Exception as e:
            errors += 1
            logger.debug(f"  处理 {symbol} 时出错: {e}")
            continue
    
    # 6. 处理筛选结果
    logger.info("Step 6: 处理筛选结果...")
    results_df = pd.DataFrame(screened_results)
    screened_count = len(results_df)
    
    logger.info(f"✓ 筛选完成:")
    logger.info(f"  总股票数: {total_stocks}")
    logger.info(f"  通过筛选: {screened_count}")
    logger.info(f"  筛选率: {screened_count/total_stocks*100:.2f}%")
    logger.info(f"  处理错误: {errors}")
    
    # 7. 排序和限制结果
    if not results_df.empty:
        # 按得分排序
        if 'score' in results_df.columns:
            results_df = results_df.sort_values('score', ascending=False)
        
        # 应用top_n限制
        top_n = strategy_config.get('output_settings', {}).get('top_n', 50)
        if len(results_df) > top_n:
            results_df = results_df.head(top_n)
            logger.info(f"  按配置限制输出前 {top_n} 只股票")
        
        logger.info(f"✓ 最终输出 {len(results_df)} 只股票")
    
    # 8. 保存结果
    logger.info("Step 7: 保存结果...")
    output_files = _save_results(results_df, strategy_name, market, output_dir, save_format)
    
    for file_path in output_files:
        logger.info(f"  ✓ 保存到: {file_path}")
    
    # 9. 生成报告
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"=" * 80)
    logger.info(f"✓ 基本面筛选完成！耗时: {duration:.2f} 秒")
    logger.info(f"=" * 80)
    
    return {
        "status": "success",
        "total_stocks": total_stocks,
        "screened_stocks": len(results_df),
        "results": results_df,
        "output_files": output_files,
        "strategy_config": strategy_config,
        "duration_seconds": duration
    }


# 删除 mock 函数，现在使用真实的 provider.get_complete_fundamental_data


def _apply_screening(data: Dict[str, float], thresholds: Dict[str, Any]) -> tuple:
    """
    应用策略筛选条件
    
    Args:
        data: 基本面数据
        thresholds: 策略阈值配置
    
    Returns:
        tuple: (score, passed, reasons)
            - score: 总得分
            - passed: 是否通过筛选
            - reasons: 通过的理由列表
    """
    score = 0
    reasons = []
    passed = True
    
    # 估值筛选
    valuation = thresholds.get('valuation', {})
    for metric, config in valuation.items():
        if metric in data:
            value = data[metric]
            max_val = config.get('max', float('inf'))
            ideal_val = config.get('ideal', 0)
            weight = config.get('weight', 0)
            
            if value <= max_val:
                # 计算得分（越接近ideal_val得分越高）
                distance = abs(value - ideal_val) / max_val if max_val > 0 else 0
                metric_score = max(0, 1 - distance) * weight * 100
                score += metric_score
                
                if distance < 0.3:  # 接近理想值
                    reasons.append(f"{metric}={value:.2f} 优秀")
            else:
                passed = False
    
    # 盈利能力筛选
    profitability = thresholds.get('profitability', {})
    for metric, config in profitability.items():
        if metric in data:
            value = data[metric]
            min_val = config.get('min', 0)
            ideal_val = config.get('ideal', 1)
            weight = config.get('weight', 0)
            
            if value >= min_val:
                # 计算得分
                distance = abs(value - ideal_val) / ideal_val if ideal_val > 0 else 0
                metric_score = max(0, 1 - distance) * weight * 100
                score += metric_score
                
                if value >= ideal_val:
                    reasons.append(f"{metric}={value:.2%} 优秀")
            else:
                passed = False
    
    # 资产质量筛选
    asset_quality = thresholds.get('asset_quality', {})
    for metric, config in asset_quality.items():
        if metric in data:
            value = data[metric]
            max_val = config.get('max', float('inf'))
            min_val = config.get('min', 0)
            
            if min_val <= value <= max_val:
                reasons.append(f"{metric}={value:.2%} 健康")
            else:
                passed = False
    
    # 流动性筛选
    liquidity = thresholds.get('liquidity', {})
    # 这里应该检查成交额、市值等，暂时跳过
    
    # 死亡组合检查
    death_combinations = thresholds.get('death_combinations', [])
    for combo in death_combinations:
        condition = combo.get('condition', '')
        # 这里应该解析并检查死亡组合条件，暂时跳过
        pass
    
    return score, passed, reasons


def _save_results(df: pd.DataFrame, strategy_name: str, market: MarketCode, 
                  output_dir: str, save_format: str) -> List[str]:
    """
    保存筛选结果到文件
    
    Args:
        df: 筛选结果DataFrame
        strategy_name: 策略名称
        market: 市场枚举
        output_dir: 输出目录
        save_format: 保存格式
    
    Returns:
        List[str]: 保存的文件路径列表
    """
    if df.empty:
        logger.warning("没有筛选结果需要保存")
        return []
    
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            'data',
            'screening_results'
        )
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f"{strategy_name}_{market.value}_{timestamp}"
    
    output_files = []
    
    # 保存CSV格式
    if save_format in ['csv', 'both']:
        csv_path = os.path.join(output_dir, f"{base_name}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        output_files.append(csv_path)
    
    # 保存JSON格式
    if save_format in ['json', 'both']:
        json_path = os.path.join(output_dir, f"{base_name}.json")
        
        # 将DataFrame转换为可序列化的格式
        results_dict = {
            'metadata': {
                'strategy': strategy_name,
                'market': market.value,
                'timestamp': datetime.now().isoformat(),
                'total_stocks': len(df)
            },
            'stocks': df.to_dict('records')
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
        
        output_files.append(json_path)
    
    return output_files


if __name__ == "__main__":
    """测试函数"""
    import sys
    
    print("=" * 80)
    print("基本面筛选器 - 测试模式")
    print("=" * 80)
    
    # 测试参数
    test_market = MarketCode.CN
    test_strategy = "munger"
    test_max_stocks = 100  # 只测试100只股票
    
    print(f"\n测试参数:")
    print(f"  市场: {test_market.value}")
    print(f"  策略: {test_strategy}")
    print(f"  最大股票数: {test_max_stocks}")
    print(f"  输出目录: core_bak_refactored/data/screening_results")
    
    try:
        results = screen_fundamental_stocks(
            market=test_market,
            strategy_name=test_strategy,
            max_stocks=test_max_stocks,
            output_dir=None,
            save_format="both"
        )
        
        if results['status'] == 'success':
            print(f"\n✓ 测试成功!")
            print(f"  总股票数: {results['total_stocks']}")
            print(f"  筛选结果: {results['screened_stocks']}")
            print(f"  耗时: {results['duration_seconds']:.2f}秒")
            print(f"\n  输出文件:")
            for file_path in results['output_files']:
                print(f"    - {file_path}")
        else:
            print(f"\n✗ 测试失败: {results.get('message', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
