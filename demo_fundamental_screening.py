#!/usr/bin/env python3
"""
演示脚本 - 基本面筛选功能

运行方式:
    python demo_fundamental_screening.py
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core_bak_refactored.core.share.market.market_enums import MarketCode
from core_bak_refactored.core.data.providers.akshare_provider import AKShareDataProvider


def demo_get_all_symbols():
    """演示获取股票列表"""
    print("\n" + "="*80)
    print("演示 1: 获取A股股票列表")
    print("="*80)
    
    try:
        provider = AKShareDataProvider()
        
        if not provider.available:
            print("❌ AKShare 不可用")
            return False
        
        print("正在获取A股列表，请稍候...")
        df = provider.get_all_symbols(MarketCode.CN)
        
        if df.empty:
            print("❌ 未能获取股票列表")
            return False
        
        print(f"✓ 成功获取 {len(df)} 只股票")
        print("\n前10只股票:")
        print(df.head(10).to_string(index=False))
        return True
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def demo_get_fundamental_data():
    """演示获取基本面数据"""
    print("\n" + "="*80)
    print("演示 2: 获取个股基本面数据")
    print("="*80)
    
    try:
        provider = AKShareDataProvider()
        
        if not provider.available:
            print("❌ AKShare 不可用")
            return False
        
        # 使用平安银行作为示例
        symbol = "000001.SZ"
        print(f"正在获取 {symbol} 的基本面数据...")
        
        data = provider.get_complete_fundamental_data(symbol)
        
        if not data:
            print("❌ 未能获取基本面数据")
            return False
        
        print(f"✓ 成功获取 {symbol} 的基本面数据")
        print(f"\n股票名称: {data.get('name', 'N/A')}")
        print(f"市盈率(PE): {data.get('pe', 'N/A')}")
        print(f"市净率(PB): {data.get('pb', 'N/A')}")
        print(f"净资产收益率(ROE): {data.get('roe', 'N/A')}")
        print(f"资产负债率: {data.get('资产负债率', 'N/A')}")
        print(f"营业收入增长率: {data.get('revenue_growth', 'N/A')}")
        print(f"净利润增长率: {data.get('profit_growth', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_fundamental_screening():
    """演示基本面筛选"""
    print("\n" + "="*80)
    print("演示 3: 基本面筛选（使用模拟数据）")
    print("="*80)
    
    try:
        from core_bak_refactored.core.strategy.fundamentals.fundamental_screener import screen_fundamental_stocks
        
        print("正在执行芒格策略筛选...")
        
        # 使用少量股票进行演示
        result = screen_fundamental_stocks(
            market=MarketCode.CN,
            strategy_name="munger",
            output_dir="demo_output",
            max_stocks=50,  # 只处理50只股票
            save_format="csv"
        )
        
        if result['status'] == 'success':
            print(f"✓ 筛选完成！")
            print(f"  总股票数: {result['total_stocks']}")
            print(f"  筛选结果: {result['screened_stocks']}")
            print(f"  耗时: {result['duration_seconds']:.2f}秒")
            print(f"\n输出文件:")
            for file_path in result['output_files']:
                print(f"  - {file_path}")
            
            if result['screened_stocks'] > 0:
                print(f"\n筛选结果预览:")
                df = result['results']
                print(df[['symbol', 'name', 'score', 'pe', 'pb', 'roe']].to_string(index=False))
            
            return True
        else:
            print(f"❌ 筛选失败: {result.get('message', '未知错误')}")
            return False
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("="*80)
    print("基本面筛选功能演示")
    print("="*80)
    
    results = []
    
    # 运行演示
    results.append(("获取股票列表", demo_get_all_symbols()))
    results.append(("获取基本面数据", demo_get_fundamental_data()))
    results.append(("基本面筛选", demo_fundamental_screening()))
    
    # 总结
    print("\n" + "="*80)
    print("演示总结")
    print("="*80)
    
    for name, success in results:
        status = "✓ 通过" if success else "❌ 失败"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("\n🎉 所有演示通过！基本面筛选功能正常工作。")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 项演示失败，请检查错误信息。")
        return 1


if __name__ == '__main__':
    sys.exit(main())
