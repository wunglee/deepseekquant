"""简单测试数据库基础功能"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core_bak_refactored.infrastructure.database import get_database, MarketDataRepository
import pandas as pd


def test_basic_database():
    """测试数据库基础功能"""
    print("=" * 80)
    print("数据库基础功能测试")
    print("=" * 80)
    
    # 1. 创建数据库
    print("\n1. 创建数据库...")
    db_path = "data/test_market.db"
    os.makedirs("data", exist_ok=True)
    
    db = get_database('sqlite', database_path=db_path)
    db.connect()
    print(f"✅ 数据库已创建: {db_path}")
    
    # 2. 创建 Repository
    print("\n2. 创建 Repository...")
    repo = MarketDataRepository(db)
    print("✅ Repository 已创建")
    
    # 3. 插入测试数据
    print("\n3. 插入测试数据...")
    test_data = pd.DataFrame({
        'date': [(pd.Timestamp.now() - pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(10, 0, -1)],
        'open': [3200 + i * 10 for i in range(10)],
        'high': [3220 + i * 10 for i in range(10)],
        'low': [3190 + i * 10 for i in range(10)],
        'close': [3210 + i * 10 for i in range(10)],
        'volume': [1000000 + i * 10000 for i in range(10)]
    })
    
    row_count = repo.insert_prices("000300.SH", test_data, "TestProvider")
    print(f"✅ 插入成功: {row_count} 条")
    
    # 4. 查询数据
    print("\n4. 查询数据...")
    start_date = test_data['date'].min()
    end_date = test_data['date'].max()
    
    df = repo.query_prices("000300.SH", start_date, end_date)
    print(f"✅ 查询成功: {len(df)} 条")
    print(df.head(3))
    
    # 5. 获取最新日期
    print("\n5. 获取最新日期...")
    latest = repo.get_latest_date("000300.SH")
    print(f"✅ 最新日期: {latest}")
    
    # 6. 获取日期范围
    print("\n6. 获取日期范围...")
    date_range = repo.get_date_range("000300.SH")
    print(f"✅ 日期范围: {date_range}")
    
    # 7. 关闭数据库
    print("\n7. 关闭数据库...")
    db.close()
    print("✅ 数据库已关闭")
    
    print("\n" + "=" * 80)
    print("✅ 所有测试通过！")
    print("=" * 80)

if __name__ == '__main__':
    try:
        test_basic_database()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
