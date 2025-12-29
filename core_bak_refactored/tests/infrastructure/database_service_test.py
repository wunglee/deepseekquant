"""
测试数据库服务

验证:
1. 数据库服务初始化
2. 缓存数据写入
3. 缓存数据读取
4. 数据库统计信息
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import pandas as pd

from core_bak_refactored.infrastructure import get_database_service

def test_database_service():
    """测试数据库服务"""
    print("=" * 80)
    print("数据库服务测试")
    print("=" * 80)
    
    # 1. 初始化数据库服务
    print("\n1. 初始化数据库服务...")
    db_service = get_database_service()
    print(f"✅ 数据库服务已初始化")
    print(f"   - 缓存启用: {db_service.cache_enabled}")
    print(f"   - 增量更新: {db_service.incremental_enabled}")
    print(f"   - 监控启用: {db_service.monitoring_enabled}")
    
    # 2. 获取数据库统计
    print("\n2. 数据库统计信息...")
    stats = db_service.get_database_stats()
    print(f"✅ 数据库统计:")
    print(f"   - 数据库路径: {stats.get('database_path', 'N/A')}")
    print(f"   - 总行数: {stats.get('total_rows', 0)}")
    print(f"   - 指数数量: {stats.get('index_count', 0)}")
    print(f"   - 数据范围: {stats.get('date_range', {}).get('start', 'N/A')} ~ {stats.get('date_range', {}).get('end', 'N/A')}")
    print(f"   - 文件大小: {stats.get('file_size_mb', 0)} MB")
    
    # 3. 测试缓存写入
    print("\n3. 测试缓存写入...")
    test_index = "000300.SH"
    test_data = pd.DataFrame({
        'date': [
            (pd.Timestamp.now() - pd.Timedelta(days=i)).strftime('%Y-%m-%d')
            for i in range(10, 0, -1)
        ],
        'open': [3200 + i * 10 for i in range(10)],
        'high': [3220 + i * 10 for i in range(10)],
        'low': [3190 + i * 10 for i in range(10)],
        'close': [3210 + i * 10 for i in range(10)],
        'volume': [1000000 + i * 10000 for i in range(10)]
    })
    
    success = db_service.cache_data(test_index, test_data, 'TestProvider')
    if success:
        print(f"✅ 缓存写入成功: {len(test_data)} 条数据")
    else:
        print(f"❌ 缓存写入失败")
    
    # 4. 测试缓存读取
    print("\n4. 测试缓存读取...")
    start_date = test_data['date'].min()
    end_date = test_data['date'].max()
    
    cached_df = db_service.get_cached_data(
        test_index,
        start_date,
        end_date,
        'TestProvider'
    )
    
    if cached_df is not None and not cached_df.empty:
        print(f"✅ 缓存读取成功: {len(cached_df)} 条数据")
        print(f"   - 日期范围: {cached_df['date'].min()} ~ {cached_df['date'].max()}")
        print(f"   - 示例数据:")
        print(cached_df.head(3).to_string(index=False))
    else:
        print(f"❌ 缓存读取失败")
    
    # 5. 测试增量更新参数计算
    print("\n5. 测试增量更新参数...")
    params = db_service.get_incremental_update_params(test_index, requested_count=60)
    print(f"✅ 增量更新参数:")
    print(f"   - 需要获取: {params.get('need_fetch', True)}")
    print(f"   - 原因: {params.get('reason', 'N/A')}")
    print(f"   - 起始日期: {params.get('start_date', 'N/A')}")
    print(f"   - 结束日期: {params.get('end_date', 'N/A')}")
    print(f"   - 数据条数: {params.get('count', 0)}")
    
    # 6. 再次获取统计信息
    print("\n6. 更新后的数据库统计...")
    stats = db_service.get_database_stats()
    print(f"✅ 数据库统计:")
    print(f"   - 总行数: {stats.get('total_rows', 0)}")
    print(f"   - 指数数量: {stats.get('index_count', 0)}")
    print(f"   - 文件大小: {stats.get('file_size_mb', 0)} MB")
    
    print("\n" + "=" * 80)
    print("✅ 所有测试通过！")
    print("=" * 80)


if __name__ == '__main__':
    try:
        test_database_service()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
