#!/usr/bin/env python3
"""
清空数据库缓存脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core_bak_refactored.infrastructure import get_database



def clear_database_cache():
    """清空数据库缓存"""
    print("🗑️ 清空数据库缓存脚本")
    print("=" * 50)
    
    try:
        # 初始化数据库
        db_path = project_root / "data" / "market_data.db"
        database = get_database('sqlite', database_path=str(db_path))
        database.connect()
        
        print(f"📁 数据库路径: {db_path}")
        
        # 查询当前记录数
        count_result = database.fetch_one("SELECT COUNT(*) as count FROM index_prices")
        current_count = count_result['count'] if count_result else 0
        print(f"📊 当前记录数: {current_count}")
        
        if current_count == 0:
            print("✅ 数据库已为空，无需清空")
            return
        
        # 清空表数据
        print("🗑️ 正在清空缓存表...")
        database.execute("DELETE FROM index_prices")
        database.commit()
        
        # 验证清空结果
        count_result = database.fetch_one("SELECT COUNT(*) as count FROM index_prices")
        final_count = count_result['count'] if count_result else 0
        
        print(f"✅ 清空完成！剩余记录数: {final_count}")
        
        # 关闭数据库连接
        database.close()
        
    except Exception as e:
        print(f"❌ 清空失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    clear_database_cache()