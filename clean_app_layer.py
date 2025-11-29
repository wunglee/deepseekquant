#!/usr/bin/env python3
"""
删除data_fetcher.py中的应用层代码（Dashboard和APIService）
保持领域层纯净
"""

file_path = 'core_bak_refactored/core/data/data_fetcher.py'

# 读取文件
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"原始文件: {len(lines)} 行")

# 删除5845-7834行（Dashboard注释到APIService结束）
# Python索引从0开始，所以是lines[5844:7834]
new_lines = lines[:5844]  # 保留1-5844行

# 添加迁移说明
migration_note = """
# ==================================================================================
# 应用层代码已迁移说明 (2025-11-27)
# ==================================================================================
# 
# [已删除并迁移] DataQualityDashboard 类 (约695行)
#   目标位置: core_bak_refactored/app/data_quality/dashboard.py
#
# [已删除并迁移] DataQualityAPIService 类 (约1289行)
#   目标位置: core_bak_refactored/app/data_quality/api_service.py
#
# 使用方法:
#   from core_bak_refactored.app.data_quality.dashboard import DataQualityDashboard
#   from core_bak_refactored.app.data_quality.api_service import DataQualityAPIService
#
# ==================================================================================

"""

new_lines.append(migration_note)
new_lines.extend(lines[7834:])  # 添加7835行到末尾

# 写入
with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"删除行数: {7834 - 5844} 行 (Dashboard + APIService)")
print(f"新文件: {len(new_lines)} 行")
print(f"✅ 应用层代码已从领域层删除完成")
