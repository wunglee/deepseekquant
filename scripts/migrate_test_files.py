#!/usr/bin/env python3
"""
测试文件批量迁移脚本
将无法一一对应的测试文件迁移到专门分类目录
"""

import shutil
from pathlib import Path

def migrate_files():
    """执行文件迁移"""
    base = Path("/Users/wangli/Library/Mobile Documents/com~apple~CloudDocs/历史项目/projects/deepseekquant/core_bak_refactored/tests")
    
    # 创建目标目录结构
    dirs_to_create = [
        base / "integration" / "core" / "risk",
        base / "integration" / "infrastructure",
        base / "performance" / "core" / "risk",
        base / "performance" / "infrastructure",
        base / "validation" / "core" / "risk",
        base / "e2e" / "core" / "backtest",
        base / "fixtures" / "core" / "backtest",
    ]
    
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
        init_file = d / "__init__.py"
        if not init_file.exists():
            init_file.touch()
        print(f"✓ 确保目录存在: {d.relative_to(base)}")
    
    # 定义迁移映射
    migrations = [
        # 集成测试
        ("core/risk/currency_adapter_integration_test.py", "integration/core/risk/"),
        ("core/risk/currency_consistency_test.py", "integration/core/risk/"),
        ("core/risk/data_provider_integration_test.py", "integration/core/risk/"),
        ("core/risk/industry_parameter_validation_integration_test.py", "integration/core/risk/"),
        ("core/risk/international_support_test.py", "integration/core/risk/"),
        ("core/risk/risk_limits_integration_test.py", "integration/core/risk/"),
        
        # 性能测试
        ("core/risk/benchmark_parallel.py", "performance/core/risk/"),
        ("core/risk/benchmark_parallel_test.py", "performance/core/risk/"),
        ("core/risk/portfolio_risk_parallel_test.py", "performance/core/risk/"),
        ("infrastructure/risk_metrics_performance_test.py", "performance/infrastructure/"),
        
        # 业务验证
        ("core/risk/industry_parameter_analyzer_test.py", "validation/core/risk/"),
        ("core/risk/parameter_literature_validation_test.py", "validation/core/risk/"),
        ("core/risk/risk_limits_p1_3_test.py", "validation/core/risk/"),
        
        # 端到端测试
        ("core/backtest/end_to_end_integration_test.py", "e2e/core/backtest/"),
        
        # 测试辅助
        ("core/backtest/test_fixtures.py", "fixtures/core/backtest/"),
    ]
    
    moved_count = 0
    skipped_count = 0
    
    for src_rel, dst_rel in migrations:
        src = base / src_rel
        dst_dir = base / dst_rel
        dst = dst_dir / src.name
        
        if not src.exists():
            print(f"⚠️  源文件不存在: {src_rel}")
            skipped_count += 1
            continue
        
        if dst.exists():
            print(f"⚠️  目标文件已存在: {dst_rel}{src.name}")
            skipped_count += 1
            continue
        
        try:
            shutil.move(str(src), str(dst))
            print(f"✓ 迁移: {src_rel} -> {dst_rel}{src.name}")
            moved_count += 1
        except Exception as e:
            print(f"❌ 迁移失败: {src_rel} - {e}")
            skipped_count += 1
    
    print(f"\n{'='*60}")
    print(f"迁移完成:")
    print(f"  成功: {moved_count} 个文件")
    print(f"  跳过: {skipped_count} 个文件")
    print(f"{'='*60}")
    
    return moved_count, skipped_count

if __name__ == "__main__":
    migrate_files()
