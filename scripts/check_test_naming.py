#!/usr/bin/env python3
"""
测试文件命名规范检查脚本

检查规则：
1. 每个源文件必须有对应的测试文件（*_test.py）
2. 测试文件必须以 _test.py 结尾，禁止 test_*.py 格式
3. 一个源文件只能有一个对应的单元测试文件
4. 目录结构必须镜像（tests/ 镜像 core/ 或 infrastructure/）
"""

from pathlib import Path
from typing import List, Set, Tuple
import sys


class TestNamingValidator:
    """测试文件命名规范验证器"""
    
    def __init__(self, root_dir: str = "core_bak_refactored"):
        self.root = Path(root_dir)
        self.core_dir = self.root / "core"
        self.infra_dir = self.root / "infrastructure"
        self.tests_dir = self.root / "tests"
        
        # 需要一一对应的目录
        self.required_dirs = [
            "core/risk",
            "core/share",
            "infrastructure",
        ]
        
    def find_source_files(self, directory: Path) -> Set[Path]:
        """查找源文件（排除__init__.py）"""
        source_files = set()
        if directory.exists():
            for py_file in directory.rglob("*.py"):
                if py_file.name != "__init__.py":
                    source_files.add(py_file)
        return source_files
    
    def get_expected_test_file(self, source_file: Path) -> Path:
        """获取源文件对应的预期测试文件路径"""
        # 计算相对路径
        if str(source_file).startswith(str(self.core_dir)):
            rel_path = source_file.relative_to(self.core_dir)
            test_base = self.tests_dir / "units" / "core" / rel_path.parent
        elif str(source_file).startswith(str(self.infra_dir)):
            rel_path = source_file.relative_to(self.infra_dir)
            test_base = self.tests_dir / "infrastructure" / rel_path.parent
        else:
            return None
        
        # 生成测试文件名
        test_name = source_file.stem + "_test.py"
        return test_base / test_name
    
    def find_related_tests(self, source_file: Path) -> List[Path]:
        """查找与源文件相关的所有测试文件"""
        expected_test = self.get_expected_test_file(source_file)
        if not expected_test:
            return []
        
        test_dir = expected_test.parent
        if not test_dir.exists():
            return []
        
        base_name = source_file.stem
        related = []
        
        # 只查找精确匹配的测试文件，不包括带后缀的变体
        for test_file in test_dir.glob("*.py"):
            # 只匹配 {name}_test.py 或 test_{name}.py，不包括 {name}_enhanced_test.py 等变体
            if (test_file.stem == f"{base_name}_test" or
                test_file.stem == f"test_{base_name}"):
                related.append(test_file)
        
        return related
    
    def check_test_naming(self) -> List[str]:
        """检查测试文件命名规范"""
        errors = []
        
        # 检查所有需要一一对应的目录
        for dir_path in self.required_dirs:
            src_dir = self.root / dir_path
            if not src_dir.exists():
                continue
            
            # 查找所有源文件
            source_files = self.find_source_files(src_dir)
            
            for src_file in sorted(source_files):
                expected_test = self.get_expected_test_file(src_file)
                if not expected_test:
                    continue
                
                # 检查1：必须存在对应测试
                if not expected_test.exists():
                    errors.append(
                        f"❌ 缺少测试文件:\n"
                        f"   源文件: {src_file.relative_to(self.root)}\n"
                        f"   需要: {expected_test.relative_to(self.root)}"
                    )
                
                # 检查2：不允许 test_*.py 格式
                wrong_format = expected_test.parent / f"test_{src_file.stem}.py"
                if wrong_format.exists():
                    errors.append(
                        f"❌ 错误格式 (禁止 test_*.py):\n"
                        f"   错误: {wrong_format.relative_to(self.root)}\n"
                        f"   应改为: {expected_test.relative_to(self.root)}"
                    )
                
                # 检查3：不允许多个单元测试文件
                related_tests = self.find_related_tests(src_file)
                if len(related_tests) > 1:
                    errors.append(
                        f"❌ 一个源文件对应多个测试:\n"
                        f"   源文件: {src_file.relative_to(self.root)}\n"
                        f"   测试文件: {[str(t.relative_to(self.root)) for t in related_tests]}"
                    )
        
        # 检查所有测试文件
        for test_file in self.tests_dir.rglob("*_test.py"):
            # 检查4：测试文件是否在正确位置
            if not self._is_valid_test_location(test_file):
                errors.append(
                    f"⚠️  测试文件位置可能不正确:\n"
                    f"   {test_file.relative_to(self.root)}"
                )
        
        # 检查 test_*.py 格式的测试文件
        for test_file in self.tests_dir.rglob("test_*.py"):
            if self._is_unit_test_location(test_file):
                errors.append(
                    f"❌ 禁止 test_*.py 格式 (单元测试):\n"
                    f"   {test_file.relative_to(self.root)}\n"
                    f"   应改为: {test_file.stem[5:]}_test.py"
                )
        
        return errors
    
    def _is_valid_test_location(self, test_file: Path) -> bool:
        """检查测试文件是否在有效位置"""
        # 排除特殊目录（白名单）
        special_dirs = ["integration", "performance", "benchmarks", "e2e", "validation", "fixtures", "common"]
        for special in special_dirs:
            if special in test_file.parts:
                return True
        
        # 检查是否在镜像目录中
        try:
            rel_path = test_file.relative_to(self.tests_dir)
            if rel_path.parts[0] in ["units", "infrastructure"]:
                return True
        except ValueError:
            pass
        
        return False
    
    def _is_unit_test_location(self, test_file: Path) -> bool:
        """判断是否在单元测试位置（非集成/性能测试目录）"""
        special_dirs = ["integration", "performance", "benchmarks", "e2e", "validation", "fixtures", "common"]
        for special in special_dirs:
            if special in test_file.parts:
                return False
        return True
    
    def generate_report(self) -> Tuple[int, int]:
        """生成检查报告"""
        errors = self.check_test_naming()
        
        print("=" * 80)
        print("测试文件命名规范检查报告")
        print("=" * 80)
        
        if not errors:
            print("\n✅ 所有测试文件命名符合规范！")
            print(f"\n检查目录: {self.root}")
            print(f"源文件目录: {self.required_dirs}")
            return 0, 0
        
        print(f"\n发现 {len(errors)} 个问题:\n")
        for i, error in enumerate(errors, 1):
            print(f"{i}. {error}\n")
        
        print("=" * 80)
        print("修复建议:")
        print("=" * 80)
        print("1. 重命名 test_*.py 为 *_test.py")
        print("2. 合并重复的单元测试文件")
        print("3. 为缺少测试的源文件创建对应的 *_test.py")
        print("4. 集成测试应移到 tests/integration/")
        print("5. 性能测试应移到 tests/performance/")
        print("=" * 80)
        
        return len(errors), 0


def main():
    """主函数"""
    validator = TestNamingValidator()
    error_count, warning_count = validator.generate_report()
    
    if error_count > 0:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
