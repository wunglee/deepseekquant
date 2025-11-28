#!/usr/bin/env python3
import sys
import re
from pathlib import Path

REQUIRED_SECTIONS = [
    "## 📋 Phase边界声明（必须）",
    "## 📚 依赖上下文与设计文档清单（必须）",

    "## 🧩 代码评审",
    "### 上一轮答复摘要与本轮改进（必须)",
    "### ✅ 本轮改进验收清单（专家确认）",
    "### 本轮改进（清单与关键评审）（必须)",
    "## 业务视角的代码实现评审要点",
    "### 🧩 架构变更与影响（如果有）",
    "## 本轮业务问题（下一轮需解决，非本轮验收内容）",
    # 取消实施纲要强制要求
    "## 🔗 相关文件（参考）",
]

FIXED_SENTENCE = "重要：请尽可能详尽和充分，不要遗漏和简化，谢谢！"
DISALLOWED_PATTERNS = ["SPRINT.md", "TODO.md", "测试计划"]


def load_text(path: Path) -> str:
    if not path.exists():
        print(f"❌ 文件不存在: {path}")
        sys.exit(1)
    return path.read_text(encoding="utf-8", errors="ignore")


def check_chinese_language(text: str) -> list:
    errors = []
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    latin_letters = re.findall(r"[A-Za-z]", text)
    # 简单启发式：中文字符数至少300，且中文占比≥25%（避免路径/代码片段影响）
    total_chars = len(text)
    zh_count = len(chinese_chars)
    en_count = len(latin_letters)
    zh_ratio = zh_count / max(total_chars, 1)
    if zh_count < 300 or zh_ratio < 0.25:
        errors.append(f"语言不合规：中文字符数={zh_count}，中文占比={zh_ratio:.2f}")
    return errors


def check_required_sections(text: str) -> list:
    errors = []
    import re
    norm_text = re.sub(r"\s+", " ", text)
    for sec in REQUIRED_SECTIONS:
        # 忽略标题层级（# 前缀），只校验章节标题文本是否出现
        sec_title = re.sub(r"^\s*#+\s*", "", sec)
        if sec_title not in norm_text:
            errors.append(f"缺少锚点章节：{sec}")
    return errors


def check_honorific(text: str) -> list:
    errors = []
    # 要求在核心问题部分使用“请您确认”，这里检查全局至少出现4次
    count = text.count("请您确认")
    if count < 4:
        errors.append(f"称谓不统一：\"请您确认\"出现次数={count}，应≥4")
    return errors


def check_fixed_sentence(text: str) -> list:
    errors = []
    if FIXED_SENTENCE not in text:
        errors.append("缺少末尾固定语句：" + FIXED_SENTENCE)
    return errors


def check_disallowed_docs(text: str) -> list:
    errors = []
    for pat in DISALLOWED_PATTERNS:
        if pat in text:
            errors.append(f"不应在ask中出现内部项目管理文档：{pat}")
    return errors


def check_update_files_section(text: str) -> list:
    errors = []
    # “本次更新文件（必须）”已由代码评审块中的“本轮改进（清单与关键评审）/代码清单汇总”承载，此处不再强制要求
    return errors

def check_sections_adjacent(text: str) -> list:
    errors = []
    # 代码评审块在新迭代启动阶段可以整体缺省
    if "## 🧩 代码评审" not in text:
        return errors
    # 验收清单必须紧随“本轮改进（清单与关键评审）（必须）”之后
    try:
        idx_improve = text.index("### 本轮改进（清单与关键评审）（必须)")
        idx_accept = text.index("### ✅ 本轮改进验收清单（专家确认）")
    except ValueError:
        errors.append("存在代码评审块时，必须包含‘本轮改进（清单与关键评审）（必须）’和‘本轮改进验收清单（专家确认）’两个小节")
        return errors
    between = text[idx_improve:idx_accept]
    # 两者之间若出现其他二级或三级标题，则视为不相邻
    if re.search(r"\n## |\n### ", between):
        errors.append("验收清单必须紧随‘本轮改进（清单与关键评审）（必须）’之后，不得插入其他标题")
    return errors


def main():
    if len(sys.argv) < 2:
        print("用法：python scripts/validate_ask.py --path docs/ask.md")
        sys.exit(2)
    # 简单参数解析
    try:
        if sys.argv[1] == "--path":
            path = Path(sys.argv[2])
        else:
            path = Path(sys.argv[1])
    except Exception:
        print("参数错误：请使用 --path <file>")
        sys.exit(2)

    text = load_text(path)
    errors = []
    errors += check_chinese_language(text)
    errors += check_required_sections(text)
    errors += check_honorific(text)
    errors += check_fixed_sentence(text)
    errors += check_disallowed_docs(text)
    errors += check_update_files_section(text)
    errors += check_sections_adjacent(text)

    if errors:
        print("❌ ask 合规检查失败：")
        for e in errors:
            print(" - " + e)
        sys.exit(1)
    else:
        print("✅ ask 合规检查通过：中文与固定锚点校验均通过")
        sys.exit(0)


if __name__ == "__main__":
    main()
