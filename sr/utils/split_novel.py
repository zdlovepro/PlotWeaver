#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小说章节分割脚本 - 修复版
处理章节标题重复问题
"""

import re
import os
import sys
from pathlib import Path


def extract_chapter_info(line):
    """
    提取章节编号和标题
    支持：第1章绯红 / 第1章 欢迎 / 第487章 这是高手
    """
    pattern = r'^第\s*(\d+)\s*章\s*(.*)$'
    match = re.match(pattern, line.strip())

    if match:
        chapter_num = int(match.group(1))
        chapter_title = match.group(2).strip()
        return chapter_num, chapter_title
    return None, None


def get_novel_name(file_path):
    """
    从文件路径提取小说名
    """
    filename = Path(file_path).stem
    novel_name = re.sub(r'[（(].*?[）)]', '', filename)
    novel_name = novel_name.strip()
    return novel_name


def split_novel(input_file):
    """
    分割小说文件
    """
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"❌ 错误：文件不存在\n{input_file}")
        return False

    # 获取小说名并创建输出目录
    novel_name = get_novel_name(input_file)
    output_dir = input_path.parent / f"{novel_name}_分割"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📖 小说名：{novel_name}")
    print(f"📂 输出目录：{output_dir}")
    print("-" * 50)

    # 读取文件
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print("✅ 使用UTF-8编码读取")
    except UnicodeDecodeError:
        try:
            with open(input_path, 'r', encoding='gbk') as f:
                content = f.read()
            print("✅ 使用GBK编码读取")
        except Exception as e:
            print(f"❌ 读取失败：{e}")
            return False

    # 按行分析
    lines = content.split('\n')
    chapters = []
    current_chapter = None
    current_lines = []
    last_chapter_num = None  # 记录上一个章节号，用于检测重复

    print("\n🔍 正在分析章节...")

    for line_num, line in enumerate(lines, 1):
        ch_num, ch_title = extract_chapter_info(line)

        if ch_num is not None:
            # 检测重复章节标题（同一章节号连续出现）
            if ch_num == last_chapter_num and current_chapter is not None:
                # 这是重复的标题行，直接跳过，不开启新章节
                print(f"   ⚠️  跳过重复标题：第{ch_num}章")
                continue

            # 保存上一章节（如果不是重复的）
            if current_chapter is not None:
                chapters.append({
                    'num': current_chapter['num'],
                    'title': current_chapter['title'],
                    'content': '\n'.join(current_lines)
                })

            # 开启新章节
            current_chapter = {'num': ch_num, 'title': ch_title}
            current_lines = [line]
            last_chapter_num = ch_num

            title_display = ch_title if ch_title else "(无标题)"
            print(f"   第{ch_num:03d}章 - {title_display}")
        else:
            if current_chapter is not None:
                current_lines.append(line)

    # 保存最后一章
    if current_chapter and current_lines:
        chapters.append({
            'num': current_chapter['num'],
            'title': current_chapter['title'],
            'content': '\n'.join(current_lines)
        })

    if not chapters:
        print("❌ 未找到章节，请检查文件格式")
        return False

    print(f"\n📊 共找到 {len(chapters)} 个章节")
    print("-" * 50)

    # 写入文件
    success = 0
    for ch in chapters:
        filename = f"{ch['num']:03d}.txt"
        filepath = output_dir / filename

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(ch['content'])
            success += 1
        except Exception as e:
            print(f"❌ 第{ch['num']:03d}章写入失败: {e}")

    print(f"\n✅ 分割完成！成功 {success}/{len(chapters)} 个文件")
    print(f"📁 输出位置：{output_dir.absolute()}")

    # 显示前几个文件名示例
    print(f"\n📋 文件示例：")
    for ch in chapters[:3]:
        title = ch['title'] if ch['title'] else "无标题"
        print(f"   {ch['num']:03d}.txt (第{ch['num']}章 {title})")
    if len(chapters) > 3:
        print(f"   ... 共 {len(chapters)} 个文件")

    return True


def main():
    print("=" * 60)
    print("📚 小说章节分割工具（修复重复标题版）")
    print("=" * 60)

    # 默认路径
    default_path = r"D:\_study\小说大纲\第三轮尝试\一念永恒(1-500章).txt"


    # 获取输入
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        use_default = input(f"使用默认路径？\n{default_path}\n(Y/n): ").strip().lower()
        if use_default in ('', 'y', 'yes'):
            input_file = default_path
        else:
            input_file = input("请输入文件路径: ").strip().strip('"')

    print()
    split_novel(input_file)

    input("\n按回车键退出...")


if __name__ == "__main__":
    main()