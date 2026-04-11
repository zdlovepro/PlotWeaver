#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小说情节提取脚本
从章节大纲中提取：章节标题 + 因果链条
"""

import re
import os
import sys
from pathlib import Path


def extract_chapter_number(filename):
    """
    从文件名提取章节数字
    001.txt -> 1
    """
    match = re.match(r'(\d+)\.txt$', filename)
    if match:
        return int(match.group(1))
    return None


def extract_content(text):
    """
    提取章节标题和因果链条
    """
    result = {
        'chapter_title': '',
        'causal_chain': ''
    }

    # 提取章节标题 - 匹配 # 第X章《标题》或 # 第X章 标题
    title_pattern = r'#\s*第(\d+)章[《\s]*(.*?)[》\s]*\n'
    title_match = re.search(title_pattern, text)
    if title_match:
        chapter_num = title_match.group(1)
        chapter_name = title_match.group(2).strip()
        result['chapter_title'] = f"第{chapter_num}章《{chapter_name}》"

    # 提取因果链条部分
    # 查找 "## 一、因果链条" 到下一个 "##" 或 "---" 之间的内容
    causal_pattern = r'## 一、因果链条(.*?)(?=##|---|$)'
    causal_match = re.search(causal_pattern, text, re.DOTALL)

    if causal_match:
        causal_content = causal_match.group(1).strip()

        # 清理内容：移除多余的空行，但保留结构
        lines = causal_content.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped:
                cleaned_lines.append(line)

        result['causal_chain'] = '\n'.join(cleaned_lines)

    return result


def process_files(input_dir, output_dir):
    """
    处理输入目录中的所有txt文件
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"❌ 错误：输入目录不存在\n{input_dir}")
        return False

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取所有txt文件并按数字排序
    txt_files = []
    for file in input_path.glob('*.txt'):
        num = extract_chapter_number(file.name)
        if num is not None:
            txt_files.append((num, file))

    txt_files.sort(key=lambda x: x[0])

    if not txt_files:
        print(f"❌ 未找到任何txt文件在：{input_dir}")
        return False

    print(f"📂 输入目录：{input_path}")
    print(f"📂 输出目录：{output_path}")
    print(f"📊 找到 {len(txt_files)} 个文件")
    print("-" * 60)

    success_count = 0
    skip_count = 0

    for num, file_path in txt_files:
        try:
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ 读取失败 {file_path.name}: {e}")
            continue

        # 提取内容
        extracted = extract_content(content)

        # 检查是否提取成功
        if not extracted['chapter_title'] or not extracted['causal_chain']:
            print(f"⚠️  跳过 {file_path.name}：未找到章节标题或因果链条")
            skip_count += 1
            continue

        # 构建输出内容
        output_content = f"""{extracted['chapter_title']}

## 因果链条

{extracted['causal_chain']}
"""

        # 保存文件
        output_filename = f"{num:03d}.txt"
        output_file_path = output_path / output_filename

        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(output_content)
            success_count += 1
            print(f"✅ {output_filename} <- {extracted['chapter_title']}")
        except Exception as e:
            print(f"❌ 写入失败 {output_filename}: {e}")

    print("-" * 60)
    print(f"✅ 完成！成功 {success_count}，跳过 {skip_count}，总计 {len(txt_files)}")
    print(f"📁 输出位置：{output_path.absolute()}")

    return True


def main():
    print("=" * 60)
    print("📚 小说情节提取工具")
    print("提取内容：章节标题 + 因果链条")
    print("=" * 60)

    # 默认路径
    default_input = r"D:\_study\小说大纲\第三轮尝试\仙逆_章"
    default_output = r"D:\_study\小说大纲\第三轮尝试\仙逆_章_情节"

    # 获取输入路径
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    else:
        use_default = input(f"使用默认输入路径？\n{default_input}\n(Y/n): ").strip().lower()
        if use_default in ('', 'y', 'yes'):
            input_dir = default_input
        else:
            input_dir = input("请输入输入目录路径: ").strip().strip('"')

    # 获取输出路径
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        use_default_out = input(f"使用默认输出路径？\n{default_output}\n(Y/n): ").strip().lower()
        if use_default_out in ('', 'y', 'yes'):
            output_dir = default_output
        else:
            output_dir = input("请输入输出目录路径: ").strip().strip('"')

    print()
    process_files(input_dir, output_dir)

    input("\n按回车键退出...")


if __name__ == "__main__":
    main()