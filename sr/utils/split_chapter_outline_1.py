#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小说情节提取脚本 - 修复版
修复章节标题识别问题，支持多种格式
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


def extract_section(content, section_title):
    """
    提取指定章节的内容
    """
    # 构建正则表达式，匹配 ## 一、XXX 或 ## 二、XXX 等格式
    pattern = rf'## [一二三四五六七八九十]+、[^\n]*{section_title}[^\n]*\n(.*?)(?=## [一二三四五六七八九十]+、|---|$)'
    match = re.search(pattern, content, re.DOTALL)

    if match:
        section_content = match.group(1).strip()
        # 清理内容：移除多余空行，但保留结构
        lines = section_content.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped:
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)

    # 如果没找到，尝试更宽松的匹配（不带序号）
    pattern2 = rf'##[^\n]*{section_title}[^\n]*\n(.*?)(?=##|---|$)'
    match2 = re.search(pattern2, content, re.DOTALL)
    if match2:
        section_content = match2.group(1).strip()
        lines = section_content.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped:
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)

    return ""


def extract_chapter_title(text):
    """
    提取章节标题 - 支持多种格式
    """
    # 格式1: # 第三章《测试》详细梗概
    pattern1 = r'#\s*第(\d+)章[《](.*?)[》]\s*详细梗概'
    match1 = re.search(pattern1, text)
    if match1:
        chapter_num = match1.group(1)
        chapter_name = match1.group(2).strip()
        return f"第{chapter_num}章《{chapter_name}》"

    # 格式2: # 第六章 势利 详细梗概（没有书名号）
    pattern2 = r'#\s*第(\d+)章\s+([^\n《》]+?)\s*详细梗概'
    match2 = re.search(pattern2, text)
    if match2:
        chapter_num = match2.group(1)
        chapter_name = match2.group(2).strip()
        return f"第{chapter_num}章《{chapter_name}》"

    # 格式3: # 第15章《怀疑》详细梗概（原始格式）
    pattern3 = r'#\s*第(\d+)章[《\s]*(.*?)[》\s]*\n'
    match3 = re.search(pattern3, text)
    if match3:
        chapter_num = match3.group(1)
        chapter_name = match3.group(2).strip()
        # 移除可能的后缀
        chapter_name = re.sub(r'详细梗概.*$', '', chapter_name).strip()
        if chapter_name:
            return f"第{chapter_num}章《{chapter_name}》"

    # 格式4: 更宽松的匹配，只要找到 第X章 就行
    pattern4 = r'第\s*(\d+)\s*章'
    match4 = re.search(pattern4, text)
    if match4:
        chapter_num = match4.group(1)
        # 尝试在这一行找标题
        line = text[:text.find('\n')].strip()
        # 提取章后面的内容
        title_match = re.search(rf'第\s*{chapter_num}\s*章[《\s]*(.*?)[》\s]*', line)
        if title_match:
            chapter_name = title_match.group(1).strip()
            # 移除详细梗概等后缀
            chapter_name = re.sub(r'详细梗概.*$', '', chapter_name).strip()
            if chapter_name and chapter_name != f'第{chapter_num}章':
                return f"第{chapter_num}章《{chapter_name}》"
        return f"第{chapter_num}章"

    return ""


def extract_content(text):
    """
    提取章节标题、因果链条和人物动机
    """
    result = {
        'chapter_title': '',
        'causal_chain': '',
        'character_motivation': ''
    }

    # 提取章节标题
    result['chapter_title'] = extract_chapter_title(text)

    # 提取因果链条
    result['causal_chain'] = extract_section(text, '因果链条')

    # 提取人物动机与关系演变
    result['character_motivation'] = extract_section(text, '人物动机')

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
    partial_count = 0
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

        # 调试信息：如果标题为空，显示文件开头内容
        if not extracted['chapter_title']:
            print(f"\n🔍 调试 {file_path.name} 开头内容：")
            preview = content[:200].replace('\n', '\\n')
            print(f"   {preview}...")

        # 检查是否提取成功
        if not extracted['chapter_title']:
            print(f"⚠️  跳过 {file_path.name}：未找到章节标题")
            skip_count += 1
            continue

        if not extracted['causal_chain']:
            print(f"⚠️  跳过 {file_path.name}：未找到因果链条")
            skip_count += 1
            continue

        # 构建输出内容
        output_lines = [extracted['chapter_title'], "", "## 因果链条", "", extracted['causal_chain']]

        # 如果有人员动机部分，也添加进去
        if extracted['character_motivation']:
            output_lines.extend(["", "## 主要人物动机与关系演变", "", extracted['character_motivation']])
            status = "✅"
        else:
            status = "🟡"
            partial_count += 1

        output_content = '\n'.join(output_lines)

        # 保存文件
        output_filename = f"{num:03d}.txt"
        output_file_path = output_path / output_filename

        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(output_content)

            if status == "✅":
                success_count += 1
            print(f"{status} {output_filename} <- {extracted['chapter_title']}")

        except Exception as e:
            print(f"❌ 写入失败 {output_filename}: {e}")

    print("-" * 60)
    print(f"📈 统计：完整提取 {success_count}，部分提取 {partial_count}，跳过 {skip_count}，总计 {len(txt_files)}")
    print(f"📁 输出位置：{output_path.absolute()}")

    return True


def main():
    print("=" * 60)
    print("📚 小说情节提取工具（修复版）")
    print("修复：支持多种章节标题格式")
    print("=" * 60)

    # 默认路径
    default_input = r"D:\_study\小说大纲\第三轮尝试\一念永恒_章"
    default_output = r"D:\_study\小说大纲\第三轮尝试\一念永恒_章_情节"

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