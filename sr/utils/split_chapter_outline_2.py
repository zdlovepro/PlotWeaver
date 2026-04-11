#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小说情节提取脚本 - 修复版
修复章节标题识别问题，支持多种格式
新增：支持从“起因/冲突/转折/高潮/收束”等分段提取因果链，兼容无加粗/加粗/不同标点
"""

import re
import os
import sys
from pathlib import Path


def extract_chapter_number(filename):
    match = re.match(r'(\d+)\.txt$', filename)
    if match:
        return int(match.group(1))
    return None


def _clean_lines(block: str) -> str:
    lines = block.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            cleaned.append(stripped)
    return '\n'.join(cleaned)


def extract_causal_blocks_loose(content: str) -> str:
    """
    宽松提取：匹配起因/冲突/转折/高潮/收束，不要求粗体或行首井号。
    兼容：
      起因：...
      **起因：**
      【起因】...
      起因) ...
    """
    txt = content.replace('\r', '')
    # 捕获当前标签后的内容，直到下一个标签或文本结束
    pattern = re.compile(
        r'(?:^|\n)\s*\*{0,2}\s*(起因|冲突|转折|高潮|收束)\s*[\uFF1A:】)>）]?\s*\*{0,2}\s*'  # 标签行
        r'(.*?)(?=\n\s*\*{0,2}\s*(起因|冲突|转折|高潮|收束)\s*[\uFF1A:】)>）]?\s*\*{0,2}\s*|\Z)',
        re.S
    )
    blocks = []
    for m in pattern.finditer(txt):
        label = m.group(1)
        body = _clean_lines(m.group(2))
        if body:
            blocks.append(f"{label}：\n{body}")
    return "\n\n".join(blocks)


def extract_section(content, section_title):
    """
    提取指定章节的内容；先尝试“## ... section_title ...”标题，
    若 section_title 为因果链条，则尝试粗体分段与宽松标签。
    """
    # 1) 严格标题匹配（## 一、XXX section_title ...）
    pattern = rf'## [一二三四五六七八九十]+、[^\n]*{section_title}[^\n]*\n(.*?)(?=## [一二三四五六七八九十]+、|---|$)'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return _clean_lines(match.group(1).strip())

    # 2) 宽松标题匹配（## ...section_title...）
    pattern2 = rf'##[^\n]*{section_title}[^\n]*\n(.*?)(?=##|---|$)'
    match2 = re.search(pattern2, content, re.DOTALL)
    if match2:
        return _clean_lines(match2.group(1).strip())

    # 3) 特殊处理：因果链条缺失时，尝试粗体/宽松标签
    if section_title == '因果链条':
        # 粗体行块（起因/冲突/转折/高潮/收束）
        label_pat = re.compile(
            r'^\s*\*\*\s*(起因|冲突|转折|高潮|收束)[：:]*\s*\*\*\s*\n(.*?)(?=^\s*\*\*|\Z)',
            re.MULTILINE | re.DOTALL
        )
        blocks = []
        for m in label_pat.finditer(content):
            label = m.group(1)
            body = _clean_lines(m.group(2).strip())
            if body:
                blocks.append(f"{label}：\n{body}")
        if blocks:
            return "\n\n".join(blocks)

        # 更宽松：不要求粗体/行首
        loose = extract_causal_blocks_loose(content)
        if loose:
            return loose

    return ""


def extract_chapter_title(text):
    """
    提取章节标题 - 支持多种格式，包括纯中文“第X章”无书名号
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

    # 格式3: # 第15章《怀疑》详细梗概
    pattern3 = r'#\s*第(\d+)章[《\s]*(.*?)[》\s]*\n'
    match3 = re.search(pattern3, text)
    if match3:
        chapter_num = match3.group(1)
        chapter_name = match3.group(2).strip()
        chapter_name = re.sub(r'详细梗概.*$', '', chapter_name).strip()
        if chapter_name:
            return f"第{chapter_num}章《{chapter_name}》"

    # 格式4: 纯中文/阿拉伯数字“第X章”标题，可含空格、无书名号
    pattern4 = r'第\s*([一二三四五六七八九十百零〇两\d]+)\s*章\s*([^\n《》]*)'
    match4 = re.search(pattern4, text)
    if match4:
        chapter_num_raw = match4.group(1)
        chapter_name = match4.group(2).strip()
        chapter_name = re.sub(r'详细梗概.*$', '', chapter_name).strip()
        if chapter_name:
            return f"第{chapter_num_raw}章《{chapter_name}》"
        return f"第{chapter_num_raw}章"

    # 兜底
    pattern5 = r'第\s*(\d+)\s*章'
    match5 = re.search(pattern5, text)
    if match5:
        chapter_num = match5.group(1)
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

    result['chapter_title'] = extract_chapter_title(text)
    result['causal_chain'] = extract_section(text, '因果链条')
    result['character_motivation'] = extract_section(text, '人物动机')

    return result


def process_files(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"❌ 错误：输入目录不存在\n{input_dir}")
        return False

    output_path.mkdir(parents=True, exist_ok=True)

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
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ 读取失败 {file_path.name}: {e}")
            continue

        extracted = extract_content(content)

        if not extracted['chapter_title']:
            print(f"\n🔍 调试 {file_path.name} 开头内容：")
            preview = content[:200].replace('\n', '\\n')
            print(f"   {preview}...")

        if not extracted['chapter_title']:
            print(f"⚠️  跳过 {file_path.name}：未找到章节标题")
            skip_count += 1
            continue

        if not extracted['causal_chain']:
            print(f"⚠️  跳过 {file_path.name}：未找到因果链条")
            skip_count += 1
            continue

        output_lines = [extracted['chapter_title'], "", "## 因果链条", "", extracted['causal_chain']]

        if extracted['character_motivation']:
            output_lines.extend(["", "## 主要人物动机与关系演变", "", extracted['character_motivation']])
            status = "✅"
        else:
            status = "🟡"
            partial_count += 1

        output_content = '\n'.join(output_lines)
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
    print("修复：支持多种章节标题格式；支持起因/冲突/转折/高潮/收束分段提取因果链（宽松匹配）")
    print("=" * 60)

    default_input = r"D:\_study\小说大纲\第三轮尝试\一念永恒_章"
    default_output = r"D:\_study\小说大纲\第三轮尝试\一念永恒_章_情节"

    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    else:
        use_default = input(f"使用默认输入路径？\n{default_input}\n(Y/n): ").strip().lower()
        if use_default in ('', 'y', 'yes'):
            input_dir = default_input
        else:
            input_dir = input("请输入输入目录路径: ").strip().strip('"')

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