import re
import os


def split_novel_chapters():
    """
    将小说大纲按章节拆分为独立文件
    源文件: D:\_study\小说大纲\第三轮尝试\洗稿1\washed.txt
    输出目录: D:\_study\小说大纲\第三轮尝试\洗稿1\
    生成文件: 001.txt, 002.txt, ..., 050.txt
    """

    # 定义文件路径
    source_file = r"D:\_study\小说大纲\第三轮尝试\洗稿1\washed.txt"
    output_dir = r"D:\_study\小说大纲\第三轮尝试\洗稿1"

    # 检查源文件是否存在
    if not os.path.exists(source_file):
        print(f"错误：源文件不存在 - {source_file}")
        print("请确保文件路径正确，或修改脚本中的 source_file 变量")
        return

    # 读取源文件内容
    print(f"正在读取源文件: {source_file}")
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"文件读取成功，总字符数: {len(content)}")

    # 使用正则表达式匹配章节
    # 匹配模式：### **章节 CXXX：标题**
    chapter_pattern = r'###\s*\*\*章节\s*C(\d+)：(.+?)\*\*\n(.*?)(?=###\s*\*\*章节|\Z)'

    # 查找所有章节
    chapters = re.findall(chapter_pattern, content, re.DOTALL)

    if not chapters:
        print("未找到任何章节，请检查文件格式是否正确")
        return

    print(f"找到 {len(chapters)} 个章节")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # 处理每个章节
    success_count = 0
    for chapter_num, chapter_title, chapter_content in chapters:
        # 格式化章节号为3位数字（001, 002, ...）
        file_num = chapter_num.zfill(3)
        filename = f"{file_num}.txt"
        filepath = os.path.join(output_dir, filename)

        # 构建章节完整内容（保留原标题和格式）
        full_content = f"### **章节 C{chapter_num}：{chapter_title.strip()}**\n\n{chapter_content.strip()}\n"

        # 写入文件
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(full_content)
            print(f"✓ 已创建: {filename} - {chapter_title.strip()}")
            success_count += 1
        except Exception as e:
            print(f"✗ 创建失败: {filename} - 错误: {e}")

    print(f"\n{'=' * 50}")
    print(f"拆分完成！成功生成 {success_count}/{len(chapters)} 个文件")
    print(f"文件保存在: {output_dir}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    split_novel_chapters()