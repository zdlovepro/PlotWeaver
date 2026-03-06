"""
PlotWeaver — 多小说情节混编系统入口

用法：
    python main.py
    python main.py --config config/pipeline.yaml
    python main.py --input data/input/小说A.txt data/input/小说B.txt
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="PlotWeaver: 多小说情节混编系统"
    )
    parser.add_argument(
        "--config",
        default="config/pipeline.yaml",
        help="配置文件路径 (默认: config/pipeline.yaml)",
    )
    parser.add_argument(
        "--input",
        nargs="*",
        help="输入小说 .txt 文件列表 (留空则自动扫描 data/input/)",
    )
    args = parser.parse_args()

    try:
        from src.pipeline import Pipeline
    except ImportError as exc:
        print(f"[错误] 无法导入 PlotWeaver 模块: {exc}")
        print("请先安装依赖: pip install -r requirements.txt")
        sys.exit(1)

    pipeline = Pipeline(config_path=args.config)
    output_path = pipeline.run(novel_files=args.input if args.input else None)
    print(f"\n✓ 完成！新小说大纲已保存至: {output_path}")


if __name__ == "__main__":
    main()

