import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from src.config.setting import Config


class FileManager:
    """文件管理器"""

    def __init__(self):
        self.config = Config()
        self._setup_logging()

    def _setup_logging(self):
        """设置日志"""
        log_file = self.config.LOGS_DIR / f"fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def read_input_files(self, file_a: Optional[Path] = None, file_b: Optional[Path] = None) -> Dict[str, str]:
        """读取输入文件"""
        file_a = file_a or self.config.DEFAULT_INPUT_FILES["outline_a"]
        file_b = file_b or self.config.DEFAULT_INPUT_FILES["outline_b"]

        outlines = {}

        try:
            if file_a.exists():
                with open(file_a, 'r', encoding='utf-8') as f:
                    outlines["outline_a"] = f.read().strip()
            else:
                self.logger.warning(f"文件 {file_a} 不存在，将使用默认示例")
                outlines["outline_a"] = self._get_default_outline_a()

            if file_b.exists():
                with open(file_b, 'r', encoding='utf-8') as f:
                    outlines["outline_b"] = f.read().strip()
            else:
                self.logger.warning(f"文件 {file_b} 不存在，将使用默认示例")
                outlines["outline_b"] = self._get_default_outline_b()

        except Exception as e:
            self.logger.error(f"读取输入文件时出错: {e}")
            outlines = {
                "outline_a": self._get_default_outline_a(),
                "outline_b": self._get_default_outline_b()
            }

        return outlines

    def _get_default_outline_a(self) -> str:
        """获取默认大纲A"""
        return """
        主角李明是一个普通的程序员，某天发现自己能看见别人的情绪颜色。
        他利用这个能力帮助同事解决心理问题，但逐渐发现这个能力有副作用。
        最终他学会了控制这个能力，并找到了真正的自我。
        """

    def _get_default_outline_b(self) -> str:
        """获取默认大纲B"""
        return """
        一个神秘组织"色彩猎手"在追捕有特殊能力的人。
        他们试图控制李明，利用他的能力进行犯罪。
        李明在对抗中发现组织的惊天秘密。
        """

    def save_result(self, result: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """保存结果到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fusion_result_{timestamp}.json"

        output_file = self.config.RESULTS_DIR / filename

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            self.logger.info(f"结果已保存到: {output_file}")
            return output_file

        except Exception as e:
            self.logger.error(f"保存结果时出错: {e}")
            raise

    def save_outline_text(self, outline: str, filename: Optional[str] = None) -> Path:
        """保存大纲文本"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fused_outline_{timestamp}.txt"

        output_file = self.config.RESULTS_DIR / filename

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(outline)

            self.logger.info(f"大纲文本已保存到: {output_file}")
            return output_file

        except Exception as e:
            self.logger.error(f"保存大纲文本时出错: {e}")
            raise

    def list_previous_results(self) -> list:
        """列出之前的结果文件"""
        result_files = list(self.config.RESULTS_DIR.glob("fusion_result_*.json"))
        return sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True)

    def load_previous_result(self, filename: str) -> Dict[str, Any]:
        """加载之前的结果"""
        result_file = self.config.RESULTS_DIR / filename

        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"加载之前的结果时出错: {e}")
            raise