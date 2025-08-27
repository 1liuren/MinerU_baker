#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成单独JSONL文件的脚本
从results文件夹中读取已处理的PDF数据，生成单独的JSONL文件

功能：
1. 遍历输入文件夹中的所有子文件夹
2. 读取每个文件夹中的.md文件和_extracted_metadata.json文件
3. 可选：重新过滤HTML转markdown格式
4. 可选：检查和转换lang字段格式
5. 为每个PDF生成单独的JSONL文件
"""

import argparse
import json
import hashlib
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
from loguru import logger
from datetime import datetime
import traceback


class IndividualJsonlGenerator:
    """生成单独JSONL文件的生成器"""

    def __init__(self, input_dir: Path, output_dir: Path, convert_html: bool = False, fix_lang: bool = False):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.convert_html = convert_html
        self.fix_lang = fix_lang

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化HTML转换器（如果需要）
        self.html_converter = None
        if self.convert_html:
            try:
                from html_converter import HTMLToMarkdownConverter
                self.html_converter = HTMLToMarkdownConverter()
                logger.info("HTML转换器初始化成功")
            except ImportError as e:
                logger.warning(f"无法导入HTML转换器: {e}，跳过HTML转换")

    def normalize_lang_code(self, lang: str) -> str:
        """标准化语言代码"""
        if not lang:
            return "zh"

        lang = lang.lower().strip()

        # 常见的语言代码映射
        lang_mapping = {
            "english": "en",
            "chinese": "zh",
            "zh-cn": "zh",
            "zh-tw": "zh",
            "zhs": "zh",
            "zht": "zh",
            "jp": "ja",
            "japanese": "ja",
            "ko": "ko",
            "korean": "ko",
            "fr": "fr",
            "french": "fr",
            "de": "de",
            "german": "de",
            "es": "es",
            "spanish": "es",
            "it": "it",
            "italian": "it",
            "pt": "pt",
            "portuguese": "pt",
            "ru": "ru",
            "russian": "ru",
            "ar": "ar",
            "arabic": "ar"
        }

        return lang_mapping.get(lang, lang)

    def load_metadata(self, metadata_file: Path) -> Dict:
        """加载元数据文件"""
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # 如果启用lang修复，处理lang字段
            if self.fix_lang and "lang" in metadata:
                original_lang = metadata["lang"]
                metadata["lang"] = self.normalize_lang_code(original_lang)
                if original_lang != metadata["lang"]:
                    logger.info(f"语言代码已修复: {original_lang} -> {metadata['lang']}")

            # 确保基本字段存在
            metadata.setdefault("book_name", metadata_file.stem.replace("_extracted_metadata", ""))
            metadata.setdefault("lang", "zh")
            metadata.setdefault("type", "书籍")
            metadata.setdefault("processing_date", datetime.now().strftime("%Y-%m-%d"))

            return metadata

        except Exception as e:
            logger.warning(f"读取元数据文件失败 {metadata_file}: {e}")
            return {
                "book_name": metadata_file.stem.replace("_extracted_metadata", ""),
                "lang": "zh",
                "type": "书籍",
                "processing_date": datetime.now().strftime("%Y-%m-%d"),
            }

    def load_content(self, md_file: Path) -> str:
        """加载markdown内容"""
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 如果启用HTML转换，转换HTML内容
            if self.convert_html and self.html_converter:
                try:
                    original_content = content
                    content = self.html_converter.convert_html_in_text(content)
                    if original_content != content:
                        logger.success(f"HTML转换成功: {md_file.name}")
                except Exception as e:
                    logger.warning(f"HTML转换失败 {md_file}: {e}")

            return content

        except Exception as e:
            logger.error(f"读取markdown文件失败 {md_file}: {e}")
            return ""

    def create_jsonl_record(self, content: str, metadata: Dict) -> Dict:
        """创建JSONL记录"""
        return {
            "id": hashlib.sha256(content.encode('utf-8')).hexdigest(),
            "text": metadata.get("text", ""),
            "meta": {
                "data_info": {
                    "lang": metadata.get("lang", "zh"),
                    "source": metadata.get("publisher", ""),
                    "type": metadata.get("type", "书籍"),
                    "book_name": metadata.get("title", ""),
                    "book_content": content,
                    "author": metadata.get("author", ""),
                    "processing_date": metadata.get("processing_date", datetime.now().strftime("%Y-%m-%d"))
                },
                "knowledge_info": {
                    "category": metadata.get("category", []),
                    "knowledge": metadata.get("knowledge", [])
                }
            }
        }

    def process_single_pdf(self, pdf_dir: Path) -> bool:
        """处理单个PDF文件夹"""
        pdf_name = pdf_dir.name
        logger.info(f"开始处理: {pdf_name}")

        try:
            # 检查必要文件
            md_file = pdf_dir / f"{pdf_name}.md"
            metadata_file = pdf_dir / f"{pdf_name}_extracted_metadata.json"

            if not md_file.exists():
                logger.warning(f"跳过 {pdf_name}: 未找到markdown文件")
                return False

            # 加载内容和元数据
            content = self.load_content(md_file)
            if not content.strip():
                logger.warning(f"跳过 {pdf_name}: 内容为空")
                return False

            metadata = self.load_metadata(metadata_file) if metadata_file.exists() else {
                "book_name": pdf_name,
                "lang": "zh",
                "type": "书籍",
                "processing_date": datetime.now().strftime("%Y-%m-%d"),
            }

            # 创建JSONL记录
            jsonl_record = self.create_jsonl_record(content, metadata)

            # 生成输出文件
            output_file = self.output_dir / f"{pdf_name}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json.dumps(jsonl_record, ensure_ascii=False) + '\n')

            logger.success(f"成功生成: {output_file}")
            return True

        except Exception as e:
            logger.error(f"处理失败 {pdf_name}: {e}")
            logger.debug(traceback.format_exc())
            return False

    def generate_all_jsonl(self) -> Tuple[int, int]:
        """生成所有JSONL文件"""
        logger.info(f"开始批量生成JSONL文件")
        logger.info(f"输入目录: {self.input_dir}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"HTML转换: {'启用' if self.convert_html else '禁用'}")
        logger.info(f"语言修复: {'启用' if self.fix_lang else '禁用'}")

        if not self.input_dir.exists():
            logger.error(f"输入目录不存在: {self.input_dir}")
            return 0, 0

        success_count = 0
        total_count = 0

        # 遍历所有PDF子目录
        for pdf_dir in sorted(self.input_dir.iterdir()):
            if pdf_dir.is_dir():
                total_count += 1
                if self.process_single_pdf(pdf_dir):
                    success_count += 1

        logger.info(f"处理完成: {success_count}/{total_count} 成功")
        return success_count, total_count


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成单独的JSONL文件")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="输入目录路径，包含PDF子文件夹"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_jsonl",
        help="输出目录路径，默认为results_jsonl"
    )
    parser.add_argument(
        "--convert_html",
        action="store_true",
        help="是否重新过滤HTML转markdown格式"
    )
    parser.add_argument(
        "--fix_lang",
        action="store_true",
        help="是否检查和修复lang字段格式"
    )

    args = parser.parse_args()

    # 转换路径
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # 创建生成器并执行
    generator = IndividualJsonlGenerator(
        input_dir=input_dir,
        output_dir=output_dir,
        convert_html=args.convert_html,
        fix_lang=args.fix_lang
    )

    success_count, total_count = generator.generate_all_jsonl()

    if success_count > 0:
        logger.success(f"批量生成完成: {success_count}/{total_count} 个文件成功生成")
    else:
        logger.error("没有成功生成任何JSONL文件")


if __name__ == "__main__":
    main()
