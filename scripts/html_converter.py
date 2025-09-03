#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTML转Markdown转换器模块
使用 markdownify 开源库进行更强大的HTML到Markdown转换
"""

import re
from loguru import logger


from markdownify import markdownify as md  # type: ignore



class HTMLToMarkdownConverter:
    """HTML转Markdown转换器，支持表格和通用HTML转换"""

    def __init__(self):
        # 如果有 markdownify，使用它；否则使用简单的正则表达式
        self.use_markdownify = md is not None
        if self.use_markdownify:
            logger.info("使用 markdownify 进行 HTML 转换")
        else:
            logger.info("回退到正则表达式处理 HTML")

        # 备用：简单的HTML表格转换正则表达式
        self.html_table_pattern = re.compile(r'<table[^>]*>(.*?)</table>', re.DOTALL | re.IGNORECASE)
        self.table_row_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
        self.table_cell_pattern = re.compile(r'<t[hd][^>]*>(.*?)</t[hd]>', re.DOTALL | re.IGNORECASE)

    def convert_html_table_to_markdown(self, html_content: str) -> str:
        """将HTML表格转换为Markdown表格"""
        if self.use_markdownify:
            # 使用 markdownify 处理表格
            try:
                # 提取表格部分
                table_match = self.html_table_pattern.search(html_content)
                if table_match:
                    table_html = table_match.group(0)
                    # markdownify 可以很好地处理表格
                    markdown_result = md(table_html, heading_style="ATX", bullets="-")
                    return markdown_result.strip()
                else:
                    return html_content
            except Exception as e:
                logger.warning(f"markdownify 表格转换失败，回退到正则: {e}")
                return self._convert_table_fallback(html_content)
        else:
            return self._convert_table_fallback(html_content)


    def convert_html_in_text(self, text: str) -> str:
        """转换文本中的HTML为Markdown"""
        if not text or not isinstance(text, str):
            return text

        try:
            if self.use_markdownify:
                # 使用 markdownify 进行完整的HTML转换
                # 配置选项：保留表格，转换为ATX风格标题，使用-作为列表符号
                result = md(text,
                           heading_style="ATX",
                           bullets="-",
                           code_language="",
                           keep_html_if_not_valid=True)

                # 清理一些不需要的转换
                result = re.sub(r'\n{3,}', '\n\n', result)  # 减少多余的空行
                return result.strip()

        except Exception as e:
            logger.error(f"HTML转换失败: {e}")
            return text
