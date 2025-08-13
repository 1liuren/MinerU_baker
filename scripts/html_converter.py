#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTML转Markdown转换器模块
"""

import re
from loguru import logger


class HTMLToMarkdownConverter:
    """HTML表格转Markdown转换器"""
    
    def __init__(self):
        # 简单的HTML表格转换正则表达式
        self.html_table_pattern = re.compile(r'<html><body><table>(.*?)</table></body></html>', re.DOTALL)
        self.table_row_pattern = re.compile(r'<tr>(.*?)</tr>', re.DOTALL)
        self.table_cell_pattern = re.compile(r'<t[hd]>(.*?)</t[hd]>', re.DOTALL)
    
    def convert_html_table_to_markdown(self, html_content: str) -> str:
        """将HTML表格转换为Markdown表格"""
        try:
            # 提取表格内容
            table_match = self.html_table_pattern.search(html_content)
            if not table_match:
                return "[表格内容]"
            
            table_content = table_match.group(1)
            rows = self.table_row_pattern.findall(table_content)
            
            if not rows:
                return "[表格内容]"
            
            markdown_rows = []
            for i, row in enumerate(rows):
                cells = self.table_cell_pattern.findall(row)
                if cells:
                    # 清理单元格内容
                    cleaned_cells = [re.sub(r'<[^>]+>', '', cell).strip() for cell in cells]
                    markdown_row = "| " + " | ".join(cleaned_cells) + " |"
                    markdown_rows.append(markdown_row)
                    
                    # 添加表头分隔符
                    if i == 0 and len(cleaned_cells) > 0:
                        separator = "| " + " | ".join(["---"] * len(cleaned_cells)) + " |"
                        markdown_rows.append(separator)
            
            return "\n".join(markdown_rows) if markdown_rows else "[表格内容]"
            
        except Exception as e:
            logger.warning(f"HTML表格转换失败: {e}")
            return "[表格内容]"
    
    def convert_html_in_text(self, text: str) -> str:
        """转换文本中的HTML表格为Markdown"""
        def replace_table(match):
            return self.convert_html_table_to_markdown(match.group(0))
        
        return self.html_table_pattern.sub(replace_table, text)
