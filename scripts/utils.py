
# -*- coding: utf-8 -*-
"""
工具函数模块
"""

import re
import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict
from loguru import logger
try:
    from .config import OpenAI, BookMetadata
except ImportError:
    # 支持直接运行测试
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from config import OpenAI, BookMetadata


def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)}分{remaining_seconds:.2f}秒"
    else:
        hours = seconds // 3600
        remaining = seconds % 3600
        minutes = remaining // 60
        seconds = remaining % 60
        return f"{int(hours)}小时{int(minutes)}分{seconds:.2f}秒"


def find_files(input_dir: Path, extensions: List[str]) -> List[Path]:
    """查找指定扩展名的文件"""
    files = []
    for ext in extensions:
        files.extend(input_dir.rglob(f"*.{ext.lower()}"))
        files.extend(input_dir.rglob(f"*.{ext.upper()}"))
    return sorted(set(files))


def clean_markdown_text(text):
    # 保留标题（# 开头的行）
    lines = text.split('\n')
    cleaned_lines = []

    # 允许的Unicode范围（基本中英文字符、标点、LaTeX、常见单位符号）
    allowed_unicode_ranges = (
        r'\u4e00-\u9fff'  # 中文
        r'\u0000-\u007F'  # ASCII
        r'\u00A0-\u00FF'  # 拉丁补充
        r'\u2000-\u206F'  # 常用符号
        r'\u2100-\u214F'  # 单位、货币等
        r'\u2190-\u21FF'  # 箭头符号
        r'\u2200-\u22FF'  # 数学符号
        r'\u2B00-\u2BFF'  # 箭头和图形
        r'\u25A0-\u25FF'  # 几何图形
        r'\u0300-\u036F'  # LaTeX重音
    )

    # 正则表达式规则
    valid_char_pattern = re.compile(f'[^{allowed_unicode_ranges}]+')
    latex_inline_pattern = re.compile(r'(\$.*?\$)')  # 保留 LaTeX 语句
    punctuation_map = {
        '，': ',', '。': '.', '？': '?', '！': '!', '：': ':', '；': ';',
        '（': '(', '）': ')', '【': '[', '】': ']', '“': '"', '”': '"',
        '‘': "'", '’': "'", '、': ','
    }

    for line in lines:
        original_line = line

        # 保留标题行原样
        if line.strip().startswith('#'):
            cleaned_lines.append(line.rstrip())
            continue

        # 提取并保留LaTeX内容
        latex_parts = latex_inline_pattern.findall(line)
        line = latex_inline_pattern.sub('[LATEX]', line)

        # 替换中文标点为英文标点
        for zh, en in punctuation_map.items():
            line = line.replace(zh, en)

        # 去除不正常的空格和缩进
        line = line.strip()
        line = re.sub(r'[ \t]{2,}', ' ', line)

        # 去除连续特殊符号、表情、艺术字符（除允许范围和LATEX外）
        line = valid_char_pattern.sub('', line)

        # 恢复LaTeX语句
        for latex in latex_parts:
            line = line.replace('[LATEX]', latex, 1)

        cleaned_lines.append(line)

    # 去除连续多行空行
    final_text = '\n'.join(cleaned_lines)
    final_text = re.sub(r'\n{2,}', '\n', final_text)

    return final_text.strip()


def clean_markdown_file(input_path, output_path):
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        cleaned = clean_markdown_text(content)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        return True
    except Exception as e:
        print(f"处理文件 {input_path} 时出错: {str(e)}")
        return False


def extract_metadata_with_llm(text: str, api_url: str) -> Dict:
    """使用大模型提取元数据"""
    logger.info(f"开始元数据提取，API URL: {api_url}")
    
    if not OpenAI:
        logger.warning("OpenAI模块未安装，跳过元数据提取")
        return {}
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        logger.error("DASHSCOPE_API_KEY环境变量未设置")
        return {}
    
    # logger.info(f"API密钥已获取，长度: {len(api_key)}")
    
    try:
        # logger.info("正在创建OpenAI客户端...")
        client = OpenAI(
            base_url=api_url,
            api_key=api_key
        )
        logger.info("OpenAI客户端创建成功")
        
        prompt = f"""请你阅读以下书籍内容片段，并提取该书的关键信息。

请严格按照以下格式返回一个 **纯 JSON 对象**：

{{
    "title": "书籍标题",
    "author": "作者姓名",
    "publisher": "出版社名称",
    "lang": "语言（例如：中文）",
    "category": ["分类1", "分类2"],
    "knowledge": ["知识点1", "知识点2"]
}}

以下是书籍内容片段：
----------------------------------------
{text[:1000]}
...
{text[-1000:]}
----------------------------------------

请只输出标准 JSON 字符串，不要添加说明或注释。
"""
        
        # logger.info("正在调用大模型API...")
        response = client.chat.completions.create(
            model="qwen-max-latest",
            messages=[
                {"role": "system", "content": "你是一个专业的中文语言助理。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            stream=False,
            temperature=0.3,
            extra_body={"enable_thinking": False},
        )
        # logger.info("API调用成功")
        
        content = response.choices[0].message.content.strip()
        # 清理可能的markdown包装
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        
        if OpenAI and BookMetadata:
            data = BookMetadata.model_validate_json(content)
            return data.dict()
        else:
            return json.loads(content)
            
    except Exception as e:
        logger.error(f"元数据提取失败: {e}")
        return {}
