#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置和日志模块
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from loguru import logger


def configure_logging(log_level: str = "INFO", log_file: str = None):
    """配置loguru日志"""
    logger.remove()  # 移除默认处理器
    
    # 控制台日志
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level.upper()
    )
    
    # 文件日志
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="7 days"
        )
        logger.info(f"日志文件: {log_path}")


def setup_project_path():
    """设置项目路径"""
    # 添加项目根目录到Python路径
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))


def import_optional_modules():
    """导入可选模块"""
    try:
        from openai import OpenAI
        from pydantic import BaseModel, Field
        from dotenv import load_dotenv
        load_dotenv()
        return OpenAI, BaseModel, Field
    except ImportError:
        logger.warning("OpenAI模块未安装，将跳过元数据提取功能")
        return None, None, None


# 书籍元数据模型
def create_book_metadata_model(BaseModel, Field):
    """创建书籍元数据模型"""
    if not BaseModel or not Field:
        return None
        
    class BookMetadata(BaseModel):
        title: str = Field(description="书籍标题")
        author: str = Field(description="作者名称")
        publisher: str = Field(description="出版社")
        lang: str = Field(default="中文", description="书籍语言")
        category: list = Field(default_factory=list, description="书籍分类")
        knowledge: list = Field(default_factory=list, description="主要知识点")
    
    return BookMetadata


# 默认配置
configure_logging()
setup_project_path()

# 导入可选模块
OpenAI, BaseModel, Field = import_optional_modules()
BookMetadata = create_book_metadata_model(BaseModel, Field) if BaseModel and Field else None
