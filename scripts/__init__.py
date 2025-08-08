#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF处理流水线脚本包
"""

__version__ = "1.0.0"
__author__ = "MinerU Team"
__description__ = "优化版PDF处理流水线 - MinerU集成版"

from .config import configure_logging, BookMetadata
from .status_manager import ProcessingStatus
from .html_converter import HTMLToMarkdownConverter
from .utils import format_time, find_files, clean_markdown_text, extract_metadata_with_llm
from .pdf_pipeline import OptimizedPDFPipeline
from .main import main

__all__ = [
    "configure_logging",
    "BookMetadata",
    "ProcessingStatus",
    "HTMLToMarkdownConverter",
    "format_time",
    "find_files",
    "clean_markdown_text",
    "extract_metadata_with_llm",
    "OptimizedPDFPipeline",
    "main"
]