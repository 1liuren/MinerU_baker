#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF图像截取器批量处理服务启动脚本
专为远程Linux服务器部署设计
"""

import uvicorn
import argparse
import os
from pdf_image_extractor_demo import app

def main():
    parser = argparse.ArgumentParser(description="PDF图像截取器批量处理服务")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=2221, help="服务器端口")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数量")
    parser.add_argument("--reload", action="store_true", help="启用热重载（开发模式）")
    parser.add_argument("--log-level", default="info", help="日志级别")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PDF图像截取器批量处理服务")
    print("=" * 60)
    print(f"服务监听地址: {args.host}:{args.port}")
    print(f"外部访问地址: http://10.10.50.24:{args.port}")
    print(f"单文件处理: http://10.10.50.24:{args.port}/")
    print(f"批量处理: http://10.10.50.24:{args.port}/batch")
    print(f"API文档: http://10.10.50.24:{args.port}/docs")
    print("=" * 60)
    print("功能特性:")
    print("- 单文件上传处理")
    print("- 批量文件夹处理")
    print("- 自动PDF文件搜索匹配")
    print("- 支持嵌套目录结构")
    print("- 实时处理进度反馈")
    print("=" * 60)
    print("按 Ctrl+C 停止服务")
    print("")
    
    # 启动服务
    uvicorn.run(
        "pdf_image_extractor_demo:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level,
        access_log=True
    )

if __name__ == "__main__":
    main()
