#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF处理流水线主程序
"""

import argparse
import sys
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings("ignore")

# 支持直接执行和模块导入两种方式
try:
    from .config import configure_logging
    from .pdf_pipeline import OptimizedPDFPipeline
except ImportError:
    # 直接执行时使用绝对导入
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.config import configure_logging
    from scripts.pdf_pipeline import OptimizedPDFPipeline


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="优化版PDF处理流水线 - MinerU集成版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例用法:
  python scripts/main.py --input /path/to/pdfs --output /path/to/output
  python scripts/main.py --input /path/to/pdfs --output /path/to/output --backend vlm-sglang-client --server-url http://localhost:30000
  python scripts/main.py --input /path/to/pdfs --output /path/to/output --batch-size 50 --concurrent-batches 2
        """
    )
    
    # 必需参数
    parser.add_argument(
        "--input", "-i", 
        required=True, 
        help="输入目录路径（包含PDF或EPUB文件）"
    )
    parser.add_argument(
        "--output", "-o", 
        required=True, 
        help="输出目录路径"
    )
    
    # 处理配置
    parser.add_argument(
        "--backend", 
        default="vlm-http-client", 
        choices=["vlm-sglang-client", "pipeline","vlm-http-client"],
        help="处理后端 (默认: vlm-sglang-client)"
    )
    parser.add_argument(
        "--server-url", 
        default="http://10.10.50.50:30000",
        help="sglang服务器URL，支持逗号分隔多个地址(例如: http://host1:30000,http://host2:30000)"
    )
    parser.add_argument(
        "--lang", 
        default="ch", 
        choices=["ch", "en"],
        help="处理语言 (默认: ch)"
    )
    parser.add_argument(
        "--api-url", 
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="大模型API URL (默认: 阿里云DashScope)"
    )
    # -d "/path/a.json,/path/b.json" 或 -d "/path/a.json"
    parser.add_argument(
        "-d", "--data-json-path",
        default=None,
        help="筛选合格数据的json路径(data_info.json)，仅处理ok_status为'合格'的文件"
    )
    
    # 性能配置
    parser.add_argument(
        "--max-workers", 
        type=int, 
        default=4, 
        help="最大工作线程数 (默认: 4)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1, 
        help="每批次处理的文件数量 (默认: 100)"
    )
    parser.add_argument(
        "--concurrent-batches", 
        type=int, 
        default=10, 
        help="同时处理的批次数量 (默认: 4)"
    )
    parser.add_argument(
        "--batches-per-round", 
        type=int, 
        default=None, 
        help="每轮处理的批次数量，处理完后重建进程池 (默认: max(concurrent_batches*10, 50))"
    )
    parser.add_argument(
        "--lb-strategy",
        choices=["round_robin", "random","ewma"],
        default="round_robin",
        help="多服务器URL的负载均衡策略 (默认: round_robin)"
    )

    
    # 其他配置
    parser.add_argument(
        "--log-level", 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)"
    )
    parser.add_argument(
        "--version", 
        action="version", 
        version="PDF处理流水线 v1.0.0"
    )
    
    return parser.parse_args()


def validate_args(args):
    """验证命令行参数"""
    # 检查输入目录
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"输入目录不存在: {input_path}")
        return False
    if not input_path.is_dir():
        logger.error(f"输入路径不是目录: {input_path}")
        return False
    
    # 检查输出目录的父目录是否存在
    output_path = Path(args.output)
    if not output_path.parent.exists():
        logger.error(f"输出目录的父目录不存在: {output_path.parent}")
        return False
    
    # 检查性能参数
    if args.max_workers < 1:
        logger.error("max-workers 必须大于 0")
        return False
    if args.batch_size < 1:
        logger.error("batch-size 必须大于 0")
        return False
    if args.concurrent_batches < 1:
        logger.error("concurrent-batches 必须大于 0")
        return False

    
    # 检查后端配置
    if args.backend == "vlm-sglang-client" and not args.server_url:
        logger.warning("使用 vlm-sglang-client 后端但未指定 server-url，将使用默认配置")
    
    return True


def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_args()
        
        # 配置日志
        configure_logging(args.log_level)
        
        # 验证参数
        if not validate_args(args):
            sys.exit(1)
        
        # 显示配置信息
        logger.info("PDF处理流水线启动")
        logger.info(f"输入目录: {args.input}")
        logger.info(f"输出目录: {args.output}")
        logger.info(f"处理后端: {args.backend}")
        if args.server_url:
            logger.info(f"服务器URL: {args.server_url}")
        logger.info(f"语言: {args.lang}")
        logger.info(f"性能配置: {args.max_workers} 线程, {args.batch_size} 批次大小, {args.concurrent_batches} 并发批次")
        
        # 创建并运行流水线
        pipeline = OptimizedPDFPipeline(
            input_dir=args.input,
            output_dir=args.output,
            max_workers=args.max_workers,
            backend=args.backend,
            server_url=args.server_url,
            lang=args.lang,
            api_url=args.api_url,
            batch_size=args.batch_size,
            concurrent_batches=args.concurrent_batches,
            data_json_path=args.data_json_path,
            batches_per_round=args.batches_per_round,
            lb_strategy=args.lb_strategy
        )
        
        # 运行流水线
        success = pipeline.run_pipeline()
        
        if success:
            logger.success("流水线执行成功完成")
            sys.exit(0)
        else:
            logger.error("流水线执行失败")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("用户中断操作")
        sys.exit(130)
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
