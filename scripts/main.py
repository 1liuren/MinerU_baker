#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF处理流水线主程序
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

# 支持直接执行和模块导入两种方式
try:
    from config import configure_logging
    from pdf_pipeline import OptimizedPDFPipeline
except ImportError:
    try:
        from .config import configure_logging
        from .pdf_pipeline import OptimizedPDFPipeline
    except ImportError:
        # 最后尝试从scripts包导入
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
        default="vlm-sglang-client", 
        choices=["vlm-sglang-client", "pipeline"],
        help="处理后端 (默认: vlm-sglang-client)"
    )
    parser.add_argument(
        "--server-url", 
        default="http://10.10.50.50:30000",
        help="sglang服务器URL (例如: http://localhost:30000)"
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
        default=100, 
        help="每批次处理的文件数量 (默认: 100)"
    )
    parser.add_argument(
        "--concurrent-batches", 
        type=int, 
        default=4, 
        help="同时处理的批次数量 (默认: 4)"
    )

    # 多卡/分片设置
    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        help="逗号分隔的GPU索引列表，例如: 0,1,2。留空表示单卡/自动"
    )

    
    # 健康检查配置
    parser.add_argument(
        "--health-preset",
        choices=["conservative", "aggressive", "development", "production"],
        help="健康监控预设配置"
    )
    parser.add_argument(
        "--disable-auto-restart",
        action="store_true",
        help="禁用自动重启功能"
    )
    parser.add_argument(
        "--max-restart-attempts",
        type=int,
        default=3,
        help="最大重启尝试次数 (默认: 3)"
    )
    parser.add_argument(
        "--restart-cooldown",
        type=int,
        default=60,
        help="重启冷却时间（秒） (默认: 60)"
    )
    parser.add_argument(
        "--health-check-interval",
        type=int,
        default=30,
        help="健康检查间隔（秒） (默认: 30)"
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

    # 子进程内部使用的分片参数（不会在帮助中展示给普通用户）
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=None,
        help=argparse.SUPPRESS
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

    # 校验 gpus 参数格式
    if args.gpus:
        try:
            _ = [int(x) for x in args.gpus.split(',') if x.strip() != ""]
        except ValueError:
            logger.error("--gpus 参数格式错误，应为逗号分隔的整数列表，如: 0,1,2")
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
        
        # 是否启用脚本内自动分卡并发
        if args.gpus:
            import os
            import subprocess
            import json
            try:
                from health_monitor import HealthMonitor
                from health_config import get_health_config
            except ImportError:
                try:
                    from .health_monitor import HealthMonitor
                    from .health_config import get_health_config
                except ImportError:
                    from scripts.health_monitor import HealthMonitor
                    from scripts.health_config import get_health_config

            gpu_list = [int(x) for x in args.gpus.split(',') if x.strip() != ""]
            shard_count = len(gpu_list)

            logger.info(f"启用脚本内多卡并发: GPUs={gpu_list}, 分片数={shard_count}")

            # 获取健康监控配置
            health_config = get_health_config(args.health_preset)
            
            # 应用命令行参数覆盖
            if args.disable_auto_restart:
                health_config.enable_auto_restart = False
            if hasattr(args, 'max_restart_attempts'):
                health_config.max_restart_attempts = args.max_restart_attempts
            if hasattr(args, 'restart_cooldown'):
                health_config.restart_cooldown = args.restart_cooldown
            if hasattr(args, 'health_check_interval'):
                health_config.check_interval = args.health_check_interval
            
            # 记录配置信息
            health_config.log_config()

            # 创建健康监控器
            health_monitor = HealthMonitor(
                max_restart_attempts=health_config.max_restart_attempts,
                restart_cooldown=health_config.restart_cooldown,
                check_interval=health_config.check_interval,
                memory_threshold=health_config.memory_threshold,
                enable_auto_restart=health_config.enable_auto_restart
            )

            # 启动所有子进程
            for shard_index, gpu_id in enumerate(gpu_list):
                env = os.environ.copy()
                # 仅暴露当前子进程可见的GPU
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                # 同时给 MinerU 设置 device（对非CUDA环境可换成 cpu/mps 等）
                env["MINERU_DEVICE_MODE"] = f"cuda:0"

                cmd = [
                    sys.executable,
                    __file__,
                    "--input", args.input,
                    "--output", args.output,
                    "--backend", args.backend,
                    "--server-url", args.server_url if args.server_url else "",
                    "--lang", args.lang,
                    "--api-url", args.api_url,
                    "--batch-size", str(args.batch_size),
                    "--concurrent-batches", str(args.concurrent_batches),
                    "--max-workers", str(args.max_workers),
                    # 传递分片信息到子进程（子进程内不再传 --gpus）
                    "--shard-index", str(shard_index),
                    "--shard-count", str(shard_count)
                ]
                # 去掉空的 server-url 参数
                cmd = [c for c in cmd if c != ""]

                logger.info(f"启动子进程: GPU={gpu_id}, shard={shard_index+1}/{shard_count}")
                
                try:
                    proc = subprocess.Popen(cmd, env=env)
                    # 注册进程到健康监控器
                    health_monitor.register_process(shard_index, gpu_id, cmd, proc)
                    logger.success(f"子进程启动成功: GPU={gpu_id}, PID={proc.pid}")
                except Exception as e:
                    logger.error(f"启动子进程失败: GPU={gpu_id}, 错误: {e}")
                    continue

            # 启动健康监控
            health_monitor.start_monitoring()
            
            try:
                # 等待所有进程完成
                success = health_monitor.wait_for_completion()
                
                # 显示最终状态
                status = health_monitor.get_status()
                logger.info(f"处理完成统计: 总进程={status['total_processes']}, 存活={status['alive_processes']}, 死亡={status['dead_processes']}")
                
            except KeyboardInterrupt:
                logger.warning("用户中断，正在清理进程...")
                success = False
            finally:
                # 停止监控并清理所有进程
                health_monitor.stop_monitoring()
                health_monitor.cleanup_all_processes()
            
            # 进程收尾后做一次全局清理
            try:
                import gc
                import torch
                from mineru.utils.config_reader import get_device
                from mineru.utils.model_utils import clean_memory
                
                # 强制垃圾回收
                gc.collect()
                
                # 清理所有可用GPU的显存
                if torch.cuda.is_available():
                    for gpu_id in range(torch.cuda.device_count()):
                        try:
                            with torch.cuda.device(gpu_id):
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                                torch.cuda.synchronize()
                            logger.info(f"GPU {gpu_id} 显存清理完成")
                        except Exception as e:
                            logger.warning(f"GPU {gpu_id} 显存清理失败: {e}")
                
                # 通用设备清理
                try:
                    device = get_device()
                    clean_memory(device)
                    logger.info("全局显存清理完成")
                except Exception as e:
                    logger.warning(f"全局显存清理失败: {e}")
                    
                # 调用专用内存清理
                try:
                    from memory_utils import cleanup_process_memory
                    cleanup_process_memory()
                except ImportError:
                    try:
                        from .memory_utils import cleanup_process_memory
                        cleanup_process_memory()
                    except ImportError:
                        try:
                            from scripts.memory_utils import cleanup_process_memory
                            cleanup_process_memory()
                        except ImportError:
                            logger.warning("无法导入内存清理工具，跳过专用内存清理")
                    
            except Exception as e:
                logger.warning(f"全局清理失败: {e}")
        else:
            # 单进程（可搭配 CUDA_VISIBLE_DEVICES 外部设置）
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
                # 支持从命令行隐含传入的分片参数（被子进程用）
                shard_index=getattr(args, "shard_index", None),
                shard_count=getattr(args, "shard_count", None),
            )

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