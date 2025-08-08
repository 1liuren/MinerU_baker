#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
显存管理工具模块
用于解决多进程环境下的显存泄漏问题
"""

import gc
import os
import sys
import logging
from typing import Optional

# 使用loguru替代标准logging，确保日志正常输出
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    # 如果没有配置handler，添加一个基本的控制台输出
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

def force_cleanup_models():
    """
    强制清理所有模型和缓存
    """
    try:
        # 清理torch相关
        import torch
        if torch.cuda.is_available():
            # 清理当前设备
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            # 清理所有GPU设备
            for i in range(torch.cuda.device_count()):
                try:
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        torch.cuda.synchronize()
                except Exception as e:
                    logger.warning(f"清理GPU {i} 失败: {e}")
                    
        # 强制垃圾回收
        gc.collect()
        
        logger.info(f"进程 {os.getpid()}: 强制模型清理完成")
        
    except Exception as e:
        logger.warning(f"进程 {os.getpid()}: 强制模型清理失败: {e}")

def clear_model_singleton():
    """
    清理模型单例缓存
    """
    try:
        # 尝试清理可能的模型单例
        import sys
        
        # 查找并清理模型相关的模块缓存
        modules_to_clear = []
        for module_name in sys.modules:
            if any(keyword in module_name.lower() for keyword in 
                   ['model', 'pipeline', 'singleton', 'cache']):
                modules_to_clear.append(module_name)
        
        # 清理模块中的单例实例
        for module_name in modules_to_clear:
            try:
                module = sys.modules[module_name]
                if hasattr(module, '_instance'):
                    delattr(module, '_instance')
                if hasattr(module, '_instances'):
                    delattr(module, '_instances')
                if hasattr(module, 'instance'):
                    delattr(module, 'instance')
            except Exception:
                pass
                
        logger.info(f"进程 {os.getpid()}: 模型单例清理完成")
        
    except Exception as e:
        logger.warning(f"进程 {os.getpid()}: 模型单例清理失败: {e}")

def cleanup_process_memory():
    """
    进程级别的内存清理
    """
    try:
        # 清理模型单例
        clear_model_singleton()
        
        # 强制清理模型
        force_cleanup_models()
        
        # 使用MinerU的清理工具
        try:
            from mineru.utils.config_reader import get_device
            from mineru.utils.model_utils import clean_memory
            device = get_device()
            clean_memory(device)
        except Exception as e:
            logger.warning(f"MinerU清理工具失败: {e}")
            
        # 最终垃圾回收
        gc.collect()
        
        logger.info(f"进程 {os.getpid()}: 进程内存清理完成")
        
    except Exception as e:
        logger.error(f"进程 {os.getpid()}: 进程内存清理失败: {e}")

def monitor_gpu_memory():
    """
    监控GPU显存使用情况
    """
    try:
        import torch
        logger.info(f"PyTorch版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"检测到 {device_count} 个CUDA设备")
            
            for i in range(device_count):
                try:
                    with torch.cuda.device(i):
                        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                        cached = torch.cuda.memory_reserved(i) / 1024**3  # GB
                        total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                        
                        logger.info(f"GPU {i}: 已分配 {allocated:.2f}GB, 已缓存 {cached:.2f}GB, 总计 {total:.2f}GB")
                        
                        # 如果显存使用率超过90%，发出警告
                        usage_rate = cached / total
                        if usage_rate > 0.9:
                            logger.warning(f"GPU {i} 显存使用率过高: {usage_rate:.1%}")
                        else:
                            logger.info(f"GPU {i} 显存使用率: {usage_rate:.1%}")
                            
                except Exception as e:
                    logger.warning(f"监控GPU {i} 失败: {e}")
        else:
            logger.warning("CUDA不可用，当前PyTorch版本不支持GPU或未安装CUDA版本的PyTorch")
            logger.info("如果需要GPU支持，请安装CUDA版本的PyTorch")
                    
    except ImportError as e:
        logger.error(f"导入PyTorch失败: {e}")
    except Exception as e:
        logger.warning(f"GPU内存监控失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())

def setup_memory_monitoring():
    """
    设置内存监控
    """
    import atexit
    
    # 注册进程退出时的清理函数
    atexit.register(cleanup_process_memory)
    
    logger.info(f"进程 {os.getpid()}: 内存监控已设置")