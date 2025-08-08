#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
健康监控配置模块
用于管理健康检查的各种参数和配置
"""

import os
from dataclasses import dataclass
from typing import Optional
from loguru import logger


@dataclass
class HealthConfig:
    """健康监控配置"""
    
    # 重启配置
    max_restart_attempts: int = 3  # 最大重启尝试次数
    restart_cooldown: int = 60     # 重启冷却时间（秒）
    enable_auto_restart: bool = True  # 是否启用自动重启
    
    # 监控配置
    check_interval: int = 30       # 检查间隔（秒）
    process_timeout: int = 3600    # 进程超时时间（秒）
    
    # 资源阈值
    memory_threshold: float = 0.95  # 内存使用率阈值
    cpu_threshold: float = 0.90     # CPU使用率阈值
    gpu_memory_threshold: float = 0.95  # GPU显存使用率阈值
    
    # OOM检测配置
    enable_oom_detection: bool = True  # 是否启用OOM检测
    oom_check_interval: int = 10       # OOM检查间隔（秒）
    oom_memory_threshold: float = 0.98  # OOM内存阈值
    
    # 日志配置
    log_level: str = "INFO"        # 日志级别
    enable_detailed_logging: bool = False  # 是否启用详细日志
    
    # 清理配置
    cleanup_on_failure: bool = True    # 失败时是否清理
    force_cleanup_timeout: int = 30    # 强制清理超时时间（秒）
    
    @classmethod
    def from_env(cls) -> 'HealthConfig':
        """从环境变量创建配置"""
        return cls(
            max_restart_attempts=int(os.getenv('HEALTH_MAX_RESTART_ATTEMPTS', '3')),
            restart_cooldown=int(os.getenv('HEALTH_RESTART_COOLDOWN', '60')),
            enable_auto_restart=os.getenv('HEALTH_ENABLE_AUTO_RESTART', 'true').lower() == 'true',
            
            check_interval=int(os.getenv('HEALTH_CHECK_INTERVAL', '30')),
            process_timeout=int(os.getenv('HEALTH_PROCESS_TIMEOUT', '3600')),
            
            memory_threshold=float(os.getenv('HEALTH_MEMORY_THRESHOLD', '0.95')),
            cpu_threshold=float(os.getenv('HEALTH_CPU_THRESHOLD', '0.90')),
            gpu_memory_threshold=float(os.getenv('HEALTH_GPU_MEMORY_THRESHOLD', '0.95')),
            
            enable_oom_detection=os.getenv('HEALTH_ENABLE_OOM_DETECTION', 'true').lower() == 'true',
            oom_check_interval=int(os.getenv('HEALTH_OOM_CHECK_INTERVAL', '10')),
            oom_memory_threshold=float(os.getenv('HEALTH_OOM_MEMORY_THRESHOLD', '0.98')),
            
            log_level=os.getenv('HEALTH_LOG_LEVEL', 'INFO'),
            enable_detailed_logging=os.getenv('HEALTH_ENABLE_DETAILED_LOGGING', 'false').lower() == 'true',
            
            cleanup_on_failure=os.getenv('HEALTH_CLEANUP_ON_FAILURE', 'true').lower() == 'true',
            force_cleanup_timeout=int(os.getenv('HEALTH_FORCE_CLEANUP_TIMEOUT', '30'))
        )
    
    def validate(self) -> bool:
        """验证配置参数"""
        try:
            # 检查重启配置
            if self.max_restart_attempts < 0:
                logger.error("max_restart_attempts 必须大于等于 0")
                return False
                
            if self.restart_cooldown < 0:
                logger.error("restart_cooldown 必须大于等于 0")
                return False
            
            # 检查监控配置
            if self.check_interval <= 0:
                logger.error("check_interval 必须大于 0")
                return False
                
            if self.process_timeout <= 0:
                logger.error("process_timeout 必须大于 0")
                return False
            
            # 检查阈值配置
            if not (0 < self.memory_threshold <= 1):
                logger.error("memory_threshold 必须在 (0, 1] 范围内")
                return False
                
            if not (0 < self.cpu_threshold <= 1):
                logger.error("cpu_threshold 必须在 (0, 1] 范围内")
                return False
                
            if not (0 < self.gpu_memory_threshold <= 1):
                logger.error("gpu_memory_threshold 必须在 (0, 1] 范围内")
                return False
            
            # 检查OOM配置
            if self.oom_check_interval <= 0:
                logger.error("oom_check_interval 必须大于 0")
                return False
                
            if not (0 < self.oom_memory_threshold <= 1):
                logger.error("oom_memory_threshold 必须在 (0, 1] 范围内")
                return False
            
            # 检查清理配置
            if self.force_cleanup_timeout <= 0:
                logger.error("force_cleanup_timeout 必须大于 0")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'max_restart_attempts': self.max_restart_attempts,
            'restart_cooldown': self.restart_cooldown,
            'enable_auto_restart': self.enable_auto_restart,
            'check_interval': self.check_interval,
            'process_timeout': self.process_timeout,
            'memory_threshold': self.memory_threshold,
            'cpu_threshold': self.cpu_threshold,
            'gpu_memory_threshold': self.gpu_memory_threshold,
            'enable_oom_detection': self.enable_oom_detection,
            'oom_check_interval': self.oom_check_interval,
            'oom_memory_threshold': self.oom_memory_threshold,
            'log_level': self.log_level,
            'enable_detailed_logging': self.enable_detailed_logging,
            'cleanup_on_failure': self.cleanup_on_failure,
            'force_cleanup_timeout': self.force_cleanup_timeout
        }
    
    def log_config(self):
        """记录配置信息"""
        logger.info("健康监控配置:")
        logger.info(f"  重启配置: 最大尝试={self.max_restart_attempts}, 冷却时间={self.restart_cooldown}s, 自动重启={self.enable_auto_restart}")
        logger.info(f"  监控配置: 检查间隔={self.check_interval}s, 进程超时={self.process_timeout}s")
        logger.info(f"  资源阈值: 内存={self.memory_threshold*100:.1f}%, CPU={self.cpu_threshold*100:.1f}%, GPU显存={self.gpu_memory_threshold*100:.1f}%")
        logger.info(f"  OOM检测: 启用={self.enable_oom_detection}, 检查间隔={self.oom_check_interval}s, 阈值={self.oom_memory_threshold*100:.1f}%")
        logger.info(f"  其他配置: 日志级别={self.log_level}, 详细日志={self.enable_detailed_logging}, 失败清理={self.cleanup_on_failure}")


# 预定义配置
class HealthConfigPresets:
    """健康监控预设配置"""
    
    @staticmethod
    def conservative() -> HealthConfig:
        """保守配置 - 更少的重启，更长的间隔"""
        return HealthConfig(
            max_restart_attempts=2,
            restart_cooldown=120,
            check_interval=60,
            memory_threshold=0.90,
            cpu_threshold=0.85,
            enable_auto_restart=True
        )
    
    @staticmethod
    def aggressive() -> HealthConfig:
        """激进配置 - 更多的重启，更短的间隔"""
        return HealthConfig(
            max_restart_attempts=5,
            restart_cooldown=30,
            check_interval=15,
            memory_threshold=0.98,
            cpu_threshold=0.95,
            enable_auto_restart=True
        )
    
    @staticmethod
    def development() -> HealthConfig:
        """开发配置 - 适合开发和测试"""
        return HealthConfig(
            max_restart_attempts=1,
            restart_cooldown=10,
            check_interval=10,
            memory_threshold=0.80,
            cpu_threshold=0.80,
            enable_auto_restart=False,  # 开发时禁用自动重启
            enable_detailed_logging=True
        )
    
    @staticmethod
    def production() -> HealthConfig:
        """生产配置 - 适合生产环境"""
        return HealthConfig(
            max_restart_attempts=3,
            restart_cooldown=60,
            check_interval=30,
            memory_threshold=0.95,
            cpu_threshold=0.90,
            enable_auto_restart=True,
            enable_oom_detection=True,
            cleanup_on_failure=True
        )


def get_health_config(preset: Optional[str] = None) -> HealthConfig:
    """
    获取健康监控配置
    
    Args:
        preset: 预设配置名称 ('conservative', 'aggressive', 'development', 'production')
                如果为None，则从环境变量读取或使用默认配置
    
    Returns:
        HealthConfig: 健康监控配置
    """
    if preset:
        preset_map = {
            'conservative': HealthConfigPresets.conservative,
            'aggressive': HealthConfigPresets.aggressive,
            'development': HealthConfigPresets.development,
            'production': HealthConfigPresets.production
        }
        
        if preset in preset_map:
            config = preset_map[preset]()
            logger.info(f"使用预设配置: {preset}")
        else:
            logger.warning(f"未知的预设配置: {preset}，使用默认配置")
            config = HealthConfig()
    else:
        # 尝试从环境变量读取
        config = HealthConfig.from_env()
        logger.info("从环境变量读取健康监控配置")
    
    # 验证配置
    if not config.validate():
        logger.error("健康监控配置验证失败，使用默认配置")
        config = HealthConfig()
    
    return config