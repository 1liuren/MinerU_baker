#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
进程健康检查和自动重启模块
用于监控PDF处理流水线的进程状态，当进程挂掉时自动清理并重启
"""

import os
import sys
import time
import signal
import psutil
import subprocess
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

try:
    from .memory_utils import cleanup_process_memory, monitor_gpu_memory
except ImportError:
    from memory_utils import cleanup_process_memory, monitor_gpu_memory


@dataclass
class ProcessInfo:
    """进程信息"""
    pid: int
    gpu_id: int
    shard_index: int
    command: List[str]
    start_time: float
    restart_count: int = 0
    last_restart_time: float = 0
    process: Optional[subprocess.Popen] = None
    

class HealthMonitor:
    """健康检查监控器"""
    
    def __init__(self, 
                 max_restart_attempts: int = 3,
                 restart_cooldown: int = 60,
                 check_interval: int = 30,
                 memory_threshold: float = 0.95,
                 enable_auto_restart: bool = True):
        """
        初始化健康监控器
        
        Args:
            max_restart_attempts: 最大重启尝试次数
            restart_cooldown: 重启冷却时间（秒）
            check_interval: 检查间隔（秒）
            memory_threshold: 内存使用率阈值
            enable_auto_restart: 是否启用自动重启
        """
        self.max_restart_attempts = max_restart_attempts
        self.restart_cooldown = restart_cooldown
        self.check_interval = check_interval
        self.memory_threshold = memory_threshold
        self.enable_auto_restart = enable_auto_restart
        
        self.processes: Dict[int, ProcessInfo] = {}  # shard_index -> ProcessInfo
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"收到信号 {signum}，开始清理进程...")
        self.stop_monitoring()
        self.cleanup_all_processes()
        sys.exit(0)
        
    def register_process(self, shard_index: int, gpu_id: int, command: List[str], process: subprocess.Popen):
        """注册进程"""
        process_info = ProcessInfo(
            pid=process.pid,
            gpu_id=gpu_id,
            shard_index=shard_index,
            command=command,
            start_time=time.time(),
            process=process
        )
        self.processes[shard_index] = process_info
        logger.info(f"注册进程: shard={shard_index}, gpu={gpu_id}, pid={process.pid}")
        
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            logger.warning("监控已经在运行中")
            return
            
        self.monitoring = True
        self._stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("健康监控已启动")
        
    def stop_monitoring(self):
        """停止监控"""
        if not self.monitoring:
            return
            
        self.monitoring = False
        self._stop_event.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
            
        logger.info("健康监控已停止")
        
    def _monitor_loop(self):
        """监控循环"""
        logger.info(f"开始监控循环，检查间隔: {self.check_interval}秒")
        
        while self.monitoring and not self._stop_event.is_set():
            try:
                self._check_processes()
                self._check_system_resources()
                
                # 等待下次检查
                if self._stop_event.wait(self.check_interval):
                    break
                    
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                time.sleep(5)  # 异常时短暂等待
                
    def _check_processes(self):
        """检查进程状态"""
        dead_processes = []
        
        for shard_index, process_info in self.processes.items():
            try:
                if not self._is_process_alive(process_info):
                    logger.warning(f"检测到进程死亡: shard={shard_index}, pid={process_info.pid}")
                    dead_processes.append(shard_index)
                else:
                    # 检查进程资源使用情况
                    self._check_process_resources(process_info)
                    
            except Exception as e:
                logger.error(f"检查进程 {shard_index} 状态失败: {e}")
                dead_processes.append(shard_index)
                
        # 处理死亡进程
        for shard_index in dead_processes:
            self._handle_dead_process(shard_index)
            
    def _is_process_alive(self, process_info: ProcessInfo) -> bool:
        """检查进程是否存活"""
        try:
            if process_info.process:
                # 检查subprocess对象
                if process_info.process.poll() is not None:
                    return False
                    
            # 使用psutil检查进程
            if psutil.pid_exists(process_info.pid):
                proc = psutil.Process(process_info.pid)
                return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
            else:
                return False
                
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
        except Exception as e:
            logger.warning(f"检查进程存活状态异常: {e}")
            return False
            
    def _check_process_resources(self, process_info: ProcessInfo):
        """检查进程资源使用情况"""
        try:
            proc = psutil.Process(process_info.pid)
            
            # 检查内存使用率
            memory_percent = proc.memory_percent()
            if memory_percent > self.memory_threshold * 100:
                logger.warning(f"进程 {process_info.pid} 内存使用率过高: {memory_percent:.1f}%")
                
            # 检查CPU使用率（可选）
            cpu_percent = proc.cpu_percent()
            if cpu_percent > 90:  # CPU使用率超过90%
                logger.warning(f"进程 {process_info.pid} CPU使用率过高: {cpu_percent:.1f}%")
                
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass  # 进程可能已经结束
        except Exception as e:
            logger.warning(f"检查进程资源使用情况失败: {e}")
            
    def _check_system_resources(self):
        """检查系统资源"""
        try:
            # 检查系统内存
            memory = psutil.virtual_memory()
            if memory.percent > self.memory_threshold * 100:
                logger.warning(f"系统内存使用率过高: {memory.percent:.1f}%")
                
            # 监控GPU内存
            monitor_gpu_memory()
            
        except Exception as e:
            logger.warning(f"检查系统资源失败: {e}")
            
    def _handle_dead_process(self, shard_index: int):
        """处理死亡进程"""
        process_info = self.processes.get(shard_index)
        if not process_info:
            return
            
        logger.error(f"进程死亡: shard={shard_index}, pid={process_info.pid}, 运行时间: {time.time() - process_info.start_time:.1f}秒")
        
        # 清理死亡进程
        self._cleanup_dead_process(process_info)
        
        # 检查是否需要重启
        if self.enable_auto_restart and self._should_restart(process_info):
            self._restart_process(shard_index)
        else:
            logger.info(f"进程 {shard_index} 不会被重启")
            del self.processes[shard_index]
            
    def _cleanup_dead_process(self, process_info: ProcessInfo):
        """清理死亡进程"""
        try:
            # 强制终止进程（如果还在运行）
            if process_info.process:
                try:
                    if process_info.process.poll() is None:
                        process_info.process.terminate()
                        process_info.process.wait(timeout=10)
                        if process_info.process.poll() is None:
                            process_info.process.kill()
                            process_info.process.wait()
                except Exception as e:
                    logger.warning(f"强制终止进程失败: {e}")
                    
            # 清理GPU显存
            try:
                import torch
                if torch.cuda.is_available():
                    with torch.cuda.device(process_info.gpu_id):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        torch.cuda.synchronize()
                    logger.info(f"GPU {process_info.gpu_id} 显存清理完成")
            except Exception as e:
                logger.warning(f"GPU显存清理失败: {e}")
                
            # 调用专用内存清理
            cleanup_process_memory()
            
        except Exception as e:
            logger.error(f"清理死亡进程失败: {e}")
            
    def _should_restart(self, process_info: ProcessInfo) -> bool:
        """判断是否应该重启进程"""
        # 检查重启次数
        if process_info.restart_count >= self.max_restart_attempts:
            logger.warning(f"进程 {process_info.shard_index} 重启次数已达上限 ({self.max_restart_attempts})")
            return False
            
        # 检查冷却时间
        if time.time() - process_info.last_restart_time < self.restart_cooldown:
            logger.warning(f"进程 {process_info.shard_index} 在冷却期内，暂不重启")
            return False
            
        return True
        
    def _restart_process(self, shard_index: int):
        """重启进程"""
        process_info = self.processes.get(shard_index)
        if not process_info:
            return
            
        logger.info(f"正在重启进程: shard={shard_index}, 第{process_info.restart_count + 1}次重启")
        
        try:
            # 设置环境变量
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(process_info.gpu_id)
            env["MINERU_DEVICE_MODE"] = f"cuda:0"
            
            # 启动新进程
            new_process = subprocess.Popen(process_info.command, env=env)
            
            # 更新进程信息
            process_info.pid = new_process.pid
            process_info.process = new_process
            process_info.start_time = time.time()
            process_info.restart_count += 1
            process_info.last_restart_time = time.time()
            
            logger.success(f"进程重启成功: shard={shard_index}, new_pid={new_process.pid}")
            
        except Exception as e:
            logger.error(f"重启进程失败: {e}")
            del self.processes[shard_index]
            
    def wait_for_completion(self) -> bool:
        """等待所有进程完成"""
        logger.info("等待所有进程完成...")
        
        exit_codes = []
        for shard_index, process_info in self.processes.items():
            try:
                if process_info.process:
                    exit_code = process_info.process.wait()
                    exit_codes.append(exit_code)
                    logger.info(f"进程 {shard_index} 完成，退出码: {exit_code}")
                else:
                    exit_codes.append(-1)
                    
            except Exception as e:
                logger.error(f"等待进程 {shard_index} 完成失败: {e}")
                exit_codes.append(-1)
                
        success = all(code == 0 for code in exit_codes)
        logger.info(f"所有进程完成，成功: {success}")
        return success
        
    def cleanup_all_processes(self):
        """清理所有进程"""
        logger.info("开始清理所有进程...")
        
        for shard_index, process_info in list(self.processes.items()):
            try:
                self._cleanup_dead_process(process_info)
            except Exception as e:
                logger.error(f"清理进程 {shard_index} 失败: {e}")
                
        self.processes.clear()
        logger.info("所有进程清理完成")
        
    def get_status(self) -> Dict:
        """获取监控状态"""
        status = {
            "monitoring": self.monitoring,
            "total_processes": len(self.processes),
            "alive_processes": 0,
            "dead_processes": 0,
            "processes": []
        }
        
        for shard_index, process_info in self.processes.items():
            is_alive = self._is_process_alive(process_info)
            if is_alive:
                status["alive_processes"] += 1
            else:
                status["dead_processes"] += 1
                
            status["processes"].append({
                "shard_index": shard_index,
                "pid": process_info.pid,
                "gpu_id": process_info.gpu_id,
                "is_alive": is_alive,
                "restart_count": process_info.restart_count,
                "uptime": time.time() - process_info.start_time
            })
            
        return status