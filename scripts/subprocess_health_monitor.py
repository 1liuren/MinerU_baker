#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
子进程健康监控模块
用于监控pdf_pipeline.py中的多进程处理
"""

import os
import sys
import time
import psutil
import threading
import traceback
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, Future
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

try:
    from memory_utils import cleanup_process_memory, monitor_gpu_memory
except ImportError:
    logger.warning("无法导入memory_utils，将使用基本的内存清理")
    def cleanup_process_memory():
        import gc
        gc.collect()
    
    def monitor_gpu_memory():
        logger.info("GPU监控不可用")


@dataclass
class SubprocessInfo:
    """子进程信息"""
    future: Future
    batch_idx: int
    batch_files: List[Path]
    start_time: float
    pid: Optional[int] = None
    last_check_time: float = field(default_factory=time.time)
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    restart_count: int = 0
    status: str = "running"  # running, completed, failed, timeout, oom
    error_msg: Optional[str] = None


class SubprocessHealthMonitor:
    """子进程健康监控器"""
    
    def __init__(self, 
                 check_interval: float = 30.0,
                 memory_threshold_mb: float = 8192.0,  # 8GB
                 cpu_threshold_percent: float = 95.0,
                 timeout_minutes: float = 60.0,
                 max_restart_attempts: int = 2,
                 restart_cooldown: float = 60.0):
        
        self.check_interval = check_interval
        self.memory_threshold_mb = memory_threshold_mb
        self.cpu_threshold_percent = cpu_threshold_percent
        self.timeout_minutes = timeout_minutes
        self.max_restart_attempts = max_restart_attempts
        self.restart_cooldown = restart_cooldown
        
        # 进程管理
        self.subprocesses: Dict[str, SubprocessInfo] = {}
        self.executor: Optional[ProcessPoolExecutor] = None
        
        # 监控控制
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            "total_processes": 0,
            "completed_processes": 0,
            "failed_processes": 0,
            "restarted_processes": 0,
            "oom_kills": 0,
            "timeout_kills": 0
        }
        
        # 回调函数
        self.on_process_failure: Optional[Callable] = None
        self.on_process_restart: Optional[Callable] = None
        
        logger.info(f"子进程健康监控器初始化完成")
        logger.info(f"监控配置: 检查间隔={check_interval}s, 内存阈值={memory_threshold_mb}MB, "
                   f"CPU阈值={cpu_threshold_percent}%, 超时={timeout_minutes}min")
    
    def start_monitoring(self, executor: ProcessPoolExecutor):
        """开始监控"""
        with self._lock:
            if self.monitoring:
                logger.warning("监控已在运行中")
                return
            
            self.executor = executor
            self.monitoring = True
            
            # 启动监控线程
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="SubprocessHealthMonitor",
                daemon=True
            )
            self.monitor_thread.start()
            
            logger.info("子进程健康监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        with self._lock:
            if not self.monitoring:
                return
            
            self.monitoring = False
            
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5.0)
            
            logger.info("子进程健康监控已停止")
    
    def register_subprocess(self, future: Future, batch_idx: int, batch_files: List[Path]):
        """注册子进程"""
        process_id = f"batch_{batch_idx}"
        
        with self._lock:
            self.subprocesses[process_id] = SubprocessInfo(
                future=future,
                batch_idx=batch_idx,
                batch_files=batch_files,
                start_time=time.time()
            )
            self.stats["total_processes"] += 1
        
        logger.debug(f"注册子进程: {process_id}, 文件数: {len(batch_files)}")
    
    def _monitor_loop(self):
        """监控循环"""
        logger.info("开始子进程健康监控循环")
        
        while self.monitoring:
            try:
                self._check_all_processes()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                logger.debug(traceback.format_exc())
                time.sleep(5.0)  # 异常后短暂休息
    
    def _check_all_processes(self):
        """检查所有进程"""
        current_time = time.time()
        processes_to_remove = []
        
        with self._lock:
            for process_id, proc_info in self.subprocesses.items():
                try:
                    # 检查进程是否完成
                    if proc_info.future.done():
                        self._handle_completed_process(process_id, proc_info)
                        processes_to_remove.append(process_id)
                        continue
                    
                    # 更新进程PID（如果还没有）
                    if proc_info.pid is None:
                        proc_info.pid = self._get_process_pid(proc_info.future)
                    
                    # 检查超时
                    if self._check_timeout(proc_info, current_time):
                        self._handle_timeout_process(process_id, proc_info)
                        processes_to_remove.append(process_id)
                        continue
                    
                    # 检查资源使用
                    if proc_info.pid:
                        self._check_resource_usage(process_id, proc_info)
                    
                    proc_info.last_check_time = current_time
                    
                except Exception as e:
                    logger.error(f"检查进程 {process_id} 时出错: {e}")
            
            # 移除已完成的进程
            for process_id in processes_to_remove:
                del self.subprocesses[process_id]
    
    def _get_process_pid(self, future: Future) -> Optional[int]:
        """获取进程PID（尽力而为）"""
        try:
            # 这是一个简化的实现，实际可能需要更复杂的逻辑
            # 因为ProcessPoolExecutor不直接暴露PID
            return None
        except Exception:
            return None
    
    def _check_timeout(self, proc_info: SubprocessInfo, current_time: float) -> bool:
        """检查进程超时"""
        elapsed_time = current_time - proc_info.start_time
        timeout_seconds = self.timeout_minutes * 60
        
        if elapsed_time > timeout_seconds:
            logger.warning(f"进程超时: batch_{proc_info.batch_idx}, "
                          f"运行时间: {elapsed_time:.1f}s > {timeout_seconds}s")
            return True
        
        return False
    
    def _check_resource_usage(self, process_id: str, proc_info: SubprocessInfo):
        """检查资源使用情况"""
        if not proc_info.pid:
            return
        
        try:
            process = psutil.Process(proc_info.pid)
            
            # 检查内存使用
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            proc_info.memory_usage = memory_mb
            
            # 检查CPU使用
            cpu_percent = process.cpu_percent()
            proc_info.cpu_usage = cpu_percent
            
            # 检查内存阈值
            if memory_mb > self.memory_threshold_mb:
                logger.warning(f"进程 {process_id} 内存使用过高: {memory_mb:.1f}MB > {self.memory_threshold_mb}MB")
                self._handle_oom_process(process_id, proc_info)
            
            # 记录资源使用情况（定期）
            if time.time() - proc_info.last_check_time > 300:  # 每5分钟记录一次
                logger.debug(f"进程 {process_id} 资源使用: 内存={memory_mb:.1f}MB, CPU={cpu_percent:.1f}%")
        
        except psutil.NoSuchProcess:
            logger.debug(f"进程 {process_id} 已不存在")
        except Exception as e:
            logger.debug(f"检查进程 {process_id} 资源使用失败: {e}")
    
    def _handle_completed_process(self, process_id: str, proc_info: SubprocessInfo):
        """处理已完成的进程"""
        try:
            result = proc_info.future.result()
            proc_info.status = "completed"
            self.stats["completed_processes"] += 1
            
            elapsed_time = time.time() - proc_info.start_time
            logger.info(f"进程 {process_id} 完成, 耗时: {elapsed_time:.1f}s")
            
        except Exception as e:
            proc_info.status = "failed"
            proc_info.error_msg = str(e)
            self.stats["failed_processes"] += 1
            
            logger.error(f"进程 {process_id} 失败: {e}")
            
            if self.on_process_failure:
                self.on_process_failure(process_id, proc_info, e)
    
    def _handle_timeout_process(self, process_id: str, proc_info: SubprocessInfo):
        """处理超时进程"""
        logger.warning(f"终止超时进程: {process_id}")
        
        try:
            proc_info.future.cancel()
            proc_info.status = "timeout"
            self.stats["timeout_kills"] += 1
            
            # 尝试清理进程
            if proc_info.pid:
                try:
                    process = psutil.Process(proc_info.pid)
                    process.terminate()
                    time.sleep(2)
                    if process.is_running():
                        process.kill()
                except Exception as e:
                    logger.debug(f"清理超时进程失败: {e}")
        
        except Exception as e:
            logger.error(f"处理超时进程失败: {e}")
    
    def _handle_oom_process(self, process_id: str, proc_info: SubprocessInfo):
        """处理OOM进程"""
        logger.warning(f"终止OOM进程: {process_id}, 内存使用: {proc_info.memory_usage:.1f}MB")
        
        try:
            proc_info.future.cancel()
            proc_info.status = "oom"
            self.stats["oom_kills"] += 1
            
            # 尝试清理进程
            if proc_info.pid:
                try:
                    process = psutil.Process(proc_info.pid)
                    process.terminate()
                    time.sleep(2)
                    if process.is_running():
                        process.kill()
                except Exception as e:
                    logger.debug(f"清理OOM进程失败: {e}")
            
            # 执行内存清理
            try:
                cleanup_process_memory()
                monitor_gpu_memory()
            except Exception as e:
                logger.debug(f"内存清理失败: {e}")
        
        except Exception as e:
            logger.error(f"处理OOM进程失败: {e}")
    
    def get_status_report(self) -> Dict:
        """获取状态报告"""
        with self._lock:
            active_processes = len(self.subprocesses)
            
            # 计算平均资源使用
            total_memory = sum(proc.memory_usage for proc in self.subprocesses.values())
            avg_memory = total_memory / active_processes if active_processes > 0 else 0
            
            total_cpu = sum(proc.cpu_usage for proc in self.subprocesses.values())
            avg_cpu = total_cpu / active_processes if active_processes > 0 else 0
            
            return {
                "monitoring": self.monitoring,
                "active_processes": active_processes,
                "avg_memory_mb": avg_memory,
                "avg_cpu_percent": avg_cpu,
                "stats": self.stats.copy(),
                "processes": {
                    pid: {
                        "batch_idx": proc.batch_idx,
                        "status": proc.status,
                        "memory_mb": proc.memory_usage,
                        "cpu_percent": proc.cpu_usage,
                        "elapsed_time": time.time() - proc.start_time,
                        "restart_count": proc.restart_count
                    }
                    for pid, proc in self.subprocesses.items()
                }
            }
    
    def cleanup(self):
        """清理资源"""
        self.stop_monitoring()
        
        with self._lock:
            # 取消所有未完成的任务
            for proc_info in self.subprocesses.values():
                if not proc_info.future.done():
                    try:
                        proc_info.future.cancel()
                    except Exception:
                        pass
            
            self.subprocesses.clear()
        
        logger.info("子进程健康监控器已清理")


def create_monitored_process_pool(max_workers: int, 
                                 monitor_config: Optional[Dict] = None) -> tuple:
    """创建带监控的进程池"""
    # 创建进程池
    executor = ProcessPoolExecutor(max_workers=max_workers)
    
    # 创建监控器
    config = monitor_config or {}
    monitor = SubprocessHealthMonitor(**config)
    
    # 启动监控
    monitor.start_monitoring(executor)
    
    return executor, monitor


if __name__ == "__main__":
    # 测试代码
    import time
    
    def test_subprocess_monitor():
        """测试子进程监控"""
        logger.info("开始测试子进程监控")
        
        # 创建监控器
        monitor = SubprocessHealthMonitor(
            check_interval=5.0,
            memory_threshold_mb=1024.0,
            timeout_minutes=1.0
        )
        
        # 创建进程池
        with ProcessPoolExecutor(max_workers=2) as executor:
            monitor.start_monitoring(executor)
            
            # 提交一些测试任务
            def test_task(x):
                time.sleep(x)
                return x * 2
            
            futures = []
            for i in range(3):
                future = executor.submit(test_task, 10 + i * 5)
                monitor.register_subprocess(future, i, [Path(f"test_{i}.pdf")])
                futures.append(future)
            
            # 等待一段时间
            time.sleep(30)
            
            # 获取状态报告
            report = monitor.get_status_report()
            logger.info(f"状态报告: {report}")
            
            # 清理
            monitor.cleanup()
        
        logger.info("测试完成")
    
    test_subprocess_monitor()