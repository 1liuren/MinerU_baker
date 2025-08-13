#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理状态管理模块
"""

import json
from pathlib import Path
from datetime import datetime
from threading import Lock
from typing import Dict
from loguru import logger


class ProcessingStatus:
    """简化的处理状态管理类"""
    
    def __init__(self, status_file: str):
        self.status_file = Path(status_file)
        self.lock = Lock()
        self.status_data = self._load_status()
    
    def _load_status(self) -> Dict:
        """加载处理状态"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            "processed_files": {},
            "processing_stats": {
                "total_files": 0,
                "completed": 0,
                "failed": 0,
                "skipped": 0
            },
            "last_update": datetime.now().isoformat()
        }
    
    def _save_status(self):
        """保存处理状态"""
        try:
            self.status_data["last_update"] = datetime.now().isoformat()
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(self.status_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存状态文件失败: {e}")
    
    def is_processed(self, file_path: str, task_type: str = "default") -> bool:
        """检查文件是否已处理"""
        with self.lock:
            file_key = f"{str(Path(file_path).resolve())}"
            # 检查文件是否在已处理列表中，并且处理成功
            return (file_key in self.status_data["processed_files"] and 
                   self.status_data["processed_files"][file_key]["success"])
    
    def mark_processed(self, file_path: str, task_type: str, processing_time: float, success: bool = True, error_msg: str = None):
        """标记文件处理状态"""
        with self.lock:
            file_key = f"{str(Path(file_path).resolve())}"
            
            self.status_data["processed_files"][file_key] = {
                "timestamp": datetime.now().isoformat(),
                "processing_time": processing_time,
                "success": success,
                "error": error_msg if not success else None,
            }
            
            if success:
                self.status_data["processing_stats"]["completed"] += 1
            else:
                self.status_data["processing_stats"]["failed"] += 1
            
            self._save_status()
    
    def get_stats(self) -> Dict:
        """获取处理统计信息"""
        with self.lock:
            return self.status_data["processing_stats"].copy()
