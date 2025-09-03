#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF图像截取器 FastAPI Demo

功能：根据JSON文件中的bbox坐标从PDF中截取图像
作者：AI Assistant
版本：1.0.0
"""

import os
import json
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional, Union
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import uvicorn
from PIL import Image
import io
import logging
import shutil
import tempfile
import asyncio
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
from dataclasses import dataclass
import pickle
import uuid

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,  # 改为DEBUG级别以显示详细的目标添加信息
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="PDF图像截取器",
    description="根据JSON文件中的bbox坐标从PDF中截取图像",
    version="1.0.0"
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 创建模板目录
templates = Jinja2Templates(directory="templates")

# 进程池配置
MAX_WORKERS = min(multiprocessing.cpu_count(), 32)  # 最大进程数
PROGRESS_CHECK_INTERVAL = 1  # 进度检查间隔（秒）

# 全局任务状态存储
task_status_store = {}
task_lock = threading.Lock()

# 全局进程池和任务控制
active_executors = {}
executor_lock = threading.Lock()

@dataclass
class TaskProgress:
    """任务进度信息"""
    task_id: str
    total_books: int = 0
    processed_books: int = 0
    current_book: str = ""
    status: str = "pending"  # pending, running, completed, failed, cancelled
    message: str = ""
    results: List[Dict] = None
    start_time: float = 0
    end_time: float = 0
    checkpoint_file: str = ""
    
    def __post_init__(self):
        if self.results is None:
            self.results = []

@dataclass 
class BookTask:
    """单本书籍处理任务"""
    book_name: str
    json_file_path: str
    pdf_path: Optional[str]  # 改为可选，在worker中按需查找
    output_dir: str
    task_id: str
    pdf_base_folder: Optional[str] = None  # 添加PDF搜索基础目录

# 数据模型
class BboxInfo(BaseModel):
    """边界框信息"""
    bbox: List[float] = Field(..., description="边界框坐标 [x0, y0, x1, y1]")
    type: str = Field(..., description="目标类型")
    text: str = Field(..., description="目标文本")
    page_idx: Optional[int] = Field(None, description="页码索引")

class TargetInfo(BaseModel):
    """目标信息"""
    id: str = Field(..., description="目标标识")
    type: str = Field(..., description="目标格式")
    text: str = Field(..., description="目标文本")
    bbox: List[float] = Field(..., description="边界框坐标 [x0, y0, x1, y1]")
    page_idx: int = Field(..., description="页码索引")
    path: Optional[str] = Field(None, description="截取后保存的图像路径")

class ProcessRequest(BaseModel):
    """处理请求"""
    json_path: str = Field(..., description="JSON文件路径")
    pdf_path: Optional[str] = Field(None, description="PDF文件路径，如果不指定则从JSON中读取")
    output_dir: Optional[str] = Field(None, description="输出目录，如果不指定则使用PDF所在目录")

class ProcessResponse(BaseModel):
    """处理响应"""
    success: bool = Field(..., description="处理是否成功")
    message: str = Field(..., description="处理结果消息")
    saved_images: List[str] = Field(..., description="保存的图片文件路径列表")
    total_targets: int = Field(..., description="总目标数量")
    processed_targets: int = Field(..., description="成功处理的目标数量")

class BatchProcessRequest(BaseModel):
    """批量处理请求"""
    json_folder: str = Field(..., description="包含JSON文件的文件夹路径")
    pdf_search_folders: List[str] = Field(..., description="搜索PDF文件的文件夹列表")
    output_base_dir: Optional[str] = Field(None, description="输出基础目录")

class BatchProcessResponse(BaseModel):
    """批量处理响应"""
    success: bool = Field(..., description="处理是否成功")
    message: str = Field(..., description="处理结果消息")
    total_books: int = Field(..., description="总书籍数量")
    processed_books: int = Field(..., description="成功处理的书籍数量")
    failed_books: List[str] = Field(..., description="处理失败的书籍列表")
    results: List[Dict] = Field(..., description="每本书的处理结果")
    task_id: Optional[str] = Field(None, description="任务ID")

class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    total_books: int = Field(..., description="总书籍数量")
    processed_books: int = Field(..., description="已处理书籍数量")
    current_book: str = Field(..., description="当前处理的书籍")
    progress_percentage: float = Field(..., description="进度百分比")
    message: str = Field(..., description="状态消息")
    start_time: Optional[float] = Field(None, description="开始时间")
    estimated_remaining: Optional[float] = Field(None, description="预计剩余时间（秒）")

class PDFImageExtractor:
    """PDF图像截取器"""
    
    def __init__(self, pdf_path: str):
        """
        初始化PDF图像截取器
        
        Args:
            pdf_path: PDF文件路径
        """
        self.pdf_path = pdf_path
        self.doc = None
        self._open_pdf()
    
    def _open_pdf(self):
        """打开PDF文件"""
        try:
            self.doc = fitz.open(self.pdf_path)
            logger.info(f"成功打开PDF文件: {self.pdf_path}")
        except Exception as e:
            raise Exception(f"无法打开PDF文件 {self.pdf_path}: {e}")
    
    def extract_image_by_bbox(self, page_idx: int, bbox: List[float], output_path: str) -> bool:
        """
        根据bbox坐标从指定页截取图像
        
        Args:
            page_idx: 页码索引
            bbox: 边界框坐标 [x0, y0, x1, y1]
            output_path: 输出图像路径
            
        Returns:
            bool: 是否成功截取
        """
        try:
            if page_idx >= len(self.doc):
                logger.error(f"页码 {page_idx} 超出范围，PDF总页数: {len(self.doc)}")
                return False
            
            # 获取指定页面
            page = self.doc[page_idx]
            
            # 创建矩形区域
            rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
            
            # 获取图像配置
            try:
                from config import get_image_config
                image_config = get_image_config()
                scale_factor = image_config.get('scale_factor', 2.0)
            except ImportError:
                scale_factor = 2.0
                logger.warning("配置文件未找到，使用默认缩放因子: 2.0")
            
            # 截取图像
            pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(scale_factor, scale_factor))
            
            # 确保输出目录存在
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存图像
            pix.save(output_path)
            logger.info(f"成功截取图像: {output_path} (缩放因子: {scale_factor})")
            return True
            
        except Exception as e:
            logger.error(f"截取图像失败: {e}")
            return False
    
    def close(self):
        """关闭PDF文档"""
        if self.doc:
            self.doc.close()

def parse_json_file(json_path: str) -> Dict:
    """
    解析JSON文件
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        Dict: 解析后的JSON数据
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"成功解析JSON文件: {json_path}")
        return data
    except Exception as e:
        raise Exception(f"无法解析JSON文件 {json_path}: {e}")

def extract_targets_from_json(json_data: Dict) -> List[TargetInfo]:
    """
    从JSON数据中提取目标信息
    
    Args:
        json_data: JSON数据
        
    Returns:
        List[TargetInfo]: 目标信息列表
    """
    targets = []
    
    try:
        # 导入配置文件
        try:
            from config import get_target_types
            target_types = get_target_types()
        except ImportError:
            # 如果配置文件不存在，使用默认类型
            target_types = {"interline_equation", "table"}
            logger.warning("配置文件未找到，使用默认目标类型")
        
        logger.info(f"使用目标类型: {target_types}")
        
        # 检查是否有pdf_info字段
        if 'pdf_info' in json_data:
            pdf_info = json_data['pdf_info']
            
            for page_idx, page_data in enumerate(pdf_info):
                if 'para_blocks' in page_data:
                    for block in page_data['para_blocks']:
                        # 只处理spans中的指定类型
                        # 检查是否有嵌套的blocks结构
                        if 'blocks' in block:
                            # 处理嵌套的blocks
                            for nested_block in block['blocks']:
                                if 'lines' in nested_block:
                                    for line in nested_block['lines']:
                                        if 'spans' in line:
                                            for span_idx, span in enumerate(line['spans']):
                                                if 'bbox' in span and 'type' in span:
                                                    span_type = span.get('type', '')
                                                    logger.debug(f"检查span类型: {span_type}, 目标类型: {target_types}")
                                                    if span_type in target_types:
                                                        # 根据类型选择不同的文本字段
                                                        if span_type == 'table':
                                                            text_content = span.get('html', '')  # table类型使用html字段
                                                        else:
                                                            text_content = span.get('content', '')  # 公式类型使用content字段
                                                        
                                                        span_target = TargetInfo(
                                                            id=f"page_{page_idx}_span_{span_type}_{span_idx}",
                                                            type=f"span_{span_type}",
                                                            text=text_content,
                                                            bbox=span['bbox'],
                                                            page_idx=page_idx
                                                        )
                                                        targets.append(span_target)
                                                        logger.debug(f"添加span目标: {span_target.id} (类型: {span_target.type}, 内容: {text_content[:50]}...)")
                        elif 'lines' in block:
                            # 直接处理block中的lines（兼容旧结构）
                            for line in block['lines']:
                                if 'spans' in line:
                                    for span_idx, span in enumerate(line['spans']):
                                        if 'bbox' in span and 'type' in span:
                                            span_type = span.get('type', '')
                                            logger.debug(f"检查span类型: {span_type}, 目标类型: {target_types}")
                                            if span_type in target_types:
                                                # 根据类型选择不同的文本字段
                                                if span_type == 'table':
                                                    text_content = span.get('html', '')  # table类型使用html字段
                                                else:
                                                    text_content = span.get('content', '')  # 公式类型使用content字段
                                                
                                                span_target = TargetInfo(
                                                    id=f"page_{page_idx}_span_{span_type}_{span_idx}",
                                                    type=f"span_{span_type}",
                                                    text=text_content,
                                                    bbox=span['bbox'],
                                                    page_idx=page_idx
                                                )
                                                targets.append(span_target)
                                                logger.debug(f"添加span目标: {span_target.id} (类型: {span_target.type}, 内容: {text_content[:50]}...)")
        
        # 按类型统计目标数量
        type_counts = {}
        for target in targets:
            type_counts[target.type] = type_counts.get(target.type, 0) + 1
        
        logger.info(f"从JSON中提取了 {len(targets)} 个目标")
        for target_type, count in type_counts.items():
            logger.info(f"  类型 '{target_type}': {count} 个")
        
        return targets
        
    except Exception as e:
        logger.error(f"提取目标信息失败: {e}")
        return []

def save_checkpoint(task_progress: TaskProgress):
    """保存任务检查点"""
    try:
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, f"task_{task_progress.task_id}.pkl")
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(task_progress, f)
        
        task_progress.checkpoint_file = checkpoint_file
        logger.info(f"检查点已保存: {checkpoint_file}")
        
    except Exception as e:
        logger.error(f"保存检查点失败: {e}")

def load_checkpoint(task_id: str) -> Optional[TaskProgress]:
    """加载任务检查点"""
    try:
        checkpoint_file = os.path.join("checkpoints", f"task_{task_id}.pkl")
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                task_progress = pickle.load(f)
            logger.info(f"检查点已加载: {checkpoint_file}")
            logger.info(f"已处理 {len(task_progress.results)} 本书籍，总计 {task_progress.total_books} 本")
            return task_progress
        return None
    except Exception as e:
        logger.error(f"加载检查点失败: {e}")
        return None

def check_book_already_processed(book_name: str, output_base_dir: str) -> bool:
    """检查书籍是否已经处理过（通过检查输出文件）"""
    try:
        book_output_dir = os.path.join(output_base_dir, book_name)
        
        # 检查书籍目录是否存在
        if not os.path.exists(book_output_dir):
            return False
        
        # 检查是否有extraction_results.json文件
        result_json_path = os.path.join(book_output_dir, "extraction_results.json")
        if not os.path.exists(result_json_path):
            return False
        
        # 检查images目录是否存在且有文件
        images_dir = os.path.join(book_output_dir, "images")
        if not os.path.exists(images_dir):
            return False
        
        # 检查images目录中是否有图片文件
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
        if not image_files:
            return False
        
        logger.info(f"书籍 {book_name} 已经处理过，有 {len(image_files)} 张图片")
        return True
        
    except Exception as e:
        logger.warning(f"检查书籍 {book_name} 处理状态时出错: {e}")
        return False

def get_processed_books_from_filesystem(output_base_dir: str) -> set:
    """从文件系统获取已处理的书籍列表"""
    processed_books = set()
    
    try:
        if not os.path.exists(output_base_dir):
            return processed_books
        
        for item in os.listdir(output_base_dir):
            item_path = os.path.join(output_base_dir, item)
            if os.path.isdir(item_path):
                if check_book_already_processed(item, output_base_dir):
                    processed_books.add(item)
        
        logger.info(f"从文件系统发现 {len(processed_books)} 本已处理的书籍")
        
    except Exception as e:
        logger.error(f"从文件系统获取已处理书籍列表失败: {e}")
    
    return processed_books

def update_task_progress(task_id: str, **kwargs):
    """更新任务进度"""
    with task_lock:
        if task_id in task_status_store:
            task_progress = task_status_store[task_id]
            for key, value in kwargs.items():
                if hasattr(task_progress, key):
                    setattr(task_progress, key, value)
            
            # 保存检查点
            save_checkpoint(task_progress)

def get_task_progress(task_id: str) -> Optional[TaskProgress]:
    """获取任务进度"""
    with task_lock:
        return task_status_store.get(task_id)

def check_task_cancelled(task_id: str) -> bool:
    """检查任务是否被取消"""
    try:
        with task_lock:
            if task_id in task_status_store:
                return task_status_store[task_id].status == "cancelled"
        return False
    except:
        return False

def process_single_book_worker(book_task: BookTask) -> Dict:
    """单个书籍处理工作函数（多进程）"""
    try:
        # 处理开始前检查是否已被取消
        if check_task_cancelled(book_task.task_id):
            return {
                "book_name": book_task.book_name,
                "success": False,
                "message": "任务已被取消",
                "json_files_found": 0,
                "targets_processed": 0,
                "images_saved": 0
            }
        
        logger.info(f"开始处理书籍: {book_task.book_name}")
        
        # 更新当前处理的书籍
        update_task_progress(
            book_task.task_id,
            current_book=f"正在搜索PDF: {book_task.book_name}",
            status="running"
        )
        
        # 按需查找PDF文件
        pdf_path = book_task.pdf_path
        if pdf_path is None and book_task.pdf_base_folder:
            logger.info(f"为书籍 {book_task.book_name} 搜索PDF文件...")
            pdf_path = find_pdf_file_v2(book_task.book_name, book_task.pdf_base_folder)
            
            if pdf_path is None:
                return {
                    "book_name": book_task.book_name,
                    "success": False,
                    "message": "未找到对应的PDF文件",
                    "json_files_found": 1,
                    "targets_processed": 0,
                    "images_saved": 0
                }
            
            logger.info(f"找到PDF文件: {book_task.book_name} -> {pdf_path}")
        
        # 再次检查取消状态
        if check_task_cancelled(book_task.task_id):
            return {
                "book_name": book_task.book_name,
                "success": False,
                "message": "任务已被取消",
                "json_files_found": 0,
                "targets_processed": 0,
                "images_saved": 0
            }
        
        # 更新进度：开始图像提取
        update_task_progress(
            book_task.task_id,
            current_book=f"正在提取图像: {book_task.book_name}",
            status="running"
        )
        
        # 执行PDF图像提取
        result = process_pdf_extraction(
            book_task.json_file_path,
            pdf_path,
            book_task.output_dir
        )
        
        # 处理完成后最后检查取消状态
        if check_task_cancelled(book_task.task_id):
            return {
                "book_name": book_task.book_name,
                "success": False,
                "message": "任务已被取消",
                "json_files_found": 0,
                "targets_processed": 0,
                "images_saved": 0
            }
        
        book_result = {
            "book_name": book_task.book_name,
            "success": result.success,
            "message": result.message,
            "json_files_found": 1,
            "targets_processed": result.processed_targets,
            "images_saved": len(result.saved_images)
        }
        
        logger.info(f"书籍 {book_task.book_name} 处理完成: {result.message}")
        return book_result
        
    except Exception as e:
        error_msg = f"处理异常: {str(e)}"
        logger.error(f"处理书籍 {book_task.book_name} 异常: {e}")
        
        return {
            "book_name": book_task.book_name,
            "success": False,
            "message": error_msg,
            "json_files_found": 0,
            "targets_processed": 0,
            "images_saved": 0
        }

def process_pdf_extraction(json_path: str, pdf_path: Optional[str] = None, output_dir: Optional[str] = None) -> ProcessResponse:
    """
    处理PDF图像截取
    
    Args:
        json_path: JSON文件路径
        pdf_path: PDF文件路径
        output_dir: 输出目录
        
    Returns:
        ProcessResponse: 处理结果
    """
    try:
        # 解析JSON文件
        json_data = parse_json_file(json_path)
        
        # 确定PDF路径
        if not pdf_path:
            # 尝试从JSON中获取PDF路径
            if 'pdf_path' in json_data:
                pdf_path = json_data['pdf_path']
            else:
                # 使用JSON文件同目录下的同名PDF
                json_file = Path(json_path)
                pdf_path = str(json_file.parent / f"{json_file.stem}.pdf")
        
        # 检查PDF文件是否存在
        if not os.path.exists(pdf_path):
            raise Exception(f"PDF文件不存在: {pdf_path}")
        
        # 确定书籍名称（从PDF文件名提取）
        pdf_basename = os.path.basename(pdf_path)
        book_name = os.path.splitext(pdf_basename)[0]  # 移除.pdf扩展名
        
        # 确定输出目录结构：输出目录/书籍名称/
        if not output_dir or output_dir.strip() == "":
            # 使用PDF文件所在目录下的书籍名称子目录
            base_output_dir = str(Path(pdf_path).parent)
        else:
            # 确保输出目录是绝对路径或相对于当前工作目录的路径
            if not os.path.isabs(output_dir):
                # 如果是相对路径，转换为绝对路径
                base_output_dir = str(Path.cwd() / output_dir)
            else:
                base_output_dir = output_dir
        
        # 创建书籍专用目录：输出目录/书籍名称/
        book_output_dir = os.path.join(base_output_dir, book_name)
        os.makedirs(book_output_dir, exist_ok=True)
        
        # 将PDF文件复制到书籍目录下
        pdf_copy_path = os.path.join(book_output_dir, pdf_basename)
        if not os.path.exists(pdf_copy_path):
            shutil.copy2(pdf_path, pdf_copy_path)
            logger.info(f"已将PDF文件复制到书籍目录: {pdf_copy_path}")
        
        # 将JSON文件也复制到书籍目录下
        json_basename = os.path.basename(json_path)
        json_copy_path = os.path.join(book_output_dir, json_basename)
        if not os.path.exists(json_copy_path):
            shutil.copy2(json_path, json_copy_path)
            logger.info(f"已将JSON文件复制到书籍目录: {json_copy_path}")
        # 提取目标信息
        targets = extract_targets_from_json(json_data)
        
        if not targets:
            return ProcessResponse(
                success=False,
                message="未找到可处理的目标",
                saved_images=[],
                total_targets=0,
                processed_targets=0
            )
        
        # 创建PDF图像截取器
        extractor = PDFImageExtractor(pdf_path)
        
        saved_images = []
        processed_count = 0
        # 在书籍目录下创建images子目录
        images_dir = os.path.join(book_output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        # 保存结果到JSON文件（在书籍目录下）
        result_json_path = os.path.join(book_output_dir, "extraction_results.json")
        try:
            # 处理每个目标
            for target in targets:
                # 生成输出路径
                output_path = os.path.join(images_dir, f"{target.id}.png")
                # 截取图像
                if extractor.extract_image_by_bbox(target.page_idx, target.bbox, output_path):
                    saved_images.append(output_path)
                    processed_count += 1
                    logger.info(f"成功处理目标 {target.id}")
                else:
                    logger.warning(f"处理目标 {target.id} 失败")
            
        finally:
            extractor.close()
        
        try:
            result_data = {
                "pdf_path": pdf_path,
                "json_source": json_path,
                "extraction_time": str(Path(__file__).stat().st_mtime),
                "total_targets": len(targets),
                "processed_targets": processed_count,
                "targets": []
            }
            
            # 添加每个目标的详细信息
            for target in targets:
                target_info = {
                    "id": target.id,
                    "type": target.type,
                    "text": target.text,
                    "bbox": target.bbox,
                    "page_idx": target.page_idx,
                    "image_path": os.path.join(images_dir, f"{target.id}.png") if target.id in [Path(img).stem for img in saved_images] else None
                }
                result_data["targets"].append(target_info)
            
            # 保存JSON文件
            with open(result_json_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"结果已保存到: {result_json_path}")
            
        except Exception as e:
            logger.warning(f"保存结果JSON失败: {e}")
        
        # 返回处理结果
        return ProcessResponse(
            success=True,
            message=f"成功处理 {processed_count}/{len(targets)} 个目标，输出目录: {book_output_dir}",
            saved_images=saved_images,
            total_targets=len(targets),
            processed_targets=processed_count
        )
        
    except Exception as e:
        logger.error(f"处理PDF图像截取失败: {e}")
        return ProcessResponse(
            success=False,
            message=f"处理失败: {str(e)}",
            saved_images=[],
            total_targets=0,
            processed_targets=0
        )

def find_pdf_file(book_name: str, search_folders: List[str]) -> Optional[str]:
    """
    在指定文件夹中查找PDF文件
    
    Args:
        book_name: 书籍名称
        search_folders: 搜索的文件夹列表
        
    Returns:
        Optional[str]: 找到的PDF文件路径，未找到返回None
    """
    import glob
    
    # 常见的PDF文件模式
    pdf_patterns = [
        f"{book_name}.pdf",
        f"{book_name}_*.pdf", 
        f"*{book_name}*.pdf",
        f"*{book_name}.pdf"
    ]
    
    for search_folder in search_folders:
        if not os.path.exists(search_folder):
            logger.warning(f"搜索文件夹不存在: {search_folder}")
            continue
            
        for pattern in pdf_patterns:
            # 构建搜索路径
            search_pattern = os.path.join(search_folder, "**", pattern)
            
            # 使用glob递归搜索
            matches = glob.glob(search_pattern, recursive=True)
            
            if matches:
                # 返回第一个匹配的文件
                pdf_path = matches[0]
                logger.info(f"找到PDF文件: {book_name} -> {pdf_path}")
                return pdf_path
    
    logger.warning(f"未找到PDF文件: {book_name}")
    return None

def build_pdf_cache(base_folder: str, target_books: list = None) -> dict:
    """
    构建PDF文件缓存，一次性搜索所有PDF文件
    
    Args:
        base_folder: 基础搜索文件夹
        target_books: 目标书籍名称列表，如果提供则优先搜索这些书籍
        
    Returns:
        dict: 文件名到路径的映射缓存
    """
    import subprocess
    
    pdf_cache = {}
    
    if not os.path.exists(base_folder):
        logger.warning(f"搜索基础文件夹不存在: {base_folder}")
        return pdf_cache
    
    try:
        # 如果提供了目标书籍列表，先尝试精确搜索
        if target_books:
            logger.info(f"开始为 {len(target_books)} 本目标书籍构建PDF缓存，搜索目录: {base_folder}")
            
            # 先尝试精确匹配目标书籍
            for book_name in target_books:
                patterns = [
                    f"{book_name}.pdf",
                    f"{book_name}_*.pdf", 
                    f"*{book_name}*.pdf"
                ]
                
                for pattern in patterns:
                    try:
                        cmd = ["find", base_folder, "-name", pattern, "-type", "f"]
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                        
                        if result.returncode == 0 and result.stdout.strip():
                            files = result.stdout.strip().split('\n')
                            for pdf_path in files:
                                pdf_path = pdf_path.strip()
                                if pdf_path and pdf_path not in pdf_cache.values():
                                    filename = os.path.basename(pdf_path)
                                    name_without_ext = os.path.splitext(filename)[0]
                                    
                                    # 存储多种匹配方式
                                    pdf_cache[filename] = pdf_path
                                    pdf_cache[name_without_ext] = pdf_path
                                    pdf_cache[book_name] = pdf_path  # 直接映射书籍名
                                    
                                    # 处理带下划线的情况
                                    if '_' in name_without_ext:
                                        main_part = name_without_ext.split('_')[0]
                                        if main_part not in pdf_cache:
                                            pdf_cache[main_part] = pdf_path
                                    
                                    break  # 找到一个就够了
                            
                            if book_name in pdf_cache:
                                break  # 已找到该书籍，跳到下一本
                                
                    except Exception as e:
                        logger.debug(f"搜索书籍 {book_name} 时出错: {e}")
                        continue
            
            # 统计精确搜索的结果
            found_books = len([book for book in target_books if book in pdf_cache])
            logger.info(f"精确搜索完成，找到 {found_books}/{len(target_books)} 本目标书籍的PDF文件")
            
            # 如果精确搜索找到的比例较高，就不进行全量搜索
            if found_books / len(target_books) >= 0.8:
                logger.info(f"精确搜索成功率较高 ({found_books/len(target_books)*100:.1f}%)，跳过全量搜索")
                return pdf_cache
        
        # 执行全量搜索（当没有目标书籍或精确搜索成功率低时）
        logger.info(f"开始全量PDF文件缓存构建，搜索目录: {base_folder}")
        
        cmd = ["find", base_folder, "-name", "*.pdf", "-type", "f"]
        logger.info(f"执行搜索命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0 and result.stdout.strip():
            pdf_files = result.stdout.strip().split('\n')
            
            for pdf_path in pdf_files:
                pdf_path = pdf_path.strip()
                if pdf_path:
                    # 提取文件名（不含扩展名）
                    filename = os.path.basename(pdf_path)
                    name_without_ext = os.path.splitext(filename)[0]
                    
                    # 存储多种匹配方式
                    pdf_cache[filename] = pdf_path
                    pdf_cache[name_without_ext] = pdf_path
                    
                    # 处理带下划线的情况，提取主要部分
                    if '_' in name_without_ext:
                        main_part = name_without_ext.split('_')[0]
                        if main_part not in pdf_cache:
                            pdf_cache[main_part] = pdf_path
            
            logger.info(f"PDF文件缓存构建完成，找到 {len(pdf_files)} 个PDF文件，缓存条目 {len(pdf_cache)} 个")
            
        else:
            logger.warning(f"未找到任何PDF文件或搜索失败")
            
    except subprocess.TimeoutExpired:
        logger.error(f"构建PDF缓存超时，目录可能过大: {base_folder}")
    except Exception as e:
        logger.error(f"构建PDF缓存出错: {e}")
    
    return pdf_cache

def find_pdf_from_cache(book_name: str, pdf_cache: dict) -> Optional[str]:
    """
    从PDF缓存中查找文件
    
    Args:
        book_name: 书籍名称
        pdf_cache: PDF文件缓存
        
    Returns:
        Optional[str]: 找到的PDF文件路径，未找到返回None
    """
    # 尝试多种匹配方式
    search_keys = [
        book_name,  # 精确匹配
        f"{book_name}.pdf",  # 带扩展名匹配
    ]
    
    # 添加模糊匹配
    for key in pdf_cache.keys():
        if book_name in key or key in book_name:
            search_keys.append(key)
    
    # 按优先级查找
    for search_key in search_keys:
        if search_key in pdf_cache:
            pdf_path = pdf_cache[search_key]
            logger.info(f"从缓存找到PDF文件: {book_name} -> {pdf_path}")
            return pdf_path
    
    logger.warning(f"在缓存中未找到PDF文件: {book_name}")
    return None

def find_pdf_file_v2(book_name: str, base_folder: str) -> Optional[str]:
    """
    在基础文件夹中使用find命令搜索PDF文件（向后兼容）
    
    Args:
        book_name: 书籍名称
        base_folder: 基础搜索文件夹
        
    Returns:
        Optional[str]: 找到的PDF文件路径，未找到返回None
    """
    import subprocess
    
    if not os.path.exists(base_folder):
        logger.warning(f"搜索基础文件夹不存在: {base_folder}")
        return None
    
    # 使用find命令搜索PDF文件
    search_patterns = [
        f"{book_name}*.pdf",
        f"*{book_name}*.pdf"
    ]
    
    for pattern in search_patterns:
        try:
            # 构建find命令
            cmd = ["find", base_folder, "-name", pattern, "-type", "f"]
            logger.debug(f"执行搜索命令: {' '.join(cmd)}")
            
            # 执行命令
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout.strip():
                # 获取第一个匹配的文件
                files = result.stdout.strip().split('\n')
                pdf_path = files[0].strip()
                logger.info(f"找到PDF文件: {book_name} -> {pdf_path}")
                return pdf_path
                
        except subprocess.TimeoutExpired:
            logger.warning(f"搜索PDF文件超时: {book_name}")
        except Exception as e:
            logger.error(f"搜索PDF文件出错: {e}")
    
    logger.warning(f"未找到PDF文件: {book_name}")
    return None

def process_batch_extraction(json_folder: str, pdf_search_folders: List[str], output_base_dir: Optional[str] = None) -> BatchProcessResponse:
    """
    批量处理PDF图像截取
    
    Args:
        json_folder: 包含JSON文件的文件夹路径
        pdf_search_folders: 搜索PDF文件的文件夹列表
        output_base_dir: 输出基础目录
        
    Returns:
        BatchProcessResponse: 批量处理结果
    """
    try:
        if not os.path.exists(json_folder):
            raise Exception(f"JSON文件夹不存在: {json_folder}")
        
        # 确定输出基础目录
        if not output_base_dir:
            output_base_dir = os.path.join(json_folder, "batch_output")
        
        os.makedirs(output_base_dir, exist_ok=True)
        
        # 获取所有子文件夹（书籍名称）
        book_folders = []
        for item in os.listdir(json_folder):
            item_path = os.path.join(json_folder, item)
            if os.path.isdir(item_path):
                book_folders.append(item)
        
        if not book_folders:
            raise Exception(f"在文件夹 {json_folder} 中未找到任何子文件夹")
        
        logger.info(f"找到 {len(book_folders)} 个书籍文件夹: {book_folders}")
        
        total_books = len(book_folders)
        processed_books = 0
        failed_books = []
        results = []
        
        for book_name in book_folders:
            book_result = {
                "book_name": book_name,
                "success": False,
                "message": "",
                "json_files_found": 0,
                "targets_processed": 0,
                "images_saved": 0
            }
            
            try:
                logger.info(f"开始处理书籍: {book_name}")
                
                # 查找JSON文件
                book_folder_path = os.path.join(json_folder, book_name)
                json_files = []
                
                # 递归查找JSON文件
                for root, dirs, files in os.walk(book_folder_path):
                    for file in files:
                        if file.endswith('.json'):
                            json_files.append(os.path.join(root, file))
                
                if not json_files:
                    book_result["message"] = f"未找到JSON文件"
                    failed_books.append(book_name)
                    results.append(book_result)
                    continue
                
                book_result["json_files_found"] = len(json_files)
                logger.info(f"书籍 {book_name} 找到 {len(json_files)} 个JSON文件")
                
                # 查找对应的PDF文件
                pdf_path = find_pdf_file(book_name, pdf_search_folders)
                if not pdf_path:
                    book_result["message"] = f"未找到对应的PDF文件"
                    failed_books.append(book_name)
                    results.append(book_result)
                    continue
                
                total_targets_for_book = 0
                total_processed_for_book = 0
                total_images_for_book = 0
                
                # 处理每个JSON文件
                for json_file in json_files:
                    try:
                        logger.info(f"处理JSON文件: {json_file}")
                        
                        # 处理单个JSON文件（传递基础输出目录，让process_pdf_extraction自己创建书籍目录）
                        result = process_pdf_extraction(json_file, pdf_path, output_base_dir)
                        
                        if result.success:
                            total_targets_for_book += result.total_targets
                            total_processed_for_book += result.processed_targets
                            total_images_for_book += len(result.saved_images)
                            logger.info(f"成功处理JSON文件 {json_file}: {result.processed_targets}/{result.total_targets} 个目标")
                        else:
                            logger.warning(f"处理JSON文件失败 {json_file}: {result.message}")
                            
                    except Exception as e:
                        logger.error(f"处理JSON文件异常 {json_file}: {e}")
                        continue
                
                book_result["targets_processed"] = total_processed_for_book
                book_result["images_saved"] = total_images_for_book
                
                if total_processed_for_book > 0:
                    book_result["success"] = True
                    book_result["message"] = f"成功处理 {total_processed_for_book}/{total_targets_for_book} 个目标，保存 {total_images_for_book} 张图片"
                    processed_books += 1
                    logger.info(f"书籍 {book_name} 处理完成: {book_result['message']}")
                else:
                    book_result["message"] = "未找到可处理的目标"
                    failed_books.append(book_name)
                
            except Exception as e:
                book_result["message"] = f"处理异常: {str(e)}"
                failed_books.append(book_name)
                logger.error(f"处理书籍 {book_name} 异常: {e}")
            
            results.append(book_result)
        
        # 生成总结报告
        summary_message = f"批量处理完成: {processed_books}/{total_books} 本书籍处理成功"
        if failed_books:
            summary_message += f"，失败的书籍: {', '.join(failed_books)}"
        
        return BatchProcessResponse(
            success=processed_books > 0,
            message=summary_message,
            total_books=total_books,
            processed_books=processed_books,
            failed_books=failed_books,
            results=results
        )
        
    except Exception as e:
        logger.error(f"批量处理失败: {e}")
        return BatchProcessResponse(
            success=False,
            message=f"批量处理失败: {str(e)}",
            total_books=0,
            processed_books=0,
            failed_books=[],
            results=[]
        )

def process_batch_extraction_v2(results_folder: str, pdf_base_folder: str, output_base_dir: Optional[str] = None) -> BatchProcessResponse:
    """
    批量处理PDF图像截取 (第二版，针对新的目录结构)
    
    Args:
        results_folder: 包含书籍结果的文件夹路径 (书籍名称/书籍名称_middle.json)
        pdf_base_folder: PDF文件搜索基础目录
        output_base_dir: 输出基础目录
        
    Returns:
        BatchProcessResponse: 批量处理结果
    """
    try:
        if not os.path.exists(results_folder):
            raise Exception(f"结果文件夹不存在: {results_folder}")
        
        if not os.path.exists(pdf_base_folder):
            raise Exception(f"PDF基础文件夹不存在: {pdf_base_folder}")
        
        # 确定输出基础目录
        if not output_base_dir:
            output_base_dir = os.path.join(results_folder, "batch_output")
        
        os.makedirs(output_base_dir, exist_ok=True)
        
        # 获取所有书籍文件夹
        book_folders = []
        for item in os.listdir(results_folder):
            item_path = os.path.join(results_folder, item)
            if os.path.isdir(item_path):
                book_folders.append(item)
        
        if not book_folders:
            raise Exception(f"在文件夹 {results_folder} 中未找到任何书籍文件夹")
        
        logger.info(f"找到 {len(book_folders)} 个书籍文件夹: {book_folders}")
        
        total_books = len(book_folders)
        processed_books = 0
        failed_books = []
        results = []
        
        for book_name in book_folders:
            book_result = {
                "book_name": book_name,
                "success": False,
                "message": "",
                "json_files_found": 0,
                "targets_processed": 0,
                "images_saved": 0
            }
            
            try:
                logger.info(f"开始处理书籍: {book_name}")
                
                # 查找JSON文件 (书籍名称_middle.json)
                book_folder_path = os.path.join(results_folder, book_name)
                json_file_pattern = f"{book_name}_middle.json"
                json_file_path = os.path.join(book_folder_path, json_file_pattern)
                
                if not os.path.exists(json_file_path):
                    book_result["message"] = f"未找到JSON文件: {json_file_pattern}"
                    failed_books.append(book_name)
                    results.append(book_result)
                    continue
                
                book_result["json_files_found"] = 1
                logger.info(f"书籍 {book_name} 找到JSON文件: {json_file_path}")
                
                # 使用find命令查找对应的PDF文件
                pdf_path = find_pdf_file_v2(book_name, pdf_base_folder)
                if not pdf_path:
                    book_result["message"] = f"未找到对应的PDF文件"
                    failed_books.append(book_name)
                    results.append(book_result)
                    continue
                
                # 处理JSON文件
                try:
                    logger.info(f"处理JSON文件: {json_file_path}")
                    
                    # 处理单个JSON文件（传递基础输出目录，让process_pdf_extraction自己创建书籍目录）
                    result = process_pdf_extraction(json_file_path, pdf_path, output_base_dir)
                    
                    if result.success:
                        book_result["targets_processed"] = result.processed_targets
                        book_result["images_saved"] = len(result.saved_images)
                        book_result["success"] = True
                        book_result["message"] = f"成功处理 {result.processed_targets}/{result.total_targets} 个目标，保存 {len(result.saved_images)} 张图片"
                        processed_books += 1
                        logger.info(f"书籍 {book_name} 处理完成: {book_result['message']}")
                    else:
                        book_result["message"] = f"处理失败: {result.message}"
                        failed_books.append(book_name)
                        
                except Exception as e:
                    book_result["message"] = f"处理JSON文件异常: {str(e)}"
                    failed_books.append(book_name)
                    logger.error(f"处理JSON文件异常 {json_file_path}: {e}")
                
            except Exception as e:
                book_result["message"] = f"处理异常: {str(e)}"
                failed_books.append(book_name)
                logger.error(f"处理书籍 {book_name} 异常: {e}")
            
            results.append(book_result)
        
        # 生成总结报告
        summary_message = f"批量处理完成: {processed_books}/{total_books} 本书籍处理成功"
        if failed_books:
            summary_message += f"，失败的书籍: {', '.join(failed_books)}"
        
        return BatchProcessResponse(
            success=processed_books > 0,
            message=summary_message,
            total_books=total_books,
            processed_books=processed_books,
            failed_books=failed_books,
            results=results
        )
        
    except Exception as e:
        logger.error(f"批量处理失败: {e}")
        return BatchProcessResponse(
            success=False,
            message=f"批量处理失败: {str(e)}",
            total_books=0,
            processed_books=0,
            failed_books=[],
            results=[]
        )

# API路由
@app.get("/")
async def root(request: Request):
    """主页 - 显示文件上传界面"""
    return templates.TemplateResponse("process.html", {"request": request})

@app.get("/api")
async def api_info():
    """API信息"""
    return {
        "message": "PDF图像截取器 API",
        "version": "1.0.0",
        "endpoints": {
            "/": "主页 - 文件上传界面",
            "/api": "API信息",
            "/process": "处理页面",
            "/process-upload": "文件上传处理",
            "/batch": "批量处理页面",
            "/batch-process": "批量处理接口",
            "/health": "健康检查"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "message": "服务运行正常"}

@app.get("/process", response_class=HTMLResponse)
async def process_pdf_extraction_page(request: Request):
    """
    提供一个网页界面来上传JSON和PDF文件
    """
    return templates.TemplateResponse("process.html", {"request": request})

@app.post("/process-upload", response_model=ProcessResponse)
async def process_pdf_extraction_upload(
    json_file: UploadFile = File(...),
    pdf_file: UploadFile = File(...),
    output_dir: Optional[str] = Form(None)
):
    """
    通过网页上传JSON和PDF文件进行处理
    """
    try:
        # 验证文件类型
        if not json_file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="JSON文件必须是.json格式")
        if not pdf_file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="PDF文件必须是.pdf格式")
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        json_path = os.path.join(temp_dir, json_file.filename)
        pdf_path = os.path.join(temp_dir, pdf_file.filename)
        
        try:
            # 保存上传的文件
            with open(json_path, "wb") as f:
                shutil.copyfileobj(json_file.file, f)
            with open(pdf_path, "wb") as f:
                shutil.copyfileobj(pdf_file.file, f)
            
            # 处理PDF图像截取
            result = process_pdf_extraction(json_path, pdf_path, output_dir)
            
            if result.success:
                return result
            else:
                raise HTTPException(status_code=500, detail=result.message)
                
        finally:
            # 清理临时文件和目录
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"清理临时目录失败: {e}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件上传处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

@app.post("/process-json", response_model=ProcessResponse)
async def process_pdf_extraction_json(request: ProcessRequest):
    """
    处理PDF图像截取API（JSON格式）
    
    Args:
        request: 处理请求
        
    Returns:
        ProcessResponse: 处理结果
    """
    try:
        # 检查JSON文件是否存在
        if not os.path.exists(request.json_path):
            raise HTTPException(status_code=400, detail=f"JSON文件不存在: {request.json_path}")
        
        # 处理PDF图像截取
        result = process_pdf_extraction(
            request.json_path, 
            request.pdf_path, 
            request.output_dir
        )
        
        if result.success:
            return result
        else:
            raise HTTPException(status_code=500, detail=result.message)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

@app.get("/batch", response_class=HTMLResponse)
async def batch_process_page(request: Request):
    """
    批量处理页面
    """
    return templates.TemplateResponse("batch.html", {"request": request})

@app.post("/batch-process", response_model=BatchProcessResponse)
async def batch_process_api(
    json_folder: str = Form(..., description="包含JSON文件的文件夹路径"),
    pdf_search_folders: str = Form(..., description="搜索PDF文件的文件夹列表（用逗号分隔）"),
    output_base_dir: Optional[str] = Form(None, description="输出基础目录")
):
    """
    批量处理API
    """
    try:
        # 解析PDF搜索文件夹列表
        pdf_folders = [folder.strip() for folder in pdf_search_folders.split(',') if folder.strip()]
        
        if not pdf_folders:
            raise HTTPException(status_code=400, detail="必须提供至少一个PDF搜索文件夹")
        
        # 执行批量处理
        result = process_batch_extraction(json_folder, pdf_folders, output_base_dir)
        
        if result.success:
            return result
        else:
            raise HTTPException(status_code=500, detail=result.message)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量处理API失败: {e}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

@app.get("/get-subdirs-count")
async def get_subdirs_count(results_folder: str):
    """
    获取结果文件夹下的子目录数量
    """
    try:
        if not os.path.exists(results_folder):
            raise HTTPException(status_code=404, detail=f"结果文件夹不存在: {results_folder}")
        
        # 获取所有子目录
        subdirs = []
        for item in os.listdir(results_folder):
            item_path = os.path.join(results_folder, item)
            if os.path.isdir(item_path):
                subdirs.append(item)
        
        return {
            "success": True,
            "total_books": len(subdirs),
            "book_names": subdirs
        }
        
    except Exception as e:
        logger.error(f"获取子目录数量失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取子目录数量失败: {str(e)}")

@app.get("/check-processed-books")
async def check_processed_books(results_folder: str, output_base_dir: str):
    """
    检查已处理的书籍状态
    """
    try:
        if not os.path.exists(results_folder):
            raise HTTPException(status_code=404, detail=f"结果文件夹不存在: {results_folder}")
        
        # 获取所有书籍文件夹
        all_books = []
        for item in os.listdir(results_folder):
            item_path = os.path.join(results_folder, item)
            if os.path.isdir(item_path):
                all_books.append(item)
        
        # 检查已处理的书籍
        processed_books = get_processed_books_from_filesystem(output_base_dir)
        
        # 分类书籍状态
        pending_books = [book for book in all_books if book not in processed_books]
        
        return {
            "success": True,
            "total_books": len(all_books),
            "processed_books": len(processed_books),
            "pending_books": len(pending_books),
            "processed_list": sorted(list(processed_books)),
            "pending_list": sorted(pending_books),
            "can_resume": len(processed_books) > 0,
            "completion_percentage": (len(processed_books) / len(all_books) * 100) if all_books else 0
        }
        
    except Exception as e:
        logger.error(f"检查已处理书籍失败: {e}")
        raise HTTPException(status_code=500, detail=f"检查已处理书籍失败: {str(e)}")

@app.post("/batch-process-v2", response_model=BatchProcessResponse)
async def batch_process_api_v2(
    results_folder: str = Form(..., description="包含书籍结果的文件夹路径"),
    pdf_base_folder: str = Form(..., description="PDF文件搜索基础目录"),
    output_base_dir: Optional[str] = Form(None, description="输出基础目录")
):
    """
    批量处理API (第二版，针对新的目录结构)
    """
    try:
        # 执行批量处理
        result = process_batch_extraction_v2(results_folder, pdf_base_folder, output_base_dir)
        
        if result.success:
            return result
        else:
            raise HTTPException(status_code=500, detail=result.message)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量处理API v2失败: {e}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

@app.post("/batch-process-v3", response_model=BatchProcessResponse)
async def batch_process_api_v3(
    background_tasks: BackgroundTasks,
    results_folder: str = Form(..., description="包含书籍结果的文件夹路径"),
    pdf_base_folder: str = Form(..., description="PDF文件搜索基础目录"),
    output_base_dir: Optional[str] = Form(None, description="输出基础目录"),
    resume_task_id: Optional[str] = Form(None, description="要恢复的任务ID")
):
    """
    批量处理API (第三版，支持多进程和中断重续)
    """
    try:
        # 执行批量处理
        result = await process_batch_extraction_v3(results_folder, pdf_base_folder, output_base_dir, resume_task_id)
        
        if result.success or result.task_id:
            return result
        else:
            raise HTTPException(status_code=500, detail=result.message)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量处理API v3失败: {e}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

@app.get("/task-status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    获取任务状态和进度
    """
    try:
        task_progress = get_task_progress(task_id)
        
        if not task_progress:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
        
        # 计算进度百分比
        progress_percentage = 0
        if task_progress.total_books > 0:
            progress_percentage = (task_progress.processed_books / task_progress.total_books) * 100
        
        # 计算预计剩余时间
        estimated_remaining = None
        if task_progress.start_time and task_progress.processed_books > 0 and task_progress.status == "running":
            elapsed_time = time.time() - task_progress.start_time
            avg_time_per_book = elapsed_time / task_progress.processed_books
            remaining_books = task_progress.total_books - task_progress.processed_books
            estimated_remaining = avg_time_per_book * remaining_books
        
        return TaskStatusResponse(
            task_id=task_id,
            status=task_progress.status,
            total_books=task_progress.total_books,
            processed_books=task_progress.processed_books,
            current_book=task_progress.current_book,
            progress_percentage=progress_percentage,
            message=task_progress.message,
            start_time=task_progress.start_time,
            estimated_remaining=estimated_remaining
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取任务状态失败: {str(e)}")

@app.post("/cancel-task/{task_id}")
async def cancel_task(task_id: str, force: bool = False):
    """
    取消任务
    
    Args:
        task_id: 任务ID
        force: 是否强制取消（会中断进程池）
    """
    try:
        task_progress = get_task_progress(task_id)
        
        if not task_progress:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
        
        if task_progress.status in ["completed", "failed", "cancelled"]:
            return {"message": f"任务已完成，无法取消。当前状态: {task_progress.status}"}
        
        # 更新任务状态为取消
        update_task_progress(
            task_id,
            status="cancelled",
            message="任务已被用户取消",
            end_time=time.time()
        )
        
        # 如果是强制取消，尝试关闭相关的进程池
        if force:
            with executor_lock:
                if task_id in active_executors:
                    try:
                        executor = active_executors[task_id]
                        executor.shutdown(wait=False)  # 不等待，立即关闭
                        del active_executors[task_id]
                        logger.info(f"强制关闭任务 {task_id} 的进程池")
                    except Exception as e:
                        logger.warning(f"关闭进程池失败: {e}")
        
        return {
            "message": "任务取消成功" + ("（已强制中断进程）" if force else ""),
            "task_id": task_id,
            "force_cancelled": force
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"取消任务失败: {str(e)}")

@app.get("/list-tasks")
async def list_tasks():
    """
    列出所有任务
    """
    try:
        with task_lock:
            tasks = []
            for task_id, task_progress in task_status_store.items():
                progress_percentage = 0
                if task_progress.total_books > 0:
                    progress_percentage = (task_progress.processed_books / task_progress.total_books) * 100
                
                tasks.append({
                    "task_id": task_id,
                    "status": task_progress.status,
                    "total_books": task_progress.total_books,
                    "processed_books": task_progress.processed_books,
                    "progress_percentage": progress_percentage,
                    "start_time": task_progress.start_time,
                    "end_time": task_progress.end_time,
                    "message": task_progress.message
                })
            
            # 按开始时间倒序排列
            tasks.sort(key=lambda x: x["start_time"], reverse=True)
            
            return {"tasks": tasks}
            
    except Exception as e:
        logger.error(f"列出任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"列出任务失败: {str(e)}")

# 主函数
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF图像截取器")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--reload", action="store_true", help="启用热重载")
    
    args = parser.parse_args()
    
    # 启动服务器
    uvicorn.run(
        "pdf_image_extractor_demo:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

async def process_batch_extraction_v3(
    results_folder: str, 
    pdf_base_folder: str, 
    output_base_dir: Optional[str] = None,
    resume_task_id: Optional[str] = None
) -> BatchProcessResponse:
    """
    批量处理PDF图像截取 (第三版，支持多进程和中断重续)
    
    Args:
        results_folder: 包含书籍结果的文件夹路径
        pdf_base_folder: PDF文件搜索基础目录
        output_base_dir: 输出基础目录
        resume_task_id: 要恢复的任务ID
        
    Returns:
        BatchProcessResponse: 批量处理结果
    """
    try:
        # 生成或恢复任务ID
        if resume_task_id:
            task_id = resume_task_id
            task_progress = load_checkpoint(task_id)
            if not task_progress:
                raise Exception(f"无法找到任务检查点: {task_id}")
            logger.info(f"恢复任务: {task_id}")
        else:
            task_id = str(uuid.uuid4())
            task_progress = TaskProgress(task_id=task_id)
            logger.info(f"创建新任务: {task_id}")
        
        # 初始化任务进度
        with task_lock:
            task_status_store[task_id] = task_progress
        
        task_progress.start_time = time.time()
        task_progress.status = "running"
        
        # 验证文件夹
        if not os.path.exists(results_folder):
            raise Exception(f"结果文件夹不存在: {results_folder}")
        if not os.path.exists(pdf_base_folder):
            raise Exception(f"PDF基础文件夹不存在: {pdf_base_folder}")
        
        # 确定输出基础目录
        if not output_base_dir:
            output_base_dir = os.path.join(results_folder, "batch_output")
        os.makedirs(output_base_dir, exist_ok=True)
        
        # 获取所有书籍文件夹
        book_folders = []
        for item in os.listdir(results_folder):
            item_path = os.path.join(results_folder, item)
            if os.path.isdir(item_path):
                book_folders.append(item)
        
        if not book_folders:
            raise Exception(f"在文件夹 {results_folder} 中未找到任何书籍文件夹")
        
        task_progress.total_books = len(book_folders)
        logger.info(f"找到 {len(book_folders)} 个书籍文件夹")
        
        # 准备任务列表
        book_tasks = []
        processed_books = set()
        
        # 获取已处理的书籍列表（优先级：检查点 > 文件系统）
        if resume_task_id and task_progress.results:
            # 从检查点获取已处理的书籍
            checkpoint_processed = {result["book_name"] for result in task_progress.results if result.get("success", False)}
            processed_books.update(checkpoint_processed)
            logger.info(f"从检查点发现 {len(checkpoint_processed)} 本已处理的书籍")
        
        # 从文件系统检查已处理的书籍（包括检查点之外完成的）
        filesystem_processed = get_processed_books_from_filesystem(output_base_dir)
        processed_books.update(filesystem_processed)
        
        # 计算需要处理的书籍（跳过已处理的）
        pending_books = [book_name for book_name in book_folders if book_name not in processed_books]
        logger.info(f"总计 {len(book_folders)} 本书籍，已处理 {len(processed_books)} 本，待处理 {len(pending_books)} 本")
        
        if processed_books:
            logger.info(f"总计跳过 {len(processed_books)} 本已处理的书籍: {', '.join(sorted(processed_books))}")
        
        # 从检查点中加载成功处理的结果
        existing_results = []
        if task_progress.results:
            existing_results = [result for result in task_progress.results if result.get("success", False)]
        
        # 为文件系统中已处理但不在检查点中的书籍创建结果记录
        for book_name in filesystem_processed:
            if not any(result["book_name"] == book_name for result in existing_results):
                existing_results.append({
                    "book_name": book_name,
                    "success": True,
                    "message": "从已有文件恢复",
                    "json_files_found": 1,
                    "targets_processed": 0,  # 无法准确统计，设为0
                    "images_saved": len([f for f in os.listdir(os.path.join(output_base_dir, book_name, "images")) 
                                       if f.endswith('.png')]) if os.path.exists(os.path.join(output_base_dir, book_name, "images")) else 0
                })
        
        # 预处理：验证JSON文件存在性
        logger.info(f"开始预处理 {len(pending_books)} 本待处理书籍的JSON文件验证...")
        
        valid_books = []
        for book_name in pending_books:
            # 检查JSON文件
            book_folder_path = os.path.join(results_folder, book_name)
            json_file_pattern = f"{book_name}_middle.json"
            json_file_path = os.path.join(book_folder_path, json_file_pattern)
            
            if not os.path.exists(json_file_path):
                logger.warning(f"跳过书籍 {book_name}: 未找到JSON文件 {json_file_pattern}")
                continue
            
            # 暂时不查找PDF，留给多线程处理时按需查找
            valid_books.append((book_name, json_file_path))
        
        logger.info(f"JSON验证完成，{len(valid_books)} 本书籍准备处理（PDF将在处理时按需查找）")
        
        # 创建任务（PDF路径设为None，在worker中按需查找）
        for book_name, json_file_path in valid_books:
            book_task = BookTask(
                book_name=book_name,
                json_file_path=json_file_path,
                pdf_path=None,  # 在worker中按需查找
                output_dir=output_base_dir,
                task_id=task_id,
                pdf_base_folder=pdf_base_folder  # 传递PDF搜索基础目录
            )
            book_tasks.append(book_task)
        
        if not book_tasks and not processed_books:
            raise Exception("没有找到可处理的书籍")
        
        logger.info(f"准备处理 {len(book_tasks)} 本书籍，使用 {MAX_WORKERS} 个进程")
        
        # 使用多进程处理
        failed_books = []
        all_results = list(existing_results)  # 从已有结果开始
        
        if book_tasks:
            executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
            
            try:
                # 注册进程池到全局管理器
                with executor_lock:
                    active_executors[task_id] = executor
                
                # 提交所有任务
                future_to_book = {
                    executor.submit(process_single_book_worker, book_task): book_task
                    for book_task in book_tasks
                }
                
                # 处理完成的任务
                for future in as_completed(future_to_book):
                    # 检查任务是否被取消
                    if check_task_cancelled(task_id):
                        logger.info(f"任务 {task_id} 被取消，停止处理")
                        # 取消所有未完成的future
                        for f in future_to_book:
                            if not f.done():
                                f.cancel()
                        break
                    
                    book_task = future_to_book[future]
                    try:
                        result = future.result(timeout=1)  # 设置超时避免阻塞
                        all_results.append(result)
                        
                        if result["success"]:
                            task_progress.processed_books += 1
                        else:
                            failed_books.append(result["book_name"])
                        
                        # 更新进度
                        current_successful = len([r for r in all_results if r.get("success", False)])
                        update_task_progress(
                            task_id,
                            processed_books=current_successful,
                            results=all_results,
                            current_book=f"已完成: {result['book_name']} ({current_successful}/{task_progress.total_books})"
                        )
                        
                        logger.info(f"完成处理: {result['book_name']} ({'成功' if result['success'] else '失败'})")
                        
                    except TimeoutError:
                        # 超时，跳过这次循环但不中断整体处理
                        continue
                    except Exception as e:
                        logger.error(f"处理任务失败: {e}")
                        failed_books.append(book_task.book_name)
                        
                        error_result = {
                            "book_name": book_task.book_name,
                            "success": False,
                            "message": f"任务执行异常: {str(e)}",
                            "json_files_found": 0,
                            "targets_processed": 0,
                            "images_saved": 0
                        }
                        all_results.append(error_result)
                        
            finally:
                # 清理进程池
                try:
                    with executor_lock:
                        if task_id in active_executors:
                            del active_executors[task_id]
                    
                    executor.shutdown(wait=True)  # 等待完成
                except Exception as e:
                    logger.warning(f"关闭进程池时出错: {e}")
                    try:
                        executor.shutdown(wait=False)  # 强制关闭
                    except:
                        pass
        
        # 统计最终结果
        successful_books = len([r for r in all_results if r.get("success", False)])
        
        # 完成任务
        task_progress.end_time = time.time()
        task_progress.status = "completed"
        task_progress.processed_books = successful_books
        task_progress.results = all_results
        
        summary_message = f"批量处理完成: {successful_books}/{task_progress.total_books} 本书籍处理成功"
        if failed_books:
            summary_message += f"，失败的书籍: {', '.join(failed_books)}"
        
        task_progress.message = summary_message
        update_task_progress(task_id, **task_progress.__dict__)
        
        return BatchProcessResponse(
            success=successful_books > 0,
            message=summary_message,
            total_books=task_progress.total_books,
            processed_books=successful_books,
            failed_books=failed_books,
            results=all_results,
            task_id=task_id
        )
        
    except Exception as e:
        logger.error(f"批量处理失败: {e}")
        
        # 更新任务状态为失败
        if 'task_id' in locals():
            update_task_progress(
                task_id,
                status="failed",
                message=f"批量处理失败: {str(e)}",
                end_time=time.time()
            )
        
        return BatchProcessResponse(
            success=False,
            message=f"批量处理失败: {str(e)}",
            total_books=0,
            processed_books=0,
            failed_books=[],
            results=[],
            task_id=task_id if 'task_id' in locals() else None
        )
