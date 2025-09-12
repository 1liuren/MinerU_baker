#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF图像截取器 - 核心功能模块
只保留命令行批量处理的核心功能，移除所有Web相关代码

功能：根据JSON文件中的bbox坐标从PDF中截取图像
作者：AI Assistant
版本：2.0.0 (简化版)
"""

import os
import json
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from loguru import logger
import shutil
import time
import fnmatch
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import threading
from tqdm import tqdm

import sys

# 移除默认的 sink，防止重复
logger.remove()

# 重新把终端 sink 加回来，并设定等级
logger.add(sys.stderr, level="INFO") 

"""使用 tqdm 默认实现作为进度条"""

# 全局配置
MAX_WORKERS = min(multiprocessing.cpu_count(), 32)
PROGRESS_CHECK_INTERVAL = 1

# 任务状态存储
task_status_store = {}
task_lock = threading.Lock()

# 全局PDF缓存（文件名stem -> 绝对路径）
PDF_CACHE: dict[str, str] = {}

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
    pdf_path: Optional[str]
    output_dir: str
    task_id: str
    pdf_base_folder: Optional[str] = None
    include_text: bool = False

@dataclass
class TargetInfo:
    """目标信息"""
    id: str
    type: str
    text: str
    bbox: List[float]
    page_idx: int
    path: Optional[str] = None

@dataclass
class ProcessResult:
    """处理结果"""
    success: bool
    message: str
    saved_images: List[str]
    total_targets: int
    processed_targets: int

@dataclass
class BatchResult:
    """批量处理结果"""
    success: bool
    message: str
    total_books: int
    processed_books: int
    failed_books: List[str]
    results: List[Dict]
    task_id: Optional[str] = None


class PDFImageExtractor:
    """PDF图像截取器"""
    
    def __init__(self, pdf_path: str):
        """初始化PDF图像截取器"""
        self.pdf_path = pdf_path
        self.doc = None
        self._open_pdf()
    
    def _open_pdf(self):
        """打开PDF文件"""
        try:
            self.doc = fitz.open(self.pdf_path)
            logger.debug(f"成功打开PDF文件: {self.pdf_path}")
        except Exception as e:
            raise Exception(f"无法打开PDF文件 {self.pdf_path}: {e}")
    
    def extract_image_by_bbox(self, page_idx: int, bbox: List[float], output_path: str) -> bool:
        """根据bbox坐标从指定页截取图像"""
        try:
            if page_idx >= len(self.doc):
                logger.error(f"页码 {page_idx} 超出范围，PDF总页数: {len(self.doc)}")
                return False
            
            # 获取指定页面
            page = self.doc[page_idx]
            
            # 创建矩形区域
            rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
            
            # 获取缩放因子（默认2.0）
            scale_factor = 2.0
            
            # 截取图像
            pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(scale_factor, scale_factor))
            
            # 确保输出目录存在
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存图像
            pix.save(output_path)
            logger.debug(f"成功截取图像: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"截取图像失败: {e}")
            return False
    
    def close(self):
        """关闭PDF文档"""
        if self.doc:
            self.doc.close()


def get_target_types() -> set:
    """获取目标类型配置"""
    try:
        from config import get_target_types
        return get_target_types()
    except ImportError:
        # 默认目标类型
        return {"interline_equation", "table"}


def parse_json_file(json_path: str) -> Dict:
    """解析JSON文件"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"成功解析JSON文件: {json_path}")
        return data
    except Exception as e:
        raise Exception(f"无法解析JSON文件 {json_path}: {e}")
def _stem_key(name: str) -> str:
    return (name or "").strip()



def extract_targets_from_json(json_data: Dict, include_text: bool = False) -> List[TargetInfo]:
    """从JSON数据中提取目标信息"""
    targets: List[TargetInfo] = []
    target_types = get_target_types()
    # 可选启用 text 类型
    if include_text:
        target_types = set(target_types) | {"text"}

    # 每页每类型的递增计数，当中间JSON缺少 index 时回退使用
    page_type_counters: dict[tuple[int, str], int] = {}

    try:
        if 'pdf_info' in json_data:
            pdf_info = json_data['pdf_info']

            for page_idx, page_data in enumerate(pdf_info):
                if 'para_blocks' not in page_data:
                    continue

                for block in page_data['para_blocks']:
                    # 处理两种结构：block 内直接含 lines，或含 blocks[nested] 再含 lines
                    if 'blocks' in block and isinstance(block['blocks'], list):
                        for nested_block in block['blocks']:
                            lines_list = nested_block.get('lines', []) if isinstance(nested_block, dict) else []
                            block_index = nested_block.get('index') if isinstance(nested_block, dict) else None
                            block_type = nested_block.get('type') if isinstance(nested_block, dict) else None

                            for line_idx, line in enumerate(lines_list or []):
                                spans = line.get('spans') if isinstance(line, dict) else None
                                if not spans:
                                    continue

                                # 先输出表格（受 target_types 控制）
                                for s_idx, s in enumerate(spans):
                                    if not isinstance(s, dict):
                                        continue
                                    if s.get('type') == 'table' and 'bbox' in s and ('table' in target_types):
                                        targets.append(
                                            TargetInfo(
                                                id=f"page_{page_idx}_span_table_{block_index}",
                                                type='span_table',
                                                text=s.get('html', ''),
                                                bbox=s['bbox'],
                                                page_idx=page_idx,
                                            )
                                        )

                                # 行间公式（受 target_types 控制），单独输出，不参与合并
                                for s_idx, s in enumerate(spans):
                                    if not isinstance(s, dict):
                                        continue
                                    if s.get('type') == 'interline_equation' and 'bbox' in s and ('interline_equation' in target_types):
                                        targets.append(
                                            TargetInfo(
                                                id=f"page_{page_idx}_span_interline_equation_{block_index}",
                                                type='span_interline_equation',
                                                text=s.get('content', ''),
                                                bbox=s['bbox'],
                                                page_idx=page_idx,
                                            )
                                        )

                    else:
                        lines_list = block.get('lines', []) if isinstance(block, dict) else []
                        block_index = block.get('index') if isinstance(block, dict) else None
                        block_type = block.get('type') if isinstance(block, dict) else None

                        for line_idx, line in enumerate(lines_list or []):
                            spans = line.get('spans') if isinstance(line, dict) else None
                            if not spans:
                                continue

                            # 表格（受 target_types 控制）
                            for s_idx, s in enumerate(spans):
                                if not isinstance(s, dict):
                                    continue
                                if s.get('type') == 'table' and 'bbox' in s and ('table' in target_types):
                                    targets.append(
                                        TargetInfo(
                                            id=f"page_{page_idx}_span_table_{block_index}",
                                            type='span_table',
                                            text=s.get('html', ''),
                                            bbox=s['bbox'],
                                            page_idx=page_idx,
                                        )
                                    )

                            # 行间公式（受 target_types 控制）
                            for s_idx, s in enumerate(spans):
                                if not isinstance(s, dict):
                                    continue
                                if s.get('type') == 'interline_equation' and 'bbox' in s and ('interline_equation' in target_types):
                                    targets.append(
                                        TargetInfo(
                                            id=f"page_{page_idx}_span_interline_equation_{block_index}",
                                            type='span_interline_equation',
                                            text=s.get('content', ''),
                                            bbox=s['bbox'],
                                            page_idx=page_idx,
                                        )
                                    )

                            # 合并文本/行内公式（仅当启用 text 提取；标题由上层 block 类型控制）
                            if 'text' in target_types:
                                parts = []
                                boxes = []
                                is_title = (block_type == 'title')
                                for s in spans:
                                    if not isinstance(s, dict):
                                        continue
                                    st = s.get('type', '')
                                    if st in ('text', 'inline_equation'):
                                        content = s.get('content', '')
                                        if st == 'inline_equation' and content:
                                            content = f"${content}$"
                                        if content:
                                            parts.append(content)
                                        if 'bbox' in s:
                                            boxes.append(s['bbox'])

                                if parts:
                                    merged = ''.join(parts)
                                    if boxes:
                                        xs1 = [b[0] for b in boxes]
                                        ys1 = [b[1] for b in boxes]
                                        xs2 = [b[2] for b in boxes]
                                        ys2 = [b[3] for b in boxes]
                                        mbox = [min(xs1), min(ys1), max(xs2), max(ys2)]
                                    else:
                                        mbox = spans[0].get('bbox', [0, 0, 0, 0])

                                    out_type = 'title_text' if is_title else 'text'
                                    out_text = f"# {merged}" if is_title else merged
                                    targets.append(
                                        TargetInfo(
                                            id=f"page_{page_idx}_line_text_{block_index}",
                                            type=out_type,
                                            text=out_text,
                                            bbox=mbox,
                                            page_idx=page_idx,
                                        )
                                    )

        logger.info(f"从JSON中提取了 {len(targets)} 个目标")
        return targets

    except Exception as e:
        logger.error(f"提取目标信息失败: {e}")
        return []


def find_pdf_file_v2(book_name: str, base_folder: str) -> Optional[str]:
    """查找PDF文件：优先使用一次性构建的缓存；必要时回退到遍历。"""
    if not os.path.exists(base_folder):
        logger.warning(f"搜索基础文件夹不存在: {base_folder}")
        return None

    key = _stem_key(book_name)

    # 1) 优先查全局缓存
    if PDF_CACHE:
        path = PDF_CACHE.get(key)
        if path and os.path.exists(path):
            return path

    # 2) 回退：遍历（规范化比较）
    try:
        for root, dirs, files in os.walk(base_folder):
            for filename in files:
                if not filename.lower().endswith('.pdf'):
                    continue
                name_without_ext = os.path.splitext(filename)[0]
                name_key = _stem_key(name_without_ext)

                if name_key == key:
                    pdf_path = os.path.join(root, filename)
                    logger.info(f"找到PDF文件（精确匹配）: {book_name} -> {pdf_path}")
                    return pdf_path
    except Exception as e:
        logger.error(f"搜索PDF文件出错: {e}")

    logger.warning(f"未找到PDF文件: {book_name}")
    return None


def build_pdf_cache(base_folder: str, target_books: list = None) -> dict:
    """构建PDF文件缓存，一次性搜索所有PDF文件"""
    pdf_cache: dict[str, str] = {}
    
    if not os.path.exists(base_folder):
        logger.warning(f"搜索基础文件夹不存在: {base_folder}")
        return pdf_cache
    
    try:
        # 如果提供了目标书籍列表，先尝试精确搜索
        if target_books:
            logger.info(f"开始为 {len(target_books)} 本目标书籍构建PDF缓存，搜索目录: {base_folder}")
            
            with tqdm(target_books, desc="🔍 搜索目标书籍PDF", unit="本") as pbar:
                for book_name in pbar:
                    pbar.set_description(f"🔍 搜索: {book_name[:20]}...")
                    
                    patterns = [
                        f"{book_name}.pdf"
                    ]
                    
                    found_for_book = False
                    for root, dirs, files in os.walk(base_folder):
                        if found_for_book:
                            break
                            
                        for pattern in patterns:
                            for filename in files:
                                if filename.lower().endswith('.pdf') and fnmatch.fnmatch(filename, pattern):
                                    pdf_path = os.path.join(root, filename)
                                    name_without_ext = os.path.splitext(filename)[0]
                                    
                                    # 存储stem键
                                    pdf_cache[_stem_key(name_without_ext)] = pdf_path
                                    
                                    pbar.set_description(f"✅ 找到: {book_name[:20]}...")
                                    found_for_book = True
                                    break
                            if found_for_book:
                                break
            
            found_books = len([book for book in target_books if book in pdf_cache])
            logger.info(f"精确搜索完成，找到 {found_books}/{len(target_books)} 本目标书籍的PDF文件")
    except Exception as e:
        logger.error(f"构建PDF缓存出错: {e}")
        return pdf_cache



def check_book_already_processed(book_name: str, output_base_dir: str) -> bool:
    """检查书籍是否已经处理过"""
    try:
        book_output_dir = os.path.join(output_base_dir, book_name)
        
        if not os.path.exists(book_output_dir):
            return False
        
        result_json_path = os.path.join(book_output_dir, "extraction_results.json")
        if not os.path.exists(result_json_path):
            return False
        
        # 统计各分类目录下的 png
        categories = [
            os.path.join(book_output_dir, "table_images"),
            os.path.join(book_output_dir, "equation_images"),
            os.path.join(book_output_dir, "text_images"),
            os.path.join(book_output_dir, "images"),  # 兼容旧结构/未知类型
        ]
        num_png = 0
        for cat in categories:
            if not os.path.exists(cat):
                continue
            for root, _, files in os.walk(cat):
                num_png += sum(1 for f in files if f.lower().endswith('.png'))
        if num_png == 0:
            return False
        
        logger.debug(f"书籍 {book_name} 已经处理过，有 {num_png} 张图片")
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


def process_pdf_extraction(json_path: str, pdf_path: Optional[str] = None, output_dir: Optional[str] = None, include_text: bool = False) -> ProcessResult:
    """处理PDF图像截取"""
    try:
        # 解析JSON文件
        json_data = parse_json_file(json_path)
        
        # 确定PDF路径
        if not pdf_path:
            if 'pdf_path' in json_data:
                pdf_path = json_data['pdf_path']
            else:
                json_file = Path(json_path)
                pdf_path = str(json_file.parent / f"{json_file.stem}.pdf")
        
        if not os.path.exists(pdf_path):
            raise Exception(f"PDF文件不存在: {pdf_path}")
        
        # 确定书籍名称
        pdf_basename = os.path.basename(pdf_path)
        book_name = os.path.splitext(pdf_basename)[0]
        
        # 确定输出目录
        if not output_dir or output_dir.strip() == "":
            base_output_dir = str(Path(pdf_path).parent)
        else:
            if not os.path.isabs(output_dir):
                base_output_dir = str(Path.cwd() / output_dir)
            else:
                base_output_dir = output_dir
        
        # 创建书籍专用目录
        book_output_dir = os.path.join(base_output_dir, book_name)
        os.makedirs(book_output_dir, exist_ok=True)
        
        # 复制文件到书籍目录
        #pdf_copy_path = os.path.join(book_output_dir, pdf_basename)
        #if not os.path.exists(pdf_copy_path):
        #    shutil.copy2(pdf_path, pdf_copy_path)
        
        json_basename = os.path.basename(json_path)
        json_copy_path = os.path.join(book_output_dir, json_basename)
        if not os.path.exists(json_copy_path):
            shutil.copy2(json_path, json_copy_path)
        
        # 提取目标信息
        targets = extract_targets_from_json(json_data, include_text=include_text)
        
        if not targets:
            return ProcessResult(
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
        # 分类子目录
        table_dir = os.path.join(book_output_dir, "table_images")
        equation_dir = os.path.join(book_output_dir, "equation_images") 
        text_dir = os.path.join(book_output_dir, "text_images")
        os.makedirs(table_dir, exist_ok=True)
        os.makedirs(equation_dir, exist_ok=True)
        os.makedirs(text_dir, exist_ok=True)
        # id 到图片路径的映射
        id_to_image_path: dict[str, str] = {}
        
        try:
            # 处理每个目标（不显示进度条，避免在多进程中混乱）
            for target in targets:
                # 根据类型选择分类目录
                if target.type == 'span_table':
                    target_dir = table_dir
                elif target.type == 'span_interline_equation':
                    target_dir = equation_dir
                elif target.type in ('text', 'title_text'):
                    target_dir = text_dir
                else:
                    target_dir = os.path.join(book_output_dir, "images")

                output_path = os.path.join(target_dir, f"{target.id}.png")
                if extractor.extract_image_by_bbox(target.page_idx, target.bbox, output_path):
                    saved_images.append(output_path)
                    id_to_image_path[target.id] = output_path
                    processed_count += 1
                    logger.debug(f"成功截取: {target.type}#{target.page_idx}")
                else:
                    logger.warning(f"截取失败: {target.type}#{target.page_idx}")
            
        finally:
            extractor.close()
        
        # 保存结果JSON（总汇总 + 按类型拆分）
        # result_json_path = os.path.join(book_output_dir, "extraction_results.json")
        table_json_path = os.path.join(book_output_dir, "table.json")
        equation_json_path = os.path.join(book_output_dir, "equation.json")
        text_json_path = os.path.join(book_output_dir, "text.json")
        try:
            result_data = {
                "pdf_path": pdf_path,
                "json_source": json_path,
                "extraction_time": str(time.time()),
                "total_targets": len(targets),
                "processed_targets": processed_count,
                "targets": []
            }
            
            for target in targets:
                # 使用实际保存时记录的路径，若不存在则为空
                img_path = id_to_image_path.get(target.id)
                target_info = {
                    "id": target.id,
                    "type": target.type,
                    "text": target.text,
                    "bbox": target.bbox,
                    "page_idx": target.page_idx,
                    "image_path": img_path
                }
                result_data["targets"].append(target_info)
            
            # with open(result_json_path, 'w', encoding='utf-8') as f:
            #     json.dump(result_data, f, ensure_ascii=False, indent=2)

            # 写出拆分后的 JSON
            def _filter_targets(tt: str) -> list[dict]:
                return [t for t in result_data["targets"] if t.get("type") == tt]

            table_payload = {
                "pdf_path": pdf_path,
                "json_source": json_path,
                "extraction_time": result_data["extraction_time"],
                "total_targets": len(_filter_targets("span_table")),
                "processed_targets": len(_filter_targets("span_table")),
                "targets": _filter_targets("span_table"),
            }
            equation_payload = {
                "pdf_path": pdf_path,
                "json_source": json_path,
                "extraction_time": result_data["extraction_time"],
                "total_targets": len(_filter_targets("span_interline_equation")),
                "processed_targets": len(_filter_targets("span_interline_equation")),
                "targets": _filter_targets("span_interline_equation"),
            }
            text_targets = [t for t in result_data["targets"] if t.get("type") in ("text", "title_text")]
            text_payload = {
                "pdf_path": pdf_path,
                "json_source": json_path,
                "extraction_time": result_data["extraction_time"],
                "total_targets": len(text_targets),
                "processed_targets": len(text_targets),
                "targets": text_targets,
            }

            with open(table_json_path, 'w', encoding='utf-8') as f:
                json.dump(table_payload, f, ensure_ascii=False, indent=2)
            with open(equation_json_path, 'w', encoding='utf-8') as f:
                json.dump(equation_payload, f, ensure_ascii=False, indent=2)
            if include_text and text_targets:
                with open(text_json_path, 'w', encoding='utf-8') as f:
                    json.dump(text_payload, f, ensure_ascii=False, indent=2)

            logger.info(f"结果已保存到: {result_json_path}")
            logger.info(f"拆分结果已保存到: {table_json_path}, {equation_json_path}{', ' + text_json_path if include_text and text_targets else ''}")
            
        except Exception as e:
            logger.warning(f"保存结果JSON失败: {e}")
        
        return ProcessResult(
            success=True,
            message=f"成功处理 {processed_count}/{len(targets)} 个目标，输出目录: {book_output_dir}",
            saved_images=saved_images,
            total_targets=len(targets),
            processed_targets=processed_count
        )
        
    except Exception as e:
        logger.error(f"处理PDF图像截取失败: {e}")
        return ProcessResult(
            success=False,
            message=f"处理失败: {str(e)}",
            saved_images=[],
            total_targets=0,
            processed_targets=0
        )


def process_single_book_worker(book_task: BookTask) -> Dict:
    """单个书籍处理工作函数（多进程）"""
    start_time = time.time()
    try:
        logger.info(f"🔄 开始处理书籍: {book_task.book_name}")
        
        # 按需查找PDF文件
        pdf_path = book_task.pdf_path
        if pdf_path is None and book_task.pdf_base_folder:
            logger.info(f"🔍 为书籍 {book_task.book_name} 搜索PDF文件...")
            pdf_path = find_pdf_file_v2(book_task.book_name, book_task.pdf_base_folder)
            
            if pdf_path is None:
                logger.warning(f"❌ 未找到PDF文件: {book_task.book_name}")
                return {
                    "book_name": book_task.book_name,
                    "success": False,
                    "message": "未找到对应的PDF文件",
                    "json_files_found": 1,
                    "targets_processed": 0,
                    "images_saved": 0
                }
            
            logger.info(f"✅ 找到PDF文件: {os.path.basename(pdf_path)}")
        
        # 执行PDF图像提取
        logger.info(f"🎯 开始截取图像: {book_task.book_name}")
        result = process_pdf_extraction(
            book_task.json_file_path,
            pdf_path,
            book_task.output_dir,
            include_text=book_task.include_text,
        )
        
        # 计算处理时间
        elapsed_time = time.time() - start_time
        
        if result.success:
            logger.info(f"✅ 书籍处理成功: {book_task.book_name} - {result.processed_targets}目标/{len(result.saved_images)}图片 ({elapsed_time:.1f}秒)")
        else:
            logger.warning(f"❌ 书籍处理失败: {book_task.book_name} - {result.message}")
        
        return {
            "book_name": book_task.book_name,
            "success": result.success,
            "message": result.message,
            "json_files_found": 1,
            "targets_processed": result.processed_targets,
            "images_saved": len(result.saved_images),
            "processing_time": elapsed_time
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"💥 处理书籍异常: {book_task.book_name} - {str(e)} ({elapsed_time:.1f}秒)")
        return {
            "book_name": book_task.book_name,
            "success": False,
            "message": f"处理异常: {str(e)}",
            "json_files_found": 0,
            "targets_processed": 0,
            "images_saved": 0,
            "processing_time": elapsed_time
        }


def batch_process_books(
    results_folder: str, 
    pdf_base_folder: str, 
    output_base_dir: Optional[str] = None,
    max_workers: Optional[int] = None,
    include_text: bool = False,
) -> BatchResult:
    """批量处理PDF图像截取（简化版）"""
    try:
        # 验证文件夹
        if not os.path.exists(results_folder):
            raise Exception(f"结果文件夹不存在: {results_folder}")
        if not os.path.exists(pdf_base_folder):
            raise Exception(f"PDF基础文件夹不存在: {pdf_base_folder}")
        
        # 确定输出目录，增加总文件夹层级
        if not output_base_dir:
            output_base_dir = os.path.join(results_folder, "batch_output")
        
        # 创建总文件夹：使用results_folder的父目录文件名
        results_folder_path = Path(results_folder)
        total_folder_name = results_folder_path.parent.name
        if not total_folder_name or total_folder_name == ".":
            # 如果父目录是根目录，使用results_folder本身的名字
            total_folder_name = results_folder_path.name
        
        # 最终输出目录结构：output_base_dir/总文件夹/
        final_output_dir = os.path.join(output_base_dir, total_folder_name)
        os.makedirs(final_output_dir, exist_ok=True)
        
        logger.info(f"输出目录结构: {output_base_dir}/{total_folder_name}/各书籍文件夹")
        logger.info(f"总文件夹名: {total_folder_name} (来自: {results_folder_path.parent})")
        
        # 更新output_base_dir为包含总文件夹的路径
        output_base_dir = final_output_dir
        
        # 获取所有书籍文件夹
        book_folders = []
        for item in os.listdir(results_folder):
            item_path = os.path.join(results_folder, item)
            if os.path.isdir(item_path):
                book_folders.append(item)
        
        if not book_folders:
            raise Exception(f"在文件夹 {results_folder} 中未找到任何书籍文件夹")
        
        logger.info(f"找到 {len(book_folders)} 个书籍文件夹")
        
        # 智能恢复：检查已处理的书籍
        processed_books = get_processed_books_from_filesystem(output_base_dir)
        pending_books = [book_name for book_name in book_folders if book_name not in processed_books]
        
        logger.info(f"总计 {len(book_folders)} 本书籍，已处理 {len(processed_books)} 本，待处理 {len(pending_books)} 本")
        
        if processed_books:
            logger.info(f"跳过 {len(processed_books)} 本已处理的书籍")
        
        # 验证JSON文件
        valid_books = []
        with tqdm(pending_books, desc="📋 验证JSON文件", unit="本") as pbar:
            for book_name in pbar:
                pbar.set_description(f"📋 验证: {book_name[:20]}...")
                
                book_folder_path = os.path.join(results_folder, book_name)
                json_file_pattern = f"{book_name}_middle.json"
                json_file_path = os.path.join(book_folder_path, json_file_pattern)
                
                if os.path.exists(json_file_path):
                    valid_books.append((book_name, json_file_path))
                    pbar.set_description(f"✅ 有效: {book_name[:20]}...")
                else:
                    pbar.set_description(f"❌ 跳过: {book_name[:20]}...")
        
        logger.info(f"JSON验证完成，{len(valid_books)} 本书籍准备处理")
        
        # 在处理前构建或更新一次性PDF缓存（针对待处理书籍，可快速命中）
        try:
            logger.info("构建PDF文件缓存（一次性）...")
            # 优先用待处理书籍列表作为目标，以加速缓存构建
            cache = build_pdf_cache(pdf_base_folder, target_books=pending_books)
            # 若命中率较低，build_pdf_cache 会自动做全量扫描
            global PDF_CACHE
            PDF_CACHE = cache
            logger.info(f"PDF缓存条目数: {len(PDF_CACHE)}")
        except Exception as e:
            logger.warning(f"构建PDF缓存失败，后续将回退逐本查找: {e}")

        # 创建任务
        book_tasks = []
        for book_name, json_file_path in valid_books:
            book_task = BookTask(
                book_name=book_name,
                json_file_path=json_file_path,
                pdf_path=None,
                output_dir=output_base_dir,
                task_id="batch_task",
                pdf_base_folder=pdf_base_folder,
                include_text=include_text,
            )
            book_tasks.append(book_task)
        
        # 为已处理的书籍创建结果记录
        all_results = []
        for book_name in processed_books:
            # 统计分类目录下的图片数量
            base_dir = os.path.join(output_base_dir, book_name)
            cat_dirs = [
                os.path.join(base_dir, "table_images"),
                os.path.join(base_dir, "equation_images"),
                os.path.join(base_dir, "text_images"),
                os.path.join(base_dir, "images"),
            ]
            img_count = 0
            for cat in cat_dirs:
                if not os.path.exists(cat):
                    continue
                for _, _, files in os.walk(cat):
                    img_count += sum(1 for f in files if f.lower().endswith('.png'))

            all_results.append({
                "book_name": book_name,
                "success": True,
                "message": "智能恢复：跳过已处理文件",
                "json_files_found": 1,
                "targets_processed": 0,
                "images_saved": img_count
            })
        
        if not book_tasks:
            if processed_books:
                return BatchResult(
                    success=True,
                    message=f"所有 {len(processed_books)} 本书籍已处理完成",
                    total_books=len(book_folders),
                    processed_books=len(processed_books),
                    failed_books=[],
                    results=all_results
                )
            else:
                raise Exception("没有找到可处理的书籍")
        
        use_workers = MAX_WORKERS if (max_workers is None or max_workers <= 0) else max(1, max_workers)
        logger.info(f"准备处理 {len(book_tasks)} 本书籍，使用 {use_workers} 个进程")
        
        # 多进程处理
        failed_books = []
        
        with ProcessPoolExecutor(max_workers=use_workers) as executor:
            # 提交所有任务
            future_to_book = {
                executor.submit(process_single_book_worker, book_task): book_task
                for book_task in book_tasks
            }
            
            # 计算总的处理进度（包括已处理的书籍）
            total_books_to_process = len(book_folders)
            already_processed_count = len(processed_books)
            
            # 使用进度条显示整体处理进度
            with tqdm(
                total=total_books_to_process, 
                initial=already_processed_count,  # 从已处理的书籍数开始
                desc="📚 批量处理进度", 
                unit="本", 
                dynamic_ncols=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            ) as pbar:
                
                # 设置初始状态
                pbar.set_postfix({
                    '已跳过': already_processed_count,
                    '待处理': len(book_tasks),
                    '成功': 0,
                    '失败': 0
                })
                
                completed_count = 0  # 新完成的任务数
                
                for future in as_completed(future_to_book):
                    book_task = future_to_book[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                        completed_count += 1
                        
                        if result["success"]:
                            status_msg = f"✅ 成功: {result['book_name'][:20]}"
                            current_successful = len([r for r in all_results if r.get("success", False) and r.get("message") != "智能恢复：跳过已处理文件"])
                        else:
                            failed_books.append(result["book_name"])
                            status_msg = f"❌ 失败: {result['book_name'][:20]}"
                            current_successful = len([r for r in all_results if r.get("success", False) and r.get("message") != "智能恢复：跳过已处理文件"])
                        
                        # 更新进度条描述和后缀
                        pbar.set_description(f"📚 {status_msg}")
                        pbar.set_postfix({
                            '已跳过': already_processed_count,
                            '成功': current_successful,
                            '失败': len(failed_books),
                            '进度': f"{completed_count}/{len(book_tasks)}"
                        })
                        
                        # 更新进度
                        pbar.update(1)
                        
                        # 显示详细信息
                        targets = result.get("targets_processed", 0)
                        images = result.get("images_saved", 0)
                        logger.info(f"完成处理 ({completed_count}/{len(book_tasks)}): {result['book_name']} - {targets}目标/{images}图片")
                        
                    except Exception as e:
                        logger.error(f"处理任务失败: {e}")
                        failed_books.append(book_task.book_name)
                        completed_count += 1
                        
                        error_result = {
                            "book_name": book_task.book_name,
                            "success": False,
                            "message": f"任务执行异常: {str(e)}",
                            "json_files_found": 0,
                            "targets_processed": 0,
                            "images_saved": 0
                        }
                        all_results.append(error_result)
                        
                        # 更新进度条
                        pbar.set_description(f"📚 💥 异常: {book_task.book_name[:20]}")
                        pbar.set_postfix({
                            '已跳过': already_processed_count,
                            '成功': len([r for r in all_results if r.get("success", False) and r.get("message") != "智能恢复：跳过已处理文件"]),
                            '失败': len(failed_books),
                            '进度': f"{completed_count}/{len(book_tasks)}"
                        })
                        pbar.update(1)
                
                # 完成后的最终状态
                final_successful = len([r for r in all_results if r.get("success", False)])
                pbar.set_description(f"📚 🎉 批量处理完成")
                pbar.set_postfix({
                    '总成功': final_successful,
                    '总失败': len(failed_books),
                    '完成度': '100%'
                })
        
        # 统计结果
        successful_books = len([r for r in all_results if r.get("success", False)])
        
        summary_message = f"批量处理完成: {successful_books}/{len(book_folders)} 本书籍处理成功"
        if failed_books:
            summary_message += f"，失败的书籍: {', '.join(failed_books)}"
        
        return BatchResult(
            success=successful_books > 0,
            message=summary_message,
            total_books=len(book_folders),
            processed_books=successful_books,
            failed_books=failed_books,
            results=all_results
        )
        
    except Exception as e:
        logger.error(f"批量处理失败: {e}")
        return BatchResult(
            success=False,
            message=f"批量处理失败: {str(e)}",
            total_books=0,
            processed_books=0,
            failed_books=[],
            results=[]
        )


if __name__ == "__main__":
    # 简单的命令行接口
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF图像截取器 - 核心功能")
    parser.add_argument("--results-folder", required=True, help="结果文件夹路径")
    parser.add_argument("--pdf-base-folder", required=True, help="PDF搜索基础目录")
    parser.add_argument("--output-base-dir", help="输出基础目录")
    parser.add_argument("--max-workers", type=int, default=0, help="并发进程数（默认0表示自动）")
    parser.add_argument("--include-text", action="store_true", help="是否同时提取普通文本(text)目标")
    
    args = parser.parse_args()
    
    print("🚀 PDF图像截取器 - 批量处理开始")
    print("="*80)
    print(f"📂 结果文件夹: {args.results_folder}")
    print(f"📁 PDF基础目录: {args.pdf_base_folder}")
    print(f"📤 输出目录: {args.output_base_dir or '自动创建'}")
    print("="*80)
    
    try:
        result = batch_process_books(
            results_folder=args.results_folder,
            pdf_base_folder=args.pdf_base_folder,
            output_base_dir=args.output_base_dir,
            max_workers=args.max_workers,
            include_text=args.include_text,
        )
        
        # 显示结果
        print("\n" + "="*80)
        print("🎉 批量处理完成!")
        print("="*80)
        
        success_rate = (result.processed_books / result.total_books * 100) if result.total_books > 0 else 0
        
        print(f"📊 处理状态: {'✅ 成功' if result.success else '❌ 失败'}")
        print(f"📚 总书籍数: {result.total_books}")
        print(f"✅ 成功处理: {result.processed_books} ({success_rate:.1f}%)")
        print(f"❌ 失败数量: {len(result.failed_books)}")
        
        if result.failed_books:
            print(f"💔 失败书籍: {', '.join(result.failed_books[:3])}")
            if len(result.failed_books) > 3:
                print(f"   ... 以及其他 {len(result.failed_books) - 3} 本")
        
        print(f"💬 消息: {result.message}")
        print("="*80)
        
        # 显示详细结果
        if result.results:
            print("\n📋 最近处理结果:")
            recent_results = result.results[-5:] if len(result.results) > 5 else result.results
            for book_result in recent_results:
                status = "✅" if book_result.get("success", False) else "❌"
                book_name = book_result.get("book_name", "未知")[:30]
                targets = book_result.get("targets_processed", 0)
                images = book_result.get("images_saved", 0)
                
                if book_result.get("success", False):
                    print(f"  {status} {book_name}: {targets}目标/{images}图片")
                else:
                    message = book_result.get("message", "无消息")[:50]
                    print(f"  {status} {book_name}: {message}")
            
            if len(result.results) > 5:
                print(f"  📝 ... 以及其他 {len(result.results) - 5} 个结果")
        
        print("\n🚀 批量处理成功完成！")
        
    except Exception as e:
        print(f"❌ 批量处理异常: {e}")
        import traceback
        traceback.print_exc()
