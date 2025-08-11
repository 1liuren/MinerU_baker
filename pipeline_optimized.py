#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF批处理流水线 - MinerU集成版
使用MinerU进行PDF处理，支持进度条显示，移除超分功能
"""

from math import fabs
import os
import sys
import json
import time
import argparse
import traceback
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Tuple, Optional
import re
from tqdm import tqdm
from loguru import logger

def configure_logging(log_level: str = "INFO", log_file: str = None):
    """配置loguru日志"""
    logger.remove()  # 移除默认处理器
    
    # 控制台日志
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level.upper()
    )
    
    # 文件日志
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="7 days"
        )
        logger.info(f"日志文件: {log_path}")

# 默认配置
configure_logging()

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 延迟导入MinerU相关模块，避免启动时的导入错误
# 在实际使用时再导入

# 可选的OpenAI模块（用于元数据提取）
try:
    from openai import OpenAI
    from pydantic import BaseModel, Field
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    OpenAI = None
    logger.warning("OpenAI模块未安装，将跳过元数据提取功能")

# 书籍元数据模型
if OpenAI:
    class BookMetadata(BaseModel):
        title: str = Field(description="书籍标题")
        author: str = Field(description="作者名称")
        publisher: str = Field(description="出版社")
        lang: str = Field(default="中文", description="书籍语言")
        category: List[str] = Field(default_factory=list, description="书籍分类")
        knowledge: List[str] = Field(default_factory=list, description="主要知识点")


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
            file_key = f"{str(Path(file_path).resolve())}_{task_type}"
            # 检查文件是否在已处理列表中，并且处理成功
            return (file_key in self.status_data["processed_files"] and 
                   self.status_data["processed_files"][file_key]["success"])
    
    def mark_processed(self, file_path: str, task_type: str, processing_time: float, success: bool = True, error_msg: str = None):
        """标记文件处理状态"""
        with self.lock:
            file_key = f"{str(Path(file_path).resolve())}_{task_type}"
            
            self.status_data["processed_files"][file_key] = {
                "timestamp": datetime.now().isoformat(),
                "processing_time": processing_time,
                "success": success,
                "error": error_msg if not success else None,
                "task_type": task_type
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


class HTMLToMarkdownConverter:
    """HTML表格转Markdown转换器"""
    
    def __init__(self):
        # 简单的HTML表格转换正则表达式
        self.html_table_pattern = re.compile(r'<html><body><table>(.*?)</table></body></html>', re.DOTALL)
        self.table_row_pattern = re.compile(r'<tr>(.*?)</tr>', re.DOTALL)
        self.table_cell_pattern = re.compile(r'<t[hd]>(.*?)</t[hd]>', re.DOTALL)
    
    def convert_html_table_to_markdown(self, html_content: str) -> str:
        """将HTML表格转换为Markdown表格"""
        try:
            # 提取表格内容
            table_match = self.html_table_pattern.search(html_content)
            if not table_match:
                return "[表格内容]"
            
            table_content = table_match.group(1)
            rows = self.table_row_pattern.findall(table_content)
            
            if not rows:
                return "[表格内容]"
            
            markdown_rows = []
            for i, row in enumerate(rows):
                cells = self.table_cell_pattern.findall(row)
                if cells:
                    # 清理单元格内容
                    cleaned_cells = [re.sub(r'<[^>]+>', '', cell).strip() for cell in cells]
                    markdown_row = "| " + " | ".join(cleaned_cells) + " |"
                    markdown_rows.append(markdown_row)
                    
                    # 添加表头分隔符
                    if i == 0 and len(cleaned_cells) > 0:
                        separator = "| " + " | ".join(["---"] * len(cleaned_cells)) + " |"
                        markdown_rows.append(separator)
            
            return "\n".join(markdown_rows) if markdown_rows else "[表格内容]"
            
        except Exception as e:
            logger.warning(f"HTML表格转换失败: {e}")
            return "[表格内容]"
    
    def convert_html_in_text(self, text: str) -> str:
        """转换文本中的HTML表格为Markdown"""
        def replace_table(match):
            return self.convert_html_table_to_markdown(match.group(0))
        
        return self.html_table_pattern.sub(replace_table, text)


class OptimizedPDFPipeline:
    """优化版PDF处理流水线 - MinerU集成版"""
    
    def __init__(self, input_dir: str, output_dir: str, max_workers: int = 4,
                 backend: str = "vlm-sglang-client", server_url: str = None,
                 lang: str = "ch", api_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 batch_size: int = 100, concurrent_batches: int = 4):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.backend = backend
        self.server_url = server_url
        self.lang = lang
        self.api_url = api_url
        self.batch_size = batch_size
        self.concurrent_batches = concurrent_batches

        
        # 创建输出目录结构
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建必要的子目录
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 重新配置日志，输出到项目目录
        log_file = self.logs_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        configure_logging("INFO", str(log_file))
        
        # 状态管理
        self.status = ProcessingStatus(self.output_dir / "processing_status.json")
        
        # HTML转换器
        self.html_converter = HTMLToMarkdownConverter()
        
        logger.info("初始化MinerU流水线")
        logger.info(f"Backend: {self.backend}")
        logger.info(f"Server URL: {self.server_url}")
        logger.info(f"Language: {self.lang}")
        logger.info(f"Max Workers: {self.max_workers}")

    
    def format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.2f}秒"
        elif seconds < 3600:
            minutes = seconds // 60
            remaining_seconds = seconds % 60
            return f"{int(minutes)}分{remaining_seconds:.2f}秒"
        else:
            hours = seconds // 3600
            remaining = seconds % 3600
            minutes = remaining // 60
            seconds = remaining % 60
            return f"{int(hours)}小时{int(minutes)}分{seconds:.2f}秒"
    
    def find_files(self, extensions: List[str]) -> List[Path]:
        """查找指定扩展名的文件"""
        files = []
        for ext in extensions:
            files.extend(self.input_dir.rglob(f"*.{ext.lower()}"))
            files.extend(self.input_dir.rglob(f"*.{ext.upper()}"))
        return sorted(set(files))
    
    def clean_markdown_text(self, text: str) -> str:
        """清洗Markdown文本"""
        lines = text.split('\n')
        cleaned_lines = []

        # 允许的Unicode范围
        allowed_unicode_ranges = (
            r'\u4e00-\u9fff'  # 中文
            r'\u0000-\u007F'  # ASCII
            r'\u00A0-\u00FF'  # 拉丁补充
            r'\u2000-\u206F'  # 常用符号
            r'\u2100-\u214F'  # 单位、货币等
            r'\u2190-\u21FF'  # 箭头符号
            r'\u2200-\u22FF'  # 数学符号
            r'\u2B00-\u2BFF'  # 箭头和图形
            r'\u25A0-\u25FF'  # 几何图形
            r'\u0300-\u036F'  # LaTeX重音
        )

        valid_char_pattern = re.compile(f'[^{allowed_unicode_ranges}]+')
        latex_inline_pattern = re.compile(r'(\$.*?\$)')
        punctuation_map = {
            '，': ',', '。': '.', '？': '?', '！': '!', '：': ':', '；': ';',
            '（': '(', '）': ')', '【': '[', '】': ']', '"': '"', '"': '"',
            ''': "'", ''': "'", '、': ','
        }

        for line in lines:
            if line.strip().startswith('#'):
                cleaned_lines.append(line.rstrip())
                continue

            latex_parts = latex_inline_pattern.findall(line)
            line = latex_inline_pattern.sub('[LATEX]', line)

            for zh, en in punctuation_map.items():
                line = line.replace(zh, en)

            line = line.strip()
            line = re.sub(r'[ \t]{2,}', ' ', line)
            line = valid_char_pattern.sub('', line)

            for latex in latex_parts:
                line = line.replace('[LATEX]', latex, 1)

            cleaned_lines.append(line)

        final_text = '\n'.join(cleaned_lines)
        final_text = re.sub(r'\n{2,}', '\n', final_text)
        return final_text.strip()
    
    def extract_metadata_with_llm(self, text: str) -> Dict:
        """使用大模型提取元数据"""
        if not OpenAI:
            logger.warning("OpenAI模块未安装，跳过元数据提取")
            return {}
        
        try:
            client = OpenAI(
                base_url=self.api_url,
                api_key=os.getenv("DASHSCOPE_API_KEY")
            )
            
            prompt = f"""请你阅读以下书籍内容片段，并提取该书的关键信息。

请严格按照以下格式返回一个 **纯 JSON 对象**：

{{
    "title": "书籍标题",
    "author": "作者姓名",
    "publisher": "出版社名称",
    "lang": "语言（例如：中文）",
    "category": ["分类1", "分类2"],
    "knowledge": ["知识点1", "知识点2"]
}}

以下是书籍内容片段：
----------------------------------------
{text[:1000]}
...
{text[-1000:]}
----------------------------------------

请只输出标准 JSON 字符串，不要添加说明或注释。
"""
            
            response = client.chat.completions.create(
                model="qwen-max-latest",
                messages=[
                    {"role": "system", "content": "你是一个专业的中文语言助理。"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                stream = False,
                temperature=0.3,
                extra_body={"enable_thinking": False},
            )
            
            content = response.choices[0].message.content.strip()
            # 清理可能的markdown包装
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            
            if OpenAI and BookMetadata:
                data = BookMetadata.model_validate_json(content)
                return data.dict()
            else:
                return json.loads(content)
                
        except Exception as e:
            logger.error(f"元数据提取失败: {e}")
            return {}
    
    # 移除了单个PDF处理方法，现在使用批量处理模式
    
    def convert_epub_to_pdf(self) -> bool:
        """转换EPUB文件为PDF"""
        epub_files = self.find_files(['epub'])
        if not epub_files:
            logger.info("未找到EPUB文件，跳过转换步骤")
            return True
        
        logger.info(f"找到 {len(epub_files)} 个EPUB文件，开始转换...")
        
        try:
            # 使用现有的batch_epub_to_pdf函数
            from scripts.batch_epub_to_pdf import batch_convert_epub_to_pdf
            
            batch_convert_epub_to_pdf(
                source_dir=str(self.input_dir),
                output_dir=str(self.input_dir / "converted_pdfs"),
                max_workers=self.max_workers,
                skip_existing=True
            )
            logger.success("EPUB转PDF完成")
            return True
        except Exception as e:
            logger.error(f"EPUB转PDF失败: {e}")
            return False
    
    def process_all_pdfs(self) -> Tuple[bool, List[Dict]]:
        """处理所有PDF文件 - 使用demo.py的批量处理方式"""
        # 查找所有PDF文件
        pdf_files = []
        pdf_files.extend(self.find_files(['pdf']))
        
        # 添加转换后的PDF文件
        converted_dir = self.input_dir / "converted_pdfs"
        if converted_dir.exists():
            pdf_files.extend(converted_dir.rglob("*.pdf"))
        
        pdf_files = sorted(set(pdf_files))
        
        if not pdf_files:
            logger.error("未找到PDF文件")
            return False, []
        
        logger.info(f"找到 {len(pdf_files)} 个PDF文件，使用批量处理模式...")
        logger.info("服务端自动处理多进程，无需客户端多线程")
        
        return self._process_with_batch_mode(pdf_files)
    
    def _process_with_batch_mode(self, pdf_files: List[Path]) -> Tuple[bool, List[Dict]]:
        """使用分批次异步处理模式处理PDF文件"""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        success_count = 0
        processed_data = []
        
        # 初始化统计信息
        self.status.status_data["processing_stats"]["total_files"] = len(pdf_files)
        self.status._save_status()
        
        # 检查已处理的文件（跳过）
        files_to_process = []
        logger.info("检查已处理的文件...")
        
        with tqdm(pdf_files, desc="检查文件状态", unit="文件") as pbar:
            for pdf_file in pbar:
                if self.status.is_processed(str(pdf_file), "pdf_processing"):
                    logger.debug(f"跳过已处理文件: {pdf_file.name}")
                    self.status.status_data["processing_stats"]["skipped"] += 1
                    continue
                files_to_process.append(pdf_file)
        
        if not files_to_process:
            logger.success("所有文件已处理完成")
            return True, []
        
        total_files = len(files_to_process)
        logger.info(f"需要处理 {total_files} 个文件（跳过 {len(pdf_files) - total_files} 个已处理）")
        logger.info(f"批次配置: 每批次 {self.batch_size} 个文件，同时处理 {self.concurrent_batches} 个批次")
        
        # 将文件分成批次
        batches = []
        for i in range(0, total_files, self.batch_size):
            batch = files_to_process[i:i + self.batch_size]
            batches.append(batch)
        
        logger.info(f"共分为 {len(batches)} 个批次")
        
        try:
            # 使用异步处理多个批次
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                success, data = loop.run_until_complete(
                    self._process_batches_async(batches)
                )
                return success, data
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"批量处理失败: {e}")
            logger.debug(traceback.format_exc())
            return False, []
    
    # 移除 _process_large_batch 相关逻辑（不再限制单次调用文件数）
    
    # 移除 _process_group_directly（不再按调用组分层）
    
    async def _process_batches_async(self, batches: List[List[Path]]) -> Tuple[bool, List[Dict]]:
        """异步处理多个批次"""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        all_processed_data = []
        total_success = 0
        
        # 使用信号量控制并发数量
        semaphore = asyncio.Semaphore(self.concurrent_batches)
        
        async def process_single_batch(batch_idx: int, batch_files: List[Path]):
            async with semaphore:
                logger.info(f"开始处理批次 {batch_idx + 1}/{len(batches)} ({len(batch_files)} 个文件)")
                
                # 在线程池中执行同步的批次处理
                with ThreadPoolExecutor(max_workers=1) as executor:
                    loop = asyncio.get_event_loop()
                    success, data = await loop.run_in_executor(
                        executor, 
                        self._process_single_batch_sync, 
                        batch_idx, 
                        batch_files
                    )
                
                return success, data
        
        # 创建所有批次的任务
        tasks = [
            process_single_batch(i, batch) 
            for i, batch in enumerate(batches)
        ]
        
        # 等待所有批次完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 收集结果
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"批次 {i + 1} 处理失败: {result}")
            else:
                success, data = result
                if success:
                    all_processed_data.extend(data)
                    total_success += len(data)
        
        logger.success(f"所有批次处理完成，总共成功处理 {total_success} 个文件")
        return total_success > 0, all_processed_data
    
    def _process_single_batch_sync(self, batch_idx: int, batch_files: List[Path]) -> Tuple[bool, List[Dict]]:
        """同步处理单个批次"""
        from mineru.cli.common import read_fn
        import time
        
        batch_start_time = time.time()
        success_count = 0
        processed_data = []
        
        try:
            # 准备文件列表和参数
            file_name_list = []
            pdf_bytes_list = []
            lang_list = []
            valid_files = []
            
            logger.info(f"批次 {batch_idx + 1}: 读取 {len(batch_files)} 个PDF文件...")
            
            # 文件读取阶段计时
            read_start_time = time.time()
            
            with tqdm(batch_files, desc=f"读取批次{batch_idx + 1}文件", unit="文件", leave=False) as pbar:
                for pdf_file in pbar:
                    pbar.set_postfix({"当前": pdf_file.name[:20]})
                    try:
                        file_read_start = time.time()
                        file_name = str(pdf_file.stem)
                        pdf_bytes = read_fn(str(pdf_file))
                        file_read_time = time.time() - file_read_start
                        
                        file_name_list.append(file_name)
                        pdf_bytes_list.append(pdf_bytes)
                        lang_list.append(self.lang)
                        valid_files.append(pdf_file)
                        
                        logger.debug(f"批次 {batch_idx + 1}: 读取文件 {pdf_file.name} 耗时 {file_read_time:.2f}s, 大小 {len(pdf_bytes)} bytes")
                    except Exception as e:
                        logger.error(f"读取文件失败 {pdf_file.name}: {e}")
                        self.status.mark_processed(
                            str(pdf_file), 
                            "pdf_processing", 
                            0, 
                            False, 
                            f"读取文件失败: {str(e)}"
                        )
                        continue
            
            read_total_time = time.time() - read_start_time
            logger.debug(f"批次 {batch_idx + 1}: 文件读取总耗时 {self.format_time(read_total_time)}")
            
            if not pdf_bytes_list:
                logger.error(f"批次 {batch_idx + 1}: 没有可处理的文件")
                return False, []
            
            logger.info(f"批次 {batch_idx + 1}: 开始处理 {len(pdf_bytes_list)} 个文件...")
            
            # 动态导入do_parse，避免Python版本兼容性问题
            try:
                import importlib.util
                import sys
                
                # 动态加载demo.py模块
                demo_path = Path(__file__).parent / "demo" / "demo.py"
                if not demo_path.exists():
                    logger.error(f"demo.py文件不存在: {demo_path}")
                    return False, []
                
                spec = importlib.util.spec_from_file_location("demo", demo_path)
                demo_module = importlib.util.module_from_spec(spec)
                
                # 临时修改sys.path以确保导入成功
                original_path = sys.path.copy()
                sys.path.insert(0, str(demo_path.parent.parent))
                
                try:
                    spec.loader.exec_module(demo_module)
                    do_parse = demo_module.do_parse
                    logger.debug("成功动态导入do_parse函数")
                except Exception as import_error:
                    logger.error(f"动态导入demo模块失败: {import_error}")
                    logger.debug(traceback.format_exc())
                    return False, []
                finally:
                    sys.path = original_path
                    
            except Exception as e:
                logger.error(f"无法导入demo.demo模块: {e}")
                logger.debug(traceback.format_exc())
                return False, []
            
            # 创建临时输出目录
            temp_output_dir = self.temp_dir / f"batch_{batch_idx + 1}_{int(time.time())}"
            temp_output_dir.mkdir(exist_ok=True)
            
            logger.info(f"批次 {batch_idx + 1}: 临时输出目录 {temp_output_dir}")
            
            # 记录处理开始时间
            process_start_time = time.time()
            
            # 调用HTTP服务进行批量处理（异步并发）
            try:
                logger.info(f"批次 {batch_idx + 1}: 通过HTTP服务(异步)处理 {len(valid_files)} 个文件")
                logger.info(f"批次 {batch_idx + 1}: 使用后端 {self.backend}，目标输出目录 {self.results_dir}")

            #  # 动态导入异步客户端
            #  import importlib.util as _importlib_util
            #  client_path = Path(__file__).parent / "projects" / "multi_gpu_v2" / "client.py"
            #  spec_client = _importlib_util.spec_from_file_location("mclient", client_path)
            #  mclient = _importlib_util.module_from_spec(spec_client)
            #  spec_client.loader.exec_module(mclient)
            #  mineru_parse_async = getattr(mclient, "mineru_parse_async")
                from projects.multi_gpu_v2.client import mineru_parse_async


                # 服务端/predict地址：优先环境变量MINERU_SERVICE_URL，否则默认本机
                service_url = os.getenv("MINERU_SERVICE_URL", "http://10.10.50.52:8111/predict")

                import asyncio
                import aiohttp

                async def _run_async_calls():
                    # 每批次内部的HTTP并发，使用 max_workers 控制
                    concurrency = max(1, int(self.max_workers))
                    semaphore = asyncio.Semaphore(concurrency)

                    async with aiohttp.ClientSession() as session:
                        async def _one(idx, pdf_path):
                            opts = {
                                'backend': self.backend,
                                'method': 'auto',
                                'lang': lang_list[idx],
                                'formula_enable': True,
                                'table_enable': True,
                                'start_page_id': 0,
                                'end_page_id': None,
                                # VLM后端的URL使用 vlm_server_url，避免与HTTP请求参数 server_url 冲突
                                'vlm_server_url': None,
                                # 指定服务端输出根目录为批次临时目录
                                'output_dir': str(self.results_dir),
                            }
                            async with semaphore:
                                # 修复server_url参数重复传递的问题：直接传递service_url作为位置参数
                                return await mineru_parse_async(session, str(pdf_path), service_url, **opts)

                        tasks = [
                            _one(idx, pdf_file) for idx, pdf_file in enumerate(valid_files)
                        ]
                        return await asyncio.gather(*tasks, return_exceptions=True)

                parse_start_time = time.time()
                # 独立事件循环，避免与外层冲突
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(_run_async_calls())
                finally:
                    loop.close()

                # 检查错误
                for r in results:
                    if isinstance(r, Exception):
                        raise r
                    if isinstance(r, dict) and 'error' in r:
                        raise RuntimeError(r['error'])

                parse_time = time.time() - parse_start_time
                logger.success(f"批次 {batch_idx + 1}: 服务端并发调用完成，耗时 {self.format_time(parse_time)}")
                
            except Exception as e:
                logger.error(f"批次 {batch_idx + 1}: do_parse调用失败: {e}")
                logger.error(traceback.format_exc())
                return False, []
            
            process_time = time.time() - process_start_time
            logger.info(f"批次 {batch_idx + 1}: 处理耗时 {self.format_time(process_time)}")
            
            logger.info(f"批次 {batch_idx + 1}: 收集处理结果...")
            
            # 收集处理结果 - 分阶段进行，先收集文件，再并行处理元数据
            collect_start_time = time.time()
            
            # 第一阶段：收集文件内容
            file_contents = []
            with tqdm(valid_files, desc=f"收集批次{batch_idx + 1}文件", unit="文件", leave=False) as pbar:
                for idx, pdf_file in enumerate(pbar):
                    pbar.set_postfix({"当前": pdf_file.name[:20]})
                    try:
                        file_name = file_name_list[idx]
                        
                        # 查找输出文件
                        if self.backend == "pipeline":
                            parse_method = "auto"
                        else:
                            parse_method = "vlm"
                        
                        result_dir = temp_output_dir / file_name / parse_method
                        md_file = result_dir / f"{file_name}.md"
                        json_file = result_dir / f"{file_name}_middle.json"
                        
                        if md_file.exists() and json_file.exists():
                            # 读取处理结果
                            with open(md_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                            with open(json_file, 'r', encoding='utf-8') as f:
                                middle_json = json.load(f)
                            
                            # 清洗markdown内容
                            cleaned_content = self.clean_markdown_text(content)
                            
                            # 移动文件到最终目录
                            final_dir = self.results_dir / pdf_file.stem
                            final_dir.mkdir(exist_ok=True)
                            
                            # 保存清洗后的内容到.md文件
                            import shutil
                            with open(final_dir / f"{pdf_file.stem}.md", 'w', encoding='utf-8') as f:
                                f.write(cleaned_content)
                            # 复制middle.json文件
                            shutil.copy2(json_file, final_dir / f"{pdf_file.stem}_middle.json")
                            
                            # 收集文件信息，暂不提取元数据
                            file_contents.append({
                                "pdf_file": pdf_file,
                                "content": cleaned_content,
                                "file_name": file_name,
                                "idx": idx
                            })
                            
                        else:
                            logger.error(f"未找到输出文件: {pdf_file.name}")
                            self.status.mark_processed(
                                str(pdf_file), 
                                "pdf_processing", 
                                0, 
                                False, 
                                "未找到输出文件"
                            )
                            
                    except Exception as e:
                        logger.error(f"收集结果失败 {pdf_file.name}: {e}")
                        self.status.mark_processed(
                            str(pdf_file), 
                            "pdf_processing", 
                            0, 
                            False, 
                            f"收集结果失败: {str(e)}"
                        )
            
            collect_files_time = time.time() - collect_start_time
            logger.debug(f"批次 {batch_idx + 1}: 文件收集耗时 {self.format_time(collect_files_time)}")
            
            # 第二阶段：并行提取元数据
            if file_contents:
                metadata_start_time = time.time()
                logger.info(f"批次 {batch_idx + 1}: 开始并行提取 {len(file_contents)} 个文件的元数据...")
                
                # 使用多线程并行处理元数据提取
                from concurrent.futures import ThreadPoolExecutor, as_completed
                
                def extract_metadata_for_file(file_info):
                    """为单个文件提取元数据"""
                    try:
                        metadata = self.extract_metadata_from_content(file_info["content"])
                        return file_info, metadata, None
                    except Exception as e:
                        return file_info, None, str(e)
                
                # 使用线程池并行处理
                max_metadata_workers = min(len(file_contents), 200)  # 限制并发数
                with ThreadPoolExecutor(max_workers=max_metadata_workers) as executor:
                    # 提交所有任务
                    future_to_file = {executor.submit(extract_metadata_for_file, file_info): file_info 
                                     for file_info in file_contents}
                    
                    # 收集结果
                    with tqdm(total=len(file_contents), desc=f"提取批次{batch_idx + 1}元数据", unit="文件", leave=False) as pbar:
                        for future in as_completed(future_to_file):
                            file_info, metadata, error = future.result()
                            pbar.update(1)
                            pbar.set_postfix({"当前": file_info["pdf_file"].name[:20]})
                            
                            if error:
                                logger.error(f"元数据提取失败 {file_info['pdf_file'].name}: {error}")
                                self.status.mark_processed(
                                    str(file_info["pdf_file"]), 
                                    "pdf_processing", 
                                    0, 
                                    False, 
                                    f"元数据提取失败: {error}"
                                )
                            else:
                                # 保存大模型提取的元数据到单独文件
                                final_dir = self.results_dir / file_info["pdf_file"].stem
                                metadata_file = final_dir / f"{file_info['pdf_file'].stem}_extracted_metadata.json"
                                
                                try:
                                    with open(metadata_file, 'w', encoding='utf-8') as f:
                                        json.dump(metadata or {}, f, ensure_ascii=False, indent=2)
                                    logger.debug(f"保存元数据到: {metadata_file}")
                                except Exception as e:
                                    logger.warning(f"保存元数据失败 {file_info['pdf_file'].name}: {e}")
                                
                                # 收集最终数据
                                processed_data.append({
                                    "file": file_info["pdf_file"].name,
                                    "content": file_info["content"],
                                    "metadata": metadata or {}
                                })
                                
                                # 标记为已处理
                                self.status.mark_processed(
                                    str(file_info["pdf_file"]), 
                                    "pdf_processing", 
                                    process_time / len(valid_files),  # 平均处理时间
                                    True
                                )
                                
                                success_count += 1
                
                metadata_time = time.time() - metadata_start_time
                logger.debug(f"批次 {batch_idx + 1}: 元数据提取耗时 {self.format_time(metadata_time)} (并行处理 {max_metadata_workers} 线程)")
            
            # 清理临时目录
            try:
                import shutil
                if temp_output_dir.exists():
                    shutil.rmtree(temp_output_dir)
                    logger.debug(f"批次 {batch_idx + 1}: 临时目录已清理")
            except Exception as e:
                logger.warning(f"批次 {batch_idx + 1}: 清理临时目录失败: {e}")
            
            batch_total_time = time.time() - batch_start_time
            
            # 详细的耗时汇总
            logger.success(f"批次 {batch_idx + 1} 完成: {success_count}/{len(batch_files)} 个文件成功，总耗时 {self.format_time(batch_total_time)}")
            logger.info(f"批次 {batch_idx + 1} 耗时详情:")
            logger.info(f"  - 文件读取: {self.format_time(read_total_time)} ({read_total_time/batch_total_time*100:.1f}%)")
            logger.info(f"  - PDF处理: {self.format_time(parse_time)} ({parse_time/batch_total_time*100:.1f}%)")
            logger.info(f"  - 文件收集: {self.format_time(collect_files_time)} ({collect_files_time/batch_total_time*100:.1f}%)")
            if 'metadata_time' in locals():
                logger.info(f"  - 元数据提取: {self.format_time(metadata_time)} ({metadata_time/batch_total_time*100:.1f}%)")
            logger.info(f"  - 平均每文件: {self.format_time(batch_total_time/len(batch_files))}")
            
            return success_count > 0, processed_data
            
        except Exception as e:
            logger.error(f"批次 {batch_idx + 1} 处理失败: {e}")
            logger.debug(traceback.format_exc())
            return False, []
    
    def extract_metadata_from_content(self, content: str) -> Dict:
        """从内容中提取元数据（简化版）"""
        return self.extract_metadata_with_llm(content)
    
    def create_final_jsonl(self, processed_data: List[Dict]) -> bool:
        """生成最终的JSONL文件 - 收集所有历史处理过的PDF文件数据，优先使用大模型提取的元数据"""
        logger.info("开始收集所有已处理的PDF文件数据...")
        
        try:
            # 收集results目录下所有已处理的文件数据
            all_processed_data = []
            
            # 遍历results目录下的所有子目录
            if self.results_dir.exists():
                for pdf_dir in self.results_dir.iterdir():
                    if pdf_dir.is_dir():
                        pdf_name = pdf_dir.name
                        md_file = pdf_dir / f"{pdf_name}.md"
                        extracted_metadata_file = pdf_dir / f"{pdf_name}_extracted_metadata.json"
                        middle_json_file = pdf_dir / f"{pdf_name}_middle.json"
                        
                        # 检查必要文件是否存在
                        if md_file.exists():
                            logger.debug(f"读取文件: {pdf_name}")
                            
                            # 读取markdown内容
                            try:
                                with open(md_file, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                
                                # 优先读取大模型提取的元数据
                                metadata = {}
                                if extracted_metadata_file.exists():
                                    try:
                                        with open(extracted_metadata_file, 'r', encoding='utf-8') as f:
                                            extracted_metadata = json.load(f)
                                            logger.debug(f"使用大模型提取的元数据: {pdf_name}")
                                            metadata = extracted_metadata
                                            # 确保基本字段存在
                                            metadata.setdefault("source_file", pdf_name)
                                            metadata.setdefault("lang", "zh")
                                            metadata.setdefault("type", "书籍")
                                            metadata.setdefault("processing_date", datetime.now().strftime("%Y-%m-%d"))
                                    except Exception as e:
                                        logger.warning(f"读取大模型元数据失败 {extracted_metadata_file}: {e}")
                                        metadata = None
                                
                                # 如果没有大模型元数据，则使用基本元数据（不依赖middle.json的文件信息）
                                if not metadata:
                                    logger.debug(f"使用基本元数据: {pdf_name}")
                                    # 尝试从middle.json获取基本的处理信息（非文件信息）
                                    if middle_json_file.exists():
                                        try:
                                            with open(middle_json_file, 'r', encoding='utf-8') as f:
                                                middle_data = json.load(f)
                                                # 只提取处理相关信息，不依赖文件信息

                                        except Exception as e:
                                            logger.warning(f"读取middle.json基本信息失败: {e}")
                                    
                                    metadata = {
                                        "source_file": pdf_name,
                                        "lang": "zh",
                                        "type": "书籍",
                                        "processing_date": datetime.now().strftime("%Y-%m-%d"),
                                    }
                                
                                # 如果内容不为空，则添加到处理数据中
                                if content.strip():
                                    all_processed_data.append({
                                        "content": content,
                                        "metadata": metadata
                                    })
                                    
                            except Exception as e:
                                logger.error(f"读取{md_file}失败: {e}")
                                continue
            
            # # 添加当前批次的数据（如果有）
            # if processed_data:
            #     logger.info(f"添加当前批次数据: {len(processed_data)} 条记录")
            #     all_processed_data.extend(processed_data)
            
            logger.info(f"总共收集到 {len(all_processed_data)} 条记录")
            
            if not all_processed_data:
                logger.warning("没有找到任何处理成功的数据，跳过JSONL生成")
                return True
            
            # 生成JSONL文件
            output_jsonl = self.results_dir / f"processed_books_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            
            with open(output_jsonl, 'w', encoding='utf-8') as f:
                for item in all_processed_data:
                    jsonl_record = {
                        "id": hashlib.sha256(item["content"].encode('utf-8')).hexdigest(),
                        "text": item["content"],
                        "meta": {
                            "data_info": {
                                "lang": item["metadata"].get("lang", "zh"),
                                "source": item["metadata"].get("source_file", item["metadata"].get("publisher", "")),
                                "type": item["metadata"].get("type", "书籍"),
                                "author": item["metadata"].get("author", ""),
                                "processing_date": item["metadata"].get("processing_date", datetime.now().strftime("%Y-%m-%d"))
                            },
                            "knowledge_info": {
                                "category": item["metadata"].get("category", []),
                                "knowledge": item["metadata"].get("knowledge", [])
                            }
                        }
                    }
                    f.write(json.dumps(jsonl_record, ensure_ascii=False) + '\n')
            
            logger.success(f"JSONL文件生成完成: {output_jsonl}")
            logger.info(f"包含历史所有处理过的 {len(all_processed_data)} 个PDF文件的数据")
            return True
            
        except Exception as e:
            logger.error(f"JSONL生成失败: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def run_pipeline(self) -> bool:
        """运行完整流水线"""
        logger.info("开始优化版PDF处理流水线")
        logger.info(f"输入目录: {self.input_dir}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"最大线程数: {self.max_workers}")
        logger.info(f"处理后端: {self.backend}")
        
        start_time = time.time()
        
        try:
            # 步骤1: 转换EPUB为PDF
            if not self.convert_epub_to_pdf():
                logger.error("EPUB转PDF失败")
                return False
            
            # 步骤2: 处理所有PDF文件
            success, processed_data = self.process_all_pdfs()
            if not success:
                logger.error("PDF处理失败")
                return False
            
            # 步骤3: 生成最终JSONL
            if not self.create_final_jsonl(processed_data):
                logger.error("JSONL生成失败")
                return False
            
            # 输出最终统计
            total_time = time.time() - start_time
            stats = self.status.get_stats()
            
            logger.info("\n" + "="*60)
            logger.success("优化版流水线处理完成!")
            logger.info("="*60)
            logger.info(f"总耗时: {self.format_time(total_time)}")
            logger.info(f"处理统计:")
            logger.info(f"  - 完成: {stats['completed']}")
            logger.info(f"  - 失败: {stats['failed']}")
            logger.info(f"输出目录: {self.results_dir}")
            logger.info(f"保留文件: .md (清洗后内容) + .json (middle数据)")
            
            return True
            
        except KeyboardInterrupt:
            logger.warning("用户中断操作")
            return False
        except Exception as e:
            logger.error(f"流水线执行失败: {e}")
            logger.error(traceback.format_exc())
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="PDF/EPUB批处理流水线 - MinerU集成版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MinerU集成版特性:
- 使用MinerU进行PDF处理，支持多种后端
- 智能分批次异步处理，支持大规模文件处理
- 可配置批次大小和并发数量，优化处理效率
- 自动处理超大文件集合，分多次调用避免内存溢出
- 详细的时间统计和进度跟踪
- 移除超分辨率功能，专注于文档解析
- 支持VLM和Pipeline两种处理模式

批次处理配置:
- --batch-size: 每个批次处理的文件数量 (默认100)
- --concurrent-batches: 同时处理的批次数量 (默认4)
- --max-files-per-call: 单次调用最大文件数量 (默认400)

使用示例:
  # 基本用法
  python pipeline_optimized.py -p /path/to/pdfs -o /path/to/output
  
  # 自定义批次配置
  python pipeline_optimized.py -p /path/to/pdfs -o /path/to/output --batch-size 50 --concurrent-batches 8
  
  # VLM后端处理
  python pipeline_optimized.py -p /path/to/pdfs -o /path/to/output -b vlm-sglang-client -u http://10.10.50.50:30000
  
  # 处理超大文件集合
  python pipeline_optimized.py -p /path/to/pdfs -o /path/to/output --max-files-per-call 200 --batch-size 25
        """
    )
    
    parser.add_argument("-p", "--input-dir", required=True, help="输入目录路径")
    parser.add_argument("-o", "--output-dir", required=True, help="输出目录路径")
    parser.add_argument("-b", "--backend", default="vlm-sglang-client", 
                       choices=["pipeline", "vlm-transformers", "vlm-sglang-engine", "vlm-sglang-client"],
                       help="处理后端 (默认: vlm-sglang-client)")
    parser.add_argument("-u", "--server-url", default="http://10.10.50.50:30000",help="VLM服务器URL (用于vlm-sglang-client)")
    parser.add_argument("--lang", default="ch", 
                       choices=["ch", "ch_server", "ch_lite", "en", "korean", "japan", "chinese_cht", "ta", "te", "ka"],
                       help="文档语言 (默认: ch)")
    parser.add_argument("--api-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                       help="大模型API地址")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="每个批次处理的PDF文件数量 (默认: 100)")
    parser.add_argument("--concurrent-batches", type=int, default=1,
                       help="同时处理的批次数量 (默认: 4)")
    # 移除 --max-files-per-call 参数
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别 (默认: INFO)")
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        logger.error(f"输入目录不存在 - {args.input_dir}")
        sys.exit(1)
    
    # 检查VLM后端的server_url参数
    if args.backend == "vlm-sglang-client" and not args.server_url:
        logger.error("使用vlm-sglang-client后端时必须指定--server-url参数")
        sys.exit(1)
    
    # 移除线程数验证，使用批量处理模式
    
    # 重新配置日志级别
    configure_logging(args.log_level)
    
    # 创建并运行MinerU流水线
    try:
        logger.info(f"开始初始化流水线，参数: backend={args.backend}, batch_size={args.batch_size}, concurrent_batches={args.concurrent_batches}")
        
        pipeline = OptimizedPDFPipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            max_workers=10,  # 批量处理模式不需要多线程
            backend=args.backend,
            server_url=args.server_url,
            lang=args.lang,
            api_url=args.api_url,
            batch_size=args.batch_size,
            concurrent_batches=args.concurrent_batches
        )
        
        success = pipeline.run_pipeline()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.warning("用户中断操作")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
