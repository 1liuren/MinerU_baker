#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF处理流水线模块
"""

import os
import sys
import json
import time
import traceback
import shutil
import hashlib
import asyncio
import importlib.util
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from multiprocessing import Manager, Queue
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from loguru import logger

from .config import configure_logging, OpenAI, BookMetadata
from .status_manager import ProcessingStatus
from .html_converter import HTMLToMarkdownConverter
from .utils import format_time, find_files, clean_markdown_text, extract_metadata_with_llm


def process_batch_worker(batch_data):
    """多进程工作函数，处理单个批次"""
    import importlib.util
    import sys
    import time
    import traceback
    import shutil
    import json
    import os
    from pathlib import Path
    from loguru import logger
    
    batch_idx, batch_files, config = batch_data
    
    try:
        # 配置子进程的日志 - 确保子进程有正确的日志配置
        from .config import configure_logging
        from datetime import datetime
        
        # 重新配置子进程的日志
        log_level = config.get('log_level', 'INFO')
        # 为子进程创建独立的日志文件
        log_file = Path(config['output_dir']) / "logs" / f"worker_{os.getpid()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        configure_logging(log_level, str(log_file))
        
        # 确保子进程有正确的环境变量
        if config.get('dashscope_api_key'):
            os.environ['DASHSCOPE_API_KEY'] = config['dashscope_api_key']
        
        # 导入必要的模块
        from mineru.cli.common import read_fn
        from .utils import format_time
        
        logger.info(f"进程 {os.getpid()}: 开始处理批次 {batch_idx + 1} ({len(batch_files)} 个文件)")
        
        # 准备文件列表和参数
        file_name_list = []
        pdf_bytes_list = []
        lang_list = []
        valid_files = []
        
        # 读取文件
        for pdf_file in batch_files:
            try:
                file_name = str(pdf_file.stem)
                pdf_bytes = read_fn(str(pdf_file))
                
                file_name_list.append(file_name)
                pdf_bytes_list.append(pdf_bytes)
                lang_list.append(config['lang'])
                valid_files.append(pdf_file)
                
            except Exception as e:
                logger.error(f"进程 {os.getpid()}: 读取文件失败 {pdf_file.name}: {e}")
                continue
        
        if not pdf_bytes_list:
            logger.error(f"进程 {os.getpid()}: 批次 {batch_idx + 1} 没有可处理的文件")
            return False, []
        
        # 动态导入do_parse
        demo_path = Path(config['demo_path'])
        spec = importlib.util.spec_from_file_location("demo", demo_path)
        demo_module = importlib.util.module_from_spec(spec)
        
        original_path = sys.path.copy()
        sys.path.insert(0, str(demo_path.parent.parent))
        
        try:
            spec.loader.exec_module(demo_module)
            do_parse = demo_module.do_parse
        finally:
            sys.path = original_path
        
        # 创建临时输出目录
        temp_output_dir = Path(config['temp_dir']) / f"batch_{batch_idx + 1}_{int(time.time())}_{os.getpid()}"
        temp_output_dir.mkdir(exist_ok=True)
        
        logger.info(f"进程 {os.getpid()}: 批次 {batch_idx + 1} 开始调用do_parse处理 {len(pdf_bytes_list)} 个文件")
        
        # 调用do_parse
        parse_start_time = time.time()
        do_parse(
            output_dir=str(temp_output_dir),
            pdf_file_names=file_name_list,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=lang_list,
            backend=config['backend'],
            parse_method="auto",
            formula_enable=True,
            table_enable=True,
            server_url=config['server_url'],
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=True,
            f_dump_middle_json=True,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=False
        )
        parse_time = time.time() - parse_start_time
        logger.success(f"进程 {os.getpid()}: 批次 {batch_idx + 1} do_parse调用完成，耗时 {format_time(parse_time)}")
        
        # 收集处理结果 - 分阶段进行，先收集文件，再并行处理元数据
        file_contents = []
        
        # 第一阶段：收集所有文件内容
        for idx, pdf_file in enumerate(valid_files):
            try:
                file_name = file_name_list[idx]
                
                # 查找输出文件
                if config['backend'] == "pipeline":
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
                    from .utils import clean_markdown_text
                    cleaned_content = clean_markdown_text(content)
                    
                    # 移动文件到最终目录
                    final_dir = Path(config['results_dir']) / pdf_file.stem
                    final_dir.mkdir(exist_ok=True)
                    
                    # 保存清洗后的内容到.md文件
                    with open(final_dir / f"{pdf_file.stem}.md", 'w', encoding='utf-8') as f:
                        f.write(cleaned_content)
                    # 复制middle.json文件
                    shutil.copy2(json_file, final_dir / f"{pdf_file.stem}_middle.json")
                    
                    # 收集文件信息，准备并行处理元数据
                    file_contents.append({
                        "pdf_file": pdf_file,
                        "content": cleaned_content,
                        "file_name": file_name,
                        "idx": idx,
                        "final_dir": final_dir
                    })
                    
                else:
                    logger.error(f"进程 {os.getpid()}: 未找到输出文件: {pdf_file.name}")
                    
            except Exception as e:
                logger.error(f"进程 {os.getpid()}: 收集结果失败 {pdf_file.name}: {e}")
        
        # 第二阶段：使用线程池并行处理元数据
        processed_data = []
        if file_contents:
            logger.info(f"进程 {os.getpid()}: 开始并行提取 {len(file_contents)} 个文件的元数据")
            
            def extract_metadata_for_file(file_info):
                """为单个文件提取元数据"""
                try:
                    import sys
                    from pathlib import Path
                    sys.path.insert(0, str(Path(__file__).parent))
                    from utils import extract_metadata_with_llm
                    
                    api_url = config.get('api_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
                    metadata = extract_metadata_with_llm(file_info["content"], api_url)
                    
                    # 保存元数据到单独的JSON文件
                    metadata_file = file_info["final_dir"] / f"{file_info['pdf_file'].stem}_extracted_metadata.json"
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                    
                    logger.debug(f"进程 {os.getpid()}: 元数据提取成功 {file_info['pdf_file'].name}")
                    return file_info, metadata, None
                except Exception as e:
                    logger.error(f"进程 {os.getpid()}: 元数据提取失败 {file_info['pdf_file'].name}: {e}")
                    return file_info, {}, str(e)
            
            # 使用线程池并行处理
            from concurrent.futures import ThreadPoolExecutor, as_completed
            max_metadata_workers = min(len(file_contents), 10)  # 限制并发数
            
            with ThreadPoolExecutor(max_workers=max_metadata_workers) as executor:
                # 提交所有任务
                future_to_file = {executor.submit(extract_metadata_for_file, file_info): file_info 
                                 for file_info in file_contents}
                
                # 收集结果
                for future in as_completed(future_to_file):
                    file_info, metadata, error = future.result()
                    
                    if error:
                        logger.error(f"进程 {os.getpid()}: 元数据提取失败 {file_info['pdf_file'].name}: {error}")
                    
                    processed_data.append({
                        "pdf_file": file_info["pdf_file"],
                        "content": file_info["content"],
                        "metadata": metadata,
                        "file_name": file_info["file_name"],
                        "idx": file_info["idx"]
                    })
            
            logger.info(f"进程 {os.getpid()}: 元数据提取完成，处理了 {len(file_contents)} 个文件")
        
        # 清理临时目录
        try:
            shutil.rmtree(temp_output_dir)
        except Exception as e:
            logger.warning(f"进程 {os.getpid()}: 清理临时目录失败: {e}")
        
        logger.success(f"进程 {os.getpid()}: 批次 {batch_idx + 1} 处理完成，成功处理 {len(processed_data)} 个文件")
        return True, processed_data, parse_time  # 返回真正的批次处理时间
        
    except Exception as e:
        logger.error(f"进程 {os.getpid()}: 批次 {batch_idx + 1} 处理失败: {e}")
        logger.error(traceback.format_exc())
        # 如果parse_time未定义，使用0作为默认值
        parse_time = locals().get('parse_time', 0)
        return False, [], parse_time


class OptimizedPDFPipeline:
    """优化版PDF处理流水线 - MinerU集成版"""
    
    def __init__(self, input_dir: str, output_dir: str, max_workers: int = 4,
                 backend: str = "vlm-sglang-client", server_url: str = None,
                 lang: str = "ch", api_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 batch_size: int = 100, concurrent_batches: int = 4,
                 shard_index: Optional[int] = None, shard_count: Optional[int] = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.backend = backend
        self.server_url = server_url
        self.lang = lang
        self.api_url = api_url
        self.batch_size = batch_size
        self.concurrent_batches = concurrent_batches
        self.shard_index = shard_index
        self.shard_count = shard_count

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
        if self.shard_index is not None and self.shard_count is not None:
            logger.info(f"Shard: {self.shard_index+1}/{self.shard_count}")

    def convert_epub_to_pdf(self) -> bool:
        """转换EPUB文件为PDF"""
        epub_files = find_files(self.input_dir, ['epub'])
        if not epub_files:
            logger.info("未找到EPUB文件，跳过转换步骤")
            return True
        
        logger.info(f"找到 {len(epub_files)} 个EPUB文件，开始转换...")
        
        try:
            # 使用现有的batch_epub_to_pdf函数
            from .batch_epub_to_pdf import batch_convert_epub_to_pdf
            
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
        pdf_files.extend(find_files(self.input_dir, ['pdf']))
        
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

        # 按分片过滤文件
        if self.shard_index is not None and self.shard_count is not None and self.shard_count > 1:
            files_sharded = [f for idx, f in enumerate(pdf_files) if idx % self.shard_count == self.shard_index]
            logger.info(f"分片过滤后: {len(files_sharded)}/{len(pdf_files)} 个文件将由当前分片处理")
            pdf_files = files_sharded

        return self._process_with_batch_mode(pdf_files)
    
    def _process_with_batch_mode(self, pdf_files: List[Path]) -> Tuple[bool, List[Dict]]:
        """使用分批次异步处理模式处理PDF文件"""
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
            # 使用多进程处理多个批次
            success, data = self._process_batches_multiprocess(batches)
            return success, data
                
        except Exception as e:
            logger.error(f"批量处理失败: {e}")
            logger.debug(traceback.format_exc())
            return False, []
    

    
    def _process_batches_multiprocess(self, batches: List[List[Path]]) -> Tuple[bool, List[Dict]]:
        """使用多进程处理多个批次"""
        all_processed_data = []
        total_success = 0
        
        # 准备配置数据
        config = {
            'backend': self.backend,
            'server_url': self.server_url,
            'lang': self.lang,
            'api_url': self.api_url,
            'results_dir': str(self.results_dir),
            'temp_dir': str(self.temp_dir),
            'demo_path': str(Path(__file__).parent.parent / "demo" / "demo.py"),
            'output_dir': str(self.output_dir),
            'log_level': "DEBUG",  # 使用DEBUG级别以获取更详细的日志
            'dashscope_api_key': os.getenv('DASHSCOPE_API_KEY')  # 显式传递API密钥
        }
        
        # 准备批次数据
        batch_data_list = [(i, batch, config) for i, batch in enumerate(batches)]
        
        logger.info(f"使用多进程处理 {len(batches)} 个批次，最大进程数: {self.concurrent_batches}")
        
        # 使用进程池处理批次
        with ProcessPoolExecutor(max_workers=self.concurrent_batches) as executor:
            # 提交所有批次任务
            future_to_batch = {}
            
            for i, batch_data in enumerate(batch_data_list):
                future = executor.submit(process_batch_worker, batch_data)
                future_to_batch[future] = (i, batch_data[1])
            
            # 收集结果
            with tqdm(total=len(batches), desc="处理批次", unit="批次") as pbar:
                for future in as_completed(future_to_batch):
                    batch_idx, batch_files = future_to_batch[future]
                    
                    try:
                        success, data, parse_time = future.result()  # 接收真正的批次处理时间
                        if success:
                            all_processed_data.extend(data)
                            total_success += len(data)
                            # 更新处理状态
                            for item in data:
                                pdf_file = item["pdf_file"]
                                self.status.mark_processed(
                                    str(pdf_file), 
                                    "pdf_processing", 
                                    parse_time,  # 使用真正的批次处理时间
                                    success=True
                                )
                        else:
                            # 标记批次中的文件为失败
                            for pdf_file in batch_files:
                                self.status.mark_processed(
                                    str(pdf_file), 
                                    "pdf_processing", 
                                    parse_time,  # 使用真正的批次处理时间
                                    success=False, 
                                    error_msg="批次处理失败"
                                )
                        pbar.set_postfix({"成功": total_success})
                    except Exception as e:
                        logger.error(f"批次 {batch_idx + 1} 处理异常: {e}")
                        # 标记批次中的文件为失败，使用默认时间0
                        for pdf_file in batch_files:
                            self.status.mark_processed(
                                str(pdf_file), 
                                "pdf_processing", 
                                0,  # 异常情况下使用默认时间
                                success=False, 
                                error_msg=f"处理异常: {e}"
                            )
                    finally:
                        pbar.update(1)
        
        logger.success(f"所有批次处理完成，总共成功处理 {total_success} 个文件")
        return total_success > 0, all_processed_data

    
    def extract_metadata_from_content(self, content: str) -> Dict:
        """从内容中提取元数据（简化版）"""
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent))
            from utils import extract_metadata_with_llm
            return extract_metadata_with_llm(content, self.api_url)
        except Exception as e:
            logger.warning(f"元数据提取失败: {e}")
            return {}
    
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
                                
                                # 如果没有大模型元数据，则使用基本元数据
                                if not metadata:
                                    logger.debug(f"使用基本元数据: {pdf_name}")
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
                                logger.warning(f"读取文件失败 {pdf_name}: {e}")
                                continue
            
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
        if self.shard_index is not None and self.shard_count is not None:
            logger.info(f"当前分片: {self.shard_index+1}/{self.shard_count}")
        
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
            logger.info(f"总耗时: {format_time(total_time)}")
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