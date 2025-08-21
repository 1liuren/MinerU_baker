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
        # 子进程不再单独配置日志，复用主进程配置
        
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
            return False, [], 0
        
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
                    try:
                        content = HTMLToMarkdownConverter.convert_html_in_text(content)
                    except Exception:
                        pass
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
            logger.info(f"进程 {os.getpid()} batch {batch_idx + 1}: 开始提取 {len(file_contents)} 个文件的元数据")
            
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
                    if not metadata:
                        return file_info, {}, '元数据提取失败'
                    metadata_file = file_info["final_dir"] / f"{file_info['pdf_file'].stem}_extracted_metadata.json"
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                    
                    # logger.info(f"进程 {os.getpid()}: batch {batch_idx + 1} 元数据提取成功 {file_info['pdf_file'].name}")
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
                    
                    # 仅返回必要的最小信息以减少跨进程传输负载
                    processed_data.append({
                        "pdf_file": file_info["pdf_file"]
                    })
            
            logger.success(f"进程 {os.getpid()}: 元数据提取完成，处理了 {len(file_contents)} 个文件")
        
        # 清理临时目录与局部变量，尽量释放内存
        try:
            shutil.rmtree(temp_output_dir)
        except Exception as e:
            logger.warning(f"进程 {os.getpid()}: 清理临时目录失败: {e}")
        
        # 主动触发垃圾回收
        try:
            import gc
            del file_name_list, pdf_bytes_list, lang_list, valid_files
            del file_contents
            gc.collect()
        except Exception:
            pass
        
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
                 data_json_path: Optional[str] = None, batches_per_round: Optional[int] = None,
                 lb_strategy: str = "round_robin"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.backend = backend
        # 多URL解析
        if server_url and "," in server_url:
            urls = [u.strip() for u in server_url.split(",") if u.strip()]
        else:
            urls = [server_url] if server_url else []
        self.server_urls: list[str] = urls
        self._rr_idx = 0
        self.lb_strategy = lb_strategy
        self.lang = lang
        self.api_url = api_url
        self.batch_size = batch_size
        self.concurrent_batches = concurrent_batches
        self.data_json_path = data_json_path
        self.batches_per_round = batches_per_round

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
        if self.server_urls and len(self.server_urls) > 1:
            logger.info(f"启用多服务器: {self.server_urls}，策略: {self.lb_strategy}")
        logger.info(f"Language: {self.lang}")
        logger.info(f"Max Workers: {self.max_workers}")
    def _load_ok_file_stems(self, filter_json_path=None):
        """从一个或多个JSON文件加载允许处理的文件stem集合（ok_status == '合格'）。仅支持标准JSON列表。

        支持形式：
        - None: 使用默认 data/data_info.json
        - 字符串: 单个路径，或逗号分隔的多个路径
        - 列表/元组: 多个路径
        """
        try:
            from pathlib import Path
            import json

            # 归一为路径列表
            if filter_json_path is None:
                paths = [Path(__file__).parent / "data" / "data_info.json"]
            elif isinstance(filter_json_path, (list, tuple)):
                paths = [Path(p) for p in filter_json_path]
            elif isinstance(filter_json_path, str) and "," in filter_json_path:
                paths = [Path(p.strip()) for p in filter_json_path.split(",") if p.strip()]
            else:
                paths = [Path(str(filter_json_path))]

            # 若传入为目录，则递归查找目录下所有的data_info.json
            normalized_paths: list[Path] = []
            for p in paths:
                try:
                    if p.is_dir():
                        # 递归查找所有data_info.json文件
                        for json_file in p.rglob("data_info.json"):
                            normalized_paths.append(json_file)
                    else:
                        normalized_paths.append(p)
                except Exception:
                    normalized_paths.append(p)
            paths = normalized_paths

            ok_stems: set[str] = set()
            any_found = False

            for p in paths:
                if not p.exists():
                    logger.info(f"未找到过滤文件: {p}")
                    continue
                any_found = True
                try:
                    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                except Exception as e:
                    logger.warning(f"过滤文件解析失败(非标准JSON列表): {p}, err={e}")
                    continue
                if not isinstance(data, list):
                    logger.warning(f"过滤文件格式异常(非list): {p}")
                    continue
                for item in data:
                    try:
                        file_path = str(item.get("file_path", "")).strip()
                        ok_status = str(item.get("ok_status", "")).strip()
                        if file_path and ok_status == "合格":
                            ok_stems.add(Path(file_path).stem)
                        # error_list = item.get("error_list", [])
                        
                        # # 只处理仅有"分类错误"一个错误的文件
                        # if file_path and ok_status == "不合格" and len(error_list) == 1:
                        #     error_item = error_list[0]
                        #     if isinstance(error_item, dict) and error_item.get("1") == "分类错误":
                        #         ok_stems.add(Path(file_path).stem)
                    except Exception:
                        continue

            if not any_found:
                logger.info("未找到任何有效的过滤文件，跳过过滤")
                return None
            logger.info(f"过滤表已加载(合并多文件)，可处理文件数: {len(ok_stems)}")
            return ok_stems if ok_stems else None
        except Exception as e:
            logger.warning(f"加载过滤文件失败: {e}")
            return None

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

        # 按文件名去重（仅保留首个出现的路径）
        seen_stems = set()
        unique_pdf_files = []
        for p in pdf_files:
            stem = p.stem
            if stem not in seen_stems:
                seen_stems.add(stem)
                unique_pdf_files.append(p)
        pdf_files = unique_pdf_files

        # 基于数据筛选文件（ok_status == "合格"）
        ok_stems = self._load_ok_file_stems(self.data_json_path)
        if ok_stems:
            before_cnt = len(pdf_files)
            pdf_files = [p for p in pdf_files if p.stem in ok_stems]
            logger.info(f"按过滤表筛选: {before_cnt} -> {len(pdf_files)}")
        
        if not pdf_files:
            logger.error("未找到PDF文件")
            return False, []
        
        logger.info(f"找到 {len(pdf_files)} 个PDF文件，使用批量处理模式...")
        logger.info("服务端自动处理多进程，无需客户端多线程")
        
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
                # 检查状态文件中的处理记录
                if self.status.is_processed(str(pdf_file), "pdf_processing"):
                    logger.debug(f"跳过已处理文件(状态记录): {pdf_file.name}")
                    self.status.status_data["processing_stats"]["skipped"] += 1
                    continue
                
                # 检查results目录下是否已存在对应的输出目录
                output_dir = self.results_dir / pdf_file.stem
                if output_dir.exists() and output_dir.is_dir():
                    # 进一步检查必要的输出文件是否存在
                    md_file = output_dir / f"{pdf_file.stem}.md"
                    middle_json = output_dir / f"{pdf_file.stem}_middle.json"
                    extracted_metadata_file = output_dir / f"{pdf_file.stem}_extracted_metadata.json"
                    if md_file.exists() and middle_json.exists() and extracted_metadata_file.exists():
                        logger.debug(f"跳过已处理文件(输出存在): {pdf_file.name}")
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
        total_success = 0
        
        # 准备配置数据
        config = {
            'backend': self.backend,
            'server_url': None,
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
        # 为每个批次分配server_url
        import random
        def pick_url() -> Optional[str]:
            if not self.server_urls:
                return self.server_url
            if self.lb_strategy == "random":
                return random.choice(self.server_urls)
            # 默认round robin
            url = self.server_urls[self._rr_idx % len(self.server_urls)]
            self._rr_idx += 1
            return url

        batch_data_list = []
        for i, batch in enumerate(batches):
            cfg = dict(config)
            cfg['server_url'] = pick_url()
            batch_data_list.append((i, batch, cfg))
        
        # 每轮处理的批次数量（每处理这么多批次后重建进程池）
        batches_per_round = self.batches_per_round or max(self.concurrent_batches * 10, 50)
        total_batches = len(batches)
        
        logger.info(f"使用多进程分轮次处理 {total_batches} 个批次，最大进程数: {self.concurrent_batches}")
        logger.info(f"每轮处理 {batches_per_round} 个批次后重建进程池")
        
        with tqdm(total=total_batches, desc="处理批次", unit="批次") as pbar:
            # 分轮次处理
            for round_start in range(0, total_batches, batches_per_round):
                round_end = min(round_start + batches_per_round, total_batches)
                current_round_data = batch_data_list[round_start:round_end]
                round_num = (round_start // batches_per_round) + 1
                
                logger.info(f"开始第 {round_num} 轮处理，批次范围: {round_start+1}-{round_end}")
                
                # 为这一轮创建新的进程池
                with ProcessPoolExecutor(max_workers=self.concurrent_batches) as executor:
                    future_to_batch = {}
                    # 提交这一轮的所有批次
                    for batch_data in current_round_data:
                        batch_idx = batch_data[0]
                        future = executor.submit(process_batch_worker, batch_data)
                        future_to_batch[future] = (batch_idx, batch_data[1])

                    # 收集这一轮的结果
                    for future in as_completed(future_to_batch):
                        batch_idx, batch_files = future_to_batch[future]
                        try:
                            success, data, parse_time = future.result()
                            if success:
                                total_success += len(data)
                                for item in data:
                                    pdf_file = item["pdf_file"]
                                    self.status.mark_processed(
                                        str(pdf_file),
                                        "pdf_processing",
                                        parse_time,
                                        success=True,
                                    )
                            else:
                                for pdf_file in batch_files:
                                    self.status.mark_processed(
                                        str(pdf_file),
                                        "pdf_processing",
                                        parse_time,
                                        success=False,
                                        error_msg="批次处理失败",
                                    )
                            pbar.set_postfix({"成功": total_success})
                        except Exception as e:
                            logger.error(f"批次 {batch_idx + 1} 处理异常: {e}")
                            for pdf_file in batch_files:
                                self.status.mark_processed(
                                    str(pdf_file),
                                    "pdf_processing",
                                    0,
                                    success=False,
                                    error_msg=f"处理异常: {e}",
                                )
                        finally:
                            pbar.update(1)
                
                logger.info(f"第 {round_num} 轮处理完成，进程池已重建")
        
        logger.success(f"所有批次处理完成，总共成功处理 {total_success} 个文件")
        return total_success > 0, []

    
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
                                            metadata.setdefault("book_name", pdf_name)
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
                        "text": item["metadata"].get("text", ""),
                        "meta": {
                            "data_info": {
                                "lang": item["metadata"].get("lang", "zh"),
                                "source": item["metadata"].get("publisher", ""),
                                "type": item["metadata"].get("type", "书籍"),
                                "book_name": item["metadata"].get("title", ""),
                                "book_content": item["content"],
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
            if not self.create_final_jsonl([]):
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
