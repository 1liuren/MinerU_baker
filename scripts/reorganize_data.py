#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据整理脚本 - 将hash命名的处理结果重组为分类目录结构
"""

import json
import shutil
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from loguru import logger
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def load_file_path_mapping(data_json_path: str) -> Dict[str, str]:
    """
    加载文件路径映射关系
    
    Args:
        data_json_path: JSON文件路径或包含JSON文件的目录
        
    Returns:
        字典映射: {文件stem: 完整file_path}
    """
    mapping = {}
    path = Path(data_json_path)
    
    # 收集所有需要处理的JSON文件
    json_files = []
    if path.is_file():
        json_files = [path]
    elif path.is_dir():
        # 递归查找所有JSON文件
        json_files = list(path.rglob("*.json"))
        if not json_files:
            # 如果没找到，尝试非递归查找
            json_files = list(path.glob("*.json"))
    else:
        logger.error(f"指定的路径不存在: {data_json_path}")
        return mapping
    
    if not json_files:
        logger.warning(f"未找到JSON文件: {data_json_path}")
        logger.info(f"请检查路径是否正确，或该目录下是否包含JSON文件")
        # 列出目录内容以帮助调试
        if path.is_dir():
            try:
                files = list(path.iterdir())[:10]  # 只显示前10个
                logger.info(f"目录 {path} 包含文件: {[f.name for f in files]}")
            except Exception:
                pass
        return mapping
    
    logger.info(f"找到 {len(json_files)} 个JSON文件,开始加载映射关系...")
    
    total_records = 0
    qualified_records = 0
    
    for json_file in tqdm(json_files, desc="加载JSON文件"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                logger.warning(f"JSON文件格式异常(非list): {json_file}")
                continue
            
            for item in data:
                total_records += 1
                try:
                    file_path = str(item.get("file_path", "")).strip()
                    ok_status = str(item.get("ok_status", "")).strip()
                    
                    if file_path and ok_status == "合格":
                        stem = Path(file_path).stem
                        mapping[stem] = file_path
                        qualified_records += 1
                except Exception as e:
                    logger.debug(f"解析记录失败: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"读取JSON文件失败 {json_file}: {e}")
            continue
    
    logger.info(f"映射关系加载完成: 总记录数={total_records}, 合格记录数={qualified_records}, 唯一文件数={len(mapping)}")
    return mapping


def extract_category_path(file_path: str, levels: int = 4) -> Optional[str]:
    """
    从文件路径中提取最后N级路径组成部分
    
    Args:
        file_path: 完整文件路径
        levels: 要提取的层级数(默认4: 大类/小类/语言/文件名)
        
    Returns:
        提取的路径字符串,如 "工业技术相关书籍数据集成/原子能技术/en/6739a0359335ce188ed06631028d7e3a"
    """
    try:
        path = Path(file_path)
        # 获取路径部分(不含扩展名)
        parts = list(path.parts)
        
        # 提取最后levels个部分,最后一个去掉扩展名
        if len(parts) < levels:
            logger.warning(f"路径层级不足 {levels} 级: {file_path}")
            # 如果层级不足,使用所有可用的部分
            extracted_parts = parts[:-1] + [path.stem]
        else:
            extracted_parts = parts[-(levels):][:-1] + [path.stem]
        
        return "/".join(extracted_parts)
    except Exception as e:
        logger.error(f"提取路径失败 {file_path}: {e}")
        return None


def copy_directory_contents(src_dir: Path, dest_dir: Path, file_stem: str) -> bool:
    """
    复制处理结果目录的内容
    
    Args:
        src_dir: 源目录(hash命名的结果目录)
        dest_dir: 目标目录
        file_stem: 文件stem(用于匹配文件名)
        
    Returns:
        是否成功复制
    """
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制markdown文件
        md_file = src_dir / f"{file_stem}.md"
        if md_file.exists():
            shutil.copy2(md_file, dest_dir / md_file.name)
        else:
            logger.warning(f"未找到markdown文件: {md_file}")
        
        # 复制middle.json
        middle_json = src_dir / f"{file_stem}_middle.json"
        if middle_json.exists():
            shutil.copy2(middle_json, dest_dir / middle_json.name)
        
        # 复制extracted_metadata.json
        metadata_json = src_dir / f"{file_stem}_extracted_metadata.json"
        if metadata_json.exists():
            shutil.copy2(metadata_json, dest_dir / metadata_json.name)
        
        # 复制images目录
        images_dir = src_dir / "images"
        if images_dir.exists() and images_dir.is_dir():
            dest_images = dest_dir / "images"
            if dest_images.exists():
                shutil.rmtree(dest_images)
            shutil.copytree(images_dir, dest_images)
        
        return True
    except Exception as e:
        logger.error(f"复制目录内容失败 {src_dir} -> {dest_dir}: {e}")
        return False


def find_original_pdf(input_dir: Path, file_path: str) -> Optional[Path]:
    """
    在输入目录中查找原始PDF文件
    
    Args:
        input_dir: 输入目录
        file_path: 原始文件路径
        
    Returns:
        找到的PDF文件路径,未找到返回None
    """
    try:
        # 方法1: 直接使用file_path(如果它是绝对路径且存在)
        original_path = Path(file_path)
        if original_path.exists() and original_path.is_file():
            return original_path
        
        # 方法2: 在输入目录中递归查找同名文件
        file_name = original_path.name
        stem = original_path.stem
        
        # 先尝试快速查找: 检查是否存在相同的相对路径结构
        possible_paths = list(input_dir.rglob(file_name))
        if possible_paths:
            # 如果找到多个,优先返回路径最短的(通常是最相关的)
            return min(possible_paths, key=lambda p: len(str(p)))
        
        # 尝试查找同stem的PDF文件
        possible_paths = list(input_dir.rglob(f"{stem}.pdf"))
        if possible_paths:
            return possible_paths[0]
        
        logger.warning(f"未找到原始PDF文件: {file_path}")
        return None
        
    except Exception as e:
        logger.error(f"查找原始PDF失败 {file_path}: {e}")
        return None


def process_single_directory(
    result_dir: Path,
    mapping: Dict[str, str],
    input_path: Path,
    organized_path: Path,
    levels: int,
    stats_lock: threading.Lock
) -> Tuple[str, Dict[str, int]]:
    """
    处理单个结果目录（线程工作函数）
    
    Returns:
        (状态, 统计信息字典)
    """
    local_stats = {
        "matched": 0,
        "unmatched": 0,
        "success": 0,
        "failed": 0,
        "pdf_found": 0,
        "pdf_not_found": 0,
        "skipped": 0
    }
    
    dir_name = result_dir.name
    
    try:
        # 在映射中查找对应的file_path
        if dir_name not in mapping:
            local_stats["unmatched"] += 1
            return "unmatched", local_stats
        
        local_stats["matched"] += 1
        file_path = mapping[dir_name]
        
        # 提取分类路径
        category_path = extract_category_path(file_path, levels)
        if not category_path:
            local_stats["failed"] += 1
            return "failed", local_stats
        
        # 创建目标目录
        target_dir = organized_path / category_path
        
        # 检查是否已处理(检查关键文件是否存在)
        md_file_target = target_dir / f"{dir_name}.md"
        metadata_file_target = target_dir / f"{dir_name}_extracted_metadata.json"
        if md_file_target.exists() and metadata_file_target.exists():
            local_stats["skipped"] += 1
            return "skipped", local_stats
        
        # 复制处理结果
        if copy_directory_contents(result_dir, target_dir, dir_name):
            local_stats["success"] += 1
            
            # 查找并复制原始PDF
            original_pdf = find_original_pdf(input_path, file_path)
            if original_pdf:
                try:
                    shutil.copy2(original_pdf, target_dir / original_pdf.name)
                    local_stats["pdf_found"] += 1
                except Exception as e:
                    logger.debug(f"复制PDF失败 {original_pdf}: {e}")
                    local_stats["pdf_not_found"] += 1
            else:
                local_stats["pdf_not_found"] += 1
            return "success", local_stats
        else:
            local_stats["failed"] += 1
            return "failed", local_stats
            
    except Exception as e:
        logger.error(f"处理目录失败 {dir_name}: {e}")
        local_stats["failed"] += 1
        return "error", local_stats


def reorganize_results(
    input_dir: str,
    output_dir: str,
    organized_output_dir: str,
    data_json_path: str,
    levels: int = 4,
    max_workers: int = 8
) -> bool:
    """
    重组处理结果目录结构
    
    Args:
        input_dir: 原始PDF输入目录
        output_dir: MinerU处理结果目录(hash命名)
        organized_output_dir: 整理后的输出目录
        data_json_path: 数据JSON路径
        levels: 提取路径层级数
        
    Returns:
        是否成功完成
    """
    # 转换为Path对象
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    organized_path = Path(organized_output_dir)
    
    # 验证输入
    if not input_path.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        return False
    
    if not output_path.exists():
        logger.error(f"处理结果目录不存在: {output_dir}")
        return False
    
    # 创建整理后的输出目录
    organized_path.mkdir(parents=True, exist_ok=True)
    
    # 加载文件路径映射
    logger.info("步骤 1/3: 加载文件路径映射...")
    mapping = load_file_path_mapping(data_json_path)
    if not mapping:
        logger.error("未能加载任何文件路径映射,无法继续")
        return False
    
    # 查找results子目录(如果存在)
    results_dir = output_path / "results"
    if results_dir.exists() and results_dir.is_dir():
        logger.info(f"检测到results子目录,使用: {results_dir}")
        scan_dir = results_dir
    else:
        logger.info(f"未检测到results子目录,直接扫描: {output_path}")
        scan_dir = output_path
    
    # 扫描处理结果目录
    logger.info("步骤 2/3: 扫描处理结果目录...")
    result_dirs = [d for d in scan_dir.iterdir() if d.is_dir()]
    logger.info(f"找到 {len(result_dirs)} 个结果目录")
    
    # 统计信息
    stats = {
        "total": len(result_dirs),
        "matched": 0,
        "unmatched": 0,
        "success": 0,
        "failed": 0,
        "pdf_found": 0,
        "pdf_not_found": 0,
        "skipped": 0
    }
    
    stats_lock = threading.Lock()
    
    # 使用多线程处理
    logger.info(f"步骤 3/3: 重组目录结构 (使用 {max_workers} 个线程)...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_dir = {
            executor.submit(
                process_single_directory,
                result_dir,
                mapping,
                input_path,
                organized_path,
                levels,
                stats_lock
            ): result_dir
            for result_dir in result_dirs
        }
        
        # 使用tqdm显示进度
        with tqdm(total=len(result_dirs), desc="处理进度", unit="个") as pbar:
            for future in as_completed(future_to_dir):
                result_dir = future_to_dir[future]
                try:
                    status, local_stats = future.result()
                    
                    # 合并统计信息
                    with stats_lock:
                        for key, value in local_stats.items():
                            stats[key] += value
                    
                except Exception as e:
                    logger.error(f"处理失败 {result_dir.name}: {e}")
                    with stats_lock:
                        stats["failed"] += 1
                
                pbar.update(1)
    
    # 输出统计信息
    logger.info("=" * 60)
    logger.info("处理完成! 统计信息:")
    logger.info(f"  总目录数: {stats['total']}")
    logger.info(f"  匹配成功: {stats['matched']}")
    logger.info(f"  未匹配: {stats['unmatched']}")
    logger.info(f"  跳过已处理: {stats['skipped']}")
    logger.info(f"  复制成功: {stats['success']}")
    logger.info(f"  复制失败: {stats['failed']}")
    logger.info(f"  PDF找到: {stats['pdf_found']}")
    logger.info(f"  PDF未找到: {stats['pdf_not_found']}")
    logger.info(f"  输出目录: {organized_path}")
    logger.info("=" * 60)
    
    return stats["success"] > 0


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="数据整理工具 - 将hash命名的处理结果重组为分类目录结构",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例用法:
  python scripts/reorganize_data.py \\
    -i /data/liupan/coder-server/xyzhwb/oss-cn-huhehaote.aliyuncs.com/20250807 \\
    -o data_55/2025自有库_0807_vllm \\
    --organized-output data_55/2025自有库_0807_vllm_organized \\
    --data-json /data/liupan/coder-server/suanfa/20250807
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="原始PDF输入目录"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="MinerU处理结果目录(hash命名)"
    )
    parser.add_argument(
        "--organized-output",
        required=True,
        help="整理后的输出目录"
    )
    parser.add_argument(
        "--data-json", "-d",
        required=True,
        help="数据JSON路径(文件或目录)"
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=4,
        help="提取路径层级数(默认: 4)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="最大线程数(默认: 8)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别(默认: INFO)"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    try:
        args = parse_args()
        
        # 配置日志
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            level=args.log_level
        )
        
        logger.info("数据整理工具启动")
        logger.info(f"输入目录: {args.input}")
        logger.info(f"处理结果目录: {args.output}")
        logger.info(f"整理输出目录: {args.organized_output}")
        logger.info(f"数据JSON: {args.data_json}")
        logger.info(f"路径层级: {args.levels}")
        logger.info(f"最大线程数: {args.max_workers}")
        
        # 执行重组
        success = reorganize_results(
            input_dir=args.input,
            output_dir=args.output,
            organized_output_dir=args.organized_output,
            data_json_path=args.data_json,
            levels=args.levels,
            max_workers=args.max_workers
        )
        
        if success:
            logger.success("数据整理完成!")
            sys.exit(0)
        else:
            logger.error("数据整理失败")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("用户中断操作")
        sys.exit(130)
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

