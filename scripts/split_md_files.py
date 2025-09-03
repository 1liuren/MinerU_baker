#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按行数拆分MD文件的脚本

将指定目录（包括子目录）下的所有.md文件按5000行一份拆分，
拆分后的文件命名为原文件名-1、-2等，保存在原文件同级目录下。
"""

import os
import sys
from pathlib import Path
from typing import List
from loguru import logger
import argparse


def split_md_file(md_file: Path, lines_per_chunk: int = 5000) -> List[Path]:
    """
    按行数拆分单个MD文件

    Args:
        md_file: 要拆分的MD文件路径
        lines_per_chunk: 每个拆分块的行数

    Returns:
        生成的拆分文件列表
    """
    try:
        # 读取原始文件内容
        with open(md_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            logger.info(f"文件为空，跳过: {md_file}")
            return []

        total_lines = len(lines)
        logger.info(f"开始拆分 {md_file.name} ({total_lines} 行)")

        # 计算需要的拆分数
        num_chunks = (total_lines + lines_per_chunk - 1) // lines_per_chunk  # 向上取整

        generated_files = []
        stem = md_file.stem  # 不含扩展名的文件名

        for chunk_idx in range(num_chunks):
            start_line = chunk_idx * lines_per_chunk
            end_line = min((chunk_idx + 1) * lines_per_chunk, total_lines)

            # 生成新文件名
            if num_chunks == 1:
                # 如果只有一块，不需要添加序号
                new_filename = f"{stem}.md"
            else:
                new_filename = f"{stem}-{chunk_idx + 1}.md"

            new_file_path = md_file.parent / new_filename

            # 写入拆分内容
            chunk_lines = lines[start_line:end_line]
            with open(new_file_path, 'w', encoding='utf-8') as f:
                f.writelines(chunk_lines)

            generated_files.append(new_file_path)
            logger.info(f"生成文件: {new_filename} ({len(chunk_lines)} 行)")

        logger.success(f"拆分完成: {md_file.name} -> {len(generated_files)} 个文件")
        return generated_files

    except Exception as e:
        logger.error(f"拆分文件失败 {md_file}: {e}")
        return []


def find_md_files(root_dir: Path) -> List[Path]:
    """递归查找所有.md文件"""
    md_files = []
    for md_file in root_dir.rglob("*.md"):
        if md_file.is_file():
            md_files.append(md_file)
    return md_files


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="按行数拆分MD文件")
    parser.add_argument("directory", help="要处理的目录路径")
    parser.add_argument("--lines-per-chunk", type=int, default=5000,
                        help="每个拆分块的行数 (默认: 5000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅显示将要进行的操作，不实际拆分文件")

    args = parser.parse_args()
    root_dir = Path(args.directory)
    lines_per_chunk = args.lines_per_chunk if hasattr(args, 'lines_per_chunk') else 5000
    dry_run = getattr(args, 'dry_run', False)

    if not root_dir.exists():
        print(f"错误: 目录不存在 - {root_dir}")
        sys.exit(1)

    if not root_dir.is_dir():
        print(f"错误: 指定的路径不是目录 - {root_dir}")
        sys.exit(1)

    # 查找所有MD文件
    md_files = find_md_files(root_dir)

    if not md_files:
        print(f"在 {root_dir} 中未找到任何 .md 文件")
        return

    logger.info(f"在 {root_dir} 中找到 {len(md_files)} 个 .md 文件")

    if dry_run:
        logger.info("--- 试运行模式 ---")
        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
                num_chunks = (line_count + lines_per_chunk - 1) // lines_per_chunk
                if num_chunks > 1:
                    print(f"将拆分: {md_file.name} ({line_count} 行) -> {num_chunks} 个文件")
                else:
                    print(f"无需拆分: {md_file.name} ({line_count} 行)")
            except Exception as e:
                print(f"读取失败: {md_file.name} - {e}")
        return

    # 统计信息
    total_files_processed = 0
    total_chunks_generated = 0

    # 处理每个MD文件
    for md_file in md_files:
        logger.info(f"处理文件: {md_file}")
        generated_files = split_md_file(md_file, lines_per_chunk)
        total_files_processed += 1
        total_chunks_generated += len(generated_files)

    # 输出统计结果
    print("\n" + "="*60)
    print("拆分完成统计")
    print("="*60)
    print(f"处理的文件数: {total_files_processed}")
    print(f"生成的拆分文件数: {total_chunks_generated}")
    print(f"每块行数: {lines_per_chunk}")
    print(f"输入目录: {root_dir}")

    if total_files_processed > 0:
        print(".1f")


if __name__ == "__main__":
    main()
