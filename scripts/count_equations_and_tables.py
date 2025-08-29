#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计PDF结果中公式和表格数量的脚本
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm


def count_special_blocks_in_structure(data_structure, counts: Dict[str, int]):
    """
    递归统计数据结构中的特殊块数量

    Args:
        data_structure: 要搜索的数据结构（dict或list）
        counts: 计数器字典
    """
    if isinstance(data_structure, dict):
        # 检查当前字典是否有type字段
        if 'type' in data_structure:
            block_type = data_structure['type']
            if block_type in counts:
                counts[block_type] += 1
                counts['total'] += 1

        # 递归处理字典中的所有值
        for value in data_structure.values():
            count_special_blocks_in_structure(value, counts)

    elif isinstance(data_structure, list):
        # 递归处理列表中的所有项
        for item in data_structure:
            count_special_blocks_in_structure(item, counts)


def count_special_blocks(json_file_path: str) -> Dict[str, int]:
    """
    统计单个JSON文件中的特殊块数量

    Args:
        json_file_path: JSON文件路径

    Returns:
        包含各种块类型计数的字典
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        counts = {
            'inline_equation': 0,
            'interline_equation': 0,
            'table': 0,
            'total': 0
        }

        # 递归搜索整个数据结构
        count_special_blocks_in_structure(data, counts)

        # 调试输出
        # print(f"  统计结果: 内联公式={counts['inline_equation']}, 行间公式={counts['interline_equation']}, 表格={counts['table']}, 总计={counts['total']}")

        return counts

    except Exception as e:
        print(f"处理文件 {json_file_path} 时出错: {e}")
        return {'inline_equation': 0, 'interline_equation': 0, 'table': 0, 'total': 0}


def analyze_results_directory(results_dir: str, top_n: int = 10) -> List[Tuple[str, Dict[str, int]]]:
    """
    分析结果目录中的所有PDF

    Args:
        results_dir: 结果目录路径
        top_n: 返回前N个结果

    Returns:
        排序后的PDF统计结果列表
    """
    results_dir_path = Path(results_dir)
    pdf_stats = []

    # 获取所有子目录总数
    subdirs = [d for d in results_dir_path.iterdir() if d.is_dir()]
    
    # 使用tqdm显示进度
    for subdir in tqdm(subdirs, desc="正在处理PDF文件"):
        # 查找_middle.json文件
        middle_json = subdir / f"{subdir.name}_middle.json"
        if middle_json.exists():
            counts = count_special_blocks(str(middle_json))
            pdf_stats.append((subdir.name, counts))
        else:
            print(f"警告: 在 {subdir.name} 中未找到 _middle.json 文件")


    # 按总数量降序排序
    pdf_stats.sort(key=lambda x: x[1]['total'], reverse=True)

    return pdf_stats[:top_n]


def print_results(results: List[Tuple[str, Dict[str, int]]], results_dir: str):
    """
    打印统计结果

    Args:
        results: 统计结果列表
        results_dir: 结果目录路径
    """
    print(f"\n{'='*80}")
    print(f"分析结果目录: {results_dir}")
    print(f"{'='*80}")
    print(f"{'排名':<4} {'PDF名称':<30} {'内联公式':<8} {'行间公式':<8} {'表格':<6} {'总计':<6}")
    print(f"{'-'*80}")

    for i, (pdf_name, counts) in enumerate(results, 1):
        print(f"{i:<4} {pdf_name:<30} {counts['inline_equation']:<8} {counts['interline_equation']:<8} {counts['table']:<6} {counts['total']:<6}")

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='统计PDF结果中公式和表格的数量')
    parser.add_argument('results_dir', help='包含PDF结果的目录路径')
    parser.add_argument('-n', '--top_n', type=int, default=10,
                       help='显示前N个结果 (默认: 10)')
    parser.add_argument('-o', '--output', help='输出结果到指定文件')

    args = parser.parse_args()

    results_dir = args.results_dir
    top_n = args.top_n

    if not os.path.exists(results_dir):
        print(f"错误: 目录 {results_dir} 不存在")
        return

    if not os.path.isdir(results_dir):
        print(f"错误: {results_dir} 不是一个目录")
        return

    # 分析结果
    results = analyze_results_directory(results_dir, top_n)

    if not results:
        print(f"在目录 {results_dir} 中未找到任何有效的PDF结果")
        return

    # 打印结果
    print_results(results, results_dir)

    # 如果指定了输出文件，保存结果
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(f"分析结果目录: {results_dir}\n")
                f.write(f"显示前{top_n}个结果\n\n")
                f.write(f"{'排名':<4} {'PDF名称':<30} {'内联公式':<8} {'行间公式':<8} {'表格':<6} {'总计':<6}\n")
                f.write(f"{'-'*80}\n")

                for i, (pdf_name, counts) in enumerate(results, 1):
                    f.write(f"{i:<4} {pdf_name:<30} {counts['inline_equation']:<8} {counts['interline_equation']:<8} {counts['table']:<6} {counts['total']:<6}\n")

            print(f"\n结果已保存到: {args.output}")
        except Exception as e:
            print(f"保存结果时出错: {e}")

    # 显示统计信息
    total_pdfs = len(results)
    total_inline = sum(counts['inline_equation'] for _, counts in results)
    total_interline = sum(counts['interline_equation'] for _, counts in results)
    total_tables = sum(counts['table'] for _, counts in results)
    grand_total = sum(counts['total'] for _, counts in results)

    print(f"\n统计汇总:")
    print(f"  处理的PDF数量: {total_pdfs}")
    print(f"  内联公式总数: {total_inline}")
    print(f"  行间公式总数: {total_interline}")
    print(f"  表格总数: {total_tables}")
    print(f"  总计: {grand_total}")


if __name__ == "__main__":
    main()
