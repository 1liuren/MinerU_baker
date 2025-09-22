#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF统计脚本（参数化、多文件汇总）
用于统计一个或多个 JSON 列表文件中的 PDF 总数、合格数、语言分布与不合格原因。
仅支持标准 JSON 列表格式（不解析 JSONL）。
"""

import json
import os
from collections import Counter
import argparse
from pathlib import Path

def analyze_pdf_data(items):
    """对传入的条目列表做统计。items 必须是 list[dict]。"""
    if not isinstance(items, list):
        print("错误：输入数据必须是 JSON 列表")
        return None
    
    # 基本统计
    total_pdfs = len(items)
    qualified_pdfs = sum(1 for item in items if item.get('ok_status') == '合格')
    unqualified_pdfs = total_pdfs - qualified_pdfs
    
    # 计算合格率
    qualification_rate = (qualified_pdfs / total_pdfs * 100) if total_pdfs > 0 else 0
    
    # 统计错误类型
    error_counter = Counter()
    for item in items:
        if item.get('ok_status') == '不合格':
            error_dict = item.get('error_dict', {})
            for error_code, error_msg in error_dict.items():
                error_counter[error_msg] += 1
    
    # 统计页数分布
    page_ranges = {
        "1-100": 0,
        "101-200": 0,
        "201-300": 0,
        "301-400": 0,
        "401-500": 0,
    }
    large_page_ranges = {}  # 存储500页以上的分布
    
    for item in items:
        page_total = item.get('page_total', 0)
        if page_total <= 0:
            continue
        
        if page_total <= 500:
            range_key = f"{((page_total-1)//100)*100+1}-{((page_total-1)//100+1)*100}"
            page_ranges[range_key] = page_ranges.get(range_key, 0) + 1
        else:
            range_key = f"{((page_total-1)//100)*100+1}-{((page_total-1)//100+1)*100}"
            large_page_ranges[range_key] = large_page_ranges.get(range_key, 0) + 1
    
    # 统计语言分布与按语言合格数量
    language_counter = Counter()
    qualified_language_counter = Counter()
    for item in items:
        file_path = item.get('file_path', '')
        if '/en/' in file_path:
            lang = '英文'
        elif '/zh/' in file_path:
            lang = '中文'
        else:
            lang = '未知'

        language_counter[lang] += 1
        if item.get('ok_status') == '合格':
            qualified_language_counter[lang] += 1
    
    return {
        'total_pdfs': total_pdfs,
        'qualified_pdfs': qualified_pdfs,
        'unqualified_pdfs': unqualified_pdfs,
        'qualification_rate': qualification_rate,
        'error_types': dict(error_counter),
        'language_distribution': dict(language_counter),
        'qualified_by_language': dict(qualified_language_counter),
        'page_distribution': {
            'normal': dict(page_ranges),
            'large': dict(sorted(large_page_ranges.items()))  # 按页数范围排序
        }
    }

def print_statistics(stats):
    """
    打印统计信息
    
    Args:
        stats: 统计信息字典
    """
    if not stats:
        return
    
    print("="*60)
    print("PDF数据统计报告")
    print("="*60)
    
    print(f"\n📊 基本统计:")
    print(f"  总PDF数量: {stats['total_pdfs']:,}")
    print(f"  合格数量:   {stats['qualified_pdfs']:,}")
    print(f"  不合格数量: {stats['unqualified_pdfs']:,}")
    print(f"  合格率:     {stats['qualification_rate']:.2f}%")
    
    print(f"\n📚 页数分布:")
    print("  500页以下:")
    for range_key, count in stats['page_distribution']['normal'].items():
        if count > 0:  # 只显示有数据的范围
            percentage = (count / stats['total_pdfs'] * 100) if stats['total_pdfs'] > 0 else 0
            print(f"    {range_key}页: {count:,} ({percentage:.1f}%)")
    
    if stats['page_distribution']['large']:
        print("\n  500页以上:")
        for range_key, count in stats['page_distribution']['large'].items():
            percentage = (count / stats['total_pdfs'] * 100) if stats['total_pdfs'] > 0 else 0
            print(f"    {range_key}页: {count:,} ({percentage:.1f}%)")

    print(f"\n🌍 语言分布:")
    for lang, count in stats['language_distribution'].items():
        percentage = (count / stats['total_pdfs'] * 100) if stats['total_pdfs'] > 0 else 0
        print(f"  {lang}: {count:,} ({percentage:.1f}%)")
    
    # 按语言合格数量
    if 'qualified_by_language' in stats:
        print(f"\n✅ 按语言合格数量:")
        print(f"  中文合格: {stats['qualified_by_language'].get('中文', 0):,}")
        print(f"  英文合格: {stats['qualified_by_language'].get('英文', 0):,}")
    
    if stats['error_types']:
        print(f"\n❌ 不合格原因统计:")
        sorted_errors = sorted(stats['error_types'].items(), key=lambda x: x[1], reverse=True)
        for error_msg, count in sorted_errors:
            percentage = (count / stats['unqualified_pdfs'] * 100) if stats['unqualified_pdfs'] > 0 else 0
            print(f"  {error_msg}: {count:,} ({percentage:.1f}%)")
    
    print("="*60)

def read_json_list(path: Path):
    """读取标准 JSON 列表文件，失败返回 None。"""
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(data, list):
            return data
        print(f"警告：{path} 不是JSON列表，已跳过")
        return None
    except Exception as e:
        print(f"警告：读取 {path} 失败 - {e}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="统计一个或多个 data_info.json 文件（标准 JSON 列表）",
    )
    parser.add_argument(
        "-i", "--input", nargs="+", required=False, default=["data/data_info.json"],
        help="输入JSON路径，支持多个（空格分隔）或目录（将自动寻找其中的*.json）",
    )
    parser.add_argument(
        "-o", "--output", default="pdf_statistics_report.json",
        help="统计结果输出路径（JSON）",
    )
    parser.add_argument(
        "--glob", action="store_true",
        help="当输入为目录时，递归匹配该目录下的 *.json 文件",
    )
    return parser.parse_args()


def expand_inputs(inputs, use_glob: bool):
    """将输入的文件/目录展开为实际 JSON 文件列表。"""
    files: list[Path] = []
    for s in inputs:
        p = Path(s)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            if use_glob:
                files.extend(p.rglob("*.json"))
            else:
                # 非 glob 模式下取目录中 data_info.json（若存在）
                candidate = p / "data_info.json"
                if candidate.exists():
                    files.append(candidate)
        else:
            # 尝试按逗号分隔
            parts = [Path(x.strip()) for x in s.split(",") if x.strip()]
            for q in parts:
                if q.exists():
                    files.append(q)
    # 去重
    uniq = []
    seen = set()
    for f in files:
        sp = str(f.resolve())
        if sp not in seen:
            uniq.append(f)
            seen.add(sp)
    return uniq


def main():
    args = parse_args()

    files = expand_inputs(args.input, args.glob)
    if not files:
        print("错误：未找到任何有效的JSON文件")
        return

    print(f"将统计以下 {len(files)} 个文件：")
    for f in files:
        print(f"  - {f}")

    # 汇总数据
    merged: list[dict] = []
    for f in files:
        data = read_json_list(f)
        if data:
            merged.extend(data)

    if not merged:
        print("错误：没有可统计的数据（文件内容为空或格式不正确）")
        return

    stats = analyze_pdf_data(merged)
    if not stats:
        print("分析失败，请检查文件格式")
        return

    print_statistics(stats)

    # 保存统计结果
    try:
        Path(args.output).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"\n📄 详细统计报告已保存到: {args.output}")
    except Exception as e:
        print(f"警告：保存统计报告失败 - {e}")

if __name__ == "__main__":
    main()