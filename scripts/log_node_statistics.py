#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
节点调用统计脚本
统计日志中各个节点的分配数据量
"""

import sys
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import argparse

def parse_timestamp(timestamp_str):
    """
    解析时间戳字符串，返回datetime对象
    支持格式: YYYY-MM-DD HH:mm:ss
    """
    try:
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except ValueError as e:
        print(f"时间戳解析失败: {timestamp_str}, 错误: {e}")
        return None

def extract_node_from_log_line(line):
    """
    从日志行中提取节点URL
    查找模式: "按权重随机选择节点: http://10.10.50.55:30000"
    """
    pattern = r"按权重随机选择节点:\s*(http://\d+\.\d+\.\d+\.\d+:\d+)"
    match = re.search(pattern, line)
    if match:
        return match.group(1)
    return None

def extract_node_identifier(node_url):
    """
    从节点URL中提取节点标识符
    例如: http://10.10.50.55:30000 -> 55节点
    """
    if not node_url:
        return None

    pattern = r"http://\d+\.\d+\.\d+\.(\d+):\d+"
    match = re.search(pattern, node_url)
    if match:
        return f"{match.group(1)}节点"
    return node_url  # 如果无法提取，返回原URL

def analyze_log_file(log_file_path, hours=1):
    """
    分析日志文件，统计指定时间范围内的节点调用情况

    Args:
        log_file_path: 日志文件路径
        hours: 统计最近多少小时的数据，默认1小时

    Returns:
        dict: 节点统计结果
    """
    if not os.path.exists(log_file_path):
        print(f"错误: 日志文件不存在: {log_file_path}")
        return {}

    # 计算时间范围
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)

    print(f"统计时间范围: {start_time.strftime('%Y-%m-%d %H:%M:%S')} 到 {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"日志文件: {log_file_path}")

    # 节点统计
    node_stats = defaultdict(int)
    total_calls = 0
    processed_lines = 0

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                processed_lines += 1

                # 解析时间戳
                timestamp_match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line.strip())
                if not timestamp_match:
                    continue

                log_time = parse_timestamp(timestamp_match.group(1))
                if not log_time:
                    continue

                # 检查是否在时间范围内
                if log_time < start_time or log_time > end_time:
                    continue

                # 提取节点信息
                node_url = extract_node_from_log_line(line)
                if node_url:
                    node_id = extract_node_identifier(node_url)
                    node_stats[node_id] += 1
                    total_calls += 1

    except Exception as e:
        print(f"读取日志文件时出错: {e}")
        return {}

    return node_stats, total_calls, processed_lines

def print_statistics(node_stats, total_calls, processed_lines):
    """打印统计结果"""
    print("\n" + "="*50)
    print("节点调用统计结果")
    print("="*50)
    print(f"总处理行数: {processed_lines}")
    print(f"节点调用总数: {total_calls}")

    if not node_stats:
        print("未找到节点调用记录")
        return

    print("\n各节点调用统计:")
    print("-" * 30)

    # 按调用次数降序排序
    sorted_stats = sorted(node_stats.items(), key=lambda x: x[1], reverse=True)

    for node, count in sorted_stats:
        percentage = (count / total_calls * 100) if total_calls > 0 else 0
        print(f"{node}: {count} 次 ({percentage:.1f}%)")

def find_recent_log_files(log_dir=".", pattern="*.log"):
    """查找最近的日志文件"""
    log_files = []
    log_path = Path(log_dir)

    if log_path.is_file():
        return [str(log_path)]

    for log_file in log_path.glob(pattern):
        if log_file.is_file():
            log_files.append(str(log_file))

    # 按修改时间降序排序
    log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return log_files

def main():
    parser = argparse.ArgumentParser(description="统计日志中各个节点的调用情况")
    parser.add_argument("-f", "--file", help="指定日志文件路径")
    parser.add_argument("-d", "--dir", default=".", help="日志文件目录 (默认: 当前目录)")
    parser.add_argument("-p", "--pattern", default="*.log", help="日志文件匹配模式 (默认: *.log)")
    parser.add_argument("-t", "--hours", type=int, default=1, help="统计最近多少小时的数据 (默认: 1)")
    parser.add_argument("-a", "--all", action="store_true", help="统计所有找到的日志文件")

    args = parser.parse_args()

    if args.file:
        # 指定了具体文件
        log_files = [args.file]
    else:
        # 查找日志文件
        log_files = find_recent_log_files(args.dir, args.pattern)
        if not log_files:
            print(f"未找到日志文件在目录: {args.dir}")
            return

        if not args.all:
            # 只使用最新的日志文件
            log_files = log_files[:1]

    print(f"找到 {len(log_files)} 个日志文件")

    all_stats = defaultdict(int)
    total_all_calls = 0
    total_processed_lines = 0

    for log_file in log_files:
        print(f"\n处理日志文件: {log_file}")
        node_stats, total_calls, processed_lines = analyze_log_file(log_file, args.hours)

        if args.all and len(log_files) > 1:
            # 如果有多个文件，累加统计
            for node, count in node_stats.items():
                all_stats[node] += count
            total_all_calls += total_calls
            total_processed_lines += processed_lines

            print_statistics(node_stats, total_calls, processed_lines)
        else:
            # 单个文件或最后的文件
            all_stats = node_stats
            total_all_calls = total_calls
            total_processed_lines = processed_lines

    if args.all and len(log_files) > 1:
        print(f"\n{'='*50}")
        print("所有日志文件汇总统计")
        print_statistics(all_stats, total_all_calls, total_processed_lines)
    elif not args.all:
        print_statistics(all_stats, total_all_calls, total_processed_lines)

if __name__ == "__main__":
    main()
