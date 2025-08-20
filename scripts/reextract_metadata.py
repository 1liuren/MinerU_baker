#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新提取并生成 extracted_metadata.json 脚本

功能：
- 输入一个目录，该目录本身是 pipeline 输出根目录，或直接是其中的 results 目录
- 脚本会进入 results 目录下的每个子目录，读取同名 .md 文件
- 使用 scripts.utils.extract_metadata_with_llm 重新提取元数据
- 生成/覆盖 <name>_extracted_metadata.json

要求：
- 需要设置环境变量 DASHSCOPE_API_KEY
"""

import os
import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse


def import_extractor():
    """导入 extract_metadata_with_llm（支持作为模块或脚本直接执行）。"""
    try:
        # 作为包内相对导入
        from .utils import extract_metadata_with_llm  # type: ignore
        return extract_metadata_with_llm
    except Exception:
        # 直接执行时，调整 sys.path 后再导入
        sys.path.insert(0, str(Path(__file__).parent))
        from utils import extract_metadata_with_llm  # type: ignore
        return extract_metadata_with_llm


def parse_args():
    parser = argparse.ArgumentParser(
        description="重新提取并生成 results 下的 *_extracted_metadata.json"
    )
    parser.add_argument(
        "-p", "--path",
        required=True,
        help="pipeline 输出根目录（包含 results 子目录）或直接指定 results 目录"
    )
    parser.add_argument(
        "--api-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="大模型API URL (默认: 阿里云DashScope)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=8,
        help="并行线程数 (默认: 8)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="已存在提取文件则跳过"
    )
    return parser.parse_args()


def resolve_results_dir(base_path: Path) -> Path:
    """根据用户传入路径解析 results 目录。"""
    if (base_path / "results").exists():
        return base_path / "results"
    return base_path


def reextract_all(results_dir: Path, api_url: str, workers: int, skip_existing: bool) -> None:
    from tqdm import tqdm
    extract_metadata_with_llm = import_extractorsafe()

    print("扫描任务...")
    # 收集任务
    tasks: list[tuple[Path, Path, Path]] = []
    skipped = 0
    for item_dir in sorted(results_dir.iterdir()):
        if not item_dir.is_dir():
            continue
        name = item_dir.name
        md_file = item_dir / f"{name}.md"
        out_file = item_dir / f"{name}_extracted_metadata.json"
        if not md_file.exists():
            skipped += 1
            continue
        if skip_existing and out_file.exists():
            skipped += 1
            continue
        tasks.append((item_dir, md_file, out_file))

    total = len(tasks)
    if total == 0:
        print(f"没有需要处理的任务（已跳过 {skipped} 个）")
        return

    print(f"共 {total} 个任务（已跳过 {skipped} 个），使用 {workers} 线程并行处理")

    def worker(md_path: Path, out_path: Path) -> tuple[bool, str]:
        try:
            content = md_path.read_text(encoding="utf-8")
        except Exception as e:
            return False, f"读取MD失败: {md_path} - {e}"
        try:
            metadata = extract_metadata_with_llm(content, api_url)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(metadata or {}, f, ensure_ascii=False, indent=2)
            return True, f"更新 {out_path}"
        except Exception as e:
            return False, f"提取失败: {md_path} - {e}"

    updated = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # 提交所有任务
        future_to_item = {
            executor.submit(worker, md_path, out_path): (item_dir, md_path, out_path)
            for (item_dir, md_path, out_path) in tasks
        }

        # 使用tqdm显示进度
        with tqdm(total=total, desc="提取元数据", unit="个") as pbar:
            for future in as_completed(future_to_item):
                ok, msg = future.result()
                if ok:
                    updated += 1
                else:
                    failed += 1
                # 更新进度条和统计
                pbar.set_postfix({
                    "成功": f"{updated}/{total}",
                    "失败": failed
                }, refresh=True)
                pbar.update(1)
                # 在进度条下方打印详细信息
                tqdm.write(("[OK] " if ok else "[ERR]") + msg)

    print("\n===== 完成 =====")
    print(f"总任务数: {total}")
    print(f"已更新:   {updated}")
    print(f"失败:     {failed}")


def import_extractorsafe():
    # 单独封装，避免顶层引用顺序问题
    return import_extractors()


def import_extractors():
    # 实际导入
    return import_extractor()


def main():
    args = parse_args()
    base_path = Path(args.path)
    if not base_path.exists():
        print(f"错误：路径不存在: {base_path}")
        sys.exit(1)

    # 检查API Key
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("警告：未设置 DASHSCOPE_API_KEY，提取可能失败")

    results_dir = resolve_results_dir(base_path)
    if not results_dir.exists():
        print(f"错误：未找到 results 目录: {results_dir}")
        sys.exit(1)

    print(f"开始重新提取元数据，目录: {results_dir}")
    reextract_all(results_dir, args.api_url, args.workers, args.skip_existing)


if __name__ == "__main__":
    main()


