import argparse
import json
import os
import random
import shutil
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "递归扫描输入目录内所有 extraction_results.json，统计 span_table / "
            "span_interline_equation（可选 span_text）数量，按比例抽样，复制图片到输出目录，"
            "并生成 table.json / equation.json（可选 text.json）。"
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="输入根目录，递归扫描包含 extraction_results.json 的所有子目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=(
            "输出根目录，内将创建 table_images / equation_images 目录，以及 table.json / equation.json"
        ),
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=3 / 10000.0,
        help="抽样比例，默认 3/10000",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，保证可复现，默认 42",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印更详细的调试信息",
    )
    parser.add_argument(
        "--table_to_md",
        action="store_true",
        help="是否将表格 text 从 HTML 转为 Markdown",
    )
    parser.add_argument(
        "--include_text",
        action="store_true",
        help="是否处理 span_text（普通文本）并输出 text.json 与 text_images",
    )
    return parser.parse_args()


def find_json_files(root: Path) -> List[Path]:
    results: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn == "extraction_results.json":
                results.append(Path(dirpath) / fn)
    return results


def is_target_type(item: Dict[str, Any], include_text: bool = False) -> bool:
    t = item.get("type")
    base = {"span_table", "span_interline_equation"}
    if include_text:
        base.add("span_text")
    return t in base


def collect_targets(json_path: Path, include_text: bool = False, verbose: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"[WARN] 读取失败: {json_path} -> {e}")
        return [], [], []

    targets = data.get("targets", [])
    if not isinstance(targets, list):
        if verbose:
            print(f"[WARN] 文件结构异常(缺少 targets 列表): {json_path}")
        return [], [], []

    tables: List[Dict[str, Any]] = []
    equations: List[Dict[str, Any]] = []
    texts: List[Dict[str, Any]] = []

    for item in targets:
        if not isinstance(item, Dict):
            continue
        if not is_target_type(item, include_text=include_text):
            continue
        if item.get("type") == "span_table":
            tables.append(item)
        elif item.get("type") == "span_interline_equation":
            equations.append(item)
        elif include_text and item.get("type") == "span_text":
            texts.append(item)

    return tables, equations, texts


def resolve_image_path(item: Dict[str, Any], json_path: Path, verbose: bool = False) -> Optional[Path]:
    raw_path = item.get("image_path")
    if not isinstance(raw_path, str) or not raw_path:
        return None

    # 优先按原始路径
    p = Path(raw_path)
    if p.exists():
        return p

    # 兼容 Linux 风格的绝对路径在 Windows 上无效的情况，取 basename 在邻近目录中搜索
    base = Path(os.path.basename(raw_path))

    # 常见相对位置：与 json 同目录的 images/ 或其子层级
    candidates: List[Path] = []
    parent = json_path.parent
    candidates.append(parent / base)
    candidates.append(parent / "images" / base)

    # 扩展：在 json 父目录内一层深度搜索 images 目录
    for name in ("images", "image", "imgs"):  # 常见命名
        candidates.append(parent / name / base)
        candidates.append(parent.parent / name / base)

    for cand in candidates:
        if cand.exists():
            if verbose:
                print(f"[INFO] 回退匹配图片: {raw_path} -> {cand}")
            return cand

    # 最后尝试在 json 同级目录树中浅层次查找同名文件（限制深度以避免过慢）
    try:
        for dirpath, _, filenames in os.walk(parent):
            if base.name in filenames:
                hit = Path(dirpath) / base.name
                if verbose:
                    print(f"[INFO] 遍历匹配图片: {raw_path} -> {hit}")
                return hit
    except Exception:
        pass

    if verbose:
        print(f"[WARN] 未找到图片: {raw_path} (来源: {json_path})")
    return None


def sample_items(items: List[Dict[str, Any]], ratio: float) -> List[Dict[str, Any]]:
    if not items or ratio <= 0:
        return []
    k = int(len(items) * ratio)
    if k <= 0 and len(items) > 0:
        k = 1
    if k >= len(items):
        return items
    return random.sample(items, k)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_and_rewrite(item: Dict[str, Any], src_img: Path, dst_dir: Path, prefix: str, index: int) -> Tuple[Dict[str, Any], Optional[Path]]:
    # 生成稳定文件名，避免重复：前缀 + 序号 + 原名
    dst_name = f"{prefix}_{index:06d}_{src_img.name}"
    dst_path = dst_dir / dst_name

    ensure_dir(dst_dir)
    shutil.copy2(src_img, dst_path)

    # 拷贝一个条目并重写 image_path（相对输出根）
    new_item = dict(item)
    # 使用输出目录相对路径，统一为 POSIX 风格
    new_item["image_path"] = f"{dst_dir.name}/{dst_name}"
    return new_item, dst_path


def main() -> None:
    args = parse_args()

    random.seed(args.seed)

    input_root = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()

    table_img_dir = output_root / "table_images"
    eq_img_dir = output_root / "equation_images"
    text_img_dir = output_root / "text_images"

    ensure_dir(table_img_dir)
    ensure_dir(eq_img_dir)
    if args.include_text:
        ensure_dir(text_img_dir)

    json_files = find_json_files(input_root)
    if args.verbose:
        print(f"[INFO] 发现 JSON 文件 {len(json_files)} 个")

    all_tables: List[Tuple[Path, Dict[str, Any]]] = []  # (json_path, item)
    all_equations: List[Tuple[Path, Dict[str, Any]]] = []
    all_texts: List[Tuple[Path, Dict[str, Any]]] = []

    for jp in json_files:
        tables, equations, texts = collect_targets(jp, include_text=args.include_text, verbose=args.verbose)
        all_tables.extend((jp, it) for it in tables)
        all_equations.extend((jp, it) for it in equations)
        if args.include_text:
            all_texts.extend((jp, it) for it in texts)

    total_tables = len(all_tables)
    total_equations = len(all_equations)
    total_texts = len(all_texts) if args.include_text else 0

    print(f"总表格(span_table): {total_tables}")
    print(f"总公式(span_interline_equation): {total_equations}")
    if args.include_text:
        print(f"总文本(span_text): {total_texts}")

    sampled_tables_items = sample_items([it for _, it in all_tables], args.ratio)
    sampled_equations_items = sample_items([it for _, it in all_equations], args.ratio)
    sampled_text_items: List[Dict[str, Any]] = []
    if args.include_text:
        sampled_text_items = sample_items([it for _, it in all_texts], args.ratio)

    print(f"抽样表格: {len(sampled_tables_items)}")
    print(f"抽样公式: {len(sampled_equations_items)}")
    if args.include_text:
        print(f"抽样文本: {len(sampled_text_items)}")

    # 建立 item -> json_path 的映射，便于解析图片路径
    table_path_map: Dict[int, Path] = {}
    eq_path_map: Dict[int, Path] = {}
    text_path_map: Dict[int, Path] = {}
    for jp, it in all_tables:
        table_path_map[id(it)] = jp
    for jp, it in all_equations:
        eq_path_map[id(it)] = jp
    if args.include_text:
        for jp, it in all_texts:
            text_path_map[id(it)] = jp

    # 执行复制与改写
    written_tables: List[Dict[str, Any]] = []
    written_equations: List[Dict[str, Any]] = []
    written_texts: List[Dict[str, Any]] = []

    converter: Optional[Any] = None
    if args.table_to_md:
        try:
            from scripts.html_converter import HTMLToMarkdownConverter  # type: ignore
        except ModuleNotFoundError:
            repo_root = Path(__file__).resolve().parents[1]
            sys.path.insert(0, str(repo_root))
            try:
                from scripts.html_converter import HTMLToMarkdownConverter  # type: ignore
            except Exception:
                logger.error("HTMLToMarkdownConverter 导入失败")
                HTMLToMarkdownConverter = None  # type: ignore
        except Exception:
            logger.error("HTMLToMarkdownConverter 导入异常")
            HTMLToMarkdownConverter = None  # type: ignore

        if 'HTMLToMarkdownConverter' in locals() and HTMLToMarkdownConverter is not None:
            try:
                converter = HTMLToMarkdownConverter()
            except Exception:
                logger.error("HTMLToMarkdownConverter 初始化失败")
                converter = None

    copied_table = 0
    for idx, item in enumerate(sampled_tables_items):
        src = resolve_image_path(item, table_path_map.get(id(item), input_root), verbose=args.verbose)
        if src is None:
            continue
        new_item, _ = copy_and_rewrite(item, src, table_img_dir, prefix="table", index=idx)
        if converter is not None:
            text_value = new_item.get("text")
            if isinstance(text_value, str):
                try:
                    new_item["text"] = converter.convert_html_table_to_markdown(text_value)
                except Exception:
                    logger.error("HTMLToMarkdownConverter 转换失败")
                    pass
        written_tables.append(new_item)
        copied_table += 1

    copied_equation = 0
    for idx, item in enumerate(sampled_equations_items):
        src = resolve_image_path(item, eq_path_map.get(id(item), input_root), verbose=args.verbose)
        if src is None:
            continue
        new_item, _ = copy_and_rewrite(item, src, eq_img_dir, prefix="equation", index=idx)
        written_equations.append(new_item)
        copied_equation += 1

    copied_text = 0
    if args.include_text:
        for idx, item in enumerate(sampled_text_items):
            src = resolve_image_path(item, text_path_map.get(id(item), input_root), verbose=args.verbose)
            if src is None:
                continue
            new_item, _ = copy_and_rewrite(item, src, text_img_dir, prefix="text", index=idx)
            written_texts.append(new_item)
            copied_text += 1

    # 写出 JSON，保持与示例一致的外层结构；仅更新 targets 与计数
    table_json_path = output_root / "table.json"
    equation_json_path = output_root / "equation.json"
    text_json_path = output_root / "text.json" if args.include_text else None

    now_ts = str(time.time())

    table_payload = {
        "pdf_path": "",
        "json_source": "",
        "extraction_time": now_ts,
        "total_targets": len(written_tables),
        "processed_targets": len(written_tables),
        "targets": written_tables,
    }
    equation_payload = {
        "pdf_path": "",
        "json_source": "",
        "extraction_time": now_ts,
        "total_targets": len(written_equations),
        "processed_targets": len(written_equations),
        "targets": written_equations,
    }
    text_payload = None
    if args.include_text:
        text_payload = {
            "pdf_path": "",
            "json_source": "",
            "extraction_time": now_ts,
            "total_targets": len(written_texts),
            "processed_targets": len(written_texts),
            "targets": written_texts,
        }

    with table_json_path.open("w", encoding="utf-8") as f:
        json.dump(table_payload, f, ensure_ascii=False, indent=2)
    with equation_json_path.open("w", encoding="utf-8") as f:
        json.dump(equation_payload, f, ensure_ascii=False, indent=2)
    if args.include_text and text_json_path is not None and text_payload is not None:
        with text_json_path.open("w", encoding="utf-8") as f:
            json.dump(text_payload, f, ensure_ascii=False, indent=2)

    print(f"已写出: {table_json_path} ({len(written_tables)} 项)")
    print(f"已写出: {equation_json_path} ({len(written_equations)} 项)")
    if args.include_text and text_json_path is not None:
        print(f"已写出: {text_json_path} ({len(written_texts)} 项)")
    if args.include_text:
        print(f"图片输出目录: {table_img_dir} / {eq_img_dir} / {text_img_dir}")
    else:
        print(f"图片输出目录: {table_img_dir} / {eq_img_dir}")


if __name__ == "__main__":
    main()
