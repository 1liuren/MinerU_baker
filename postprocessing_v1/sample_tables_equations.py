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
        help="是否处理 title_text（标题文本）和 text（普通文本）并输出 text.json 与 text_images",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help=(
            "批次根目录（可选）。若提供，则按每个批次的小于10000固定抽10个，"
            "大于等于10000按 --ratio 抽样；并在每个目标条目中写入 pdf_path"
        ),
    )
    return parser.parse_args()


def find_json_files(root: Path) -> List[Path]:
    results: List[Path] = []
    target_names = {"table.json", "equation.json", "text.json"}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn in target_names:
                results.append(Path(dirpath) / fn)
    return results


def is_target_type(item: Dict[str, Any], include_text: bool = False) -> bool:
    t = item.get("type")
    base = {"span_table", "span_interline_equation"}
    if include_text:
        base.add("title_text")
        base.add("text")
    return t in base


def collect_targets(
    json_path: Path, include_text: bool = False, verbose: bool = False
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], str]:
    pdf_path = ""
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"[WARN] 读取失败: {json_path} -> {e}")
        return [], [], [], pdf_path

    targets = data.get("targets", [])
    if not isinstance(targets, list):
        if verbose:
            print(f"[WARN] 文件结构异常(缺少 targets 列表): {json_path}")
        return [], [], [], pdf_path

    pdf_path_value = data.get("pdf_path")
    if isinstance(pdf_path_value, str):
        pdf_path = pdf_path_value

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
        elif include_text and item.get("type") in ("title_text", "text"):
            texts.append(item)

    return tables, equations, texts, pdf_path


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
    # 新结构：按类型子目录分类
    for sub in ("images", "table_images", "equation_images", "text_images"):
        candidates.append(parent / sub / base)

    # # 扩展：在 json 父目录内一层深度搜索 images 目录
    # for name in ("images", "image", "imgs", "table_images", "equation_images", "text_images"):  # 常见命名
    #     candidates.append(parent / name / base)
    #     candidates.append(parent.parent / name / base)

    for cand in candidates:
        if cand.exists():
            if verbose:
                print(f"[INFO] 回退匹配图片: {raw_path} -> {cand}")
            return cand

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


def choose_k_for_batch(total: int, ratio: float) -> int:
    if total <= 0:
        return 0
    if total < 10000:
        return min(10, total)
    k = int(total * ratio)
    if k <= 0:
        k = 1
    if k > total:
        k = total
    return k


def load_batch_index(batch_root: Path, verbose: bool = False) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, List[str]]]:
    """
    扫描批次根目录：每个子目录视为批次ID，读取其 data_info.json（列表）。
    返回：
      - md5_to_info: md5 -> (pdf_path, batch_id)
      - batch_to_md5s: batch_id -> [md5, ...]
    """
    md5_to_info: Dict[str, Tuple[str, str]] = {}
    batch_to_md5s: Dict[str, List[str]] = {}

    if not batch_root.exists():
        if verbose:
            print(f"[WARN] 批次根目录不存在: {batch_root}")
        return md5_to_info, batch_to_md5s

    for child in batch_root.iterdir():
        if not child.is_dir():
            continue
        batch_id = child.name
        data_info_path = child / "data_info.json"
        if not data_info_path.exists():
            if verbose:
                print(f"[WARN] 批次 {batch_id} 缺少 data_info.json: {data_info_path}")
            continue
        try:
            with data_info_path.open("r", encoding="utf-8") as f:
                info_list = json.load(f)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"[WARN] 读取失败: {data_info_path} -> {e}")
            continue
        if not isinstance(info_list, list):
            if verbose:
                print(f"[WARN] 数据格式错误(需为列表): {data_info_path}")
            continue

        md5s: List[str] = []
        for entry in info_list:
            if not isinstance(entry, dict):
                continue
            md5 = entry.get("md5sum")
            pdf_path = entry.get("file_path")
            if not md5 or not isinstance(md5, str) or not pdf_path or not isinstance(pdf_path, str):
                continue
            md5_to_info[md5] = (pdf_path, batch_id)
            md5s.append(md5)
        if md5s:
            batch_to_md5s[batch_id] = md5s

    return md5_to_info, batch_to_md5s


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

    pdf_path_map: Dict[int, str] = {}

    # 批次模式
    if args.batch:
        batch_root = Path(args.batch).resolve()
        md5_to_info, batch_to_md5s = load_batch_index(batch_root, verbose=args.verbose)

        # 汇总未抽样前的总量
        all_tables: List[Tuple[Path, Dict[str, Any]]] = []
        all_equations: List[Tuple[Path, Dict[str, Any]]] = []
        all_texts: List[Tuple[Path, Dict[str, Any]]] = []

        # 选择后的集合
        sel_tables: List[Tuple[Path, Dict[str, Any]]] = []
        sel_equations: List[Tuple[Path, Dict[str, Any]]] = []
        sel_texts: List[Tuple[Path, Dict[str, Any]]] = []

        # 记录每个批次的抽取数量
        batch_selected_counts: Dict[str, Dict[str, int]] = {}

        for batch_id, md5_list in batch_to_md5s.items():
            batch_tables: List[Tuple[Path, Dict[str, Any]]] = []
            batch_equations: List[Tuple[Path, Dict[str, Any]]] = []
            batch_texts: List[Tuple[Path, Dict[str, Any]]] = []

            for md5 in md5_list:
                md5_dir = input_root / md5
                # 表格
                table_json = md5_dir / "table.json"
                if table_json.exists():
                    tbs, _, _, pdf_path = collect_targets(
                        table_json, include_text=False, verbose=args.verbose
                    )
                    for it in tbs:
                        batch_tables.append((table_json, it))
                        all_tables.append((table_json, it))
                        if pdf_path:
                            pdf_path_map[id(it)] = pdf_path
                # 公式
                equation_json = md5_dir / "equation.json"
                if equation_json.exists():
                    _, eqs, _, pdf_path = collect_targets(
                        equation_json, include_text=False, verbose=args.verbose
                    )
                    for it in eqs:
                        batch_equations.append((equation_json, it))
                        all_equations.append((equation_json, it))
                        if pdf_path:
                            pdf_path_map[id(it)] = pdf_path
                # 文本（可选）
                if args.include_text:
                    text_json = md5_dir / "text.json"
                    if text_json.exists():
                        _, _, txts, pdf_path = collect_targets(
                            text_json, include_text=True, verbose=args.verbose
                        )
                        for it in txts:
                            batch_texts.append((text_json, it))
                            all_texts.append((text_json, it))
                            if pdf_path:
                                pdf_path_map[id(it)] = pdf_path

            # 按规则选取
            k_t = choose_k_for_batch(len(batch_tables), args.ratio)
            k_e = choose_k_for_batch(len(batch_equations), args.ratio)
            k_x = choose_k_for_batch(len(batch_texts), args.ratio) if args.include_text else 0

            if k_t > 0 and len(batch_tables) > 0:
                sel_tables.extend(random.sample(batch_tables, k_t))
            if k_e > 0 and len(batch_equations) > 0:
                sel_equations.extend(random.sample(batch_equations, k_e))
            if args.include_text and k_x > 0 and len(batch_texts) > 0:
                sel_texts.extend(random.sample(batch_texts, k_x))

            batch_selected_counts[batch_id] = {
                "tables": k_t if len(batch_tables) > 0 else 0,
                "equations": k_e if len(batch_equations) > 0 else 0,
                "texts": (k_x if len(batch_texts) > 0 else 0) if args.include_text else 0,
            }

        total_tables = len(all_tables)
        total_equations = len(all_equations)
        total_texts = len(all_texts) if args.include_text else 0

        print(f"总表格(span_table): {total_tables}")
        print(f"总公式(span_interline_equation): {total_equations}")
        if args.include_text:
            print(f"总文本(span_text): {total_texts}")

        sampled_tables_items = [it for _, it in sel_tables]
        sampled_equations_items = [it for _, it in sel_equations]
        sampled_text_items: List[Dict[str, Any]] = []
        if args.include_text:
            sampled_text_items = [it for _, it in sel_texts]

        print(f"抽样表格: {len(sampled_tables_items)}")
        print(f"抽样公式: {len(sampled_equations_items)}")
        if args.include_text:
            print(f"抽样文本: {len(sampled_text_items)}")

        # 建立 item -> json_path / pdf_path 的映射
        table_path_map: Dict[int, Path] = {}
        eq_path_map: Dict[int, Path] = {}
        text_path_map: Dict[int, Path] = {}

        for jp, it in sel_tables:
            table_path_map[id(it)] = jp
            md5 = jp.parent.name
            pdf_path = md5_to_info.get(md5, ("", ""))[0]
            if pdf_path:
                pdf_path_map.setdefault(id(it), pdf_path)
        for jp, it in sel_equations:
            eq_path_map[id(it)] = jp
            md5 = jp.parent.name
            pdf_path = md5_to_info.get(md5, ("", ""))[0]
            if pdf_path:
                pdf_path_map.setdefault(id(it), pdf_path)
        if args.include_text:
            for jp, it in sel_texts:
                text_path_map[id(it)] = jp
                md5 = jp.parent.name
                pdf_path = md5_to_info.get(md5, ("", ""))[0]
                if pdf_path:
                    pdf_path_map.setdefault(id(it), pdf_path)
    else:
        # 原有（非批次）模式
        json_files = find_json_files(input_root)
        if args.verbose:
            print(f"[INFO] 发现 JSON 文件 {len(json_files)} 个")

        all_tables: List[Tuple[Path, Dict[str, Any]]] = []  # (json_path, item)
        all_equations: List[Tuple[Path, Dict[str, Any]]] = []
        all_texts: List[Tuple[Path, Dict[str, Any]]] = []

        for jp in json_files:
            tables, equations, texts, pdf_path = collect_targets(
                jp, include_text=args.include_text, verbose=args.verbose
            )
            for it in tables:
                all_tables.append((jp, it))
                if pdf_path:
                    pdf_path_map[id(it)] = pdf_path
            for it in equations:
                all_equations.append((jp, it))
                if pdf_path:
                    pdf_path_map[id(it)] = pdf_path
            if args.include_text:
                for it in texts:
                    all_texts.append((jp, it))
                    if pdf_path:
                        pdf_path_map[id(it)] = pdf_path

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
        # 批次模式下补充 pdf_path
        pdf_path_value = pdf_path_map.get(id(item), "")
        if pdf_path_value:
            new_item["pdf_path"] = pdf_path_value
        if converter is not None:
            # import html
            text_value = new_item.get("text")
            import pandas as pd
            dfs = pd.read_html(text_value)
            df = dfs[0]
            df.columns = [""] * len(df.columns)
            new_item["text"] = df.to_markdown(index=False)
            # text_value = html.unescape(text_value).replace('\\"', '"')
            # if isinstance(text_value, str):
            #     try:
            #         new_item["text"] = converter.convert_html_table_to_markdown(text_value)
            #     except Exception:
            #         logger.error("HTMLToMarkdownConverter 转换失败")
            #         pass
        written_tables.append(new_item)
        copied_table += 1

    copied_equation = 0
    for idx, item in enumerate(sampled_equations_items):
        src = resolve_image_path(item, eq_path_map.get(id(item), input_root), verbose=args.verbose)
        if src is None:
            continue
        new_item, _ = copy_and_rewrite(item, src, eq_img_dir, prefix="equation", index=idx)
        pdf_path_value = pdf_path_map.get(id(item), "")
        if pdf_path_value:
            new_item["pdf_path"] = pdf_path_value
        written_equations.append(new_item)
        copied_equation += 1

    copied_text = 0
    if args.include_text:
        for idx, item in enumerate(sampled_text_items):
            src = resolve_image_path(item, text_path_map.get(id(item), input_root), verbose=args.verbose)
            if src is None:
                continue
            new_item, _ = copy_and_rewrite(item, src, text_img_dir, prefix="text", index=idx)
            pdf_path_value = pdf_path_map.get(id(item), "")
            if pdf_path_value:
                new_item["pdf_path"] = pdf_path_value
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

    # 批次模式下，打印每个批次抽取数量
    if args.batch:
        try:
            # batch_selected_counts 在批次模式分支定义
            for batch_id, cnt in locals().get("batch_selected_counts", {}).items():
                t = cnt.get("tables", 0)
                e = cnt.get("equations", 0)
                x = cnt.get("texts", 0)
                if args.include_text:
                    print(f"批次 {batch_id}: 表格 {t}，公式 {e}，文本 {x}")
                else:
                    print(f"批次 {batch_id}: 表格 {t}，公式 {e}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
