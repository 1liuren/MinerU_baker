from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from bs4 import BeautifulSoup
from tqdm import tqdm


LATEX_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\\\\\([\\s\\S]*?\\\\\)"),
    re.compile(r"\\\\\[[\\s\\S]*?\\\\\]"),
    re.compile(r"\$\$[\s\S]*?\$\$"),
    re.compile(r"(?<!\$)\$[^\$]+\$(?!\$)"),
    re.compile(r"\\\\begin\{([a-zA-Z*]+)\}[\s\S]*?\\\\end\{\\1\}"),
)


@dataclass
class TableScore:
    """Container for the scoring metrics of a single HTML table."""

    table_id: str
    row_count: int
    column_count: int
    grid_size: int
    merged_cell_count: int
    rowspan_usage: int
    colspan_usage: int
    latex_formula_count: int
    total_score: int


def _parse_span(value: str | None) -> int:
    try:
        parsed = int(value) if value else 1
    except (TypeError, ValueError):
        parsed = 1
    return max(parsed, 1)


def _count_latex_formulas(table_tag) -> int:
    content = table_tag.get_text(separator=" ", strip=False)
    spans = set()
    for pattern in LATEX_PATTERNS:
        for match in pattern.finditer(content):
            spans.add((match.start(), match.end()))
    return len(spans)


def _score_single_table(table_html: str, table_id: str) -> TableScore:
    soup = BeautifulSoup(table_html, "html.parser")
    table_tag = soup.find("table")
    if table_tag is None:
        raise ValueError(f"未找到 table 标签: {table_id}")

    occupancy: set[Tuple[int, int]] = set()
    rowspan_usage = 0
    colspan_usage = 0
    merged_cell_count = 0

    rows = table_tag.find_all("tr")
    for row_index, row in enumerate(rows):
        column_index = 0
        for cell in row.find_all(["td", "th"]):
            while (row_index, column_index) in occupancy:
                column_index += 1

            rowspan = _parse_span(cell.get("rowspan"))
            colspan = _parse_span(cell.get("colspan"))

            if rowspan > 1:
                rowspan_usage += 1
            if colspan > 1:
                colspan_usage += 1
            if rowspan > 1 or colspan > 1:
                merged_cell_count += 1

            for delta_r in range(rowspan):
                for delta_c in range(colspan):
                    occupancy.add((row_index + delta_r, column_index + delta_c))

            column_index += colspan

    if occupancy:
        max_row_index = max(row for row, _ in occupancy)
        max_column_index = max(col for _, col in occupancy)
        row_count = max(max_row_index + 1, len(rows))
        column_count = max_column_index + 1
    else:
        row_count = len(rows)
        column_count = 0

    grid_size = row_count * column_count
    latex_formula_count = _count_latex_formulas(table_tag)

    return TableScore(
        table_id=table_id,
        row_count=row_count,
        column_count=column_count,
        grid_size=grid_size,
        merged_cell_count=merged_cell_count,
        rowspan_usage=rowspan_usage,
        colspan_usage=colspan_usage,
        latex_formula_count=latex_formula_count,
        total_score=grid_size
        + merged_cell_count
        + rowspan_usage
        + colspan_usage
        + latex_formula_count,
    )


def score_tables_from_targets(targets: Sequence[dict], show_progress: bool = False) -> List[TableScore]:
    scores: List[TableScore] = []
    iterator = tqdm(targets, desc="处理表格", unit="table") if show_progress else targets
    for target in iterator:
        table_id = target.get("id", "unknown_table")
        table_html = target.get("text", "")
        if "<table" not in table_html:
            continue
        scores.append(_score_single_table(table_html, table_id))
    return scores


def score_tables_from_file(json_path: Path, show_progress: bool = False) -> tuple[list[TableScore], dict]:
    with json_path.open("r", encoding="utf-8") as source:
        payload = json.load(source)

    targets = payload.get("targets", [])
    if not isinstance(targets, Iterable):
        raise ValueError("JSON 中的 targets 字段格式不正确")

    scores = score_tables_from_targets(list(targets), show_progress=show_progress)
    return scores, payload


def _inject_scores_to_payload(payload: dict, scores: Sequence[TableScore]) -> None:
    score_map = {item.table_id: asdict(item) for item in scores}
    for target in payload.get("targets", []):
        table_id = target.get("id")
        if table_id in score_map:
            target["table_score"] = score_map[table_id]
    payload["table_scores"] = list(score_map.values())


def _iter_table_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"未找到输入路径: {input_path}")
    return sorted(path for path in input_path.rglob("table.json") if path.is_file())


def _resolve_output_path(
    output_path: Path | None,
    input_path: Path,
    target_file: Path,
    multiple_inputs: bool,
) -> Path | None:
    if output_path is None:
        return None

    if output_path.suffix.lower() == ".json" and not multiple_inputs and not output_path.is_dir():
        return output_path

    base_dir = output_path if output_path.suffix == "" else output_path.parent
    base_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        relative = target_file.name
    else:
        relative = target_file.relative_to(input_path)

    destination = base_dir / Path(relative).with_suffix(".scores.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="基于 HTML 的表格复杂度打分")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="包含表格 HTML 的 JSON 文件路径，或目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="可选，输出评分结果的文件或目录路径",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="禁用进度条显示",
    )

    args = parser.parse_args()

    input_path = args.input.resolve()
    table_files = _iter_table_files(input_path)
    if not table_files:
        raise ValueError(f"输入路径中未找到 table.json: {input_path}")

    multiple_inputs = len(table_files) > 1
    show_progress = not args.no_progress

    # 如果有多个文件，显示文件级进度条
    file_iterator = tqdm(table_files, desc="处理文件", unit="file") if show_progress and multiple_inputs else table_files
    
    for table_file in file_iterator:
        # 单个文件内的表格进度条（只在单文件或禁用文件进度条时显示）
        show_table_progress = show_progress and not multiple_inputs
        scores, payload = score_tables_from_file(table_file, show_progress=show_table_progress)

        destination: Path | None = None
        if args.output is not None:
            destination = _resolve_output_path(args.output, input_path, table_file, multiple_inputs)

        if destination:
            result = [asdict(item) for item in scores]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            _inject_scores_to_payload(payload, scores)
            table_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

