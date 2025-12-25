#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 PDF 解析生成的 Markdown，抽取图/表及其引用段落，生成：
- QA 对（围绕图/表的专业问答）
- 多轮对话（3-8 轮，含追问、指代、简单推理）
- caption（短<=60字、长4-6句）

输入：
- 指定 markdown 路径（如 demo/QandA_output/10.1002_asia.200800173/10.1002_asia.200800173.md）
- 关联图片目录（从 md 内的相对路径如 images/*.jpg 提取）

输出：
- JSONL 或 JSON（按 --format 选择）
- 结构遵循用户示例：
  {
    "id": "book123_page45_fig7",
    "figure_path": "fig/book123_page45_fig7.png",
    "caption": "图7：2020-2023 年 AI 论文发表量",
    "referring_paragraphs": ["如圖7所示，2023年發表量達到15.2萬篇..."],
    "qa_pairs": [{"q": "2023年AI论文发表量是多少？", "a": "15.2万篇"}],
    "dialogue": [
      {"role": "user", "content": "图7中哪年增长率最高？"},
      {"role": "assistant", "content": "2021年，增长率达32%。"}
    ],
    "short_caption": "2020-2023年AI论文数量逐年增加",
    "long_caption": "本图展示了2020至2023年间人工智能领域论文发表量的增长趋势..."
  }

实现策略（启发式，纯本地规则）：
- 图像：匹配形如 ![](images/xxx.jpg) 的行，向下寻找下一行以 "Figure X." 或 "图 X."、"Scheme" 开头的说明作为原始 caption。
- 表格：匹配以 "Table X." 或 "表 X." 开头，后接 <table> HTML 的块。
- 引用段落：从图/表附近上下若干段（默认前后各2段）中，选择含有 "Figure X"/"Table X"/"图X"/"表X" 的段落；若无，选紧邻上一段。
- QA/对话/Caption：基于 caption 与引用段落的关键词进行模板化生成；若没有明确数值，给出描述性问答。

注意：
- 遵循项目 Python 风格（导入顺序、命名、类型注解、logger）。
- 路径统一使用 pathlib.Path。
"""

from __future__ import annotations

import argparse
import json
import re
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger

try:
    # 可选导入，多模态调用
    from dashscope import MultiModalConversation  # type: ignore
    import dashscope  # type: ignore
    _DASHSCOPE_AVAILABLE = True
except Exception:
    _DASHSCOPE_AVAILABLE = False


FIG_PAT = re.compile(r"^\s*!\[[^\]]*\]\((?P<src>[^)]+\.(?:png|jpe?g|svg))\)\s*$", re.IGNORECASE)
HTML_TABLE_START = re.compile(r"<table[^>]*>", re.IGNORECASE)
HTML_TABLE_END = re.compile(r"</table>\s*", re.IGNORECASE)


@dataclass
class FigureItem:
    index: str
    path: str
    line_no: int


@dataclass
class TableItem:
    index: str
    html: str
    start_line: int
    end_line: int


def read_file_lines(md_path: Path) -> List[str]:
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.split("\n")


def read_file_text(md_path: Path) -> str:
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def find_figure_items(lines: List[str]) -> List[FigureItem]:
    results: List[FigureItem] = []
    for i, line in enumerate(lines):
        m = FIG_PAT.match(line)
        if not m:
            continue
        img_src = m.group("src").strip()
        results.append(FigureItem(index=str(len(results) + 1), path=img_src, line_no=i))
    return results


def find_html_tables_in_text(md_text: str) -> List[TableItem]:
    """使用跨行正则在整份 Markdown 文本中提取 <table>…</table> 片段，并计算行号。"""
    results: List[TableItem] = []
    # 跨行不贪婪匹配
    pattern = re.compile(r"<table\b[\s\S]*?</table>", re.IGNORECASE)
    for m in pattern.finditer(md_text):
        start_char = m.start()
        end_char = m.end()
        html = m.group(0)
        # 计算起止行号（0-based）
        start_line = md_text.count("\n", 0, start_char)
        end_line = md_text.count("\n", 0, end_char)
        results.append(TableItem(index=str(len(results) + 1), html=html, start_line=start_line, end_line=end_line))
    return results


def _find_tables_from_middle_json(md_path: Path) -> List[TableItem]:
    """从同目录的 *_middle.json 中抽取表格（type==table）。
    兼容不同结构：列表/字典，递归查找包含 html/markdown/cells 的表格节点，优先使用 html。
    无法定位在 md 中的行号时，用 start_line=end_line=-1。
    """
    json_path = md_path.with_name(md_path.stem + "_middle.json")
    if not json_path.exists():
        return []
    try:
        text = json_path.read_text(encoding="utf-8", errors="ignore")
        data = json.loads(text)
    except Exception as e:
        logger.warning(f"读取 middle.json 失败: {json_path} -> {e}")
        return []

    tables: List[TableItem] = []

    def _walk(node: Any):
        if isinstance(node, dict):
            if str(node.get("type", "")).lower() == "table":
                html = node.get("html") or node.get("table_html") or node.get("markdown") or node.get("md")
                if not html:
                    # 退化为JSON片段供LLM理解
                    html = json.dumps(node, ensure_ascii=False)
                tables.append(TableItem(index=str(len(tables) + 1), html=str(html), start_line=-1, end_line=-1))
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(data)

    return tables


def collect_context_text(lines: List[str], anchor_line: int, context_chars: int) -> str:
    before_chars = context_chars
    after_chars = context_chars
    # 向上收集
    up_buf: List[str] = []
    total = 0
    i = anchor_line - 1
    while i >= 0 and total < before_chars:
        t = lines[i].strip()
        if t:
            up_buf.append(t)
            total += len(t) + 1
        i -= 1
    up_text = "\n".join(reversed(up_buf))
    # 当前行
    cur_text = lines[anchor_line].strip()
    # 向下收集
    down_buf: List[str] = []
    total = 0
    j = anchor_line + 1
    while j < len(lines) and total < after_chars:
        t = lines[j].strip()
        if t:
            down_buf.append(t)
            total += len(t) + 1
        j += 1
    down_text = "\n".join(down_buf)
    parts = [s for s in [up_text, cur_text, down_text] if s]
    return "\n".join(parts)


# ===== 大模型调用 =====

def _load_openai_client(api_url: str):
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.config import OpenAI  # type: ignore
    except Exception as e:
        logger.error(f"导入 OpenAI 兼容客户端失败: {e}")
        raise
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到 DASHSCOPE_API_KEY 环境变量")
    client = OpenAI(base_url=api_url, api_key=api_key)
    return client


def _build_llm_prompt(item_type: str, md_stem: str, index: str, context_text: str, figure_path: str, table_html: str, max_dialogue_turns: int) -> str:
    example = {
        "caption": "一句话标题或说明（由模型概括）",
        "referring_paragraphs": ["从上下文中提取的1-3条原文句子（不改写）"],
        "qa_pairs": [{"q": "模型生成的专业且多样化问题", "a": "来自上下文可核对的答案"}],
        "dialogue": [
            {"role": "user", "content": "与图/表相关的追问或指代"},
            {"role": "assistant", "content": "自洽且与上下文一致的回答"}
        ],
        "short_caption": "<=60字的中文摘要",
        "long_caption": "4-6句的中文段落"
    }
    payload = {
        "item_type": item_type,
        "index": index,
        "context_text": context_text,
        "figure_path": figure_path,
        "table_html": table_html,
    }
    prompt = (
        "你是中文学术标注助手。仅基于给定上下文（不臆测文外信息），返回一个 JSON 对象，字段为：id, figure_path, caption, referring_paragraphs(1-3), "
        f"qa_pairs(3-5，多样化), dialogue(3-{max_dialogue_turns} 轮), short_caption, long_caption。"
        "注意：referring_paragraphs 必须原文摘录；其余字段为中文概括且能从上下文核对。只输出 JSON。\n\n"
        f"上下文：\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        f"示例结构：\n{json.dumps(example, ensure_ascii=False, indent=2)}\n"
    )
    return prompt


def _llm_generate_record(client, model: str, temperature: float, item_type: str, md_path: Path, index: str, context_text: str, figure_path: str, table_html: str, max_dialogue_turns: int) -> Dict[str, Any]:
    prompt = _build_llm_prompt(item_type, md_path.stem, index, context_text, figure_path, table_html, max_dialogue_turns)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个严谨的中文数据构造助手，只返回JSON对象。"},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
        stream=False,
        # extra_body={"enable_thinking": False},
        # response_format={'type': 'json_object'}
    )
    content = (response.choices[0].message.content or "").strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    data = json.loads(content)
    if not isinstance(data, dict):
        raise RuntimeError("LLM 返回非对象 JSON")
    data["id"] = f"{md_path.stem}_{'fig' if item_type=='figure' else 'table'}_{index}"
    # 标准化 figure_path
    if item_type == "figure":
        data["figure_path"] = figure_path
    else:
        data["figure_path"] = ""
        data["table_html"] = table_html
    return data


def _mm_generate_record_for_figure(md_path: Path, index: str, figure_rel_path: str, context_text: str, mm_model: str, dashscope_api_base: Optional[str], max_dialogue_turns: int) -> Dict[str, Any]:
    if not _DASHSCOPE_AVAILABLE:
        raise RuntimeError("未安装 dashscope，无法使用多模态模型")
    image_abs = (md_path.parent / figure_rel_path).resolve()
    if not image_abs.exists():
        raise FileNotFoundError(f"图片不存在: {image_abs}")
    # 使用相对路径（相对于当前工作目录）传入，多模态 SDK 会负责上传
    image_rel = os.path.relpath(str(image_abs), start=str(Path.cwd()))
    image_rel = image_rel.replace("\\", "/")
    if dashscope_api_base:
        try:
            dashscope.base_http_api_url = dashscope_api_base  # type: ignore
        except Exception as e:
            logger.warning(f"设置 dashscope 基础地址失败: {e}")
    schema_hint = {
        "caption": "...",
        "referring_paragraphs": ["..."],
        "qa_pairs": [{"q": "...", "a": "..."}],
        "dialogue": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
        "short_caption": "...",
        "long_caption": "..."
    }
    ask_text = (
        "结合图片内容与提供的上下文（不臆测上下文之外信息），构造一个包含 id, figure_path, caption, referring_paragraphs(1-3), "
        f"qa_pairs(3-5), dialogue(3-{max_dialogue_turns}), short_caption, long_caption 的 JSON 对象。referring_paragraphs 必须是原文句子。只输出 JSON。\n\n"
        f"上下文：\n{context_text}\n\n示例结构：\n{json.dumps(schema_hint, ensure_ascii=False, indent=2)}"
    )
    messages = [
        {"role": "system", "content": [{"text": "You are a helpful Chinese assistant. Output JSON only."}]},
        {"role": "user", "content": [{"image": f'file://{image_rel}'}, {"text": ask_text}]}]
    resp = MultiModalConversation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=mm_model,
        messages=messages,
        response_format={'type': 'json_object'}
    )
    content = resp["output"]["choices"][0]["message"].content[0]["text"].strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    
    # 输出token使用情况
    usage = resp["usage"]
    logger.info(f"Token使用: 输入={usage['input_tokens']}, 输出={usage['output_tokens']}, 总计={usage['total_tokens']}")
    
    print(content)
    data = json.loads(content)
    if not isinstance(data, dict):
        raise RuntimeError("多模态返回非对象 JSON")
    data["id"] = f"{md_path.stem}_fig_{index}"
    data["figure_path"] = figure_rel_path
    return data


# ===== 主流程 =====

def process_markdown(md_path: Path, api_url: str, model: str, temperature: float, max_dialogue_turns: int, mm_model: str, dashscope_api_base: Optional[str], context_chars: int) -> Dict[str, List[Dict[str, Any]]]:
    lines = read_file_lines(md_path)
    md_text = read_file_text(md_path)
    figures = find_figure_items(lines)
    tables = find_html_tables_in_text(md_text)

    logger.info(f"发现图像 {len(figures)} 个，表格 {len(tables)} 个")

    client = _load_openai_client(api_url)

    fig_records: List[Dict[str, Any]] = []
    for f in figures:
        ctx = collect_context_text(lines, f.line_no, context_chars)
        rec = _mm_generate_record_for_figure(
            md_path=md_path,
            index=f.index,
            figure_rel_path=f.path,
            context_text=ctx,
            mm_model=mm_model,
            dashscope_api_base=dashscope_api_base,
            max_dialogue_turns=max_dialogue_turns,
        )
        fig_records.append(rec)

    tab_records: List[Dict[str, Any]] = []
    for t in tables:
        # 使用表格起始行的上下文
        ctx = collect_context_text(lines, t.start_line, context_chars)
        ctx = f"{ctx}\n\n<table_context>\n{t.html}\n</table_context>"
        rec = _llm_generate_record(
            client=client,
            model=model,
            temperature=temperature,
            item_type="table",
            md_path=md_path,
            index=t.index,
            context_text=ctx,
            figure_path="",
            table_html=t.html,
            max_dialogue_turns=max_dialogue_turns,
        )
        tab_records.append(rec)

    return {"figures": fig_records, "tables": tab_records}


def write_output(data: Dict[str, List[Dict[str, Any]]], out_path: Path, fmt: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        with out_path.open("w", encoding="utf-8") as f:
            for section in ("figures", "tables"):
                for rec in data.get(section, []):
                    f.write(json.dumps(rec, ensure_ascii=False))
                    f.write("\n")
    else:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 Markdown 生成 图/表 的QA、多轮对话与caption 数据（仅LLM，图多模态/表文本）")
    parser.add_argument("--md_path", type=str, required=True, help="输入 Markdown 路径")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径，默认同名 .qa_duolun.jsonl")
    parser.add_argument("--format", type=str, choices=["json", "jsonl"], default="jsonl", help="输出格式")
    parser.add_argument("--api_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="文本 LLM（OpenAI兼容）接口地址")
    parser.add_argument("--model", type=str, default="qwen-max-latest", help="文本 LLM 模型名称")
    parser.add_argument("--temperature", type=float, default=0.2, help="文本 LLM 采样温度")
    parser.add_argument("--max_dialogue_turns", type=int, default=6, help="多轮对话最大轮次（3-8）")
    parser.add_argument("--mm_model", type=str, default="qwen3-vl-plus", help="多模态 LLM 模型名称（用于图片）")
    parser.add_argument("--dashscope_api_base", type=str, default=None, help="DashScope 基础 API 地址（如 https://dashscope-intl.aliyuncs.com/api/v1）")
    parser.add_argument("--context_chars", type=int, default=500, help="上下文字符数（上下各截取），默认500")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    md_path = Path(args.md_path)
    if not md_path.exists():
        raise FileNotFoundError(f"未找到输入 Markdown: {md_path}")
    data = process_markdown(
        md_path=md_path,
        api_url=args.api_url,
        model=args.model,
        temperature=args.temperature,
        max_dialogue_turns=max(3, min(8, args.max_dialogue_turns)),
        mm_model=args.mm_model,
        dashscope_api_base=args.dashscope_api_base,
        context_chars=max(50, args.context_chars),
    )
    out_path = Path(args.output) if args.output else md_path.with_suffix(".qa_duolun.jsonl" if args.format == "jsonl" else ".qa_duolun.json")
    write_output(data, out_path, args.format)
    logger.info(f"完成：{out_path}")


if __name__ == "__main__":
    main()
