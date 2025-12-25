#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Markdown 文件生成 QA JSON 数据（使用 LlamaIndex 的 MarkdownNodeParser 分块 + 调用 qwen-max-latest）。

功能：
- 使用 LlamaIndex 的 MarkdownNodeParser 将 .md 内容解析为节点并按目标字数聚合为块（默认 5000）
- 调用 DashScope 兼容接口（OpenAI SDK）使用 qwen-max-latest 让模型识别 domain 与 source_page，并抽取 instruction（问题/指令）与精确答案 output
- 结果输出为 JSONL（每行一个 JSON 对象），字段严格为：
  {
    "question": "由模型生成的问题/指令（尽量覆盖摘要、定义、对比、数值、推理、实验复现、图表解释、局限性等类型）",
    "original": "原文相关段落/图表描述",
    "output": "精确答案",
    "domain": "cardiology",
    "source_page": 7
  }

环境变量：
- DASHSCOPE_API_KEY 必须已设置

示例：
python QandA/generate_qa_from_md.py \
  --md_path demo/QandA_output/10.1002_asia.200800173/vlm/10.1002_asia.200800173.md \
  --output QandA/10.1002_asia.200800173.qa.jsonl \
  --chunk_size 5000 --max_items_per_chunk 3
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger


def _load_openai_client(api_url: str):
    """加载 OpenAI 兼容客户端（DashScope 兼容模式）。

    优先复用项目 scripts.config 中的 OpenAI 导入逻辑，以便共享 .env 加载等。
    """
    try:
        import sys
        # 将项目根目录加入 sys.path 以便导入 scripts.config
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


def parse_markdown_with_llamaindex(md_text: str, target_chunk_size: int = 5000) -> List[str]:
    """使用 LlamaIndex 的 MarkdownNodeParser 解析并聚合为接近 target_chunk_size 的块。

    依赖：pip install llama-index
    """
    try:
        from llama_index.core.schema import Document
        try:
            # 新版本导入路径
            from llama_index.core.node_parser import MarkdownNodeParser  # type: ignore
        except Exception:
            # 旧版本导入路径
            from llama_index.readers.file.markdown_parser import MarkdownNodeParser  # type: ignore
    except Exception as e:
        raise RuntimeError("需要安装 llama-index，请先执行: pip install -U llama-index") from e

    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents([Document(text=md_text)])
    # 逐个节点聚合到目标大小
    chunks: List[str] = []
    buf: List[str] = []
    cur = 0
    for n in nodes:
        t = n.get_text() if hasattr(n, "get_text") else getattr(n, "text", "")
        if not t:
            continue
        if cur + len(t) + (1 if buf else 0) <= target_chunk_size:
            buf.append(t)
            cur += len(t) + (1 if buf else 0)
        else:
            if buf:
                chunks.append("\n".join(buf).strip())
            buf = [t]
            cur = len(t)
    if buf:
        chunks.append("\n".join(buf).strip())
    return [c for c in chunks if c]


def build_generation_prompt(chunk: str, max_items: int) -> str:
    """构造让大模型基于 chunk 识别 domain、source_page，并抽取 question、original 与精确答案 output。

    目标输出：严格 JSON，对象结构为 {"items": [{"question": "...", "original": "...", "output": "...", "domain": "...", "source_page": 7}]}。
    约束：
    - instruction_type：尽量覆盖多样类型：摘要、定义、对比、数值、推理、实验复现、图表解释、局限性等；且要清晰、具体；
    - question: 由模型生成的中文问题/指令，尽量覆盖不同类型（摘要、定义、对比、数值、推理、实验复现、图表解释、局限性）；清晰、具体、可由 original 直接回答；
    - original: 从原文中截取能直接支撑 question 答案的内容，不要对原文进行修改；
    - output: 基于 original 能得到的“精确答案”，客观、可核对，不要添加多余解释；
    - domain: 输出英文二级学科标签，尽量具体，使用小写下划线或小写空格（如 cardiology, organic_chemistry, materials_science 等）；
    - source_page: 该证据在原文中的页码（整数）；若无明确页码信息，请结合上下文合理估计最可能页码（避免0或负数）。
    - 返回不超过 max_items 条。
    """
    prompt = f"""
你是一个严格的标注助手。基于下方文献片段，抽取若干条可用作问答的数据项。每条包含：由你生成的“question”（问题/指令）、与其对应的“证据摘录 input”、基于证据可直接得到的“精确答案 output”，并给出二级学科标签与答案出处页。严格遵循以下 JSON 模式返回：

{{
  "items": [
    {{"instruction_type": "摘要/定义/对比/数值/推理/实验复现/图表解释/局限性等", "question": "问题/指令(中文)", "original": "证据摘录", "output": "精确答案", "domain": "英文二级学科(小写)", "source_page": 整数页码}},
    ...
  ]
}}

任务：根据给定学术材料，生成高质量的自包含问答对，输出为 JSON 对象数组。每个对象包含字段：
- instruction_type, question, original, output, domain, source_page

核心要求：生成的 question 和 output 必须能够独立组成完整的问答对，即使去掉 original 字段，问答对仍然语义完整、逻辑自洽。

具体规范：

1) 问题设计（question）：
   - 必须包含足够的背景信息和关键实体，使问题在脱离原文后仍可被理解
   - 避免模糊指代词："该方法"→"苯甲酰化保护法"，"该化合物"→"雷帕霉素"
   - 问题应具体、明确，指向可验证的事实性内容
   - 模拟真实用户的自然提问方式，而非学术论文式表述

2) 答案设计（output）：
   - 必须完整回答问题，包含必要的背景说明和关键细节
   - 答案应自成体系，不依赖问题之外的任何上下文
   - 结构化表达：使用编号、要点、步骤等形式组织信息
   - 包含关键数值、术语、条件等具体信息

3) 自包含检验标准：
   - 测试：将 question 和 output 单独提取，是否仍能构成有意义的问答对？
   - 问题是否提供了足够信息让读者理解所问内容？
   - 答案是否提供了足够信息让读者理解回答内容？

4) original 字段的作用：
   - original 仅作为答案的原文依据，用于验证答案的准确性
   - original 不应成为问答对理解的必要条件
   - 从 original 提取信息时，应在 output 中完整表述，而非简单引用

5) 问题类型示例（instruction_type）：
   - 定义类：X化合物的分子结构特征是什么？
   - 机理类：Suzuki偶联反应的催化机理包括哪些关键步骤？
   - 条件类：合成化合物Y的最佳反应条件是什么？
   - 对比类：方法A与方法B在选择性方面有何差异？
   - 数值类：化合物Z的熔点和产率分别是多少？
   - 应用类：该催化剂在工业生产中有哪些优势？

6) 质量标准：
   - 问题具体明确，避免"核心问题是什么"等泛化表述
   - 答案信息密度高，包含关键技术细节
   - 专业术语使用准确，体现领域特色
   - 避免需要额外推理或常识的问题

7) 字段规范：
   - question：完整、自包含的问题，包含必要背景信息
   - original：支撑答案的原文片段（保持原文不变）
   - output：完整、自包含的答案，可独立理解
   - domain：英文二级学科名称（如 organic_chemistry, catalysis, pharmacology）
   - source_page：原文页码，无法确定时合理估计

8) 反例警示：
   - 避免："该方法的优势是什么？" → 应为："苯甲酰化保护法在糖苷合成中的优势是什么？"
   - 避免："作者提出了什么解决方案？" → 应为："针对α-糖苷选择性问题，研究中提出了什么解决方案？"
   - 避免简单复述原文的答案 → 应提供结构化、完整的回答

返回要求：生成至少 {max_items} 条高质量问答对，仅输出符合 JSON 语法的对象数组，不添加其他内容。每个问答对都应通过"独立性测试"：去掉 original 后，instruction 和 output 仍能构成完整、有价值的知识问答。

文献片段：
----------------
{chunk}
----------------

只输出 JSON 对象。
"""
    return prompt.strip()


def llm_generate_items(client, model: str, chunk: str, max_items: int) -> List[Dict[str, Any]]:
    """调用 LLM 生成 [{question, original, output, domain, source_page}, ...] 列表。"""
    prompt = build_generation_prompt(chunk, max_items)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个严谨的中文标注助手，只返回合规 JSON。"},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            stream=False,
            temperature=0.1,
            extra_body={"enable_thinking": False},
        )
        content = response.choices[0].message.content.strip()
        # 清理 Markdown 包裹
        content = re.sub(r"^```(?:json)?\\s*", "", content)
        content = re.sub(r"\\s*```$", "", content)
        data = json.loads(content)
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            logger.warning("LLM 返回格式不含 items 列表，跳过该分块。")
            return []
        results: List[Dict[str, Any]] = []
        for it in items:
            instruction_type = (it.get("instruction_type") or "").strip()
            instruction_text = (it.get("question") or "").strip()
            input_excerpt = (it.get("original") or "").strip()
            output_answer = (it.get("output") or "").strip()
            domain = (str(it.get("domain")) if it.get("domain") is not None else "").strip()
            sp = it.get("source_page")
            if not input_excerpt or not output_answer or not instruction_text or not instruction_type:
                continue
            # 容错处理：source_page 转为正整数
            try:
                sp_int = int(sp)
                if sp_int <= 0:
                    sp_int = 1
            except Exception:
                sp_int = 1
            # 生成最终记录
            results.append({
                "instruction_type": instruction_type,
                "question": instruction_text,
                "original": input_excerpt,
                "output": output_answer,
                "domain": domain or "general",
                "source_page": sp_int,
            })
        return results
    except Exception as e:
        logger.error(f"LLM 生成问答失败: {e}")
        return []


def write_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="从 Markdown 生成 QA JSONL（LlamaIndex 分块 + qwen-max-latest）")
    parser.add_argument("--md_path", type=str, required=True, help="输入的 .md 文件路径")
    parser.add_argument("--output", type=str, default=None, help="输出 JSONL 路径，默认与 md 同名")
    parser.add_argument("--chunk_size", type=int, default=5000, help="目标分块字数（默认 5000）")
    parser.add_argument("--api_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="DashScope 兼容接口地址")
    parser.add_argument("--model", type=str, default="qwen-max-latest", help="模型名称，默认 qwen-max-latest")
    parser.add_argument("--max_items_per_chunk", type=int, default=3, help="每个分块生成的记录数量上限，默认 3")

    args = parser.parse_args()

    md_path = Path(args.md_path)
    if not md_path.exists():
        raise FileNotFoundError(f"未找到输入文件: {md_path}")

    output_path = Path(args.output) if args.output else md_path.with_suffix(".qa.jsonl")
    logger.info(f"输入: {md_path}")
    logger.info(f"输出: {output_path}")

    text = md_path.read_text(encoding="utf-8", errors="ignore")
    chunks = parse_markdown_with_llamaindex(text, target_chunk_size=args.chunk_size)
    logger.info(f"共分块: {len(chunks)}，chunk_size={args.chunk_size}（LlamaIndex MarkdownNodeParser）")

    client = _load_openai_client(args.api_url)

    records: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks, start=1):
        logger.info(f"处理分块 {idx}/{len(chunks)} ...")
        items = llm_generate_items(client, args.model, chunk, args.max_items_per_chunk)
        if not items:
            continue
        records.extend(items)

    if not records:
        logger.warning("未生成任何记录，输出为空。")
    write_jsonl(records, output_path)
    logger.info(f"完成，共输出 {len(records)} 条记录 -> {output_path}")


if __name__ == "__main__":
    main()


