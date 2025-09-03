#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算表格和公式转写准确率的脚本

比较模型输出和人工校准的markdown文件，计算字符级别的准确率。
主要用于评估表格和公式的转写质量。
"""

import os
import re
import json
import html as _html
from pathlib import Path
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional
from loguru import logger
try:
    # 优先包内导入
    from scripts.html_converter import HTMLToMarkdownConverter  # type: ignore
except Exception:
    try:
        from html_converter import HTMLToMarkdownConverter  # type: ignore
    except Exception:
        HTMLToMarkdownConverter = None  # type: ignore


class AccuracyCalculator:
    """准确率计算器"""

    def __init__(self):
        # 标准化处理的正则表达式
        self.whitespace_pattern = re.compile(r'\s+')


    def normalize_text(self, text: str) -> str:
        """标准化文本内容，便于比较"""
        if not text:
            return ""

        # 移除多余的空白字符
        text = self.whitespace_pattern.sub(' ', text.strip())


        # 移除连续的空行
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 移除行尾空格
        text = re.sub(r' +\n', '\n', text)

        return text

    def _table_values_only(self, table_html: str) -> str:
        """提取表格纯值，忽略所有HTML标签和属性。"""
        try:
            row_pat = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
            cell_pat = re.compile(r'<t[hd][^>]*>(.*?)</t[hd]>', re.DOTALL | re.IGNORECASE)
            rows: list[str] = []
            for row_html in row_pat.findall(table_html):
                cells: list[str] = []
                for cell_html in cell_pat.findall(row_html):
                    text = re.sub(r'<[^>]+>', '', cell_html)
                    text = _html.unescape(text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    cells.append(text)
                rows.append('|'.join(cells))
            return '\n'.join(rows)
        except Exception:
            txt = re.sub(r'<[^>]+>', ' ', table_html)
            txt = _html.unescape(txt)
            return re.sub(r'\s+', ' ', txt).strip()

    def generate_comparison_report(self, model_tables: List[str], human_tables: List[str],
                                 model_formulas: List[str], human_formulas: List[str],
                                 model_file: str, human_file: str) -> Dict:
        """生成详细的对比报告"""
        report = {
            'files': {
                'model_file': model_file,
                'human_file': human_file
            },
            'summary': {
                'tables_count': len(model_tables),
                'formulas_count': len(model_formulas)
            },
            'tables': [],
            'formulas': []
        }

        # 按顺序对比表格
        min_table_count = min(len(model_tables), len(human_tables))
        for i in range(min_table_count):
            report['tables'].append({
                'index': i + 1,
                'model_content': model_tables[i],
                'human_content': human_tables[i]
            })

        # 按顺序对比公式
        min_formula_count = min(len(model_formulas), len(human_formulas))
        for i in range(min_formula_count):
            report['formulas'].append({
                'index': i + 1,
                'model_content': model_formulas[i],
                'human_content': human_formulas[i]
            })

        return report

    def calculate_char_accuracy(self, model_text: str, human_text: str) -> Dict:
        """
        计算字符级别的准确率

        Args:
            model_text: 模型输出的文本
            human_text: 人工校准的文本

        Returns:
            包含准确率统计的字典
        """
        # 标准化文本
        normalized_model = self.normalize_text(model_text)
        normalized_human = self.normalize_text(human_text)

        # 使用SequenceMatcher进行比较
        matcher = SequenceMatcher(None, normalized_model, normalized_human)

        # 统计字符数
        total_model_chars = len(normalized_model)
        total_human_chars = len(normalized_human)

        # 计算匹配的字符数
        correct_chars = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                correct_chars += i2 - i1

        # 计算错误字符数
        error_chars = total_model_chars - correct_chars

        # 计算准确率
        accuracy = correct_chars / total_model_chars if total_model_chars > 0 else 0.0

        # 计算其他指标
        precision = correct_chars / total_model_chars if total_model_chars > 0 else 0.0
        recall = correct_chars / total_human_chars if total_human_chars > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'total_model_chars': total_model_chars,
            'total_human_chars': total_human_chars,
            'correct_chars': correct_chars,
            'error_chars': error_chars,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'similarity_ratio': matcher.ratio()
        }

    def extract_table_content(self, text: str) -> List[str]:
        """提取文本中的表格内容"""
        tables = []

        # 提取HTML表格
        html_table_pattern = re.compile(r'<table[^>]*>(.*?)</table>', re.DOTALL | re.IGNORECASE)
        html_tables = html_table_pattern.findall(text)
        tables.extend(html_tables)

        return tables

    def extract_formula_content(self, text: str) -> List[str]:
        """提取文本中的公式内容"""
        formulas = []

        # 提取行间公式 $$...$$
        block_formulas = re.findall(r'\$\$([^$]+)\$\$', text)
        formulas.extend(block_formulas)

        # 提取行内公式 $...$
        inline_formulas = re.findall(r'\$([^$\n]+)\$', text)
        formulas.extend(inline_formulas)

        return formulas

    def calculate_table_accuracy(self, model_text: str, human_text: str,
                                 convert_to_markdown: bool = False,
                                 values_only: bool = False) -> Dict:
        """计算表格转写准确率（按顺序一对一比较）

        convert_to_markdown: 若为 True，则将抽取到的 HTML 表格转换为 Markdown 后再进行比较。
        values_only: 若为 True 且未启用 convert_to_markdown，仅比较表格中的纯文本值（忽略HTML）。
        """
        model_tables = self.extract_table_content(model_text)
        human_tables = self.extract_table_content(human_text)

        # 可选：将 HTML 表格转换为 Markdown
        if convert_to_markdown:
            if HTMLToMarkdownConverter is None:
                logger.warning("HTMLToMarkdownConverter 不可用，跳过表格到Markdown转换")
            else:
                try:
                    converter = HTMLToMarkdownConverter()
                    model_tables = [converter.convert_html_in_text(t) for t in model_tables]
                    human_tables = [converter.convert_html_in_text(t) for t in human_tables]
                except Exception as e:
                    logger.warning(f"表格到Markdown转换失败，改为直接比较HTML：{e}")
            if values_only:
                logger.info("已启用表格Markdown转换，忽略 values_only 选项。")
        else:
            # 不转换为 Markdown 时，可选仅比较值
            if values_only:
                model_tables = [self._table_values_only(t) for t in model_tables]
                human_tables = [self._table_values_only(t) for t in human_tables]

        # 调试输出：显示提取的表格内容
        logger.info("=== 表格内容对比 ===")
        logger.info(f"模型输出中提取到 {len(model_tables)} 个表格")
        # for i, table in enumerate(model_tables):
        #     logger.info(f"模型表格 {i+1} (前200字符): {table[:200]}{'...' if len(table) > 200 else ''}")

        logger.info(f"人工校准中提取到 {len(human_tables)} 个表格")
        # for i, table in enumerate(human_tables):
        #     logger.info(f"人工表格 {i+1} (前200字符): {table[:200]}{'...' if len(table) > 200 else ''}")

        # 如果不转换为 Markdown，则仅比较“值”，忽略所有 HTML 符号

        # 按顺序比较表格
        total_correct_chars = 0
        total_model_chars = 0
        total_human_chars = 0
        matched_tables = 0

        min_count = min(len(model_tables), len(human_tables))
        for i in range(min_count):
            model_table = model_tables[i]
            human_table = human_tables[i]

            # 使用SequenceMatcher比较单个表格
            matcher = SequenceMatcher(None, model_table, human_table)
            correct_chars = 0
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    correct_chars += i2 - i1

            total_correct_chars += correct_chars
            total_model_chars += len(model_table)
            total_human_chars += len(human_table)

            # 如果匹配度很高，认为匹配成功
            if matcher.ratio() > 0.8:  # 80%相似度阈值
                matched_tables += 1

            logger.info(f"表格 {i+1} 比较: 模型({len(model_table)}字符) vs 人工({len(human_table)}字符), 匹配字符数: {correct_chars}, 相似度: {matcher.ratio():.4f}")

        # 计算准确率指标
        accuracy = total_correct_chars / total_human_chars if total_human_chars > 0 else 0.0
        precision = total_correct_chars / total_model_chars if total_model_chars > 0 else 0.0
        recall = accuracy
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        logger.info(f"表格总体匹配: {matched_tables}/{min_count}, 准确率: {accuracy:.4f}")

        return {
            'table_count_model': len(model_tables),
            'table_count_human': len(human_tables),
            'matched_tables': matched_tables,
            'total_model_chars': total_model_chars,
            'total_human_chars': total_human_chars,
            'correct_chars': total_correct_chars,
            'table_accuracy': accuracy,
            'table_precision': precision,
            'table_recall': recall,
            'table_f1': f1_score
        }

    def calculate_formula_accuracy(self, model_text: str, human_text: str) -> Dict:
        """计算公式转写准确率（按顺序一对一比较）"""
        model_formulas = self.extract_formula_content(model_text)
        human_formulas = self.extract_formula_content(human_text)

        # 调试输出：显示提取的公式内容
        logger.info("=== 公式内容对比 ===")
        logger.info(f"模型输出中提取到 {len(model_formulas)} 个公式")
        for i, formula in enumerate(model_formulas):
            logger.info(f"模型公式 {i+1}: {formula}")

        logger.info(f"人工校准中提取到 {len(human_formulas)} 个公式")
        for i, formula in enumerate(human_formulas):
            logger.info(f"人工公式 {i+1}: {formula}")

        # 按顺序比较公式
        total_correct_chars = 0
        total_model_chars = 0
        total_human_chars = 0
        matched_formulas = 0

        min_count = min(len(model_formulas), len(human_formulas))
        for i in range(min_count):
            model_formula = model_formulas[i]
            human_formula = human_formulas[i]

            # 使用SequenceMatcher比较单个公式
            matcher = SequenceMatcher(None, model_formula, human_formula)
            correct_chars = 0
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    correct_chars += i2 - i1

            total_correct_chars += correct_chars
            total_model_chars += len(model_formula)
            total_human_chars += len(human_formula)

            # 如果匹配度很高，认为匹配成功
            if matcher.ratio() > 0.8:  # 80%相似度阈值
                matched_formulas += 1

            logger.info(f"公式 {i+1} 比较: 模型({len(model_formula)}字符) vs 人工({len(human_formula)}字符), 匹配字符数: {correct_chars}, 相似度: {matcher.ratio():.4f}")

        # 计算准确率指标
        accuracy = total_correct_chars / total_human_chars if total_human_chars > 0 else 0.0
        precision = total_correct_chars / total_model_chars if total_model_chars > 0 else 0.0
        recall = accuracy
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        logger.info(f"公式总体匹配: {matched_formulas}/{min_count}, 准确率: {accuracy:.4f}")

        return {
            'formula_count_model': len(model_formulas),
            'formula_count_human': len(human_formulas),
            'matched_formulas': matched_formulas,
            'total_model_chars': total_model_chars,
            'total_human_chars': total_human_chars,
            'correct_chars': total_correct_chars,
            'formula_accuracy': accuracy,
            'formula_precision': precision,
            'formula_recall': recall,
            'formula_f1': f1_score
        }

    def compare_files(self, model_file: Path, human_file: Path,
                      convert_table_html: bool = False,
                      tables_values_only: bool = False) -> Dict:
        """比较两个文件并计算各项准确率"""
        try:
            # 读取文件内容
            with open(model_file, 'r', encoding='utf-8') as f:
                model_text = f.read()
            with open(human_file, 'r', encoding='utf-8') as f:
                human_text = f.read()

            # 提取内容用于对比
            model_tables = self.extract_table_content(model_text)
            human_tables = self.extract_table_content(human_text)
            model_formulas = self.extract_formula_content(model_text)
            human_formulas = self.extract_formula_content(human_text)

            # 若需要，将表格转换为Markdown以便更直观的对比报告
            if convert_table_html and HTMLToMarkdownConverter is not None:
                try:
                    converter = HTMLToMarkdownConverter()
                    conv_model_tables = [converter.convert_html_in_text(t) for t in model_tables]
                    conv_human_tables = [converter.convert_html_in_text(t) for t in human_tables]
                except Exception as e:
                    logger.warning(f"对比报告表格Markdown转换失败，使用原HTML：{e}")
                    conv_model_tables = model_tables
                    conv_human_tables = human_tables
            else:
                # 未转换Markdown时，可根据需要仅展示值
                if tables_values_only:
                    conv_model_tables = [self._table_values_only(t) for t in model_tables]
                    conv_human_tables = [self._table_values_only(t) for t in human_tables]
                else:
                    conv_model_tables = model_tables
                    conv_human_tables = human_tables

            # 计算各项指标
            char_metrics = self.calculate_char_accuracy(model_text, human_text)
            table_metrics = self.calculate_table_accuracy(
                model_text, human_text,
                convert_to_markdown=convert_table_html,
                values_only=tables_values_only,
            )
            formula_metrics = self.calculate_formula_accuracy(model_text, human_text)

            # 生成对比报告
            comparison_report = self.generate_comparison_report(
                conv_model_tables, conv_human_tables, model_formulas, human_formulas,
                str(model_file), str(human_file)
            )

            # 合并结果
            result = {
                'file_pair': {
                    'model_file': str(model_file),
                    'human_file': str(human_file)
                },
                'char_level': char_metrics,
                'table_level': table_metrics,
                'formula_level': formula_metrics,
                'comparison_report': comparison_report
            }

            return result

        except Exception as e:
            logger.error(f"比较文件失败 {model_file} vs {human_file}: {e}")
            return {
                'file_pair': {
                    'model_file': str(model_file),
                    'human_file': str(human_file)
                },
                'error': str(e)
            }


def find_file_pairs(model_dir: Path, human_dir: Path) -> List[Tuple[Path, Path]]:
    """查找对应的文件对"""
    pairs = []

    # 遍历人工校准目录
    for human_file in human_dir.rglob("*.md"):
        if human_file.is_file():
            # 构造对应的模型输出文件路径
            relative_path = human_file.relative_to(human_dir)
            model_file = model_dir / relative_path

            if model_file.exists() and model_file.is_file():
                pairs.append((model_file, human_file))
            else:
                logger.warning(f"未找到对应的模型输出文件: {model_file}")

    return pairs


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="计算表格和公式转写准确率")
    parser.add_argument("--model-dir", help="模型输出目录")
    parser.add_argument("--human-dir", help="人工校准目录")
    parser.add_argument("--output", default="accuracy_report.json", help="输出报告文件")
    parser.add_argument("--comparison-output", default="comparison_report.json", help="输出对比报告文件")
    parser.add_argument("--single-pair", nargs=2, help="比较单个文件对 (model_file human_file)")
    parser.add_argument("--convert-table-html", action="store_true", help="将HTML表格转换为Markdown后再比较，并在对比报告中使用转换结果")
    parser.add_argument("--tables-values-only", action="store_true", help="不转换Markdown时，仅比较表格的值并在报告中仅可视化值")

    args = parser.parse_args()

    calculator = AccuracyCalculator()
    results = []

    if args.single_pair:
        # 比较单个文件对
        model_file = Path(args.single_pair[0])
        human_file = Path(args.single_pair[1])

        if not model_file.exists():
            logger.error(f"模型输出文件不存在: {model_file}")
            return
        if not human_file.exists():
            logger.error(f"人工校准文件不存在: {human_file}")
            return

        logger.info(f"比较文件: {model_file} vs {human_file}")
        result = calculator.compare_files(
            model_file, human_file,
            convert_table_html=args.convert_table_html,
            tables_values_only=args.tables_values_only,
        )
        results.append(result)

    else:
        # 批量比较
        model_dir = Path(args.model_dir)
        human_dir = Path(args.human_dir)

        if not model_dir.exists():
            logger.error(f"模型输出目录不存在: {model_dir}")
            return
        if not human_dir.exists():
            logger.error(f"人工校准目录不存在: {human_dir}")
            return

        logger.info(f"查找文件对: {model_dir} vs {human_dir}")
        file_pairs = find_file_pairs(model_dir, human_dir)

        if not file_pairs:
            logger.warning("未找到对应的文件对")
            return

        logger.info(f"找到 {len(file_pairs)} 个文件对，开始比较...")

        for model_file, human_file in file_pairs:
            logger.info(f"比较文件: {model_file.name}")
            result = calculator.compare_files(
                model_file, human_file,
                convert_table_html=args.convert_table_html,
                tables_values_only=args.tables_values_only,
            )
            results.append(result)

    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 提取对比报告并保存到单独文件
    comparison_reports = []
    for result in results:
        if 'comparison_report' in result:
            comparison_reports.append(result['comparison_report'])

    if comparison_reports:
        with open(args.comparison_output, 'w', encoding='utf-8') as f:
            json.dump(comparison_reports, f, ensure_ascii=False, indent=2)
        logger.success(f"对比报告已保存到: {args.comparison_output}")

    logger.success(f"准确率计算完成，结果已保存到: {args.output}")

    # 打印汇总统计
    if results:
        print("\n" + "="*60)
        print("汇总统计")
        print("="*60)

        total_char_accuracy = 0
        total_table_accuracy = 0
        total_formula_accuracy = 0
        valid_count = 0

        for result in results:
            if 'error' not in result:
                char_acc = result['char_level']['accuracy']
                table_acc = result['table_level']['table_accuracy']
                formula_acc = result['formula_level']['formula_accuracy']

                total_char_accuracy += char_acc
                total_table_accuracy += table_acc
                total_formula_accuracy += formula_acc
                valid_count += 1

                print(f"{Path(result['file_pair']['human_file']).name}: "
                      f"字符={char_acc:.2%}, "
                      f"表格={table_acc:.2%}, "
                      f"公式={formula_acc:.2%}")
        if valid_count > 0:
            avg_char = total_char_accuracy / valid_count
            avg_table = total_table_accuracy / valid_count
            avg_formula = total_formula_accuracy / valid_count

            print(f"平均准确率 - 字符={avg_char:.2%}, "
                  f"表格={avg_table:.2%}, "
                  f"公式={avg_formula:.2%}")


if __name__ == "__main__":
    main()
