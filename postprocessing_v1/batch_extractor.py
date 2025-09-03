#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF图像截取器 - 简化批量处理脚本
专注于命令行批量处理，提供美观的进度条和详细的结果展示

作者：AI Assistant
版本：2.0.0 (简化版)
"""

import argparse
import sys
import os
from pathlib import Path

# 添加当前目录到模块搜索路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 导入核心功能
try:
    from pdf_extractor_core import batch_process_books, logger
except ImportError as e:
    print(f"❌ 导入核心模块失败: {e}")
    print("请确保 pdf_extractor_core.py 文件存在于同一目录下")
    sys.exit(1)


def validate_paths(args):
    """验证输入路径"""
    errors = []
    
    if not os.path.exists(args.results_folder):
        errors.append(f"结果文件夹不存在: {args.results_folder}")
    
    if not os.path.exists(args.pdf_base_folder):
        errors.append(f"PDF基础文件夹不存在: {args.pdf_base_folder}")
    
    if args.output_base_dir and not os.path.isabs(args.output_base_dir):
        # 转换为绝对路径
        args.output_base_dir = str(Path(args.output_base_dir).resolve())
    
    return errors


def print_banner():
    """打印程序横幅"""
    print("="*80)
    print("📚 PDF图像截取器 - 批量处理工具 v2.0.0")
    print("="*80)
    print("🎯 功能：根据JSON文件中的bbox坐标从PDF中批量截取图像")
    print("⚡ 特性：多进程处理、智能恢复、美观进度条")
    print("="*80)


def print_config(args):
    """打印配置信息"""
    print("\n📋 处理配置:")
    print("="*60)
    print(f"📂 结果文件夹: {args.results_folder}")
    print(f"📁 PDF基础目录: {args.pdf_base_folder}")
    
    # 计算总文件夹名称
    from pathlib import Path
    results_folder_path = Path(args.results_folder)
    total_folder_name = results_folder_path.parent.name
    if not total_folder_name or total_folder_name == ".":
        total_folder_name = results_folder_path.name
    
    if args.output_base_dir:
        output_structure = f"{args.output_base_dir}/{total_folder_name}/各书籍文件夹"
        print(f"📤 输出目录: {args.output_base_dir}")
        print(f"📁 总文件夹: {total_folder_name}")
        print(f"🗂️  输出结构: {output_structure}")
    else:
        default_output = f"{args.results_folder}/batch_output"
        output_structure = f"{default_output}/{total_folder_name}/各书籍文件夹"
        print(f"📤 输出目录: 自动创建 ({default_output})")
        print(f"📁 总文件夹: {total_folder_name}")
        print(f"🗂️  输出结构: {output_structure}")
    
    # 统计信息
    try:
        if os.path.exists(args.results_folder):
            book_folders = [item for item in os.listdir(args.results_folder) 
                          if os.path.isdir(os.path.join(args.results_folder, item))]
            print(f"📊 发现书籍文件夹: {len(book_folders)} 个")
        
        if os.path.exists(args.pdf_base_folder):
            # 简单统计PDF文件数量（不递归，避免耗时）
            pdf_files = [f for f in os.listdir(args.pdf_base_folder) if f.lower().endswith('.pdf')]
            print(f"📄 PDF基础目录下直接文件: {len(pdf_files)} 个PDF")
    except Exception as e:
        print(f"⚠️  统计信息获取失败: {e}")
    
    print("="*60)


def print_summary(result):
    """打印处理结果摘要"""
    print("\n" + "="*80)
    print("🎉 批量处理完成!")
    print("="*80)
    
    # 计算成功率
    success_rate = (result.processed_books / result.total_books * 100) if result.total_books > 0 else 0
    
    print(f"📊 处理状态: {'✅ 成功' if result.success else '❌ 失败'}")
    print(f"📚 总书籍数: {result.total_books}")
    print(f"✅ 成功处理: {result.processed_books} ({success_rate:.1f}%)")
    print(f"❌ 失败数量: {len(result.failed_books)}")
    
    if result.failed_books:
        print(f"💔 失败书籍: {', '.join(result.failed_books[:3])}")
        if len(result.failed_books) > 3:
            print(f"   ... 以及其他 {len(result.failed_books) - 3} 本")
    
    print(f"💬 消息: {result.message}")
    print("="*80)
    
    # 显示处理统计
    if result.results:
        # 计算统计信息
        total_targets = sum(r.get("targets_processed", 0) for r in result.results)
        total_images = sum(r.get("images_saved", 0) for r in result.results)
        total_time = sum(r.get("processing_time", 0) for r in result.results if r.get("processing_time"))
        
        print(f"\n📊 处理统计:")
        print(f"   🎯 总目标数: {total_targets}")
        print(f"   🖼️  总图片数: {total_images}")
        if total_time > 0:
            print(f"   ⏱️  总处理时间: {total_time:.1f}秒")
            if result.processed_books > 0:
                avg_time = total_time / result.processed_books
                print(f"   📈 平均每本: {avg_time:.1f}秒")
    
    # 显示详细结果（最近处理的）
    if result.results:
        print("\n📋 处理结果详情:")
        print("-"*60)
        
        # 按成功/失败分类显示
        successful_results = [r for r in result.results if r.get("success", False) and r.get("message") != "智能恢复：跳过已处理文件"]
        failed_results = [r for r in result.results if not r.get("success", False)]
        skipped_results = [r for r in result.results if r.get("message") == "智能恢复：跳过已处理文件"]
        
        # 显示成功的结果
        if successful_results:
            print("✅ 新处理成功的书籍:")
            for book_result in successful_results[-5:]:  # 最近5个
                book_name = book_result.get("book_name", "未知")[:30]
                targets = book_result.get("targets_processed", 0)
                images = book_result.get("images_saved", 0)
                time_taken = book_result.get("processing_time", 0)
                time_str = f" ({time_taken:.1f}s)" if time_taken > 0 else ""
                print(f"   📖 {book_name}: {targets}目标/{images}图片{time_str}")
            
            if len(successful_results) > 5:
                print(f"   📝 ... 以及其他 {len(successful_results) - 5} 本成功处理的书籍")
        
        # 显示跳过的结果
        if skipped_results:
            print(f"\n⏭️  智能恢复跳过: {len(skipped_results)} 本已处理的书籍")
        
        # 显示失败的结果
        if failed_results:
            print("\n❌ 处理失败的书籍:")
            for book_result in failed_results[:3]:  # 前3个失败的
                book_name = book_result.get("book_name", "未知")[:30]
                message = book_result.get("message", "无错误信息")[:50]
                print(f"   📖 {book_name}: {message}")
            
            if len(failed_results) > 3:
                print(f"   📝 ... 以及其他 {len(failed_results) - 3} 本失败的书籍")
    
    print("\n🚀 批量处理完成！")
    
    # 性能提示
    if result.total_books > 0:
        print(f"\n💡 性能提示:")
        if success_rate >= 90:
            print("   🎯 处理成功率很高，系统运行良好！")
        elif success_rate >= 70:
            print("   ⚠️  部分书籍处理失败，建议检查失败原因")
        else:
            print("   🔧 较多书籍处理失败，建议检查配置和文件完整性")
        
        # 添加处理效率提示
        if result.results and any(r.get("processing_time") for r in result.results):
            avg_time = sum(r.get("processing_time", 0) for r in result.results) / len([r for r in result.results if r.get("processing_time")])
            if avg_time < 10:
                print("   ⚡ 处理速度很快，效率很高！")
            elif avg_time < 30:
                print("   🔄 处理速度正常")
            else:
                print("   🐌 处理速度较慢，可能是PDF文件较大或目标数量较多")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="PDF图像截取器 - 批量处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --results-folder /path/to/results --pdf-base-folder /path/to/pdfs
  %(prog)s --results-folder ./results --pdf-base-folder ./pdfs --output-base-dir ./output

注意事项:
  • 结果文件夹应包含书籍子文件夹，每个子文件夹包含对应的JSON文件
  • JSON文件命名格式：书籍名称_middle.json
  • 程序会自动跳过已处理的书籍（智能恢复）
  • 使用多进程并行处理以提高效率
        """
    )
    
    parser.add_argument(
        "--results-folder", 
        required=True, 
        help="结果文件夹路径（包含书籍子文件夹和JSON文件）"
    )
    parser.add_argument(
        "--pdf-base-folder", 
        required=True, 
        help="PDF文件搜索基础目录"
    )
    parser.add_argument(
        "--output-base-dir", 
        help="输出基础目录（默认：结果文件夹/batch_output）"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="显示详细日志信息"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="预演模式：只验证文件，不实际处理"
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logger.setLevel("DEBUG")
    
    # 打印横幅
    print_banner()
    
    # 验证路径
    errors = validate_paths(args)
    if errors:
        print("❌ 路径验证失败:")
        for error in errors:
            print(f"   • {error}")
        sys.exit(1)
    
    # 打印配置
    print_config(args)
    
    # 预演模式
    if args.dry_run:
        print("\n🧪 预演模式：验证文件结构...")
        try:
            # 简单验证
            book_folders = [item for item in os.listdir(args.results_folder) 
                          if os.path.isdir(os.path.join(args.results_folder, item))]
            
            valid_count = 0
            for book_name in book_folders:
                json_file = os.path.join(args.results_folder, book_name, f"{book_name}_middle.json")
                if os.path.exists(json_file):
                    valid_count += 1
            
            print(f"✅ 验证完成：{valid_count}/{len(book_folders)} 个书籍文件夹包含有效JSON文件")
            print("💡 使用 --verbose 参数可查看详细信息")
            sys.exit(0)
            
        except Exception as e:
            print(f"❌ 预演模式失败: {e}")
            sys.exit(1)
    
    # 确认开始处理
    print("\n🚀 准备开始批量处理...")
    try:
        input("按 Enter 键继续，或 Ctrl+C 取消...")
    except KeyboardInterrupt:
        print("\n❌ 用户取消操作")
        sys.exit(0)
    
    # 执行批量处理
    print("\n" + "="*80)
    print("🔄 开始批量处理...")
    print("="*80)
    
    try:
        result = batch_process_books(
            results_folder=args.results_folder,
            pdf_base_folder=args.pdf_base_folder,
            output_base_dir=args.output_base_dir
        )
        
        # 显示结果
        print_summary(result)
        
        # 设置退出代码
        if result.success:
            sys.exit(0)
        else:
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n❌ 用户中断处理")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 批量处理异常: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
