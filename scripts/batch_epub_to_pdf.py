import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

try:
    import pymupdf
except ImportError:
    print("❌ 错误: 需要安装 PyMuPDF")
    print("请运行: pip install PyMuPDF")
    sys.exit(1)

def format_time(seconds):
    """将秒数格式化为人类可读的时间格式"""
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)}分{remaining_seconds:.2f}秒"
    else:
        hours = seconds // 3600
        remaining = seconds % 3600
        minutes = remaining // 60
        seconds = remaining % 60
        return f"{int(hours)}小时{int(minutes)}分{seconds:.2f}秒"

def find_epub_files(source_dir):
    """递归查找所有EPUB文件"""
    epub_files = []
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"❌ 错误: 源目录不存在 - {source_dir}")
        return []
    
    # 递归查找所有.epub文件
    for epub_file in source_path.rglob("*.epub"):
        if epub_file.is_file():
            epub_files.append(epub_file)
    
    return sorted(epub_files)

def convert_single_epub_simple(args):
    """
    简化的EPUB转PDF转换函数 (避免锁竞争)
    
    Args:
        args: (epub_path, source_dir, output_dir, dpi, method, file_index, total_files)
    """
    epub_path, source_dir, output_dir, dpi, method, file_index, total_files = args
    
    start_time = time.time()
    thread_id = threading.get_ident() % 1000
    
    try:
        # 计算相对路径
        source_path = Path(source_dir)
        epub_path = Path(epub_path)
        relative_path = epub_path.relative_to(source_path)
        
        # 构建输出路径，保持目录结构
        output_path = Path(output_dir) / relative_path.with_suffix('.pdf')
        
        # 创建输出目录
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 简化的进度输出 (减少锁竞争)
        print(f"📖 [{file_index}/{total_files}] T{thread_id}: {relative_path.name}")
        
        # 打开EPUB文件并转换
        epub_doc = pymupdf.open(str(epub_path))
        
        try:
            # 直接转换
            pdf_bytes = epub_doc.convert_to_pdf()
            with open(output_path, 'wb') as f:
                f.write(pdf_bytes)
            epub_doc.close()
            
            elapsed = time.time() - start_time
            print(f"   ✅ T{thread_id}: 成功 ({elapsed:.2f}s)")
            return True, None, str(relative_path)
               
        except Exception as e:
            # 如果直接转换失败，返回错误而不尝试逐页转换（避免长时间卡顿）
            epub_doc.close()
            elapsed = time.time() - start_time
            error_msg = f"转换失败 ({elapsed:.2f}s): {str(e)}"
            print(f"   ❌ T{thread_id}: {error_msg}")
            return False, error_msg, str(relative_path)
        
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"文件处理失败 ({elapsed:.2f}s): {str(e)}"
        print(f"   ❌ T{thread_id}: {error_msg}")
        return False, error_msg, str(relative_path)

def batch_convert_epub_to_pdf(source_dir, output_dir, dpi=150, method='direct', 
                             skip_existing=True, log_file=None, max_workers=4):
    """
    批量转换EPUB文件为PDF
    
    Args:
        source_dir: 源目录路径
        output_dir: 输出目录路径
        dpi: 图像分辨率
        method: 转换方法
        skip_existing: 是否跳过已存在的文件
        log_file: 日志文件路径
        max_workers: 最大线程数
    """
    start_time = time.time()
    
    print("=" * 60)
    print("EPUB批量转换PDF工具 (多线程版)")
    print("=" * 60)
    print(f"源目录: {source_dir}")
    print(f"输出目录: {output_dir}")
    print(f"DPI设置: {dpi}")
    print(f"转换方法: {method}")
    print(f"并发线程数: {max_workers}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 查找所有EPUB文件
    print("🔍 扫描EPUB文件...")
    epub_files = find_epub_files(source_dir)
    
    if not epub_files:
        print("⚠️ 未找到任何EPUB文件")
        return
    
    print(f"📚 找到 {len(epub_files)} 个EPUB文件")
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 过滤需要转换的文件
    files_to_convert = []
    skip_count = 0
    
    for epub_file in epub_files:
        # 计算输出文件路径
        source_path = Path(source_dir)
        relative_path = epub_file.relative_to(source_path)
        output_file = Path(output_dir) / relative_path.with_suffix('.pdf')
        
        # 检查是否跳过已存在的文件
        if skip_existing and output_file.exists():
            print(f"⏭️ 跳过 (文件已存在): {relative_path}")
            skip_count += 1
            continue
        
        files_to_convert.append(epub_file)
    
    if not files_to_convert:
        print("⚠️ 没有需要转换的文件")
        return
    
    print(f"📚 需要转换 {len(files_to_convert)} 个文件 (跳过 {skip_count} 个)")
    print("🚀 开始多线程转换...\n")
    
    # 统计信息
    success_count = 0
    error_count = 0
    error_details = []
    
    # 准备任务参数列表
    task_args = []
    for i, epub_file in enumerate(files_to_convert, 1):
        task_args.append((epub_file, source_dir, output_dir, dpi, method, i, len(files_to_convert)))
    
    # 简化的多线程转换
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 使用map方法批量提交任务，更高效且更容易中断
            results = list(executor.map(convert_single_epub_simple, task_args))
            
            # 处理结果
            for i, (success, error_msg, relative_path) in enumerate(results):
                epub_file = files_to_convert[i]
        
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    error_details.append((str(epub_file), error_msg))
                
                        # 简化的日志写入
                if log_file:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        status = "成功" if success else "失败"
                        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
                                    f"{status} - {relative_path}\n")
                        if not success and error_msg:
                            f.write(f"    错误详情: {error_msg}\n")
    
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断，正在停止所有线程...")
        raise
    
    print("\n🔄 所有线程已完成")
    
    # 总结
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("多线程转换完成!")
    print("=" * 60)
    print(f"总文件数: {len(epub_files)}")
    print(f"需要转换: {len(files_to_convert)}")
    print(f"成功转换: {success_count}")
    print(f"跳过文件: {skip_count}")
    print(f"转换失败: {error_count}")
    print(f"并发线程数: {max_workers}")
    print(f"总用时: {format_time(total_time)}")
    
    if len(files_to_convert) > 0:
        avg_time = total_time / len(files_to_convert)
        print(f"平均每文件: {format_time(avg_time)}")
        print(f"多线程加速比: 约 {max_workers:.1f}x (理论值)")
    
    # 显示错误详情
    if error_details:
        print("\n❌ 失败文件详情:")
        for file_path, error_msg in error_details:
            print(f"   {file_path}")
            print(f"     -> {error_msg}")
    
    print(f"\n📁 输出目录: {os.path.abspath(output_dir)}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="批量将EPUB文件转换为PDF，保持目录结构",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python batch_epub_to_pdf.py source_folder output_folder
  python batch_epub_to_pdf.py source_folder output_folder --dpi 200 --threads 8
  python batch_epub_to_pdf.py source_folder output_folder --method page_by_page --threads 6
  python batch_epub_to_pdf.py source_folder output_folder --no-skip --log conversion.log --threads 4
        """
    )
    
    parser.add_argument("source_dir", help="源目录路径 (包含EPUB文件)")
    parser.add_argument("output_dir", help="输出目录路径")
    parser.add_argument("--dpi", type=int, default=150, 
                       help="图像分辨率DPI (默认: 150)")
    parser.add_argument("--method", choices=['direct', 'page_by_page'], 
                       default='direct',
                       help="转换方法 (默认: direct)")
    parser.add_argument("--no-skip", action="store_true",
                       help="不跳过已存在的PDF文件")
    parser.add_argument("--log", help="日志文件路径")
    parser.add_argument("--threads", type=int, default=4,
                       help="并发线程数 (默认: 4)")  
    
    args = parser.parse_args()
    
    # 检查源目录
    if not os.path.exists(args.source_dir):
        print(f"❌ 错误: 源目录不存在 - {args.source_dir}")
        sys.exit(1)
    
    # 验证线程数
    if args.threads <= 0:
        print("❌ 错误: 线程数必须大于0")
        sys.exit(1)
    elif args.threads > 8:
        print("⚠️ 警告: 推荐线程数不超过8，过多线程可能导致卡顿")
        print(f"当前设置: {args.threads} 线程，自动调整为8线程")
        args.threads = 8
    
    # 转换文件
    try:
        batch_convert_epub_to_pdf(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            dpi=args.dpi,
            method=args.method,
            skip_existing=not args.no_skip,
            log_file=args.log,
            max_workers=args.threads
        )
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()