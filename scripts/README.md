# PDF处理流水线脚本包

这是一个重构后的PDF处理流水线，将原来的 `pipeline_optimized.py` 文件按功能模块拆分为多个文件，提高代码的可维护性和可扩展性。

## 文件结构

```
scripts/
├── __init__.py              # 包初始化文件
├── config.py                # 配置和日志管理
├── status_manager.py        # 处理状态管理
├── html_converter.py        # HTML转Markdown转换器
├── utils.py                 # 工具函数集合
├── pdf_pipeline.py          # 主要的PDF处理流水线类
├── main.py                  # 主程序入口
├── batch_epub_to_pdf.py     # EPUB转PDF批处理工具
└── README.md                # 本文档
```

## 模块说明

### config.py
- 日志配置管理
- 项目路径设置
- 可选模块导入（如OpenAI）
- 书籍元数据模型定义

### status_manager.py
- `ProcessingStatus` 类：管理文件处理状态
- 支持状态持久化、进度跟踪、统计信息

### html_converter.py
- `HTMLToMarkdownConverter` 类：HTML表格转Markdown
- 支持复杂表格结构的转换

### utils.py
- `format_time`: 时间格式化
- `find_files`: 文件查找
- `clean_markdown_text`: Markdown文本清洗
- `extract_metadata_with_llm`: 大模型元数据提取

### pdf_pipeline.py
- `OptimizedPDFPipeline` 类：核心处理流水线
- 支持EPUB转PDF、批量PDF处理、异步并发处理
- 集成MinerU的 `do_parse` 函数

### main.py
- 命令行参数解析
- 流水线启动和管理
- 错误处理和日志输出

## 使用方法

### 1. 作为模块使用

```python
from scripts import OptimizedPDFPipeline

pipeline = OptimizedPDFPipeline(
    input_dir="/path/to/input",
    output_dir="/path/to/output",
    backend="vlm-sglang-client",
    server_url="http://localhost:30000"
)

success = pipeline.run_pipeline()
```

### 2. 命令行使用

```bash
# 基本用法
python -m scripts.main --input /path/to/pdfs --output /path/to/output

# 指定sglang服务器
python -m scripts.main --input /path/to/pdfs --output /path/to/output \
    --backend vlm-sglang-client --server-url http://localhost:30000

# 性能优化配置
python -m scripts.main --input /path/to/pdfs --output /path/to/output \
    --batch-size 50 --concurrent-batches 2 --max-files-per-call 200

# 查看帮助
python -m scripts.main --help
```

### 3. 参数说明

#### 必需参数
- `--input, -i`: 输入目录路径（包含PDF或EPUB文件）
- `--output, -o`: 输出目录路径

#### 处理配置
- `--backend`: 处理后端 (vlm-sglang-client, pipeline)
- `--server-url`: sglang服务器URL
- `--lang`: 处理语言 (ch, en)
- `--api-url`: 大模型API URL

#### 性能配置
- `--max-workers`: 最大工作线程数 (默认: 4)
- `--batch-size`: 每批次处理的文件数量 (默认: 100)
- `--concurrent-batches`: 同时处理的批次数量 (默认: 4)
- `--max-files-per-call`: 单次调用最大文件数量 (默认: 400)

#### 其他配置
- `--log-level`: 日志级别 (DEBUG, INFO, WARNING, ERROR)

## 性能优化建议

### 1. 针对sglang-client的优化
由于sglang-client实际上是逐个发送HTTP请求，建议：
- 将 `--batch-size` 设为 1
- 增加 `--concurrent-batches` 到 8-16
- 增加 `--max-files-per-call` 到 1000+

```bash
python -m scripts.main --input /path/to/pdfs --output /path/to/output \
    --batch-size 1 --concurrent-batches 16 --max-files-per-call 1000
```

### 2. 监控资源使用
- 监控GPU内存使用率
- 监控网络带宽
- 根据服务器性能调整并发参数

## 输出结果

处理完成后，输出目录包含：

```
output/
├── results/                 # 处理结果
│   ├── book1/
│   │   ├── book1.md        # 清洗后的Markdown内容
│   │   ├── book1_middle.json # 中间处理数据
│   │   └── book1_extracted_metadata.json # 大模型提取的元数据
│   └── book2/
│       └── ...
├── logs/                    # 处理日志
│   └── pipeline_20240101_120000.log
├── temp/                    # 临时文件（自动清理）
├── processing_status.json   # 处理状态记录
└── processed_books_20240101_120000.jsonl # 最终JSONL文件
```

## 错误处理

- 支持断点续传：重新运行时会跳过已处理的文件
- 详细的错误日志记录
- 自动清理临时文件
- 优雅的中断处理（Ctrl+C）

## 扩展性

重构后的代码具有良好的扩展性：

1. **添加新的转换器**：在 `html_converter.py` 中扩展
2. **添加新的元数据提取器**：在 `utils.py` 中扩展
3. **添加新的处理后端**：在 `pdf_pipeline.py` 中扩展
4. **添加新的状态管理功能**：在 `status_manager.py` 中扩展

## 依赖要求

- Python 3.7+
- MinerU 项目依赖
- loguru
- tqdm
- httpx (用于sglang-client)
- asyncio
- 可选：openai (用于元数据提取)