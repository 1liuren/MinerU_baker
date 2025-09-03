#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF图像截取器配置文件

用户可以在这里自定义要截取的目标类型和其他设置
"""

# 要截取的目标类型配置
TARGET_TYPES = {
    "interline_equation",   # 行间公式
    "table"                 # 表格
}

# 图像质量配置
IMAGE_CONFIG = {
    "scale_factor": 2.0,    # 图像放大倍数，提高清晰度
    "format": "PNG",        # 输出图像格式
    "dpi": 300             # 图像DPI设置
}

# 输出配置
OUTPUT_CONFIG = {
    "default_dir": "image",  # 默认输出目录名
    "naming_pattern": "{id}.png",  # 文件命名模式
    "create_subdirs": True   # 是否创建子目录
}

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",        # 日志级别: DEBUG, INFO, WARNING, ERROR
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": None            # 日志文件路径，None表示只输出到控制台
}

# 处理配置
PROCESSING_CONFIG = {
    "max_concurrent": 4,    # 最大并发处理数
    "timeout": 300,         # 处理超时时间（秒）
    "retry_count": 3        # 失败重试次数
}

def get_target_types():
    """获取要截取的目标类型"""
    return TARGET_TYPES.copy()

def add_target_type(target_type: str):
    """添加新的目标类型"""
    TARGET_TYPES.add(target_type)

def remove_target_type(target_type: str):
    """移除目标类型"""
    TARGET_TYPES.discard(target_type)

def get_image_config():
    """获取图像配置"""
    return IMAGE_CONFIG.copy()

def get_output_config():
    """获取输出配置"""
    return OUTPUT_CONFIG.copy()

def get_logging_config():
    """获取日志配置"""
    return LOGGING_CONFIG.copy()

def get_processing_config():
    """获取处理配置"""
    return PROCESSING_CONFIG.copy()

# 预定义的目标类型集合
PREDEFINED_TYPES = {
    "equations": {"inline_equation", "interline_equation"},
    "tables": {"table"},
    "figures": {"figure", "image"},
    "all": {"inline_equation", "interline_equation", "table", "figure", "image", "text"},
    "minimal": {"inline_equation", "table"}
}

def set_target_types_by_preset(preset_name: str):
    """根据预定义集合设置目标类型"""
    if preset_name in PREDEFINED_TYPES:
        global TARGET_TYPES
        TARGET_TYPES = PREDEFINED_TYPES[preset_name].copy()
        return True
    return False

def get_available_presets():
    """获取可用的预定义集合"""
    return list(PREDEFINED_TYPES.keys())

if __name__ == "__main__":
    # 配置测试
    print("=== PDF图像截取器配置 ===")
    print(f"当前目标类型: {TARGET_TYPES}")
    print(f"可用预定义集合: {get_available_presets()}")
    
    # 测试预定义集合
    print("\n测试预定义集合:")
    for preset in ["equations", "tables", "minimal"]:
        set_target_types_by_preset(preset)
        print(f"  {preset}: {TARGET_TYPES}")
    
    # 恢复默认设置
    set_target_types_by_preset("minimal")
    print(f"\n恢复默认设置: {TARGET_TYPES}")
