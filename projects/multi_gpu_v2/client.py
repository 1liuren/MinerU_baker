import base64
import requests
import os
from loguru import logger
import asyncio
import aiohttp

def mineru_parse_sync(file_path: str, server_url: str = 'http://127.0.0.1:8000/predict', **options):
    """同步调用MinerU服务端预测接口。

    Args:
        file_path: 待处理文件路径
        server_url: 服务端 /predict 接口URL
        **options: 透传给服务端的参数，包含 backend/method/lang/formula_enable/table_enable/start_page_id/end_page_id/server_url/output_dir 等

    Returns:
        dict: 成功返回 { 'output_dir': str }，失败返回 { 'error': str }
    """
    try:
        with open(file_path, 'rb') as f:
            file_b64 = base64.b64encode(f.read()).decode('utf-8')

        payload = {
            'file': file_b64,
            'options': options
        }

        resp = requests.post(server_url, json=payload, timeout=300)
        if resp.status_code == 200:
            return resp.json()
        else:
            return {'error': resp.text}
    except Exception as e:
        logger.error(f"同步调用失败 {file_path}: {e}")
        return {'error': str(e)}

async def mineru_parse_async(session, file_path, server_url='http://10.10.50.52:8111/predict', **options):
    """
    Asynchronous version of the parse function.
    """
    try:
        # Asynchronously read and encode the file
        with open(file_path, 'rb') as f:
            file_b64 = base64.b64encode(f.read()).decode('utf-8')

        payload = {
            'file': file_b64,
            'options': options
        }

        # Use the aiohttp session to send the request with timeout
        timeout = aiohttp.ClientTimeout(total=600)  # 10分钟超时
        async with session.post(server_url, json=payload, timeout=timeout) as response:
            if response.status == 200:
                result = await response.json()
                logger.info(f"✅ Processed: {file_path} -> {result.get('output_dir', 'N/A')}")
                return result
            else:
                error_text = await response.text()
                logger.error(f"❌ Server error for {file_path}: {error_text}")
                return {'error': error_text}

    except Exception as e:
        logger.error(f"❌ Failed to process {file_path}: {e}")
        return {'error': str(e)}

async def mineru_parse_by_file_path(session, file_path, server_url='http://10.10.50.52:8111/predict', **options):
    """
    使用文件路径模式调用MinerU服务端，避免创建临时文件
    
    Args:
        session: aiohttp session
        file_path: 服务端本地文件路径
        server_url: 服务端URL
        **options: 其他选项参数
    """
    try:
        payload = {
            'file_path': file_path,  # 直接传递文件路径
            'options': options
        }

        # 使用aiohttp session发送请求，设置超时
        timeout = aiohttp.ClientTimeout(total=600)  # 10分钟超时
        async with session.post(server_url, json=payload, timeout=timeout) as response:
            if response.status == 200:
                result = await response.json()
                logger.info(f"✅ Processed (file_path mode): {file_path} -> {result.get('output_dir', 'N/A')}")
                return result
            else:
                error_text = await response.text()
                logger.error(f"❌ Server error for {file_path}: {error_text}")
                return {'error': error_text}

    except Exception as e:
        logger.error(f"❌ Failed to process {file_path}: {e}")
        return {'error': str(e)}


async def main():
    """
    Main function to run all parsing tasks concurrently.
    演示两种模式：文件内容模式和文件路径模式
    """
    test_files = [
        '../../demo/pdfs/demo1.pdf',
        '../../demo/pdfs/demo2.pdf',
        '../../demo/pdfs/demo3.pdf',
        '../../demo/pdfs/small_ocr.pdf',
    ]

    test_files = [os.path.join(os.path.dirname(__file__), f) for f in test_files]
    
    existing_files = [f for f in test_files if os.path.exists(f)]
    if not existing_files:
        logger.warning("No test files found.")
        return

    # Create an aiohttp session to be reused across requests
    async with aiohttp.ClientSession() as session:
        # logger.info("=== 模式1：文件内容模式（上传文件内容到服务端） ===")
        # # === Basic Processing ===
        # basic_tasks = [mineru_parse_async(session, file_path) for file_path in existing_files[:2]]

        # # === Custom Options ===
        # custom_options = {
        #     'backend': 'pipeline', 'lang': 'ch', 'method': 'auto',
        #     'formula_enable': True, 'table_enable': True,'output_dir': './output'
        # }
        # # 'backend': 'sglang-engine' requires 24+ GB VRAM per worker

        # custom_tasks = [mineru_parse_async(session, file_path, **custom_options) for file_path in existing_files[2:]]

        # # Start all tasks
        # all_tasks = basic_tasks + custom_tasks
        # all_results = await asyncio.gather(*all_tasks)
        # logger.info(f"文件内容模式结果: {all_results}")

        logger.info("\n=== 模式2：文件路径模式（直接使用服务端本地文件） ===")
        # 文件路径模式测试 - 使用服务端本地文件路径
        server_file_paths = [
            # 这些路径应该是服务端可以访问的文件路径
            # 可以是绝对路径或相对于服务端工作目录的路径
            os.path.abspath(existing_files[0]),  # 使用绝对路径
            os.path.abspath(existing_files[1]) if len(existing_files) > 1 else None,
        ]
        
        server_file_paths = [p for p in server_file_paths if p is not None]
        
        file_path_options = {
            'backend': 'pipeline', 'lang': 'ch', 'method': 'auto',
            'formula_enable': True, 'table_enable': True,
            'output_dir': './output_file'  # 不同的输出目录
        }
        
        file_path_tasks = [mineru_parse_by_file_path(session, file_path, **file_path_options) for file_path in server_file_paths]
        file_path_results = await asyncio.gather(*file_path_tasks)
        logger.info(f"文件路径模式结果: {file_path_results}")
        
    logger.info("🎉 All processing completed!")

if __name__ == '__main__':
    # Run the async main function
    asyncio.run(main())
