import os
import base64
import tempfile
from pathlib import Path
import litserve as ls
from fastapi import HTTPException
from loguru import logger

# from mineru.cli.common import do_parse, read_fn
from mineru.cli.common import read_fn
from demo.demo import do_parse
from mineru.utils.config_reader import get_device
from mineru.utils.model_utils import get_vram
from _config_endpoint import config_endpoint

class MinerUAPI(ls.LitAPI):
    def __init__(self, output_dir='/tmp'):
        super().__init__()
        self.output_dir = output_dir

    def setup(self, device):
        """Setup environment variables exactly like MinerU CLI does"""
        logger.info(f"Setting up on device: {device}")
                
        if os.getenv('MINERU_DEVICE_MODE', None) == None:
            os.environ['MINERU_DEVICE_MODE'] = device if device != 'auto' else get_device()

        device_mode = os.environ['MINERU_DEVICE_MODE']
        if os.getenv('MINERU_VIRTUAL_VRAM_SIZE', None) == None:
            if device_mode.startswith("cuda") or device_mode.startswith("npu"):
                vram = round(get_vram(device_mode))
                os.environ['MINERU_VIRTUAL_VRAM_SIZE'] = str(vram)
            else:
                os.environ['MINERU_VIRTUAL_VRAM_SIZE'] = '1'
        logger.info(f"MINERU_VIRTUAL_VRAM_SIZE: {os.environ['MINERU_VIRTUAL_VRAM_SIZE']}")

        if os.getenv('MINERU_MODEL_SOURCE', None) in ['huggingface', None]:
            config_endpoint()
        logger.info(f"MINERU_MODEL_SOURCE: {os.environ['MINERU_MODEL_SOURCE']}")


    def decode_request(self, request):
        """Decode file and options from request"""
        options = request.get('options', {})
        
        # 支持两种模式：文件路径模式和文件内容模式
        if 'file_path' in request:
            # 文件路径模式：直接使用服务端本地文件
            input_path = request['file_path']
            temp_file_created = False
        else:
            # 文件内容模式：从base64内容创建临时文件
            file_b64 = request['file']
            file_bytes = base64.b64decode(file_b64)
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
                temp.write(file_bytes)
                input_path = str(Path(temp.name))
                temp_file_created = True
        
        return {
            'input_path': input_path,
            'temp_file_created': temp_file_created,
            'output_dir': options.get('output_dir', self.output_dir),  # 优先使用客户端传递的output_dir
            'backend': options.get('backend', 'pipeline'),
            'method': options.get('method', 'auto'),
            'lang': options.get('lang', 'ch'),
            'formula_enable': options.get('formula_enable', True),
            'table_enable': options.get('table_enable', True),
            'start_page_id': options.get('start_page_id', 0),
            'end_page_id': options.get('end_page_id', None),
            'server_url': options.get('server_url', None),
        }

    def predict(self, inputs):
        """Call MinerU's do_parse - same as CLI"""
        input_path = inputs['input_path']
        output_dir = inputs['output_dir']
        temp_file_created = inputs['temp_file_created']
        
        # 如果客户端指定了自定义output_dir，直接使用；否则创建临时目录
        if output_dir != self.output_dir:
            # 客户端指定了自定义输出目录，直接使用
            final_output_dir = Path(output_dir)
        else:
            # 使用默认行为：在服务端默认目录下创建临时子目录
            final_output_dir = Path(output_dir) / Path(input_path).stem
        
        try:
            # 检查输入文件是否存在
            if not Path(input_path).exists():
                raise FileNotFoundError(f"输入文件不存在: {input_path}")
            
            # 检查文件大小
            file_size = Path(input_path).stat().st_size
            if file_size == 0:
                raise ValueError(f"输入文件为空: {input_path}")
            
            # 检查文件是否为有效的PDF
            try:
                pdf_bytes = read_fn(Path(input_path))
                if len(pdf_bytes) == 0:
                    raise ValueError(f"PDF文件内容为空: {input_path}")
            except Exception as e:
                raise ValueError(f"无法读取PDF文件 {input_path}: {e}")
            
            os.makedirs(final_output_dir, exist_ok=True)
            
            file_name = Path(input_path).stem
            
            
            logger.info(f"开始处理文件: {input_path} (大小: {file_size / 1024 / 1024:.2f} MB)")
            
            do_parse(
                output_dir=str(final_output_dir),
                pdf_file_names=[file_name],
                pdf_bytes_list=[pdf_bytes],
                p_lang_list=[inputs['lang']],
                backend=inputs['backend'],
                parse_method=inputs['method'],
                formula_enable=inputs['formula_enable'],
                table_enable=inputs['table_enable'],
                server_url=inputs['server_url'],
                start_page_id=inputs['start_page_id'],
                end_page_id=inputs['end_page_id'],
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
                f_dump_md=True,
                f_dump_middle_json=True,
                f_dump_model_output=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=False
            )
            

            
            logger.info(f"文件处理完成: {input_path}")
            return str(final_output_dir)
            
        except FileNotFoundError as e:
            logger.error(f"文件不存在错误: {e}")
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            logger.error(f"文件格式错误: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except MemoryError as e:
            logger.error(f"内存不足: {e}")
            # 强制清理内存
            raise HTTPException(status_code=507, detail=f"内存不足，无法处理文件: {input_path}")
        except Exception as e:
            logger.error(f"处理失败 {input_path}: {e}")
            logger.error(f"错误类型: {type(e).__name__}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            
            
            raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
        finally:
            # 只有创建了临时文件才需要清理
            if temp_file_created and Path(input_path).exists():
                try:
                    Path(input_path).unlink()
                except Exception as e:
                    logger.warning(f"清理临时文件失败 {input_path}: {e}")

    def encode_response(self, response):
        return {'output_dir': response}

if __name__ == '__main__':
    server = ls.LitServer(
        MinerUAPI(output_dir='/tmp/mineru_output'),
        accelerator='auto',
        devices='auto',
        workers_per_device=1,
        timeout=False
    )
    logger.info("Starting MinerU server on port 8000")
    server.run(host='0.0.0.0',port=8111, generate_client_file=False)
