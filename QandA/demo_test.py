import os
from dashscope import MultiModalConversation
import dashscope 

# 若使用新加坡地域的模型，请释放下列注释
# dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

# 将xxx/eagle.png替换为你本地图像的绝对路径
local_path = r"demo\QandA_output\10.1002_asia.200800173\images\4ad6ae976542cbaf310a53a3d2a67204e7a9cbef17a53c8ae32bffe1d812aff2.jpg"
image_path = f"file://{local_path}"
messages = [{"role": "system",
                "content": [{"text": "You are a helpful assistant."}]},
                {'role':'user',
                'content': [{'image': image_path},
                            {'text': '图中描绘的是什么景象?'}]}]
response = MultiModalConversation.call(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    model='qwen-vl-max-latest',  # 此处以qwen-vl-max-latest为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/models
    messages=messages)
print(response["output"]["choices"][0]["message"].content[0]["text"])