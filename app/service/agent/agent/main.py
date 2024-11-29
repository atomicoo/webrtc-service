import os
import sqlite3
from typing import Annotated, Literal, TypedDict
import io
import json
import time
import base64
from pprint import pprint
import traceback
from langgraph.prebuilt import tools_condition
import aiohttp
from PIL import Image
import requests
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool, Tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_experimental.utilities import PythonREPL
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import adispatch_custom_event,dispatch_custom_event
import random 
import traceback
from langgraph.prebuilt import InjectedState
import sys, os
import threading
from datetime import datetime

# 当前文件路径
current_file_path = os.path.abspath(__file__)

# 找到 "webrtc-service/app" 的路径
target_directory = "app"
target_path = current_file_path.split(target_directory)[0] + target_directory

# 将路径加入到 sys.path
sys.path.append(target_path)

from util.logger import logger
from threadsafedict import ThreadSafeDict

session_cache = ThreadSafeDict()

webrtc_service_url = os.getenv('WEBRTC_SERVICE_URL', 'http://172.25.248.109:7956')
MEMO_URL = 'http://180.184.134.254:7169'

# 创建一个全局锁
lock = threading.Lock()

python_repl = PythonREPL()


@tool
def retrieve_memory(user_id, time_start, time_end, summarization_level, query) -> str:
    """
    当用户的问题涉及查询记忆（如历史对话、记录、会议、视频等）时，请调用此工具/函数。
    如果问题中有任何与时间相关的表述（如刚刚，前两天，上个月等），需要结合上下文中的当前时间，将其转换为精确的时间范围（time_start, time_end）。

    Parameters:
        user_id (str): 用户id(user_id)会在上下文中提供
        time_start (str): 问题中涉及的时间范围的开始，格式为："YYYY-MM-DD". 如果未提及，请设置为"None"。
        time_end (str): 问题中涉及的时间范围的结束，格式为："YYYY-MM-DD". 如果未提及，请设置为"None"。
        summarization_level (str): 如果问题和“总结”，“概括”等相关，请选择需要总结的记忆的范围大小：["total", "year", "month", "day", "event", "hour", "None"]. 如果问题与总结无关，请设置为"None"。
        query (str): 标准化的问题文本，用于检索相关记忆内容。

    Returns:
        str: 查询记忆的结果，可以用于回复涉及记忆查询的用户问题
    """
    logger.info(f"Retrieving memory for user {user_id} from {time_start} to {time_end} with summarization level {summarization_level}, query: {query}")

    url = MEMO_URL + '/get_memory'
    data = {
        'text': query,
        'user_id': user_id,
        'lang': 'cn'
    }
    response = requests.post(url, json=data)

    # Placeholder for the memory retrieval logic
    return response.json()['new_prompt']


@tool
def get_meeting_content(date, question) -> str:
    """
    Invoke this function when user ask a question about summarizing a meeting, e.g. summarize the meeting content, summarize someones speech in the meeting, etc.
    The date is necessary, if the date is not provided, ask the user to provide the date.
    If the question is not about summarizing, do not use this function.
    Parameters:
        date: date of the meeting
        question: the question about summarizing a meeting

    Returns:
        the entire content of the meeting on the given date, further summarization can be done based on the content.
    """
    logger.info(f"get the meeting content on {date}, question: {question}")
    memory_url = 'http://180.184.134.254:7169/get_hist'
    data = {
        'user_id': '3738',
        'max_conv': -1
    }
    response = requests.post(memory_url, json=data)
    hist = response.json()
    meetings = [h for k, h in hist.items() if 'meeting' in k]
    return '\n'.join(m['query'] for m in meetings[0])

# 通用 API 请求方法
def _call_api(url, payload):
    try:
        response = requests.post(url, data=json.dumps(payload), timeout=100)
        if response.status_code == 200:
            try:
                return response.json()
            except json.JSONDecodeError:
                logger.error(f"Response is not in JSON format from URL: {url}")
                return None
        else:
            logger.error(f"Request failed with status code {response.status_code} from URL: {url}")
            return None
    except Exception as e:
        logger.error(f"Error during API call: {e}")
        return None

# 发送事件的统一方法
def _dispatch_event(function_name, message):
    dispatch_custom_event(
        "dir_out_event",
        {
            "function_name": function_name,
            "arguments": {"input_question": ""},
            "tool_output_token": message,
        },
        config={"tags": ["tool_call"]},
    )

def _get_phone_call_event(content_text):
    return adispatch_custom_event(
        "dir_out_event",
        {
            "function_name": "make_phone_call",
            "arguments": {"input_question": ''},
            "tool_output_token": content_text,
        },
        config={"tags": ["tool_call"]},
    )


@tool
async def make_phone_call(target, task_description) -> str:
    """
    Invoke this function when user want to call a shop/service agent/etc. to do something.
    e.g. 帮我打个电话给某某商店，问问他们有没有某某商品，或者帮我打个电话给某某服务商，问问他们的服务价格等，比如修理某某。
    Parameters:
        target (str): Who or which shop/agent/etc. to call.
        task_description (str): The description what the user want to do.
    Returns:
        str: a concise summarization of the result of the phone call.
    """
    logger.info(f"Making a phone call, to: {target}, task: {task_description}")

    async def generate_convs():
        memory_url = 'http://180.184.134.254:7169/get_memory'
        memory_msg = {
            'user_id': '10',
            'text': ''
        }
        response = requests.post(memory_url, json=memory_msg)
        profile = response.json()['system_prompt']
        profile = '\n\n'.join(profile.split('\n\n')[1:]).replace('###', '')

        shop_url = 'http://180.184.134.254:7172/chat_stream'
        # shop_url = 'http://180.184.134.254:7166/chat_stream'
        turns = 0
        max_turn = 6
        conv_hist = []
        errors = 0
        await _get_phone_call_event('<notts>')
        while turns < max_turn and errors < 3:
            logger.info(f'>>>>>>Turn {turns}')
            await _get_phone_call_event(f'\n>>>>>>Turn {turns}\n')
            yield f'\n>>>>>>Turn {turns}\n'
            if conv_hist:
                conv_text = '\n'.join([f'**用户**: {user_input}\n**{target}**: {shop_reply}' for user_input, shop_reply in conv_hist])
            else:
                conv_text = 'Null'
            logger.info(f'>>>>>>History:\n{conv_text}')
            prompt = f'### User profile：\n```\n{profile}\n```\n\n### Task\n- 想象你是某个智能的用户助手，`User profile`中包含该用户的相关信息，用户想要完成如下任务：“{task_description}”\n- {target}可以完成该任务，你需要打电话给{target}向他提供需求，地址，联系方式和其他必要信息，并且选择方案\n- `Conversation`是你和{target}之间已经发生的对话，请你想象接下来你要说什么，续写对话，注意要符合`Constraints`中的要求\n\n### Constraints\n- 基于`Conversation`续写对话，只输出下一轮你要说的话，尽量简短，不要预测{target}的回复\n- 可以使用`User profile`中的相关信息，如果{target}询问`User profile`中没有提到的信息，请回复不清楚或者以后确认\n- 不要问太多无关信息，不要闲聊，尽快结束对话，向{target}提供必要信息和作出方案选择之后就可以结束对话\n- 如果要结束对话，请简要总结全部`Conversation`并提取关键信息，按以下格式回复：“结束对话，对话的总结和关键信息为......”\n\n### Conversation\n{conv_text}\n**用户助手**: '
            user_input = ''
            last_one = False
            await _get_phone_call_event('>>>>>>用户: ')
            yield '>>>>>>用户: '
            text = ''
            url = 'https://api.lingyiwanwu.com/v1/chat/completions'
            headers = {
                'Authorization': 'Bearer multimodel-peter',
                'Content-Type': 'application/json',
                'x-infra-sp-strategy': 'infra-sp-prd',
                'Cookie': 'acw_tc=7b39758317205914164664499e98ccd41d75333b90e0bccf5d13466813fb8b'
            }
            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "model": "gpt-4o",
                "stream": True,
                "temperature": 0.3,
                "max_tokens": 2000,
                "top_p": 0.8,
                "stream_options": {
                    "include_usage": True
                }
            }      
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=data, ssl=False) as response:
                        if response.status == 200:
                            print(f"Successfully fetched stream. Status: {response.status}")
                            async for line in response.content:
                                chunk = line.decode('utf-8').strip()
                                if chunk.startswith('data: '):
                                    chunk = chunk[6:]
                                if chunk and chunk != '[DONE]':
                                    try:
                                        parsed = json.loads(chunk)
                                        content = parsed['choices'][0]['delta'].get('content', '')
                                        if content:
                                            user_input += content
                                            await _get_phone_call_event(content)
                                            yield content
                                    except json.JSONDecodeError:
                                        print(f"无法解析JSON: {chunk}")
                        else:
                            print(f"Failed to fetch stream. Status: {response.status}")
                            yield ''
            except Exception as e:
                print(f'*********** Calling LLM failed: {e} ***************')
                yield ''

            logger.info(f'>>>>>>用户: {user_input}')
            if '结束对话' in user_input:
                break
            shop_msg = {
                'content': user_input,
                'conv_hist': conv_hist
            }
            shop_reply = ''
            try:
                shop_resp = requests.post(shop_url, json=shop_msg, stream=True)
                await _get_phone_call_event('\n>>>>>>店铺回复: ')
                yield '\n>>>>>>店铺回复: '
                if shop_resp.status_code == 200:
                    for chunk in shop_resp.iter_content(chunk_size=None):
                        if chunk:
                            chunk_data = json.loads(chunk.decode('utf-8'))
                            if chunk_data['content'] == 'error':
                                logger.error(f'Error reply from shop: {chunk_data}')
                                errors += 1
                                break
                            delta = chunk_data['delta']
                            await _get_phone_call_event(delta)
                            shop_reply += delta
                            yield delta
            except Exception as e:
                logger.error(f'Error when generating shop reply: {e}')
                errors += 1
                shop_reply = '不好意思发生了系统错误，请重复一下你的问题'
            logger.info(f'>>>>>>店铺回复: {shop_reply}\n')
            conv_hist.append([user_input, shop_reply])

            turns += 1
        shop_msg = {
            'content': '**finish**'
        }
        await _get_phone_call_event('</notts>')
        shop_resp = requests.post(shop_url, json=shop_msg)
        
    response_chunks = []
    async for chunk in generate_convs():
        response_chunks.append(chunk)
    # Join and return the accumulated response
    return ''.join(response_chunks)


@tool
def repl_tool(python_code):
    """A Python shell. Use this to execute python commands. Input should be a valid python command. 
    You can use this tool in the following situations:
    1. Data processing and analysis: When users need to process, analyze, or compute data they provide, such as calculating averages, generating statistical charts, or handling time series data.
    2. Complex mathematical calculations: When users need to perform complex mathematical or scientific calculations (like linear algebra, calculus, or probability statistics), the Python interpreter provides the necessary computational power.
    If you want to see the output of a value, you should print it out with `print(...)`.
    Parameters:
        python_code (str): The valid python code generated by LLM, which should solve the original problem, only print the useful results.
    Returns:
        res(str): The printed useful infomation in python code
    """
    res = python_repl.run(python_code)
    return res

@tool
def multiply_two_numbers(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@tool
def multiply_all_numbers(numbers) -> float:
    """multiply all numbers。计算所有数字的乘积。"""
    product = 1
    for num in numbers:
        product *= num
    return product

@tool
def sum_all_numbers(numbers) -> float:
    """Add all numbers。计算所有数字之和。"""
    return sum(numbers)

@tool
def minus(a: float, b: float) -> float:
    """Subtract two numbers. return a - b"""
    return a - b

@tool
def divide_two_numbers(a: float, b: float) -> float:
    """Divide two numbers. return a / b"""
    return a / b

@tool
def image_gen(prompt:str, shape:str):
    """
    Invoke this func when user provding a text prompt and want to create a new image according to the prompt
    return the generated image with url format.
    Parameters:
        prompt (str): The prompt is the input given to an image generation model, derived from the user's initial description. It should be expanded into a detailed and vivid depiction, using several descriptive phrases in English, usually around 50 words. The goal is to enrich the original description with relevant details that help the model create a more accurate and visually appealing image, while avoiding any unrelated information. English prompt is required! Please use English prompt.
        shape (str): the shape of the generated image, should be "vertical", "square" or "horizontal"
    """
    logger.info(f"*****************input for image gen*************\n {prompt}")
    try:
        model_name = 'flux'
        url = "http://172.25.248.111:8977/txt2img"
        shape_map = {
                    "vertical": "768x1344",
                    "square": "1024x1024",
                    "horizontal": "1344x768"
                    }
        resolution = shape_map[shape]
        payload = {
            "image_format": "jpg",
            "prompt": prompt,
            "negative_prompt": '',
            "size": resolution, 
            "n": 1,
            "response_format": 'tosUrl',
            "image_seed": random.randint(1,100),
        }
        
        response = requests.post(url, json=payload, timeout=100)
        logger.info(response)
        if response.status_code==200:
            # 反序列化数据
            response_data = response.json()
            url = response_data['data'][0].split('?')[0]
            if response_data['success']:
                logger.info(response_data)
                dispatch_custom_event(
                "dir_out_event",
                {
                    "function_name": "image_edit",
                    "arguments": {"input_question": ""},
                    "tool_output_token": f"这是画好的图片<notts>![GEN_IMAGE]({url})</notts>",
                },
                config={"tags": ["tool_call"]},
                )
                #return f"这是画好的图片地址，请返回给用户:\n![GEN_IMAGE]({response_data['data'][0]})"  #only gen one image for now
            else:
                return False
        else:
            return False
            
    except Exception as e:
        traceback.print_exc()
        logger.info(f'image gen failed with error {e}')
        return False

@tool
async def call_reatime_llm(input_question: str) -> str:
    """
    Online retrieve or search the input question which requires real-time information, such as weather, web search information, current date, etc.
    When the question is meaningful and you can't answer the question by LLM, you can try to use this function.
    例如，今天北京天气怎么样？今天是几号？距离过年还有多少天？介绍一下零一万物 等问题。
    Parameters:
        input_question (str): the question to be answered
    Returns:
        Organized retrieved information
    """
    stream = True
    url = 'https://api.lingyiwanwu.com/v1/chat/completions'
    headers = {
        'Authorization': 'Bearer multimodel-peter',
        'Content-Type': 'application/json',
        'x-infra-sp-strategy': 'infra-sp-prd',
        'Cookie': 'acw_tc=7b39758317205914164664499e98ccd41d75333b90e0bccf5d13466813fb8b'
    }
    data = {
        "messages": [
            {
                "role": "user",
                "content": input_question
            }
        ],
        "model": "yi-132b-sft-16k-v2-rag",
        "stream": stream,
        "temperature": 0.3,
        "max_tokens": 2000,
        "top_p": 0.8,
        "stream_options": {
            "include_usage": True
        }
    }
    print('search question:', input_question)

    async def fetch_stream():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data, ssl=False) as response:
                    if response.status == 200:
                        print(f"Successfully fetched stream. Status: {response.status}")
                        async for line in response.content:
                            chunk = line.decode('utf-8').strip()
                            if chunk.startswith('data: '):
                                chunk = chunk[6:]
                            if chunk and chunk != '[DONE]':
                                try:
                                    parsed = json.loads(chunk)
                                    content = parsed['choices'][0]['delta'].get('content', '')
                                    if content:
                                        await adispatch_custom_event(
                                            "call_realtime_llm",
                                            {
                                                "function_name": "call_realtime_llm",
                                                "arguments": {"input_question": input_question},
                                                "tool_output_token": content,
                                            },
                                            config={"tags": ["tool_call"]},
                                        )
                                        yield content
                                except json.JSONDecodeError:
                                    print(f"无法解析JSON: {chunk}")
                    else:
                        print(f"Failed to fetch stream. Status: {response.status}")
                        yield ''
        except Exception as e:
            print(f'*********** Calling LLM failed: {e} ***************')
            yield ''
    chunks = [chunk async for chunk in fetch_stream()]
    return "".join(chunks)

def extract_serpapi_data(data):
    """
    从给定的JSON数据中提取知识图谱和图片结果的信息，包括标题和文本，并去掉所有link字段。

    参数:
    json_data (dict): 包含搜索结果的JSON数据。

    返回:
    dict: 提取的信息，包括知识图谱和图片结果。
    """
    def clean_item(item):
        clean = {}
        for k, v in item.items():
            if k=='position' or k == 'source':continue
            if k =='images' or k == 'duration':continue
            if k =='thumbnail_width' or k=='thumbnail_height':continue
            if not isinstance(v, str) or not v.startswith("http"):
                if isinstance(v, dict):
                    clean[k] = clean_item(v)
                elif isinstance(v, list):
                    clean[k] = [clean_item(sub_item) if isinstance(sub_item, dict) else sub_item for sub_item in v]
                else:
                    clean[k] = v
        return clean

    def clean_section(section):
        return [clean_item(item) for item in section]

    cleaned_data = {}
    for key, value in data.items():
        if isinstance(value, list):
            cleaned_data[key] = clean_section(value)
    cleaned_data['visual_matches'] = cleaned_data['visual_matches'][:5]
    return cleaned_data

def get_img_url_for_other_tool(state: Annotated[dict, InjectedState]):
    """
    获取当前请求涉及到的图片URL的工具，当你发现用户的请求意图需要调用如图片编辑，图片美颜，图搜索等工具的时候。
    需要用户原始图片URL给函数进行传参的时候需要先调用此函数获取URL,然后把URL传参给其他工具
    Parameters:
        state (str): the state of the current request
    Returns:
        the url of the request input image. 
    """
    image_url=''
    found = False
    for message in state['messages']:
        if isinstance(message, HumanMessage):  # 检查是否是 HumanMessage
            # 获取 HumanMessage 的 content 属性
            message_content = message.content
            
            # 检查 content 是否为列表，且首个元素是否包含图片 URL
            if isinstance(message_content, list) and len(message_content) > 0:
                first_item = message_content[0]
                if 'type' in first_item and first_item['type'] == 'image_url':
                    image_url = first_item['image_url']['url']
                    logger.info(f"图片 URL: {image_url}")
                    found = True
    if not found:        
        logger.info("消息中没有图片 URL")
    return image_url

@tool
def reverse_image_search(state: Annotated[dict, InjectedState], timeout=20) -> str:
    """
    Use this tool to get search results from Google Lens pages when users ask for the location, place, or related content of an image or information about an image. 

    Perform a reverse image search based on the URL of an image, using the Google Lens API.
    Search results may include image matches, related videos, text, and other data.

    Parameters:
        state (dict): This parameter is passed in by a system message and does not require a model-generated field.
    Returns:
        the searched related information of the image.
    """
    url = "https://serpapi.com/search"
    
    # 从 state 中获取图片 URL，检查嵌套的内容结构
    try:
        image_url = get_img_url_for_other_tool(state)
    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"提取图片 URL 失败: {e}")
        return "未找到图片 URL"
    
    if not image_url:
        logger.error("未找到图片 URL")
        return "未找到图片 URL"
    
    try:
        logger.info(f"反向图片搜索: {image_url}")
        params = {
            "engine": "google_lens",
            "url": image_url,
            "api_key": '5c9469e3f69c2f24a6edbcbea0f699342f8dd7b44d80fa7774d11af4ab77a803'
        }
    except Exception as ex:
        logger.error(f"参数设置失败: {ex}")
        return ''
    
    try:
        response = requests.get(url, params=params, timeout=timeout)
        
        if response.status_code == 200: 
            extracted_info = extract_serpapi_data(response.json())
            return json.dumps(extracted_info, indent=2, ensure_ascii=False)
        else:
            logger.error(f"请求失败，状态码: {response.status_code}")
            return ''
    except requests.Timeout:
        logger.error("请求超时")
        return ''
    except requests.RequestException as e:
        logger.error(f"请求失败: {e}")
        return ''



# 通用的图像处理 API 请求函数
def _process_image(image_url, api_url, instruction=None):
    payload = {
        "input_img_type": "url",
        "input_img": image_url,
        "response_format": "url_json"
    }
    if instruction:
        payload['prompts'] = instruction
    return _call_api(api_url, payload)

# 图像增强/优化等操作的通用函数
def image_process(state, api_url, instruction=None, success_message="这是处理后的图片", failure_message="图片处理失败，请重试"):
    image_url = get_img_url_for_other_tool(state)
    result = _process_image(image_url, api_url, instruction)
    if result and 'data' in result:
        url = result['data'][0] if isinstance(result['data'], list) else result['data']
        message = f"{success_message}<notts>![PROCESSED_IMAGE]({url})</notts>"
    else:
        message = failure_message
    _dispatch_event("image_process", message)

# 各类图像处理函数的调用
@tool
def image_edit(state: Annotated[dict, InjectedState], instruction):
    """
        You can use this function when users ask to edit or modify the image. 
        It is particularly useful for tasks such as removing specific objects from an image or replacing the background with a different scene.

        Parameters:
            instruction (str): The text instruction describing the modification to be applied to the image.
                            Examples include:
                            - "remove the cup on the table in the image"
                            - "replace the background with a beautiful beach"
    """
    
    # API endpoint
    url = "http://172.25.250.8:8669/edit_image/"
    image_url = get_img_url_for_other_tool(state)
    logger.info(f'image_edit got input url: {image_url} and Instruction: {instruction}')
    
    # Load your image
    if image_url.startswith('http'):
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        image_bytes = image_response.content
    else:
        with open(image_url, "rb") as f:
            image_bytes = f.read()
    
    # Send the request
    response = requests.post(
        url,
        files={"file": ("image.jpg", image_bytes, "image/jpeg")},
        data={"instruction": instruction},
    )

    # Check if the request was successful
    if response.status_code == 200:
        content_str = response.content.decode('utf-8')
        content_dict = json.loads(content_str)
        url = content_dict['data']
        logger.info(f"image edit generated url:{url}")
        message = f"这是编辑后的图片<notts>![EDITED_IMAGE]({url})</notts>"
    else:
        logger.info(f"Failed to edit the image. Status code: {response.status_code}")
        message = "图片编辑失败，请重试"
    _dispatch_event("image_edit", message)

@tool
def image_beautify(state: Annotated[dict, InjectedState]):
    """
    如果用户希望对图中的人物进行美颜、磨皮，或者提到让图片中的人物更加好看等，则调用该函数
    """
    image_url = get_img_url_for_other_tool(state)
    logger.info(f"image_beautify called with image_url: {image_url}")
    url = 'http://172.25.248.128:8284/mopi'
    # response_format = 'b64_json'  # images will return in base64 strings
    response_format = 'url_json'  # images will return in TOS URLs

    payload = {
        "input_img_type": 'url',
        'input_img': image_url,
        'steps': 3,
        'cfg': 6,
        'response_format': response_format,
    }
    response = requests.post(url, data=json.dumps(payload))

    # Check if the request was successful
    if response.status_code == 200:
        try:
            data = response.json()  # This converts the response to a Python dictionary
            tos_urls_list = data['data']  # image TOS URLs list
            url = tos_urls_list[0]
            formatted_url = f"这是美颜后的图片<notts>![BEAUTIFIED_IMAGE]({url})</notts>"
        except json.JSONDecodeError:
            logger.info("image_beautify response content is not in JSON format.")
            formatted_url = '图片编辑失败，请重试'
    else:
        logger.error(f"image_beautify request failed with status code {response.status_code}")
        formatted_url = '图片编辑失败，请重试'

    _dispatch_event("image_beautify", formatted_url)


def _call_api_super_resolution(image_url):
    url = 'http://180.184.134.254:8289/superresolution'
    # 创建请求数据
    response_format = "url_json"
    payload = {
        "image_format": "jpg",
        "input_img_type": "url",
        "input_img": image_url,
        "quality": "standard",
        "response_format": response_format,
        "size": "1344x768",
        "image_seed": 666,
        "use_compel": False
    }
    response = requests.post(url, data=json.dumps(payload))
    if response.status_code == 200:
        try:
            data = response.json()  # This converts the response to a Python dictionary
        except json.JSONDecodeError:
            logger.info("_call_api_super_resolution response content is not in JSON format.")
            return ''
    else:
        logger.error(f"_call_api_super_resolution request failed with status code {response.status_code}")
        return ''
    
    if response_format == 'url_json':
        tos_url = data['data'][0] if data['success'] else ''
        return tos_url
    else:
        logger.error(f"_call_api_super_resolution response format is not supported: {response_format}. Return image TOS URL instead.")
        tos_url = data['data'][0] if data['success'] else ''
        return tos_url
@tool
def image_enhance_clarify(state: Annotated[dict, InjectedState], instruction):
    """
    如果用户明确提到需要进行去噪、去雾、去模糊，或者希望图片更加清晰、放大图片，则调用该函数
    Parameters:
        instruction (str): the instruction for image enhancement . 
    """
    image_url = get_img_url_for_other_tool(state)
    logger.info(f"image_enhance_clarify called with image_url: {image_url} and instruction: {instruction}")
    prompts = [
        'dehaze',
        'denoise',
        'deblur'
    ]
    
    result = _process_image(image_url, "http://180.184.134.254:8285/restoration", instruction=prompts)
    if result and 'data' in result:
        image_url = result['data'][0] if isinstance(result['data'], list) else result['data']
    if image_url == '':
        _dispatch_event("image_enhance_clarify", f"图片处理失败，请重试")
        return 
    logger.info(f"do image restoration at first, result image url: {image_url}")
    url = _call_api_super_resolution(image_url)

    if url == '':
        _dispatch_event("image_enhance_clarify", f"图片处理失败，请重试")
        return 
    _dispatch_event("image_enhance_clarify", f"这是清晰化后的图片<notts>![CLARIFIED_IMAGE]({url})</notts>")

@tool
def image_enhance(state: Annotated[dict, InjectedState]):
    """
    如果用户希望对图片进行美化（不是美颜）或者增强、修图、优化等，则调用该函数
    """
    prompts = ["make it better", "dehaze", "denoise", "deblur"]
    image_process(state, "http://180.184.134.254:8285/restoration", instruction=prompts)

@tool
def image_enhance_brighten(state: Annotated[dict, InjectedState], instruction):
    """
    如果用户需要提高图片的亮度，或者希望过暗的照片恢复正常，则调用该函数
    Parameters:
        instruction (str): 需要提高图片亮度的指令
    """
    prompts = ["The photo is too dark, improve exposure", "denoise"]
    image_process(state, "http://180.184.134.254:8285/restoration", instruction=prompts)

@tool
def image_transfer_recolor(state: Annotated[dict, InjectedState]):
    """
    如果用户提供一张黑白照片或者老旧照片，希望给照片上色，则调用该函数
    """
    image_process(state, "http://172.25.248.128:8293/recolor", success_message="这是上色后的图片")

    
def send_to_memory_cache(urls_to_send, user_id, start_time, headers):
    if not urls_to_send:  # 如果要发送的列表为空，则不做任何处理
        return

    memory_data = {
        "frame_urls": urls_to_send,  # 只发送选中的部分
        "start_time": start_time,
        "exp_id": f"{user_id}"
    }
    for retry in range(3):
        try:
            logger.info(f"Try to send to memory cache {memory_data}")
            response_mem = requests.post('http://14.103.17.32:7011/add_seg_to_memory', headers=headers, data=json.dumps(memory_data), timeout=200)
            status = response_mem.json()
            if status == 'success':
                logger.info(f"Send to memory cache success: {status}")
                break
        except Exception as e:
            traceback.print_exc()
            logger.error(f"retrying {retry} failed with Error: {e}")

def _record_video(user_id, record_time):
    try:
        frame_interval = 1
        start_time = int(time.time())
        data = {
            "client_user_id": user_id,
            "message": "",  # Summary for some frames, optional
            "additional_data": {
                "frames": 0,
                "start_time": start_time,
                "current_time": int(time.time()),
                "lastOne": False,  # Whether it's the last frame or not
                "role": "agent",
                "image_url": ""
            }
        }
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        frames = record_time // frame_interval
        url_list = []

        for frame in range(frames):
            user_state = session_cache.get(user_id, None)
            if user_state and user_state.get('state', None)=='stop':
                logger.info(f"Recording video for user {user_id} stopped")
                break
                
            # Update the image URL with the current frame number
            current_url = f"https://multimodal-arch.tos-cn-shanghai.volces.com/user_generated_image_donotremove/user_id_{user_id}_start_time_{start_time}_frame{frame}.jpeg"
            with lock:
                url_list.append(current_url)

            # Update the current time and frame
            data["additional_data"]["current_time"] = int(time.time())
            data["additional_data"]["frames"] = frame
            # Check if it's the last frame
            data["additional_data"]["lastOne"] = frame == frames - 1
            data["additional_data"]["image_url"] = current_url
            
            if frame % 8 == 0:
                first_frame_time = data["additional_data"]["current_time"]
                # 将时间戳转换为 datetime 对象
                dt_object = datetime.fromtimestamp(first_frame_time)

                # 格式化 datetime 对象为 "YYYYMMDD-HH:MM:SS"
                formatted_first_frame_time = dt_object.strftime('%Y%m%d-%H:%M:%S')

            # Send the request to the WebRTC service to extract and upload one frame
            with lock:
                try:
                    response = requests.post(webrtc_service_url + '/v1/rtc/frontend/message', data=json.dumps(data), headers=headers, timeout=2)
                    status = response.json()
                    if status.get('message', None) == 'success':
                        if frame % 1 == 0:
                            logger.info(f"user_id {user_id} send to webrtc frame {frame} success: {status}")
                    else:
                        logger.error(f"Send to webrtc failed: {status}")
                except Exception as e:
                    logger.error(f"Error: {e}; response status {status}")
                    continue

            # Check if we should send to memory cache
            if len(url_list) % 8 == 0:
                # 如果长度是8的倍数，取出最后8个元素
                urls_to_send = url_list[-8:]
                thread_mem = threading.Thread(target=send_to_memory_cache, args=(urls_to_send, user_id, formatted_first_frame_time, headers))
                thread_mem.start()

            elif data["additional_data"]["lastOne"]:
                # 如果是最后一帧，取出不满8个的余数帧
                remaining_frames = len(url_list) % 8
                if remaining_frames > 0:
                    urls_to_send = url_list[-remaining_frames:]  # 取出最后几帧
                    thread_mem = threading.Thread(target=send_to_memory_cache, args=(urls_to_send, user_id, formatted_first_frame_time, headers))
                    thread_mem.start()

            # Sleep for the specified frame_interval before sending the next request
            time.sleep(frame_interval)
    except:
        logger.error(f"Error: {traceback.format_exc()}")
    finally:
        session_cache.remove(user_id)
        logger.info(f"Recording video for user {user_id} finished")
        
@tool
def start_record_video(user_id, record_time: int=60):
    """
    When user want to record a video or meeting, you can use this tool.
    例如:帮我录一个3分钟的视频；录制一个会议的视频
    Parameters:
        user_id: 用户id; the user id who want to record the video.
        record_time: the time (with unit seconds) to record the video, if not set, the default value is 60s.
    """
    user_state = session_cache.get(user_id, None)
    if user_state:
        user_state.set('state','stop')
        logger.info('stop previous recording before start a new one')
        time.sleep(2)
        
    # 创建并启动后台线程
    thread = threading.Thread(target=_record_video, args=(user_id, record_time))
    user_state = ThreadSafeDict()
    user_state.set('state', 'recording')
    session_cache.set(user_id, user_state)
    thread.start()
    _dispatch_event("record_video", f"开始录制视频，时长{record_time}秒")

    
@tool
def stop_record_video(user_id):
    """
    When user want to stop recording a video or meeting, you can use this tool.
    例如:停止录制；好了别录了；结束录制
    Parameters:
        user_id: 用户id; the user id who want to stop recording the video.
    """
    # 创建并启动后台线程

    user_state = session_cache.get(user_id, None)
    if user_state:
        user_state.set('state','stop')
        try:
            data = {
                "client_user_id": user_id,
                "message": "",  # Summary for some frames, optional
                "additional_data": {
                    "current_time": int(time.time()),
                    "lastOne": True,  # Whether it's the last frame or not
                    "role": "agent",
                    "image_url": ""
                }
            }
            headers = {
                'accept': 'application/json',
                'Content-Type': 'application/json'
            }

            response = requests.post(webrtc_service_url + '/v1/rtc/frontend/message', data=json.dumps(data), headers=headers, timeout=2)
        except Exception as e:
            traceback.print_exc()
            logger.error(f"failed with Error: {e}")
        
        message = f"停止录制视频"
    else:
        logger.info(f"User {user_id} is not recording video")
        message = f"没有检测到你在录制视频"
    _dispatch_event("stop_record_video", message)


# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["direct_out_tools","common_tools", END]:
    next_node = tools_condition(state)
    if next_node == END:
        return END
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    first_tool_call = last_message.tool_calls[0]
    if first_tool_call['name'] in direct_out_tools_name:
        return "direct_out_tools"
    return "common_tools"


# Define the function that calls the model
async def call_model(state: MessagesState, config: RunnableConfig):
    messages = state["messages"]
    # Note: Passing the config through explicitly is required for python < 3.11
    # Since context var support wasn't added before then: https://docs.python.org/3/library/asyncio-task.html#creating-tasks
    logger.info(f"Calling model with messages: {messages}")
    response = await model_with_tools.ainvoke(messages, config)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

def agent_workflow(tools, checkpointer_path="./checkpoints.sqlite"):
    workflow = StateGraph(MessagesState)
    tool_node = ToolNode(tools)
    dirct_out_tools_node = ToolNode(direct_out_tools)
    common_tools_node = ToolNode(common_tools)
    workflow.add_node("direct_out_tools", dirct_out_tools_node)
    workflow.add_node("common_tools", common_tools_node)

    workflow.add_node("agent", call_model)
    # workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    # workflow.add_edge("tools", 'agent')
    workflow.add_edge(
        "direct_out_tools",
        END
    )
    workflow.add_edge(
        "common_tools",
        "agent"
    )

    if not checkpointer_path:
        logger.info('Init workflow without checkpointer')
        app = workflow.compile()
        return app
    if checkpointer_path.lower().strip() == "memory":
        logger.info('Init workflow with memory checkpointer')
        checkpointer = MemorySaver()
    else:
        logger.info('Init workflow with sqlite checkpointer')
        from sqlite import SqliteSaver
        conn = sqlite3.connect(checkpointer_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
    app = workflow.compile(checkpointer=checkpointer)
    return app

llm = ChatOpenAI(
    openai_api_base="https://api.lingyiwanwu.com/v1", 
    openai_api_key="multimodel-peter", 
    model="gpt-4o",
    stream=True,
    max_retries=3,
    timeout=60
    )
model = llm

common_tools = [
    multiply_two_numbers, multiply_all_numbers, sum_all_numbers, divide_two_numbers, minus, repl_tool, reverse_image_search, call_reatime_llm, make_phone_call, retrieve_memory
]

direct_out_tools = [
    image_edit, image_beautify, image_enhance, image_enhance_clarify, image_enhance_brighten, image_transfer_recolor, image_gen, start_record_video, stop_record_video
]

direct_out_tools_name = {t.name for t in direct_out_tools}
tools = common_tools + direct_out_tools
model_with_tools = model.bind_tools(tools)


if __name__ == '__main__':
    image_url = "https://test-content-public.tos-cn-shanghai.volces.com/agent/others/%E5%B7%B4%E5%8E%98%E5%B2%9B3.jpeg"
    image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

    # result = bing_visual_search(image_url)
    # quit()
    message = {"messages": [HumanMessage(content="计算下面所有数字之和：4654,135465,4234213,7864654,15343545。请调用工具确保计算正确。")]}    
    message = {"messages": [HumanMessage(content=[
        {"type": "text", "text": "图片中所在国家的首都今天天气怎么样？请一步一步推理，有些步骤可以调用工具，从而确保最终答案正确。"},
        {
            "type": "image_url",
            # "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            "image_url": {"url": image_url}
        },
    ]), ]}
    
    message = {"messages": [HumanMessage(content=[
        {"type": "text", "text": "计算这个数学公式？请一步一步推理，有些步骤可以调用工具，从而确保最终答案正确。"},
        {
            "type": "image_url",
            # "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            "image_url": {"url": 'https://test-content-public.tos-cn-shanghai.volces.com/agent/others/complex_cal2.png'}
        },
    ]), ]}


    final_state = agent_workflow(message=message, tools=tools)
    final_content = final_state["messages"][-1].content
    logger.info(final_state, indent=4)
    logger.info("\nfinal_content:\n", final_content, "\n")
    
