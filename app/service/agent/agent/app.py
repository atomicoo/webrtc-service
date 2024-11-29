import argparse
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from main import agent_workflow, tools  # 从 main 模块导入所需函数和变量
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import traceback
import sys, os, time, json
# 当前文件路径
current_file_path = os.path.abspath(__file__)

# 找到 "webrtc-service/app" 的路径
target_directory = "app"
target_path = current_file_path.split(target_directory)[0] + target_directory

# 将路径加入到 sys.path
sys.path.append(target_path)
print("target path",target_path)
from util.logger import logger

app = FastAPI()

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8000)
parser.add_argument('--memory_url', type=str, default='')  # set a real url to use memory (http://localhost:6169)
parser.add_argument('--context_checkpointer_path', type=str, default='')  # empty: no checkpointer; "Memory": use memory as checkpointer; a disk path: use the sqlite-based checkpointer
args = parser.parse_args()

# memory configs
get_mem_url = args.memory_url + '/get_memory'
add_mem_url = args.memory_url + '/add_memory'

SYSTEM_PROMPT = {
    'role': 'system',
    'content': 
                "You are a AI sunglass\n"
                "用户问题有可能是问你的也有可能不是，请结合上下文并具体分析用户问题进行判断：如果确定是问你的，请正常回答；如果不好判断，请主动询问用户\n\n"
                "你的回复将会被语音合成模型朗读：请确保不要使用除了逗号、句号、问号和感叹号之外的其他符号；对于需要使用序号的内容，请使用文字序号，例如“一、二、三”或者“首先、其次、最后”，并且不要出现多层序号嵌套。\n"
                "请使用口语化的文字风格，使其听起来更加自然和易于理解。\n"
}

agent = agent_workflow(tools=tools, checkpointer_path=args.context_checkpointer_path)

# class AgentRequest(BaseModel):
#     image_url: str
#     prompt: str
#     user_id: str = '10'

# class AgentResponse(BaseModel):
#     final_content: str

class AgentRequestOpenai(BaseModel):
    messages: list[dict]
    user_id: str 

class AgentResponseOpenai(BaseModel):
    content: str
    lastOne: bool

def get_user_chat_history(user_id, max_conv=5):
    try:
        logger.info(f"try to update chat history for user {user_id} >>>> ")
        memory_url = os.getenv('MEMORY_URL', 'http://172.25.248.128:7169')
        headers = {'content-type': 'application/json'}
        data = {'user_id': str(user_id), 'max_conv': int(max_conv)}
        mem_resp = requests.post(f'{memory_url}/get_hist', headers=headers, json=data, timeout=3)
        if mem_resp.status_code == 200:
            chat_history = mem_resp.json()
            if chat_history == {}:
                return []
            else:
                res = []
                for date in chat_history:
                    for entry in chat_history[date]:
                        res.append({"role": "user", "content": entry["query"]})
                        res.append({"role": "assistant", "content": entry["response"]})
                logger.info(f"update chat history >>>> {res}")
                return res
        else:
            logger.error(f"error response {mem_resp.status_code}")    
            return []
    except:
        traceback.print_exc()
        logger.info(f"get memory failed, take it as empty []")
        return []

def update_system_prompt(user_id, system_prompt):
    try:
        memory_url = os.getenv('MEMORY_URL', 'http://172.25.248.128:7169')
        headers = {'content-type': 'application/json'}
        data = {'user_id': str(user_id)}
        mem_resp = requests.post(f'{memory_url}/get_memory', headers=headers, json=data, timeout=3)
        if mem_resp.status_code == 200:
            chat_history = mem_resp.json()
            if chat_history == {}:
                return system_prompt
            else:
                new_system_prompt = {'role': 'system', 'content': f'当前日期为2024年9月13日\n用户user_id为\"{user_id}\"。\n' + chat_history['system_prompt'] + system_prompt['content']}
                # logger.info(f"update system_prompt >>>> {new_system_prompt['content']}")
                return new_system_prompt
        else:
            logger.error(f"error response {mem_resp.status_code}")
            return system_prompt
    except:
        traceback.print_exc()
        logger.error(f"get system_prompt failed, take it as default one {system_prompt}")
        return  system_prompt 

# def process_input(messages_list):
#     new_message  = []
#     for i, message in enumerate(messages_list):
#         if message['role']=='user':
#             rolesentence = []
#             for content in message['content']:
#                 #logger.info(f"request {request_id}  current content ********{content} ******")
#                 if isinstance(content, str):
#                     rolesentence.append({"type": "text", "text": message['content']})
#                     break
#                 else:
#                     if content['type']=='image_url':
#                         url = content['image_url']['url']
#                         if url !='' and url != None:
#                             content['image_url']['detail'] = content['image_url'].get('detail', None)
#                             rolesentence.append(content)
#                     elif content['type']=='text':
#                         rolesentence.append(content)
#             new_message.append(HumanMessage(content=rolesentence))
#         elif message['role'] == "assistant":
#             new_message.append(AIMessage(content=message['content']))
#         #system prompt
#         elif message['role'] == "system":
#             new_message.append(SystemMessage(content=message['content']))
#         else:
#             raise Exception(f"Unsupported role input {message['role']}")
#         #logger.info(f'{i}>>>>>>>: {new_message}')
#     return {"messages": new_message}


def process_messages_for_memory_and_img(messages, user_id):
    # get memory and update system prompt
    memory = get_user_chat_history(user_id, max_conv=6)
    # logger.info(f'user_id {user_id} memory prompt>>>>>>: {memory}')
    logger.info(f'user_id {user_id} chat content>>>> {messages}')
    updated_sys_prompt = update_system_prompt(user_id, SYSTEM_PROMPT) 
    messages = [updated_sys_prompt] + memory + messages
    
    user_msg = messages[-1]
    imgs = [(content['image_url']['url'], content['image_url'].get('display', False))
            for i in range(len(messages)) 
            for content in messages[i]['content'] 
            if isinstance(content, dict) and content['type'] == 'image_url']  #all images in current messages

    # logger.info("imgs>>>", imgs)
    text, url = '', ''
    ind = -1
    if isinstance(user_msg['content'], str):
        text = user_msg['content']
    else:
        for i in range(len(user_msg['content'])):
            content = user_msg['content'][i]
            if content['type'] == 'text':
                ind = i
                text = content['text']
                break
    if user_msg['role'] != 'user':
        return messages, text

    prompt = text
    if args.memory_url and text:
        mem_data = {
            'text': text,
            'user_id': user_id,
        }
        mem_resp = requests.post(get_mem_url, json=mem_data)
        prompt = mem_resp.json()['new_prompt']
    if imgs:
        logger.info('updating prompt with url')
        #request_id = str(int(time.time())%1000)+str(user_id)
        #prompt = f'当前请求ID:{request_id} \n 用户的请求是语音识别的结果，可能不准，如果不理解的话可以回答我没有听清，不要轻易调用工具\n请回答用户的请求：\n{prompt} \n '
        #request_id_URL_cache[request_id] = [imgs[-1][0]] #For function call input request id map to URL, only get the latest one image
        prompt = f'用户(user_id:{user_id})的请求是语音识别的结果，可能不准，如果不理解的话可以回答我没有听清\n请回答用户(user_id:{user_id})的请求：\n{prompt} \n '
        if imgs[-1][1]:
            url = imgs[-1][0]
    if ind != -1:
        user_msg['content'][ind] = {'type': 'text', 'text': prompt}
    else:
        user_msg['content'] = prompt
    return messages, text, url


@app.post("/process-agent-openai", response_model=AgentResponseOpenai)
async def process_agent(request: AgentRequestOpenai):
    TRY_COUNT=4
    async def stream_generate():
        success = False
        for try_index in range(1, TRY_COUNT+1):
            try:
                start_time = time.time()
                messages, text, url = process_messages_for_memory_and_img(request.messages, request.user_id)
                #messages = process_input(messages)
                #inputs = messages['messages']
                inputs = messages
                if try_index==TRY_COUNT:
                    inputs=[inputs[-1]]
                    logger.info(f"retry failed multiple times, only use last message for agent")
                logger.info(f"final inputs for agent:{inputs}")
                final_content = ''
                async for event in agent.astream_events({"messages": inputs}, version="v2"):
                    kind = event["event"]
                    if kind == "on_chat_model_stream":
                        content = event["data"]["chunk"].content
                        if content:
                            final_content += content
                            # Convert the dictionary to a JSON string
                            yield json.dumps({"content": final_content, "delta": content, "lastOne": False},ensure_ascii=False)
                    elif kind == "on_tool_start":
                        logger.info("\n--")
                        logger.info(f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}")
                    elif kind == "on_tool_end":
                        logger.info(f"Done tool: {event['name']}")
                        logger.info(f"Tool output was: {event['data'].get('output')}")
                        logger.info("--")
                    elif kind == 'on_custom_event':
                        # print(f'custom event: {event["name"]}')
                        if event['name'] in {'dir_out_event'}:
                            content = event['data']['tool_output_token']
                            if content:
                                final_content += content
                                # print(event['data']['tool_output_token'],end="", flush=True)
                                yield json.dumps({"content": final_content, "delta":content, "lastOne": False},ensure_ascii=False)
                                # yield json.dumps({"content": final_content, "delta":content, "lastOne": False})
                            # print(f"Memory event: {event['data']}")
                logger.info(f"Finish request using time {time.time()-start_time} and final output is: {final_content}")
                if args.memory_url:
                    mem_data = {
                        'query': text,
                        'user_id': request.user_id,
                        'response': final_content,
                        'url': url
                    }
                    mem_resp = requests.post(add_mem_url, json=mem_data)
                    
                yield json.dumps({"content": final_content, "delta":'', "lastOne": True},ensure_ascii=False)
                success=True
                break
            except Exception as e:
                traceback.print_exc()
                logger.error(e)
                time.sleep(5)
                logger.info(f"sleeping for 5 seconds before retrying, try={try_index}")
                #raise HTTPException(status_code=400, detail=str(e))
        if not success:
            raise HTTPException(status_code=400, detail="retry failed")
    return StreamingResponse(stream_generate(), media_type="application/json")


# @app.post("/process-agent", response_model=AgentResponse)
# async def process_agent(request: AgentRequest):
#     try:
#         prompt = request.prompt
#         if args.memory_url:
#             mem_data = {
#                 'text': request.prompt,
#                 'user_id': request.user_id,
#             }
#             mem_resp = requests.post(get_mem_url, json=mem_data)
#             prompt = mem_resp.json()['new_prompt']

#         if request.image_url !="":
#             prompt = prompt + "(图片url如下，用于传给需要使用图片的tools: %s)" % request.image_url
#             message = {
#             "messages": [
#                 HumanMessage(
#                     content=[
#                         {"type": "text", "text": prompt},
#                         {"type": "image_url", "image_url": {"url": request.image_url}}
#                     ]
#                 )
#             ]
#             }
#         else:
#             message = {
#             "messages": [
#                 HumanMessage(
#                     content=[
#                         {"type": "text", "text": prompt},
#                     ]
#                 )
#             ]
#             }
#         #logger.info('tools:',tools)
#         # 调用 agent_workflow 函数处理请求
#         logger.info(f'input message: {message}')
#         final_state = agent.invoke(message, config={"configurable": {"thread_id": request.user_id}})
#         logger.info('final state:', final_state)
#         final_content = final_state["messages"][-1].content

#         if args.memory_url:
#             mem_data = {
#                 'text': request.prompt,
#                 'user_id': request.user_id,
#                 'response': final_content
#             }
#             mem_resp = requests.post(add_mem_url, json=mem_data)

#         # 根据 agent_workflow 返回的结果创建响应对象
#         response = AgentResponse(final_content=final_content)
#         return response

#     except Exception as e:
#         # 处理可能的错误并返回 HTTP 400 响应
#         logger.info(e)
#         raise HTTPException(status_code=400, detail=str(e))



# 运行服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
    # curl -X POST "http://localhost:8000/process-agent" -H "Content-Type: application/json" -d '{"image_url": "https://test-content-public.tos-cn-shanghai.volces.com/agent/others/%E5%B7%B4%E5%8E%98%E5%B2%9B3.jpeg", "prompt": "这个地方所在国家的首都今天天气怎么样？请对任务进行拆解一步一步来解决。"}'

