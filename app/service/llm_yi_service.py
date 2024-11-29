#! encoding: utf-8

from conf.model_conf import LLM_CONFIG
from typing import Dict, List
from openai import Stream
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from service.llm_base import LLMBase
import requests
import os, json, typing, time

import secrets
import string
characters = string.ascii_letters + string.digits

def create_chat_completion_chunk(delta: str, content: str, lastOne: bool) -> ChatCompletionChunk:
    choice = Choice(delta=ChoiceDelta(content=delta, role='assistant'), index=0)
    unique_id = ''.join(secrets.choice(characters) for _ in range(29))
    chunk = ChatCompletionChunk(id=f"chatcmpl-{unique_id}",
                                choices=[choice], created=int(time.time()),
                                model='gpt-4o', object='chat.completion.chunk',
                                content=content, lastOne=lastOne)
    return chunk

class LLMYiService(LLMBase):
    '''
    Yi LLM service
    '''

    def __init__(self):
        super().__init__(
            LLM_CONFIG["openai_api_base"], LLM_CONFIG["openai_api_key"])

    def chat_completion(self, model_name: str, messages: List[Dict], **kwargs) -> str:
        return super().chat_completion(model_name, messages, **kwargs)

    def chat_completion_trunk(self, model_name: str, messages: List[Dict], **kwargs) -> Stream[ChatCompletionChunk]:
        return super().chat_completion_trunk(model_name, messages, **kwargs)

class LLMAgentService(LLMBase):
    '''
    Yi LLM service
    '''

    def __init__(self):
        self.agent_url = os.getenv('AGENT_URL', 'http://172.25.248.128:8008') 
        self.memory_url = os.getenv('MEMORY_URL', 'http://172.25.248.128:7169') 
        self.context_checkpointer_path = os.getenv('CONTEXT_CHECKPOINTER_PATH', '') 

    def chat_completion(self, model_name: str, messages: List[Dict], **kwargs) -> str:
        raise NotImplementedError
    
    def process_agent(self, messages, user_id, callback=None):
        # Generator for streaming response from FastAPI service
        url = f"{self.agent_url}/process-agent-openai"
        headers = {"Content-Type": "application/json"}
        data = {
            "messages": messages,
            "user_id": str(user_id)
        }
        response = requests.post(url, json=data, headers=headers, stream=True, timeout=100)
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    # Parse the JSON string from the response
                    chunk_data = json.loads(chunk.decode('utf-8'))
                    yield create_chat_completion_chunk(delta=chunk_data['delta'],
                                                       content=chunk_data['content'],
                                                       lastOne=chunk_data['lastOne'])
                if callback and isinstance(callable, typing.Callable):
                    callback()  # Call the callback function if exists
        else:
            raise requests.exceptions.RequestException(f"Error: {response.status_code} - {response.text}")
    
    def chat_completion_trunk(self, model_name: str, messages: List[Dict], **kwargs) -> Stream[ChatCompletionChunk]:
        print('calling agent api >>>>>>>>>>>>>>> ')
        # for chunk in self.process_agent(messages, kwargs['user_id']):
        #     print('chunk >>>>>>>>>>>>>>> ', chunk)
        return self.process_agent(messages, kwargs['user_id'], kwargs.get('callback', None))

llm_yi_service = LLMYiService()
llm_agent_service = LLMAgentService()


if __name__ == '__main__':
    messages = [
        {"role": "user", "content": "你能跟我聊聊天吗？"},
    ]
    generator = llm_yi_service.chat_completion_trunk('gpt-4o', messages)
    for chunk in generator:
        if chunk.lastOne:
            break
        print(chunk.choices[0].delta.content)
