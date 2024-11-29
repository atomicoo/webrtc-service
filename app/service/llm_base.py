#!encoding=utf-8
import json
import traceback
from abc import ABCMeta, abstractmethod
from typing import Dict, List

from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk

from util.logger import logger

# 接口重试次数
COMPLETION_RETRY_TIMES = 3


class LLMBase(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, openai_api_base: str, openai_api_key: str, default_headers: Dict[str, str] | None = None):
        self.openai_api_base = openai_api_base
        self.openai_api_key = openai_api_key
        self.client = OpenAI(
            base_url=self.openai_api_base,
            api_key=self.openai_api_key,
            default_headers=default_headers
        )

    @abstractmethod
    def chat_completion(self,
                        model_name: str,
                        messages: List[Dict],
                        temperature: float | None = None,
                        top_p: int | None = None,
                        frequency_penalty: float | None = None,
                        presence_penalty: float | None = None,
                        max_tokens: int | None = None,
                        response_format_type: str | None = None,
                        **kwargs) -> str:
        request_dict = {
            "model": model_name,
            "messages": messages
        }
        if temperature is not None:
            request_dict["temperature"] = temperature
        if top_p is not None:
            request_dict["top_p"] = top_p
        if frequency_penalty is not None:
            request_dict["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            request_dict["presence_penalty"] = presence_penalty
        if max_tokens is not None:
            request_dict["max_tokens"] = max_tokens
        if response_format_type is not None:
            request_dict["response_format"] = {"type": response_format_type}
        logger.debug(
            f"[chat_completion] request_dict: {json.dumps(request_dict, ensure_ascii=False)}")
        content = ""
        for retry in range(COMPLETION_RETRY_TIMES):
            try:
                completion = self.client.chat.completions.create(
                    **request_dict)
                content = completion.choices[0].message.content
                break
            except Exception as e:
                logger.warning("[chat_completion] completion %s failed, retry=%s, trace=%s",
                               model_name, retry, traceback.format_exc())
        else:
            logger.error(
                f"[chat_completion] completion {model_name} failed, request_dict: {json.dumps(request_dict, ensure_ascii=False)}")
        logger.info(f"[chat_completion] response content: {content}")
        return content

    @abstractmethod
    def chat_completion_trunk(self,
                              model_name: str,
                              messages: List[Dict],
                              temperature: float | None = None,
                              top_p: int | None = None,
                              frequency_penalty: float | None = None,
                              presence_penalty: float | None = None,
                              max_tokens: int | None = None,
                              response_format_type: str | None = None,
                              **kwargs) -> Stream[ChatCompletionChunk]:
        '''
        流式返回，return 一个 genereator
        '''
        request_dict = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            "timeout": 30
        }
        if temperature is not None:
            request_dict["temperature"] = temperature
        if top_p is not None:
            request_dict["top_p"] = top_p
        if frequency_penalty is not None:
            request_dict["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            request_dict["presence_penalty"] = presence_penalty
        if max_tokens is not None:
            request_dict["max_tokens"] = max_tokens
        if response_format_type is not None:
            request_dict["response_format"] = {"type": response_format_type}
        logger.debug(
            f"[chat_completion_trunk] request_dict: {json.dumps(request_dict, ensure_ascii=False)}")

        for retry in range(COMPLETION_RETRY_TIMES):
            try:
                completion = self.client.chat.completions.create(
                    **request_dict)
                return completion
            except Exception as e:
                logger.warning("[chat_completion_trunk] completion %s failed, retry=%s, trace=%s",
                               model_name, retry, traceback.format_exc())
        else:
            logger.error(
                f"[chat_completion_trunk] completion {model_name} failed, request_dict: {json.dumps(request_dict, ensure_ascii=False)}")
            raise ModelCompletionError(model_name=model_name)


class ModelCompletionError(Exception):
    """
    模型调用错误
    """

    def __init__(self, model_name: str):
        detail = f"completion {model_name} stream failed"
        super().__init__(detail)
