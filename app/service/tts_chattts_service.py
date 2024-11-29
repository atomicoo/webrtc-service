#! encoding: utf-8
import time
import typing
import requests
from service.tts_base import TTSBase
from util.logger import logger


class TTSChatttsService(TTSBase):
    def __init__(self):
        self.base_url = 'http://172.25.248.128:8090'

    def synthesis(self, text: str):
        '''
        Synthesize the text and return the audio file
        '''
        raise NotImplementedError('synthesize method is not implemented')

    def synthesis_stream_with_callback(self, text: str, callback: typing.Callable[[bytes, bool], None]):
        '''
        Synthesize the text and return the audio stream
        '''
        if not text:
            callback(b"", True)
            return

        body = {
            "text": text,
            "stream": True,  # streaming mode
            "lang": None,
            "skip_refine_text": True,
            "refine_text_only": False,
            "use_decoder": True,
            "audio_seed": 12345678,
            "text_seed": 87654321,
            "do_text_normalization": True,
            "do_homophone_replacement": False,
            "output_sample_rate": 16000
        }

        params_infer_code = {
            "prompt": "[speed_5]",
            "top_P": 0.1,
            "top_K": 20,
            "temperature": 0.3,
            "repetition_penalty": 1.05,
            "max_new_token": 2048,
            "min_new_token": 0,
            "show_tqdm": True,
            "ensure_non_empty": True,
            "stream_batch": 24,
            "spk_emb": None,
        }
        body["params_infer_code"] = params_infer_code

        start_time = time.time()
        first_chunk_time = 0
        response = requests.post(
            f"{self.base_url}/invoke", json=body, stream=body['stream'], timeout=30)

        if response.status_code == 200:
            index = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    index += 1
                    if index == 1:
                        chunk = chunk[44:]
                    if index == 2:
                        first_chunk_time = time.time()
                    logger.info('put chunk')
                    callback(chunk, False)
            logger.info('finish put chunk')
            callback(b"", True)
        else:
            logger.info(
                f"Request failed with status code {response.status_code}")
            logger.info(response.json())

        logger.info(f"首包延迟: {first_chunk_time - start_time} 秒")

    def synthesis_stream(self, text: str):
        if not text:
            callback(b"", True)
            return

        body = {
            "text": text,
            "stream": True,  # streaming mode
            "lang": None,
            "skip_refine_text": True,
            "refine_text_only": False,
            "use_decoder": True,
            "audio_seed": 12345678,
            "text_seed": 87654321,
            "do_text_normalization": True,
            "do_homophone_replacement": False,
            "output_sample_rate": 24000
        }

        params_infer_code = {
            "prompt": "[speed_5]",
            "top_P": 0.1,
            "top_K": 20,
            "temperature": 0.3,
            "repetition_penalty": 1.05,
            "max_new_token": 2048,
            "min_new_token": 0,
            "show_tqdm": True,
            "ensure_non_empty": True,
            "stream_batch": 24,
            "spk_emb": None,
        }
        body["params_infer_code"] = params_infer_code

        start_time = time.time()
        first_chunk_time = 0
        response = requests.post(
            f"{self.base_url}/stream", json=body, stream=body['stream'], timeout=30)

        if response.status_code == 200:
            with open("generated_audio.wav", "wb") as audio_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        if first_chunk_time == 0:
                            first_chunk_time = time.time()
                        audio_file.write(chunk)
            print("Audio has been saved to 'generated_audio.wav'.")
        else:
            print(f"Request failed with status code {response.status_code}")
            print(response.json())

        print(f"首包延迟: {first_chunk_time - start_time} 秒")


tts_chattts_service = TTSChatttsService()
