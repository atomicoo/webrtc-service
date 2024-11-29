#! encoding: utf-8
import time
import typing
import requests
from service.tts_base import TTSBase
from util.logger import logger


class TTSCosyService(TTSBase):
    def __init__(self):
        self.base_url = 'http://172.25.248.128:8190'

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
        
        data = {
            'tts': text,
            'role': '中文女',
            'chunk_size': 72,
            'sample_rate': 16000
        }

        start_time = time.time()
        first_chunk_time = 0
        response = requests.post(
            f"{self.base_url}/api/streaming/sft", data=data, stream=True, timeout=30)

        if response.status_code == 200:
            index = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    index += 1
                    if index == 1:
                        chunk = chunk[44:]
                    if index == 2:
                        first_chunk_time = time.time()
                    # logger.info('put chunk')
                    callback(chunk, False)
            logger.info('finish put chunk')
            callback(b"", True)
        else:
            logger.info(f"Request failed with status code {response.status_code}")
            logger.info(response.json())

        logger.info(f"首包延迟: {first_chunk_time - start_time} 秒")

    def synthesis_stream(self, text: str):
        if not text:
            return
        
        data = {
            'tts': text,
            'role': '中文女',
            'chunk_size': 72,
            'sample_rate': 16000
        }

        start_time = time.time()
        first_chunk_time = 0
        response = requests.post(
            f"{self.base_url}/stream", data=data, stream=True, timeout=30)

        if response.status_code == 200:
            with open("generated_audio.wav", "wb") as audio_file:
                for chunk in response.iter_content(chunk_size=320):
                    if chunk:
                        if first_chunk_time == 0:
                            first_chunk_time = time.time()
                        audio_file.write(chunk)
            print("Audio has been saved to 'generated_audio.wav'.")
        else:
            print(f"Request failed with status code {response.status_code}")
            print(response.json())

        print(f"首包延迟: {first_chunk_time - start_time} 秒")


tts_cosy_service = TTSCosyService()


import io
import numpy as np
import soxr
from conf.model_conf import MODEL_DICT
from cosyvoice.model import CosyVoice


def pack_raw(io_buffer: io.BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer


class LocalTTSCosyService(TTSBase):
    def __init__(self):
        model_path = MODEL_DICT['cosy-voice-300m-sft']
        self.tts_model = CosyVoice(model_dir=model_path)

    def synthesis(self, text: str):
        '''
        Synthesize the text and return the audio file
        '''
        raise NotImplementedError('synthesize method is not implemented')

    def synthesis_stream(self, text: str):
        '''
        Synthesize the text and return the audio stream
        '''
        raise NotImplementedError('synthesize_stream method is not implemented')

    def synthesis_stream_with_callback(self, text: str, callback: typing.Callable[[bytes, bool], None]):
        '''
        Synthesize the text and return the audio stream
        '''
        if not text:
            callback(b"", True)
            return

        start_time = time.time()
        first_chunk_time = 0

        generator = self.tts_model.stream_inference_sft(
            tts_text=text, spk_id='中文女', stream=True, stream_chunk_size=72)

        for index, chunk in enumerate(generator, start=1):
            chunk = np.array(chunk['tts_speech']).flatten()
            chunk = np.apply_along_axis(soxr.resample, axis=0, arr=chunk,
                                        in_rate=22050, out_rate=16000,
                                        quality='soxr_hq')
            chunk = (np.clip(chunk, -1.0, 1.0) * 32767).astype('int16')
            chunk = pack_raw(io.BytesIO(), chunk, 16000).getvalue()

            if index == 1:
                first_chunk_time = time.time()

            logger.info('put chunk')
            callback(chunk, False)
        logger.info('finish put chunk')
        callback(b"", True)

        logger.info(f"首包延迟: {first_chunk_time - start_time} 秒")


# tts_cosy_service = LocalTTSCosyService()

