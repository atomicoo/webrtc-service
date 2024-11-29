#! encoding: utf-8
import requests, json
import wave, io, os
from service.asr_base import ASRBase
from util.logger import logger


def wave_header(frame_input=b"", channels=1, sample_width=2, sample_rate=16000):
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)
    wav_buf.seek(0)
    return wav_buf.read(-1)


class ASRWhisperService(ASRBase):
    '''
    Whisper ASR service
    '''
    def __init__(self) -> None:
        self.base_url = os.getenv('WHISPER_API_URL', 'http://172.25.250.8:9191')

    def recognize(self, audio_bytes: bytes) -> str:
        audio_file = io.BytesIO(wave_header() + audio_bytes)
        files = {'audio_file': ('demo.wav', audio_file)}
        headers = {}#{'Content-Type': 'multipart/form-data'}
        response = requests.post(f'{self.base_url}/asr?output=json', files=files, headers=headers)
        # 确保请求成功
        if response.status_code == 200:
            result = json.loads(response.text)
            language = result['language']
            if language in ['zh', 'en']:
                return result['text']
            else:
                return ''
        else:
            logger.error(f'Error: {response.status_code} {response.text}')
            return ''
    
    def recongize_stream(self, audio_stream: list, cache: dict, is_final: bool):
        '''
        Recognize the audio stream and return the recognized text
        '''
        raise NotImplementedError('recognize_stream method is not implemented')


asr_whisper_service = ASRWhisperService()


if __name__ == "__main__":
    with open(r'example.pcm', 'rb') as pcm:
        audio_bytes = pcm.read()
        asr_text = asr_whisper_service.recognize(audio_bytes)
        print(f"Result: {asr_text}")