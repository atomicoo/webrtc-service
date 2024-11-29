
from typing import List
from util.logger import logger
from funasr import AutoModel

from conf.model_conf import MODEL_DICT


class VadFsmnService:
    def __init__(self):
        fsmn_vad = MODEL_DICT['fsmn-vad']
        logger.info(f'fsmn_vad: {fsmn_vad}')
        self.model = AutoModel(model=fsmn_vad)

    def vad_block(self, audio_file) -> List:
        '''
        Detect the voice activity in the audio file
        '''
        res = self.model.generate(input=audio_file)
        return res[0]['value']

    def vad_stream(self, audio_chunk, cache: dict, is_final: bool) -> List:
        '''
        Detect the voice activity in the audio stream
        '''
        res = self.model.generate(
            input=audio_chunk, cache=cache, is_final=is_final)
        return res[0]['value']

vad_fsmn_service = VadFsmnService()
