#!encoding: utf-8

from conf.model_conf import MODEL_DICT
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from service.asr_base import ASRBase
from util.logger import logger


class ASRSenseService(ASRBase):
    '''
    SenseVoice
    '''

    def __init__(self) -> None:
        self.asr_model = AutoModel(
            model=MODEL_DICT['sense-voice-small'],
            vad_model=MODEL_DICT['fsmn-vad'],
            vad_kwargs={"max_single_segment_time": 30000},
        )

    def recognize(self, audio_file):
        res = self.asr_model.generate(
            input=audio_file,
            cache={},
            language="zh",  # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,  #
            merge_length_s=15,
        )
        logger.info(f'raw recognise result: {res}')  # just for debug
        if res[0]['text'].startswith(('<|zh|>', '<|en|>')):
            return rich_transcription_postprocess(res[0]['text'])
        else:
            return ''

    def recongize_stream(self, audio_stream, cache: dict, is_final: bool):
        '''
        Recognize the audio stream and return the recognized text
        '''
        raise NotImplementedError('recognize_stream method is not implemented')


asr_sense_service = ASRSenseService()
