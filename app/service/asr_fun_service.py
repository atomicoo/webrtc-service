#!encoding: utf-8

from funasr import AutoModel
from conf.model_conf import MODEL_DICT
from service.asr_base import ASRBase
from util.logger import logger

STREAM_CHUNK_SIZE = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
STREAM_ENCODER_CHUNK_LOOK_BACK = 4  # number of chunks to lookback for encoder self-attention
STREAM_DECODER_CHUNK_LOOK_BACK = 1  # number of encoder chunks to lookback for decoder cross-attention
STREAM_CHUNK_SAMPLE_POINTS = 960  # 600ms

class ASRFunService(ASRBase):
    '''
    Fun ASR service
    '''

    def __init__(self) -> None:
        self.stream_chunk_size = STREAM_CHUNK_SIZE
        # number of chunks to lookback for encoder self-attention
        self.stream_encoder_chunk_look_back = STREAM_ENCODER_CHUNK_LOOK_BACK
        # number of encoder chunks to lookback for decoder cross-attention
        self.stream_decoder_chunk_look_back = STREAM_DECODER_CHUNK_LOOK_BACK
        logger.info('Loading Fun ASR non stream model')

        paraformer_zh = MODEL_DICT['paraformer-zh']
        logger.info(f'paraformer_zh: {paraformer_zh}')
        paraformer_zh_streaming = MODEL_DICT['paraformer-zh-streaming']
        logger.info(f'paraformer_zh_streaming: {paraformer_zh_streaming}')
        ct_punc = MODEL_DICT['ct-punc']
        logger.info(f'ct_punc: {ct_punc}')
        fsmn_vad = MODEL_DICT['fsmn-vad']
        logger.info(f'fsmn_vad: {fsmn_vad}')

        self.non_stream_model = AutoModel(model=paraformer_zh, vad_model=fsmn_vad, punc_model=ct_punc)
        logger.info('load model success!')
        logger.info('Loading Fun ASR stream model')
        # self.stream_model = AutoModel(model='paraformer-zh-streaming', vad_model='fsmn-vad', punc_model='ct-punc')
        self.stream_model = AutoModel(model=paraformer_zh_streaming, model_revision="v2.0.4")
        logger.info('load model success!')

        logger.info('Loading Fun ASR punctuation restore model')
        self.punc_restore_model = AutoModel(model=ct_punc)
        logger.info('load model success!')


    def recognize(self, audio_file: str):
        '''
        Recognize the audio file and return the recognized text
        '''
        logger.info('Recognizing audio file')
        res = self.non_stream_model.generate(
            input=audio_file, batch_size_s=1)
        logger.info(f'Recognized res: {res}')
        text = res[0]['text']
        return text

    def recongize_stream(self, audio_stream, cache: dict, is_final: bool):
        '''
        Recognize the audio stream and return the recognized text
        '''
        logger.info('Recognizing audio stream')

        res = self.stream_model.generate(input=audio_stream,
                                         cache=cache,
                                         is_final=is_final,
                                         chunk_size=self.stream_chunk_size,
                                         encoder_chunk_look_back=self.stream_encoder_chunk_look_back,
                                         decoder_chunk_look_back=self.stream_decoder_chunk_look_back)
        logger.info(f'Recognized text: {res}')
        return res[0]['text']

    def punctuation_restore(self, text: str):
        '''
        Restore the punctuation of the text
        '''
        if not text:
            return ''
        logger.info('Restoring punctuation')
        res = self.punc_restore_model.generate(text)
        logger.info(f'Restored text: {res}')
        return res

asr_fun_service = ASRFunService()
