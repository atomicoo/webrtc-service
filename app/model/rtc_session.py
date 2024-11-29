#! encoding: utf-8

import threading
from enum import StrEnum
from util.logger import logger
from collections import deque


class SessionState(StrEnum):
    '''
    会话状态
    '''
    INIT = 'init'
    CLINETINPUTED = 'client_inputed'
    RECOGNIZING = 'recognizing'
    RECOGNIZED = 'recognized'
    LLMING = 'llming'
    LLMED = 'llmed'
    SYNTHESIZING = 'synthesizing'
    SYNTHESIZED = 'synthesized'
    SERVEROUTPUTED = 'server_outputed'
    FINISHED = 'finished'


class RtcSession(object):
    '''
    RTC 会话类
    '''
    def __init__(self, user_id):
        self.user_id = user_id
        self.state = SessionState.INIT

        self.asr_total_text = ''
        self.asr_audio_buffer = deque()
        self.asr_slide_size = 800000 # 25s

        self.llm_total_text = ''
        self.llm_synth_text = ''

        self.tts_audio_buffer = deque()
        self.rtc_slide_size = 320  # 服务端每次发送的音频片段大小
        self.rtc_rest_slide = bytes() # 每个 chunk 切片之后，剩余部分

        self.llm_image_buffer = deque(maxlen=4)
        self.llm_image_time = 0

        self.thread_cache = deque()  # 存放线程对象，用于打断时杀死线程

        self.session_lock = threading.Lock()

    def reset(self):
        with self.session_lock:

            self.state = SessionState.INIT

            self.asr_total_text = ''
            self.asr_audio_buffer.clear()

            self.llm_total_text = ''
            self.llm_synth_text = ''

            self.tts_audio_buffer.clear()
            self.rtc_rest_slide = bytes()

            # self.llm_image_buffer.clear()
            # self.llm_image_time = 0

            self.thread_cache.clear()

    def set_state(self, state, former_state=None):
        with self.session_lock:
            if former_state:
                if self.state == former_state:
                    self.state = state
                else:
                    logger.error(f'state error, current state: {self.state}, former state: {former_state}')
            else:
                logger.info(f'set state: {state}, current state: {self.state}, former state: {former_state}')
                self.state = state
