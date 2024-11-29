#!encoding: utf-8
from abc import ABCMeta, abstractmethod

class ASRBase(metaclass=ABCMeta):
    '''
    Base class for ASR service
    '''
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def recognize(self, audio_file):
        '''
        Recognize the audio file and return the recognized text
        '''
        raise NotImplementedError('recognize method is not implemented')

    @abstractmethod
    def recongize_stream(self, audio_stream, cache: dict, is_final: bool):
        '''
        Recognize the audio stream and return the recognized text
        '''
        raise NotImplementedError('recognize_stream method is not implemented')
