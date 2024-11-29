#!encoding:utf-8
from abc import ABCMeta, abstractmethod

class TTSBase(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def synthesis(self, text: str):
        '''
        Synthesize the text and return the audio file
        '''
        raise NotImplementedError('synthesize method is not implemented')
    
    @abstractmethod
    def synthesis_stream(self, text: str):
        '''
        Synthesize the text and return the audio stream
        '''
        raise NotImplementedError('synthesize_stream method is not implemented')
    
    @abstractmethod
    def synthesis_stream_with_callback(self, text: str, callback):
        '''
        Synthesize the text and return the audio stream
        '''
        raise NotImplementedError('synthesize_stream_with_callback method is not implemented')
