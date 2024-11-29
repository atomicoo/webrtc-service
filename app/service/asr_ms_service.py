#! encoding: utf-8
from azure.cognitiveservices import speech as speechsdk

from service.asr_base import ASRBase

class ASRMsService(ASRBase):
    '''
    Microsoft ASR service
    '''
    def __init__(self) -> None:
        pass

    def recognize(self, audio_file: str):
        '''
        Recognize the audio file and return the recognized text
        '''
        speech_config = speechsdk.SpeechConfig(subscription='your-subscription-key', region='your-region')
        audio_input = speechsdk.AudioConfig(filename=audio_file)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

        result = speech_recognizer.recognize_once()
        return result.text
    
    def recongize_stream(self, audio_stream: list, cache: dict, is_final: bool):
        '''
        Recognize the audio stream and return the recognized text
        '''
        raise NotImplementedError('recognize_stream method is not implemented')