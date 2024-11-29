import torch
import torch.nn.functional as F
from typing import Dict
from silero_vad import load_silero_vad

STREAM_CHUNK_SAMPLE_POINTS: int = 512
THRESHOLD_FOR_SILERO_VAD: float = 0.90


class VADIterator:
    def __init__(self,
                 model,
                 threshold: float = 0.5,
                 sampling_rate: int = 16000,
                 min_trigger_duration_ms: int = 100,
                 min_silence_duration_ms: int = 100,
                 speech_pad_ms: int = 30
                 ):

        """
        Class for stream imitation

        Parameters
        ----------
        model: preloaded .jit/.onnx silero VAD model

        threshold: float (default - 0.5)
            Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
            It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

        sampling_rate: int (default - 16000)
            Currently silero VAD models support 8000 and 16000 sample rates

        min_silence_duration_ms: int (default - 100 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before separating it

        speech_pad_ms: int (default - 30 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side
        """

        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate

        if sampling_rate not in [8000, 16000]:
            raise ValueError('VADIterator does not support sampling rates other than [8000, 16000]')

        self.min_trigger_samples = sampling_rate * min_trigger_duration_ms / 1000
        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):

        self.model.reset_states()
        self.triggered = False
        self.temp_start = -1
        self.temp_end = 0
        self.current_sample = 0

    @torch.no_grad()
    def __call__(self, x, return_seconds=False):
        """
        x: torch.Tensor
            audio chunk (see examples in repo)

        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)
        """

        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()

        if not self.triggered:
            if speech_prob >= self.threshold:
                if self.temp_start == -1:
                    self.temp_start = self.current_sample - window_size_samples
                if self.current_sample - self.temp_start > self.min_trigger_samples:
                    self.triggered = True
                    speech_start = self.temp_start - self.speech_pad_samples
                    return {'start': int(speech_start) if not return_seconds else round(speech_start / self.sampling_rate, 1)}
            else:
                self.temp_start = -1
                return None
        else:
            if speech_prob < self.threshold - 0.15:
                if self.temp_end == 0:
                    self.temp_end = self.current_sample
                if self.current_sample - self.temp_end < self.min_silence_samples:
                    return None
                else:
                    speech_end = self.temp_end + self.speech_pad_samples - window_size_samples
                    self.temp_start = -1
                    self.temp_end = 0
                    self.triggered = False
                    return {'end': int(speech_end) if not return_seconds else round(speech_end / self.sampling_rate, 1)}
            else:
                self.temp_end = 0
                return None


class VadSileroService:
    """ Silero-VAD Service """
    def __init__(self, onnx: bool = False):
        self.vad_iterator = self.get_vad_iterator(onnx=onnx)

    def get_vad_iterator(self, onnx: bool) -> VADIterator:
        # self.vad_model, VADIterator = torch.hub.load(
        #     source="local", repo_or_dir=vad_model_path, model="silero_vad", onnx=True
        # )
        self.vad_model = load_silero_vad(onnx=onnx)
        vad_iterator: VADIterator = VADIterator(self.vad_model,
                                                threshold=THRESHOLD_FOR_SILERO_VAD,
                                                sampling_rate=16000,
                                                min_trigger_duration_ms=100,
                                                min_silence_duration_ms=100)
        return vad_iterator

    def detect_speech(self, audio_chunk: bytes) -> Dict[str, float]:
        audio_tensor: torch.tensor = torch.frombuffer(audio_chunk, dtype=torch.int16)
        audio_tensor: torch.tensor = (audio_tensor / 32768.0).to(dtype=torch.float32)
    # def detect_speech(self, audio_chunk: torch.Tensor) -> Dict[str, float]:
    #     audio_tensor: torch.tensor = torch.tensor((audio_chunk / 32768.0).float(), dtype=torch.float32)
        audio_tensor: torch.tensor = audio_tensor.unsqueeze(0)
        audio_samples: int = audio_tensor.shape[1]
        if audio_samples < STREAM_CHUNK_SAMPLE_POINTS:
            padding_length = STREAM_CHUNK_SAMPLE_POINTS - audio_samples
            padded_audio_tensor = F.pad(audio_tensor, (0, padding_length), "constant", 0)
            vad_event: Dict[str, float] = self.vad_iterator(padded_audio_tensor, return_seconds=True)
            speech_prob: float = self.vad_model(padded_audio_tensor, 16000).item()
        else:
            vad_event: Dict[str, float] = self.vad_iterator(audio_tensor, return_seconds=True)
            speech_prob = self.vad_model(audio_tensor, 16000).item()
        return speech_prob, vad_event

    def reset_states(self):
        self.vad_iterator.reset_states()


vad_silero_service = VadSileroService()


if __name__ == "__main__":
    # from silero_vad import read_audio
    # audio = read_audio(r'example.wav')
    # audio = (audio * 32768.0).to(dtype=torch.int16)
    # for i in range(0, len(audio), 512):
    #     audio_chunk = audio[i : i+512]
    #     speech_prob, vad_event = vad_silero_service.detect_speech(audio_chunk)
    #     print(speech_prob, vad_event)
    with open(r'example.pcm', 'rb') as pcm:
        audio_bytes = pcm.read()
        for i in range(0, len(audio_bytes), 1024):
            audio_chunk = audio_bytes[i : i+1024]
            speech_prob, vad_event = vad_silero_service.detect_speech(audio_chunk)
            print(speech_prob, vad_event)