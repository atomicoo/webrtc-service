from conf.model_conf import MODEL_DICT

import torch
import numpy as np
import onnxruntime as ort
# from util.logger import logger


class SsMoss2Service:
    """
    Predictor class for speech seperation using MossFormer2 and ONNX Runtime.
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):

        model_path = MODEL_DICT['mossformer2-2spk']
        # onnx.checker.check_model(onnx.load(f'{model_path}/simple_model.onnx'))
        # ort_session = ort.InferenceSession(f'{model_path}/simple_model.onnx')

        session_options = ort.SessionOptions()
        # session_options.log_severity_level = 1
        # session = ort.InferenceSession(f'{model_path}/simple_model.onnx',
        #                                providers=["CPUExecutionProvider", "CUDAExecutionProvider"],
        #                                sess_options=session_options)
        # logger.info(session.get_providers())
        if device == "cuda":
            self.model = ort.InferenceSession(
                f'{model_path}/simple_model.onnx', providers=["CUDAExecutionProvider"],
                sess_options=session_options,
            )
        elif device == "cpu":
            self.model = ort.InferenceSession(
                f'{model_path}/simple_model.onnx', providers=["CPUExecutionProvider"],
                sess_options=session_options,
            )
        else:
            raise ValueError("Device must be either 'cuda' or 'cpu'")

    def predict(self, mix):
        """
        Predict the separated sources from the input mixture signal.

        Args:
            mix (np.ndarray): Input mixture signal.

        Returns:
            tuple: Tuple containing the separated sources from the input mixture signal.
        """
        if mix.ndim > 1:
            raise ValueError('Only supports 1 channel')
        input_data = np.expand_dims(mix, axis=0).astype(np.float32)
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: input_data})
        output_data = outputs[0]
        signal1, signal2 = output_data[0, :, 0], output_data[0, :, 1]
        # signal1 = signal1 / np.abs(signal1).max() * 0.5
        # signal2 = signal2 / np.abs(signal2).max() * 0.5
        return signal1, signal2

    def separate(self, audio_bytes: bytes) -> bytes:
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_array = (audio_array / 32768.0).astype(np.float32)
        vocal1_array, vocal2_array = self.predict(mix=audio_array)
        vocal1_array = (vocal1_array * 32768.0).astype(np.int16)
        vocal2_array = (vocal2_array * 32768.0).astype(np.int16)
        return (vocal1_array.tobytes(), vocal2_array.tobytes())


ss_moss2_service = SsMoss2Service()


if __name__ == "__main__":
    with open(r'example.pcm', 'rb') as pcm:
        audio_bytes = pcm.read()
    import time; since = time.perf_counter()
    vocal1, vocal2 = ss_moss2_service.separate(audio_bytes)
    print(f'elapsed time: {time.perf_counter() - since}')
    with open(r'vocal1.pcm', 'wb') as pcm1, open(r'vocal2.pcm', 'wb') as pcm2:
        pcm1.write(vocal1)
        pcm2.write(vocal2)
