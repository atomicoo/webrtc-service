from conf.model_conf import MODEL_DICT

import torch
from df.enhance import enhance, init_df
from df.io import resample


class DnsDFN2Service:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.df, _ = init_df(MODEL_DICT['denoise-deepfilter-2'], config_allow_defaults=True)
        self.model = self.model.to(device=self.device).eval()

    def denoise(self, audio_bytes: bytes) -> bytes:
        audio_tensor = torch.frombuffer(audio_bytes, dtype=torch.int16)
        audio_tensor = (audio_tensor / 32768.0).to(dtype=torch.float32)
        audio_tensor = resample(audio_tensor, 16000, 48000)
        enhan_tensor = enhance(self.model, self.df, audio_tensor.unsqueeze(0)).squeeze(0)
        enhan_tensor = resample(enhan_tensor, 48000, 16000)
        enhan_tensor = (enhan_tensor * 32768.0).to(dtype=torch.int16)
        return enhan_tensor.detach().cpu().numpy().tobytes()


dns_dfn2_service = DnsDFN2Service()


if __name__ == "__main__":
    with open(r'example.pcm', 'rb') as pcm:
        audio_bytes = pcm.read()
    denoise = dns_dfn2_service.denoise(audio_bytes)
    with open(r'denoise.pcm', 'wb') as pcm:
        pcm.write(denoise)