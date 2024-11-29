#!encoding: utf-8

from conf.model_conf import MODEL_DICT
import torch
from funasr import AutoModel
from util.logger import logger


class SVCAMppService:
    '''
    SV CAM++ Service
    '''

    def __init__(self) -> None:
        self.sv_model = AutoModel(model=MODEL_DICT['sv-campplus'])

    def embedding(self, audio_file):
        res = self.sv_model.generate(input=audio_file)
        emb = torch.vstack([res[0]['spk_embedding'], res[0]['spk_embedding']]).mean(dim=0)
        # logger.info(f'speaker embedding: {res.shape}')  # just for debug
        return emb.detach().cpu().numpy()  # return numpy array

    def verify(self, embedding1, embedding2):
        similarity = torch.cosine_similarity(torch.from_numpy(embedding1),
                                       torch.from_numpy(embedding2),
                                       dim=0).item()
        logger.info(f'speaker similarity: {similarity:.4f}')  # just for debug
        return similarity
        """
        from scipy.spatial.distance import cdist
        distance = cdist(embedding1[None, ...], embedding2[None, ...], metric="cosine")[0, 0]
        """


sv_campp_service = SVCAMppService()


if __name__ == "__main__":
    with open(r'/ML-A800/team/mm/zhouzhiyang/WORKSPACE/SERVICE/webrtc-service/pcm/debug/zhouzhiyang/asr_2521_1724834232_29.36_80.60.pcm', 'rb') as pcm:
        audio_bytes = pcm.read()
    embedding1 = sv_campp_service.embedding(audio_bytes)
    with open(r'/ML-A800/team/mm/zhouzhiyang/WORKSPACE/SERVICE/webrtc-service/pcm/debug/zhouzhiyang/asr_2521_1725410570_32.07_72.11.pcm', 'rb') as pcm:
        audio_bytes = pcm.read()
    embedding2 = sv_campp_service.embedding(audio_bytes)
    similarity = sv_campp_service.verify(embedding1, embedding2)
    print(similarity)
